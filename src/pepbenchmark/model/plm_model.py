from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import e
import os
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

import numpy as np
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, PeftModel, get_peft_model

from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager
from pepbenchmark.evaluator import (
    Classification_Metric,
    Regression_Metric,
    evaluate_classification,
    evaluate_regression,
)
from pepbenchmark.utils.logging import get_logger
from pepbenchmark.utils.seed import set_seed


logger = get_logger()


# =========================
# Configs
# =========================

@dataclass
class LoRASettings:
    use_lora: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])


@dataclass
class RuntimeSettings:
    local_files_only: bool = False
    dataloader_num_workers: int = 0
    fp16: bool = False
    bf16: bool = False
    report_to: str = "none"


@dataclass
class TrainingSettings:
    epochs: int = 50
    learning_rate: float = 5e-5
    batch_size: int = 64
    warmup_steps: int = 0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    logging_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 5


@dataclass
class PLMConfig:
    model_name: str = "facebook/esm2_t30_150M_UR50D"
    model_type: str = "plm"
    task_type: str = "binary_classification"
    max_length: Optional[int] = None
    random_init: bool = False
    base_dir: str = "./results"
    seed: int = 42
    metrics: Optional[List[str]] = None
    lora: LoRASettings = field(default_factory=LoRASettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    model_params: Dict[str, Any] = field(default_factory=dict)

    def resolved_metrics(self) -> List[str]:
        if self.metrics is not None:
            return list(self.metrics)
        if self.task_type == "regression":
            return list(Regression_Metric)
        if self.task_type == "binary_classification":
            return list(Classification_Metric)
        raise ValueError(f"Unsupported task_type: {self.task_type}")


# =========================
# Result objects
# =========================

@dataclass
class ModelResults:
    train_metrics: Dict[str, float]
    valid_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time: float
    train_samples: int
    valid_samples: int
    test_samples: int
    random_seed: int
    fold_seed: int
    model_name: str
    split_type: str
    total_params: int
    trainable_params: int
    trainable_ratio: float
    lora_params: int
    non_lora_trainable_params: int

@dataclass
class MultiRunResults:
    results: List[ModelResults]

    def to_dataframe(self):
        import pandas as pd

        rows = []
        for res in self.results:
            row = {
                "random_seed": res.random_seed,
                "fold_seed": res.fold_seed,
                "model_name": res.model_name,
                "split_type": res.split_type,
                "train_samples": res.train_samples,
                "valid_samples": res.valid_samples,
                "test_samples": res.test_samples,
                "training_time": res.training_time,
            }

            for prefix, metrics in [
                ("train", res.train_metrics),
                ("valid", res.valid_metrics),
                ("test", res.test_metrics),
            ]:
                for k, v in metrics.items():
                    row[f"{prefix}_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_stats(self) -> Dict[str, Any]:
        df = self.to_dataframe()
        summary: Dict[str, Any] = {}

        for col in df.columns:
            if col.startswith(("train_", "valid_", "test_")) and np.issubdtype(df[col].dtype, np.number):
                series = df[col].dropna()
                if series.empty:
                    continue
                summary[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        return summary

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.to_dataframe().to_csv(os.path.join(save_dir, "multi_run_results.csv"), index=False)
        with open(os.path.join(save_dir, "multi_run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_summary_stats(), f, indent=2, ensure_ascii=False)


# =========================
# Task adapters
# =========================

class TaskAdapter(Protocol):
    task_type: str

    def get_num_labels(self) -> int:
        ...

    def get_problem_type(self) -> str:
        ...

    def encode_label(self, label: Any) -> Any:
        ...

    def prediction_to_metrics(self, pred: Any) -> Dict[str, float]:
        ...

    def logits_to_predictions(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def evaluate(self, y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        ...


class BinaryClassificationAdapter:
    task_type = "binary_classification"

    def get_num_labels(self) -> int:
        return 2

    def get_problem_type(self) -> str:
        return "single_label_classification"

    def encode_label(self, label: Any) -> int:
        return int(label)

    def prediction_to_metrics(self, pred: Any) -> Dict[str, float]:
        labels = np.array(pred.label_ids)
        logits = np.array(pred.predictions)
        preds = logits.argmax(-1)
        probs = softmax(logits, axis=-1)[:, 1]
        return evaluate_classification(y_true=labels, y_pred=preds, y_score=probs)

    def logits_to_predictions(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = logits.argmax(-1)
        probs = softmax(logits, axis=-1)[:, 1]
        return preds, probs

    def evaluate(self, y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        return evaluate_classification(y_true=y_true, y_pred=preds, y_score=probs)


class RegressionAdapter:
    task_type = "regression"

    def get_num_labels(self) -> int:
        return 1

    def get_problem_type(self) -> str:
        return "regression"

    def encode_label(self, label: Any) -> float:
        return float(label)

    def prediction_to_metrics(self, pred: Any) -> Dict[str, float]:
        labels = np.array(pred.label_ids).reshape(-1)
        preds = np.array(pred.predictions).reshape(-1)
        return evaluate_regression(labels, preds)

    def logits_to_predictions(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = logits.reshape(-1)
        return preds, preds

    def evaluate(self, y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        return evaluate_regression(y_true=y_true, y_pred=preds)


def build_task_adapter(task_type: str) -> TaskAdapter:
    if task_type == "binary_classification":
        return BinaryClassificationAdapter()
    if task_type == "regression":
        return RegressionAdapter()
    raise ValueError(f"Unsupported task_type: {task_type}")


# =========================
# Feature handling
# =========================

def get_sequences_and_labels(features: Dict[str, Any],seq_key: str="fasta",label_key: str="label") -> Tuple[List[str], List[Any], Dict[str, Any]]:
    try:
        sequences = features.get(seq_key)
        labels = features.get(label_key)
    except KeyError:
        raise KeyError(f"Features must contain keys '{seq_key}' and '{label_key}' for sequences and labels respectively.")
    return sequences, labels





# =========================
# Dataset
# =========================

class TokenizedSequenceDataset(Dataset):
    def __init__(
        self,
        encodings: Dict[str, List[Any]],
        labels: List[Any],
    ):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


class DatasetBuilder:
    def __init__(self, tokenizer: Any, task_adapter: TaskAdapter, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.task_adapter = task_adapter
        self.max_length = max_length

    def infer_max_length(
        self,
        features_list: List[Dict[str, Any]],
        extra_padding: int = 5,
    ) -> int:
        if self.max_length is not None:
            return self.max_length

        all_sequences: List[str] = []
        for features in features_list:
            sequences, _ = get_sequences_and_labels(features)
            all_sequences.extend(sequences)

        token_lengths = [
            len(
                self.tokenizer(
                    seq,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for seq in all_sequences
        ]
        if not token_lengths:
            raise ValueError("Cannot infer max_length from empty sequences.")

        return max(token_lengths) + extra_padding

    def build_dataset(
        self,
        features: Dict[str, Any],
        max_length: int,
    ) -> TokenizedSequenceDataset:
        print(features.keys())
        sequences, labels = get_sequences_and_labels(features)

        encoded_labels = [self.task_adapter.encode_label(label) for label in labels]

        encodings = self.tokenizer(
            sequences,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )

        return TokenizedSequenceDataset(
            encodings=encodings,
            labels=encoded_labels,
        )

    def build_prediction_dataset(
        self,
        features: Dict[str, Any],
        max_length: int,
    ) -> Tuple[TokenizedSequenceDataset, np.ndarray]:
        sequences, labels = get_sequences_and_labels(features)
        encoded_labels = [self.task_adapter.encode_label(label) for label in labels]

        encodings = self.tokenizer(
            sequences,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )

        return (
            TokenizedSequenceDataset(encodings=encodings, labels=encoded_labels),
            np.array(encoded_labels),
        )


# =========================
# Model / Trainer factories
# =========================

class ModelFactory:
    @staticmethod
    def build_tokenizer(config: PLMConfig):
        return AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.runtime.local_files_only,
            use_fast=False,

        )

    @staticmethod
    def build_model(config: PLMConfig, task_adapter: TaskAdapter):
        num_labels = task_adapter.get_num_labels()
        problem_type = task_adapter.get_problem_type()

        if config.random_init:
            hf_config = AutoConfig.from_pretrained(
                config.model_name,
                num_labels=num_labels,
                problem_type=problem_type,
                local_files_only=config.runtime.local_files_only,
                **config.model_params,
            )
            model = AutoModelForSequenceClassification.from_config(hf_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=num_labels,
                problem_type=problem_type,
                local_files_only=config.runtime.local_files_only,
                **config.model_params,
            )

        if config.lora.use_lora:
            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)
            logger.info(f"Use LoRA with config: {lora_config}")

        return model


class TrainerFactory:
    @staticmethod
    def build_training_arguments(
        config: PLMConfig,
        output_dir: str,
        seed: int,
    ) -> TrainingArguments:
        kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            seed=seed,
            load_best_model_at_end=True,
            report_to=config.runtime.report_to,
            save_total_limit=config.training.save_total_limit,
            logging_strategy=config.training.logging_strategy,
            save_strategy=config.training.save_strategy,
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=config.training.greater_is_better,
            dataloader_num_workers=config.runtime.dataloader_num_workers,
            fp16=config.runtime.fp16,
            bf16=config.runtime.bf16,
        )

        try:
            return TrainingArguments(
                evaluation_strategy=config.training.evaluation_strategy,
                **kwargs,
            )
        except TypeError:
            return TrainingArguments(
                eval_strategy=config.training.evaluation_strategy,
                **kwargs,
            )

    @staticmethod
    def build_eval_arguments(
        config: PLMConfig,
        output_dir: str,
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=config.training.batch_size,
            report_to="none",
            dataloader_num_workers=config.runtime.dataloader_num_workers,
            fp16=config.runtime.fp16,
            bf16=config.runtime.bf16,
        )

    @staticmethod
    def build_collator(tokenizer: Any):
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        )

    @classmethod
    def build_train_trainer(
        cls,
        model: Any,
        config: PLMConfig,
        tokenizer: Any,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        output_dir: str,
        seed: int,
        compute_metrics_fn: Any,
    ) -> Trainer:
        args = cls.build_training_arguments(config=config, output_dir=output_dir, seed=seed)
        collator = cls.build_collator(tokenizer)
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience
            )
        ]

        return Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
            data_collator=collator,
        )

    @classmethod
    def build_eval_trainer(
        cls,
        model: Any,
        config: PLMConfig,
        tokenizer: Any,
        output_dir: str,
    ) -> Trainer:
        args = cls.build_eval_arguments(config=config, output_dir=output_dir)
        collator = cls.build_collator(tokenizer)

        return Trainer(
            model=model,
            args=args,
            data_collator=collator,
        )


# =========================
# Experiment manager
# =========================

@dataclass
class RunContext:
    split_type: str
    fold_seed: int
    run_seed: int
    experiment_dir: str


class ExperimentManager:
    def __init__(self, config: PLMConfig, dataset: Optional[SinglePeptideDatasetManager]):
        self.config = config
        self.dataset = dataset

    def get_experiment_dir(
        self,
        split_type: str,
        fold_seed: int,
        run_seed: int,
    ) -> str:
        dataset_name = self.dataset.dataset_name if self.dataset is not None else "unknown_dataset"
        safe_model_name = self.config.model_name.replace("/", "_")
        return os.path.join(
            self.config.base_dir,
            dataset_name,
            safe_model_name,
            split_type,
            f"fold_{fold_seed}_seed_{run_seed}",
        )

    def create_run_context(
        self,
        split_type: str,
        fold_seed: int,
        run_seed: int,
    ) -> RunContext:
        experiment_dir = self.get_experiment_dir(split_type, fold_seed, run_seed)
        os.makedirs(experiment_dir, exist_ok=True)
        return RunContext(
            split_type=split_type,
            fold_seed=fold_seed,
            run_seed=run_seed,
            experiment_dir=experiment_dir,
        )

    def save_model(self, model: Any, tokenizer: Any, experiment_dir: str, name: str = "model"):
        model_path = os.path.join(experiment_dir, name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info(f"[SAVE MODEL] saved to {model_path}")

    def save_results(
        self,
        experiment_dir: str,
        results: ModelResults,
        model: Optional[Any] = None,
        dataset: Optional[SinglePeptideDatasetManager] = None,
        param_stats: Optional[Dict[str, Any]] = None,
    ):
        config_payload = {
            "plm_config": asdict(self.config),
            "resolved_metrics": self.config.resolved_metrics(),
            "dataset_name": dataset.dataset_name if dataset else None,
            "resolved_model_config": model.config.to_dict() if model is not None else {},
            "parameter_stats": param_stats,
        }

        with open(os.path.join(experiment_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

        with open(os.path.join(experiment_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False)


# =========================
# Main runner
# =========================

class PLMRunner:
    model_type: str = "plm"

    def __init__(
        self,
        config: PLMConfig,
        dataset: Optional[SinglePeptideDatasetManager] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.task_adapter = build_task_adapter(config.task_type)
        self.experiment_manager = ExperimentManager(config=config, dataset=dataset)

    def _compute_metrics(self, pred: Any) -> Dict[str, float]:
        return self.task_adapter.prediction_to_metrics(pred)

    def _build_runtime(
        self,
    ) -> Tuple[Any, Any, DatasetBuilder]:
        tokenizer = ModelFactory.build_tokenizer(self.config)
        model = ModelFactory.build_model(self.config, self.task_adapter)
        dataset_builder = DatasetBuilder(
            tokenizer=tokenizer,
            task_adapter=self.task_adapter,
            max_length=self.config.max_length,
        )
        return tokenizer, model, dataset_builder

    def _predict_with_trainer(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Dataset,
        output_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        trainer = TrainerFactory.build_eval_trainer(
            model=model,
            config=self.config,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )
        output = trainer.predict(dataset)
        logits = np.array(output.predictions)
        return self.task_adapter.logits_to_predictions(logits)

    def _evaluate_split(
        self,
        model: Any,
        tokenizer: Any,
        dataset_builder: DatasetBuilder,
        features: Dict[str, Any],
        output_dir: str,
        split_name: str,
        max_length: int,
    ) -> Dict[str, float]:
        dataset, labels = dataset_builder.build_prediction_dataset(
            features=features,
            max_length=max_length,
        )
        preds, probs = self._predict_with_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=output_dir,
        )
        metrics = self.task_adapter.evaluate(y_true=labels, preds=preds, probs=probs)
        logger.info(f"[METRICS] {split_name}: {metrics}")
        return metrics

    def train_once(
        self,
        train_features: Dict[str, Any],
        valid_features: Dict[str, Any],
        test_features: Dict[str, Any],
        run_context: RunContext,
    ) -> Tuple[Any, Any, DatasetBuilder, int, float]:
        tokenizer, model, dataset_builder = self._build_runtime()

        max_length = dataset_builder.infer_max_length(
            [train_features, valid_features, test_features]
        )

        train_dataset = dataset_builder.build_dataset(train_features, max_length=max_length)
        valid_dataset = dataset_builder.build_dataset(valid_features, max_length=max_length)

        trainer = TrainerFactory.build_train_trainer(
            model=model,
            config=self.config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            output_dir=run_context.experiment_dir,
            seed=run_context.run_seed,
            compute_metrics_fn=self._compute_metrics,
        )

        logger.info(
            f"[TRAIN] model={self.config.model_name}, "
            f"split={run_context.split_type}, fold={run_context.fold_seed}, seed={run_context.run_seed}"
        )
        start = time.time()
        trainer.train()
        training_time = time.time() - start
        logger.info(f"[TRAIN DONE] {self.config.model_name} in {training_time:.2f}s")

        self.experiment_manager.save_model(
            model=model,
            tokenizer=tokenizer,
            experiment_dir=run_context.experiment_dir,
        )

        return model, tokenizer, dataset_builder, max_length, training_time

    def run(
        self,
        run_seed: Optional[int] = None,
        fold_seed: int = 0,
        split_type: str = "random_split",
    ) -> ModelResults:
        if self.dataset is None:
            raise ValueError("dataset cannot be None in run().")

        effective_run_seed = self.config.seed if run_seed is None else run_seed
        set_seed(effective_run_seed)

        logger.info(
            f"[RUN] model={self.config.model_name} | "
            f"seed={effective_run_seed}, fold={fold_seed}, split={split_type}"
        )

        self.dataset.set_official_split_indices(split_type=split_type, fold_seed=fold_seed)
        train_features, valid_features, test_features = self.dataset.get_train_val_test_features("dict")

        run_context = self.experiment_manager.create_run_context(
            split_type=split_type,
            fold_seed=fold_seed,
            run_seed=effective_run_seed,
        )

        model, tokenizer, dataset_builder, max_length, training_time = self.train_once(
            train_features=train_features,
            valid_features=valid_features,
            test_features=test_features,
            run_context=run_context,
        )

        train_metrics = self._evaluate_split(
            model=model,
            tokenizer=tokenizer,
            dataset_builder=dataset_builder,
            features=train_features,
            output_dir=run_context.experiment_dir,
            split_name="train",
            max_length=max_length,
        )
        valid_metrics = self._evaluate_split(
            model=model,
            tokenizer=tokenizer,
            dataset_builder=dataset_builder,
            features=valid_features,
            output_dir=run_context.experiment_dir,
            split_name="valid",
            max_length=max_length,
        )
        test_metrics = self._evaluate_split(
            model=model,
            tokenizer=tokenizer,
            dataset_builder=dataset_builder,
            features=test_features,
            output_dir=run_context.experiment_dir,
            split_name="test",
            max_length=max_length,
        )

        train_sequences, _ = get_sequences_and_labels(train_features)
        valid_sequences, _ = get_sequences_and_labels(valid_features)
        test_sequences, _ = get_sequences_and_labels(test_features)
        
        
        
        param_stats = count_parameters(model)
        results = ModelResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            training_time=training_time,
            train_samples=len(train_sequences),
            valid_samples=len(valid_sequences),
            test_samples=len(test_sequences),
            random_seed=effective_run_seed,
            fold_seed=fold_seed,
            model_name=self.config.model_name,
            split_type=split_type,
            total_params=param_stats["total_params"],
            trainable_params=param_stats["trainable_params"],
            trainable_ratio=param_stats["trainable_ratio"],
            lora_params=param_stats["lora_params"],
            non_lora_trainable_params=param_stats["non_lora_trainable_params"],
        )

        self.experiment_manager.save_results(
            experiment_dir=run_context.experiment_dir,
            results=results,
            model=model,
            dataset=self.dataset,
            param_stats = count_parameters(model) if model is not None else {}
        )
        return results

    def run_multi(
        self,
        run_seeds: Optional[List[int]] = None,
        fold_seeds: Optional[List[int]] = None,
        split_type: str = "random_split",
        base_dir: Optional[str] = None,
    ) -> MultiRunResults:
        if base_dir is not None:
            self.config.base_dir = base_dir

        run_seeds = run_seeds or [42, 43, 44, 45, 46]
        fold_seeds = fold_seeds or [0, 1, 2, 3, 4]

        if len(run_seeds) != len(fold_seeds):
            logger.warning(
                "[RUN_MULTI] run_seeds and fold_seeds length mismatch, truncating to min length"
            )
            min_len = min(len(run_seeds), len(fold_seeds))
            run_seeds = run_seeds[:min_len]
            fold_seeds = fold_seeds[:min_len]

        all_results: List[ModelResults] = []
        for run_seed, fold_seed in zip(run_seeds, fold_seeds):
            logger.info(f"[RUN_MULTI] running seed={run_seed}, fold={fold_seed}")
            result = self.run(
                run_seed=run_seed,
                fold_seed=fold_seed,
                split_type=split_type,
            )
            all_results.append(result)

        logger.info(f"[RUN_MULTI DONE] total runs: {len(all_results)}")
        return MultiRunResults(results=all_results)

    @classmethod
    def load_model(
        cls,
        model_path: str,
        dataset: Optional[SinglePeptideDatasetManager] = None,
        **override_kwargs,
    ) -> "LoadedPLMRunner":
        if os.path.isdir(os.path.join(model_path, "model")):
            base_dir = model_path
            hf_model_path = os.path.join(model_path, "model")
        else:
            base_dir = os.path.dirname(model_path)
            hf_model_path = model_path

        config_path = os.path.join(base_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json under: {base_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_cfg = payload.get("plm_config", {})

        config = PLMConfig(
            model_name=override_kwargs.get("model_name", raw_cfg.get("model_name", hf_model_path)),
            model_type=override_kwargs.get("model_type", raw_cfg.get("model_type", "plm")),
            task_type=override_kwargs.get("task_type", raw_cfg.get("task_type", "binary_classification")),
            max_length=override_kwargs.get("max_length", raw_cfg.get("max_length")),
            random_init=False,
            base_dir=override_kwargs.get("base_dir", raw_cfg.get("base_dir", "./results")),
            seed=override_kwargs.get("seed", raw_cfg.get("seed", 42)),
            metrics=override_kwargs.get("metrics", raw_cfg.get("metrics")),
            lora=LoRASettings(**raw_cfg.get("lora", {})),
            runtime=RuntimeSettings(
                **{
                    **raw_cfg.get("runtime", {}),
                    "local_files_only": False,
                }
            ),
            training=TrainingSettings(**raw_cfg.get("training", {})),
            model_params=override_kwargs.get("model_params", raw_cfg.get("model_params", {})),
        )

        task_adapter = build_task_adapter(config.task_type)

        if config.lora.use_lora:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=task_adapter.get_num_labels(),
                problem_type=task_adapter.get_problem_type(),
                **config.model_params,
            )
            model = PeftModel.from_pretrained(base_model, hf_model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_model_path,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path,
            use_fast=False,
        )

        return LoadedPLMRunner(
            config=config,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            experiment_dir=base_dir,
        )


class LoadedPLMRunner(PLMRunner):
    def __init__(
        self,
        config: PLMConfig,
        dataset: Optional[SinglePeptideDatasetManager],
        model: Any,
        tokenizer: Any,
        experiment_dir: str,
    ):
        super().__init__(config=config, dataset=dataset)
        self.loaded_model = model
        self.loaded_tokenizer = tokenizer
        self.loaded_experiment_dir = experiment_dir

    def _build_runtime(self) -> Tuple[Any, Any, DatasetBuilder]:
        dataset_builder = DatasetBuilder(
            tokenizer=self.loaded_tokenizer,
            task_adapter=self.task_adapter,
            max_length=self.config.max_length,
        )
        return self.loaded_tokenizer, self.loaded_model, dataset_builder

    def predict(
        self,
        features: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        tokenizer, model, dataset_builder = self._build_runtime()
        max_length = dataset_builder.infer_max_length([features])
        dataset, _ = dataset_builder.build_prediction_dataset(
            features=features,
            max_length=max_length,
        )
        return self._predict_with_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=self.loaded_experiment_dir,
        )

    def evaluate(
        self,
        features: Dict[str, Any],
        split: str = "test",
    ) -> Dict[str, float]:
        tokenizer, model, dataset_builder = self._build_runtime()
        max_length = dataset_builder.infer_max_length([features])
        return self._evaluate_split(
            model=model,
            tokenizer=tokenizer,
            dataset_builder=dataset_builder,
            features=features,
            output_dir=self.loaded_experiment_dir,
            split_name=split,
            max_length=max_length,
        )


# =========================
# Backward-compatible model registry
# =========================

MODEL_REGISTRY: Dict[str, Type[PLMRunner]] = {
    "plm": PLMRunner,
}


def build_plm_model(
    model_type: str = "plm",
    dataset: Optional[SinglePeptideDatasetManager] = None,
    **kwargs,
) -> PLMRunner:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    config = PLMConfig(
        model_type=model_type,
        model_name=kwargs.pop("model_name", "facebook/esm2_t30_150M_UR50D"),
        task_type=kwargs.pop("task_type", "binary_classification"),
        max_length=kwargs.pop("max_length", None),
        random_init=kwargs.pop("random_init", False),
        base_dir=kwargs.pop("base_dir", "./results"),
        seed=kwargs.pop("seed", 42),
        metrics=kwargs.pop("metrics", None),
        lora=LoRASettings(
            use_lora=kwargs.pop("use_lora", False),
            r=kwargs.pop("lora_r", 8),
            alpha=kwargs.pop("lora_alpha", 16),
            dropout=kwargs.pop("lora_dropout", 0.1),
            target_modules=kwargs.pop("target_modules", ["query", "value"]),
        ),
        runtime=RuntimeSettings(
            local_files_only=kwargs.pop("local_files_only", False),
            dataloader_num_workers=kwargs.pop("dataloader_num_workers", 0),
            fp16=kwargs.pop("fp16", False),
            bf16=kwargs.pop("bf16", False),
            report_to=kwargs.pop("report_to", "none"),
        ),
        training=TrainingSettings(
            epochs=kwargs.pop("epochs", 30),
            learning_rate=kwargs.pop("learning_rate", 5e-5),
            batch_size=kwargs.pop("batch_size", 32),
            warmup_steps=kwargs.pop("warmup_steps", 0),
            weight_decay=kwargs.pop("weight_decay", 0.0),
            gradient_accumulation_steps=kwargs.pop("gradient_accumulation_steps", 1),
            logging_strategy=kwargs.pop("logging_strategy", "epoch"),
            evaluation_strategy=kwargs.pop("evaluation_strategy", "epoch"),
            save_strategy=kwargs.pop("save_strategy", "epoch"),
            save_total_limit=kwargs.pop("save_total_limit", 1),
            metric_for_best_model=kwargs.pop("metric_for_best_model", "eval_loss"),
            greater_is_better=kwargs.pop("greater_is_better", False),
            early_stopping_patience=kwargs.pop("early_stopping_patience", 5),
        ),
        model_params=kwargs,
    )

    return MODEL_REGISTRY[model_type](config=config, dataset=dataset)



def count_parameters(model) -> Dict[str, int]:
    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if param.requires_grad:
            trainable_params += numel

        if "lora_" in name:
            lora_params += numel

    non_lora_trainable_params = trainable_params - lora_params
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": float(trainable_ratio),
        "lora_params": int(lora_params),
        "non_lora_trainable_params": int(non_lora_trainable_params),
    }