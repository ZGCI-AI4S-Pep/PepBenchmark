import os
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from pepbenchmark.model.smiles_model import (
    DatasetBuilder,
    LoRASettings,
    ModelResults,
    MultiRunResults,
    SMILESRunner,
    TaskAdapter,
    TrainingSettings,
    build_task_adapter,
)
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


@dataclass
class RuntimeSettings:
    local_files_only: bool = False
    dataloader_num_workers: int = 0
    fp16: bool = False
    bf16: bool = False
    report_to: str = "none"
    cache_dir: Optional[str] = None

    # PepDoRA runtime options (single loading strategy; kept for compatibility)
    pepdora_load_mode: str = "auto"
    pepdora_adapter_model: str = "ChatterjeeLab/PepDoRA"
    pepdora_base_model: str = "DeepChem/ChemBERTa-77M-MLM"
    use_adapter_tokenizer: bool = True


@dataclass
class SMILESConfig:
    model_name: str = "ChatterjeeLab/PepDoRA"
    model_type: str = "smiles"
    task_type: str = "binary_classification"
    smiles_key: str = "smiles"
    label_key: str = "label"
    max_length: Optional[int] = None
    random_init: bool = False
    base_dir: str = "./results"
    seed: int = 42
    metrics: Optional[List[str]] = None
    lora: LoRASettings = field(default_factory=LoRASettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    model_params: Dict[str, Any] = field(default_factory=dict)


class PepDoRAForPrediction(nn.Module):
    """Backbone + task head wrapper compatible with HF Trainer."""

    def __init__(self, backbone: nn.Module, num_labels: int, task_type: str):
        super().__init__()
        self.backbone = backbone
        self.num_labels = int(num_labels)
        self.task_type = task_type
        hidden_size = int(getattr(backbone.config, "hidden_size", 768))

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Keep a config-like object for Trainer compatibility.
        self.config = getattr(backbone, "config", None) or PretrainedConfig()
        self.config.num_labels = self.num_labels
        self.config.problem_type = (
            "regression" if task_type == "regression" else "single_label_classification"
        )

    def _pool(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            return outputs.hidden_states[-1][:, 0]

        if isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            return outputs[0][:, 0]

        raise RuntimeError("PepDoRA backbone did not return usable hidden states for pooling.")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Newer Trainer versions may inject loss-only kwargs not accepted by backbone.
        kwargs.pop("num_items_in_batch", None)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

        pooled = self._pool(outputs)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        backbone_dir = os.path.join(save_directory, "backbone")
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(backbone_dir)

        head_path = os.path.join(save_directory, "task_head.pt")
        torch.save(
            {
                "state_dict": self.classifier.state_dict(),
                "num_labels": self.num_labels,
                "task_type": self.task_type,
            },
            head_path,
        )

        config_path = os.path.join(save_directory, "task_head_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_labels": self.num_labels,
                    "task_type": self.task_type,
                    "hidden_size": int(getattr(self.backbone.config, "hidden_size", 768)),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


class ModelFactory:
    @staticmethod
    def _get_num_labels_and_problem_type(task_adapter: TaskAdapter) -> Tuple[int, str]:
        return task_adapter.get_num_labels(), task_adapter.get_problem_type()

    @staticmethod
    def build_tokenizer(config: SMILESConfig):
        tokenizer_source = config.model_name

        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_source,
                local_files_only=config.runtime.local_files_only,
                cache_dir=config.runtime.cache_dir,
                use_fast=False,
            )
        except Exception as exc:
            logger.warning(f"Failed to load tokenizer from {tokenizer_source}: {exc}")
            if tokenizer_source != config.runtime.pepdora_base_model:
                logger.info(
                    f"Falling back tokenizer to base model: {config.runtime.pepdora_base_model}"
                )
                return AutoTokenizer.from_pretrained(
                    config.runtime.pepdora_base_model,
                    local_files_only=config.runtime.local_files_only,
                    cache_dir=config.runtime.cache_dir,
                    use_fast=False,
                )
            raise

    @staticmethod
    def build_model(config: SMILESConfig, task_adapter: TaskAdapter):
        num_labels, _ = ModelFactory._get_num_labels_and_problem_type(task_adapter)

        if config.random_init:
            hf_config = AutoConfig.from_pretrained(
                config.model_name,
                num_labels=num_labels,
                problem_type=task_adapter.get_problem_type(),
                local_files_only=config.runtime.local_files_only,
                cache_dir=config.runtime.cache_dir,
                **config.model_params,
            )
            model = AutoModelForSequenceClassification.from_config(hf_config)
        else:
            if (config.runtime.pepdora_load_mode or "auto").lower() not in {"auto", "direct"}:
                logger.warning(
                    "pepdora_load_mode is ignored in this module; always using direct AutoModel loading."
                )

            backbone = AutoModel.from_pretrained(
                config.model_name,
                local_files_only=config.runtime.local_files_only,
                cache_dir=config.runtime.cache_dir,
                **config.model_params,
            )
            logger.info("Loaded PepDoRA backbone using direct AutoModel mode.")

            model = PepDoRAForPrediction(
                backbone=backbone,
                num_labels=num_labels,
                task_type=config.task_type,
            )

        if config.lora.use_lora:
            # Optional extra LoRA on top of PepDoRA backbone.
            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            try:
                model = get_peft_model(model, lora_config)
            except Exception as exc:
                logger.warning(f"Applying extra LoRA on PepDoRA model failed, skip it: {exc}")

        return model


class SMILESPepDoRARunner(SMILESRunner):
    model_type = "smiles"

    def __init__(self, config: SMILESConfig, dataset=None):
        self.config = config
        self.dataset = dataset
        self.task_adapter = build_task_adapter(config.task_type)
        from pepbenchmark.model.smiles_model import ExperimentManager

        self.experiment_manager = ExperimentManager(config=config, dataset=dataset)

    @staticmethod
    def _extract_logits(predictions: Any) -> np.ndarray:
        # Trainer may return tuple/list when model outputs extra fields.
        if isinstance(predictions, (tuple, list)) and len(predictions) > 0:
            predictions = predictions[0]
        return np.asarray(predictions)

    def _compute_metrics(self, pred: Any) -> Dict[str, float]:
        logits = self._extract_logits(pred.predictions)
        fixed_pred = SimpleNamespace(label_ids=pred.label_ids, predictions=logits)
        return self.task_adapter.prediction_to_metrics(fixed_pred)

    def _build_runtime(self) -> Tuple[Any, Any, DatasetBuilder]:
        tokenizer = ModelFactory.build_tokenizer(self.config)
        model = ModelFactory.build_model(self.config, self.task_adapter)
        dataset_builder = DatasetBuilder(
            tokenizer=tokenizer,
            task_adapter=self.task_adapter,
            max_length=self.config.max_length,
        )
        return tokenizer, model, dataset_builder

    def _predict_with_trainer(self, model: Any, tokenizer: Any, dataset: Any, output_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        from pepbenchmark.model.smiles_model import TrainerFactory
        from transformers import Trainer

        trainer = Trainer(
            model=model,
            args=TrainerFactory.build_eval_arguments(self.config, output_dir),
            data_collator=TrainerFactory.build_collator(tokenizer),
        )
        output = trainer.predict(dataset)
        logits = self._extract_logits(output.predictions)
        return self.task_adapter.logits_to_predictions(logits)


# Keep legacy name style consistent with smiles_model.py
class SMILESModel(SMILESPepDoRARunner):
    pass


MODEL_REGISTRY: Dict[str, Type[SMILESPepDoRARunner]] = {
    "smiles": SMILESPepDoRARunner,
    "pepdora": SMILESPepDoRARunner,
}


def build_smiles_model(
    model_type: str = "smiles",
    dataset=None,
    **kwargs,
) -> SMILESPepDoRARunner:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_name = kwargs.pop("model_name", kwargs.pop("model", "ChatterjeeLab/PepDoRA"))
    # deprecated: do not expose cache_dir in public API
    kwargs.pop("cache_dir", None)
    local_model = kwargs.pop("local_model", False)

    config = SMILESConfig(
        model_type=model_type,
        model_name=model_name,
        task_type=kwargs.pop("task_type", "binary_classification"),
        smiles_key=kwargs.pop("smiles_key", kwargs.pop("smiles_field", "smiles")),
        label_key=kwargs.pop("label_key", "label"),
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
            target_modules=kwargs.pop("target_modules", ["query", "key", "value"]),
        ),
        runtime=RuntimeSettings(
            local_files_only=kwargs.pop("local_files_only", local_model),
            dataloader_num_workers=kwargs.pop("dataloader_num_workers", 0),
            fp16=kwargs.pop("fp16", False),
            bf16=kwargs.pop("bf16", False),
            report_to=kwargs.pop("report_to", "none"),
            cache_dir=None,
            pepdora_load_mode=kwargs.pop("pepdora_load_mode", "auto"),
            pepdora_adapter_model=kwargs.pop("pepdora_adapter_model", "ChatterjeeLab/PepDoRA"),
            pepdora_base_model=kwargs.pop("pepdora_base_model", "DeepChem/ChemBERTa-77M-MLM"),
            use_adapter_tokenizer=kwargs.pop("use_adapter_tokenizer", True),
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
