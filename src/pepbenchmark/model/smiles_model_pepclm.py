import os
import warnings
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, PeftModel

class BaseModel:
    pass
from pepbenchmark.evaluator import evaluate_classification, evaluate_regression
from pepbenchmark.utils.logging import get_logger

# Disable irrelevant warnings
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars",
)
os.environ["WANDB_DISABLED"] = "true"

logger = get_logger()


class SMILESDatasetWithLabels(Dataset):
    """Dataset for SMILES molecular representations."""

    def __init__(self, smiles: List[str], labels: List, tokenizer, max_len: int = 512):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        # SMILES do not need to be space-separated like protein sequences
        smiles_str = self.smiles[idx]
        
        # Encode the SMILES string
        encoded = self.tokenizer(
            smiles_str,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Remove the batch dimension, as it will be added automatically by the Dataset
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long if isinstance(self.labels[idx], int) else torch.float)
        
        return item

    def __len__(self):
        return len(self.smiles)


class SMILESModel(BaseModel):
    """SMILES-based molecular property prediction model (supports LoRA fine-tuning)."""

    def __init__(
        self,
        model: str = "seyonec/ChemBERTa-zinc-base-v1",  # Default to ChemBERTa model
        epochs: int = 50,
        metrics: Optional[List[str]] = None,
        dataset=None,
        # Training parameters expanded directly
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        gradient_accumulation_steps: int = 1,
        logging_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        save_total_limit: int = 1,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        report_to: str = "none",
        seed: int = 42,
        max_length: Optional[int] = None,
        early_stopping_patience: int = 5,
        cache_dir: str = "cache_dir=/home/batchcom/assist/.hug_cache/hub",
        # LoRA parameters
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        # Local model parameters
        local_model: bool = True,
        local_model_dir: Optional[str] = None,
        # Random initialization parameters
        random_init: bool = False,
        # SMILES-specific parameters
        smiles_field: str = "smiles",  # Name of the SMILES field in the dataset
        **kwargs,
    ):
        # Remove potentially conflicting parameters from kwargs
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'model_name'}
        
        super().__init__(
            model_name=model,
            model=None,
            epochs=epochs,
            metrics=metrics,
            dataset=dataset,
            **kwargs_clean,
        )

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_strategy = logging_strategy
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.report_to = report_to
        self.random_seed = seed
        self.max_length = max_length or 512  # SMILES are usually shorter than protein sequences
        self.early_stopping_patience = early_stopping_patience
        self.cache_dir = cache_dir

        # LoRA parameters
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # ChemBERTa uses different attention module names
        self.target_modules = target_modules or ["query", "key", "value"]

        # Local model parameters
        self.local_model = local_model
        self.local_model_dir = local_model_dir

        # Random initialization parameters
        self.random_init = random_init
        
        # SMILES-specific parameters
        self.smiles_field = smiles_field

        self.tokenizer = None
        self.trainer = None
        
        # Build model and tokenizer
        self._build_model()

    # ============ Build Model ============
    def _build_model(self) -> Any:
        # Determine model path and cache directory
        if self.local_model:
            if self.local_model_dir:
                # Use specified local directory
                model_path = os.path.join(self.local_model_dir, self.model_name.replace("/", "_"))
                if not os.path.exists(model_path):
                    logger.warning(f"Local model path {model_path} does not exist. Falling back to cache or remote.")
                    model_path = self.model_name
                    cache_dir = self.local_model_dir
                else:
                    logger.info(f"Using local model from: {model_path}")
                    cache_dir = None
            else:
                # Use default cache directory, force local cache usage
                model_path = self.model_name
                cache_dir = self.cache_dir or os.path.expanduser("~/.cache/huggingface/transformers")
                logger.info(f"Attempting to use local cached model from: {cache_dir}")
        else:
            # Default behavior: download from remote or use cache
            model_path = self.model_name
            cache_dir = self.cache_dir
            logger.info(f"Using remote model: {self.model_name}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=self.local_model,
            )
            
            # For some models, pad_token needs to be set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_path}: {e}")
            
            # If local model fails, first try loading from remote
            if self.local_model:
                try:
                    logger.info("Trying to load tokenizer from remote...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=cache_dir,
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("✅ Successfully loaded tokenizer from remote")
                    return  # Exit on successful load
                except Exception as remote_e:
                    logger.warning(f"Remote tokenizer loading also failed: {remote_e}")
            
            # Try a compatible tokenizer as a fallback
            fallback_tokenizers = [
                "DeepChem/ChemBERTa-77M-MLM",  # Common tokenizer in chemistry domain
            ]
            
            tokenizer_loaded = False
            for fallback in fallback_tokenizers:
                try:
                    logger.info(f"Trying fallback tokenizer: {fallback}")
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback, cache_dir=cache_dir)
                    
                    # For some models, pad_token needs to be set
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    logger.info(f"✅ Successfully loaded fallback tokenizer: {fallback}")
                    tokenizer_loaded = True
                    break
                    
                except Exception as fallback_e:
                    logger.warning(f"Fallback tokenizer {fallback} also failed: {fallback_e}")
                    continue
            
            if not tokenizer_loaded:
                logger.error(f"Failed to load tokenizer and all fallbacks failed")
                raise e

        # Determine number of labels
        if self.task_type == "binary_classification":
            num_labels = 2
        elif self.task_type == "regression":
            num_labels = 1
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Load model
        if self.random_init:
            # Random initialization mode: load config, then randomly initialize weights
            logger.info(f"Random initialization mode: loading config and initializing weights randomly")
            try:
                config = AutoConfig.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    local_files_only=self.local_model,
                    num_labels=num_labels,
                )
            except Exception as e:
                if self.local_model:
                    logger.error(f"Failed to load config locally: {e}")
                    logger.info("Trying to load config from remote...")
                    config = AutoConfig.from_pretrained(
                        self.model_name,
                        cache_dir=cache_dir,
                        num_labels=num_labels,
                    )
                else:
                    raise
            
            # Create model instance and randomly initialize weights
            model = AutoModelForSequenceClassification.from_config(config)
            logger.info(f"Model weights randomly initialized for {self.model_name}")
        else:
            # Pre-trained mode: use pre-trained weights
            # Check if it's a PepDoRA model
            if "PepDoRA" in self.model_name:
                logger.info(f"Detected PepDoRA model: {self.model_name}")
                try:
                    # Method 1: Load PepDoRA directly as per official documentation (second method)
                    from transformers import AutoModel
                    import torch.nn as nn
                    
                    logger.info(f"Loading PepDoRA directly: {self.model_name}")
                    
                    # Load PepDoRA model directly
                    backbone = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
                    
                    # Add classification head for the task
                    class PepDoRAForClassification(nn.Module):
                        def __init__(self, backbone, num_labels):
                            super().__init__()
                            self.backbone = backbone
                            self.classifier = nn.Sequential(
                                nn.Linear(backbone.config.hidden_size, backbone.config.hidden_size),
                                nn.Dropout(0.1),
                                nn.Linear(backbone.config.hidden_size, num_labels)
                            )
                            
                        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                            # Pass only necessary parameters to the backbone
                            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                            pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
                            logits = self.classifier(pooled_output)
                            
                            loss = None
                            if labels is not None:
                                if num_labels == 1:
                                    # Regression task
                                    loss_fn = nn.MSELoss()
                                    loss = loss_fn(logits.squeeze(), labels.float())
                                else:
                                    # Classification task
                                    loss_fn = nn.CrossEntropyLoss()
                                    loss = loss_fn(logits, labels)
                            
                            # Return output compatible with transformers
                            from transformers.modeling_outputs import SequenceClassifierOutput
                            return SequenceClassifierOutput(
                                loss=loss,
                                logits=logits,
                                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                            )
                    
                    model = PepDoRAForClassification(backbone, num_labels)
                    logger.info("✅ Successfully loaded PepDoRA with custom classification head")
                    
                except Exception as e:
                    logger.warning(f"Failed to load PepDoRA: {e}")
                    logger.info("Falling back to base model...")
                    # Fallback to base model
                    base_model_name = "DeepChem/ChemBERTa-77M-MLM"
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=num_labels,
                        cache_dir=cache_dir,
                        ignore_mismatched_sizes=True,
                    )
                    logger.warning(f"✅ Successfully loaded fallback base model: {base_model_name}")
                        
            elif "LoRA" in self.model_name.upper():
                # Handle other LoRA models
                logger.info(f"Detected LoRA-based model: {self.model_name}")
                try:
                    # Special handling for LoRA models
                    from peft import PeftModel, PeftConfig
                    
                    # Get PEFT config
                    peft_config = PeftConfig.from_pretrained(model_path)
                    base_model_name = peft_config.base_model_name_or_path
                    
                    logger.info(f"Loading base model: {base_model_name}")
                    # Load base model first
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=num_labels,
                        cache_dir=cache_dir,
                    )
                    
                    # Then load LoRA adapter
                    logger.info(f"Loading LoRA adapter from: {model_path}")
                    model = PeftModel.from_pretrained(
                        base_model,
                        model_path,
                        cache_dir=cache_dir,
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to load as PEFT model: {e}")
                    logger.info("Falling back to base model...")
                    # Fallback to base model
                    base_model_name = "DeepChem/ChemBERTa-77M-MLM"
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=num_labels,
                        cache_dir=cache_dir,
                        ignore_mismatched_sizes=True,
                    )
                    logger.warning(f"✅ Successfully loaded fallback base model: {base_model_name}")
            else:
                # Standard model loading
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=num_labels,
                        cache_dir=cache_dir,
                        local_files_only=self.local_model,
                    )
                except Exception as e:
                    if self.local_model:
                        logger.error(f"Failed to load model locally: {e}")
                        logger.info("Trying to load model from remote...")
                        model = AutoModelForSequenceClassification.from_pretrained(
                            self.model_name,
                            num_labels=num_labels,
                            cache_dir=cache_dir,
                        )
                    else:
                        raise

        # Apply LoRA (if enabled and not a pre-trained LoRA model)
        is_pretrained_lora = "PepDoRA" in self.model_name or "LoRA" in self.model_name.upper()
        
        if self.use_lora and not is_pretrained_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)
            logger.info("LoRA enabled.")
        elif is_pretrained_lora:
            logger.info(f"Using pretrained LoRA model: {self.model_name}")

        return model

    # ============ Data Preparation ============
    def _prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        train_features, valid_features, test_features = self.dataset.get_train_val_test_features("dict")

        # Get SMILES data
        # Assume 'smiles' field exists in the dataset, otherwise use 'official_fasta' as a fallback
        smiles_field = self.smiles_field
        if smiles_field not in train_features:
            # If the specified SMILES field is not present, try common field names
            possible_fields = ['smiles', 'SMILES', 'official_smiles', 'official_fasta', 'sequence']
            smiles_field = None
            for field in possible_fields:
                if field in train_features:
                    smiles_field = field
                    logger.info(f"Using '{field}' as SMILES field")
                    break
            
            if smiles_field is None:
                raise ValueError(f"No suitable SMILES field found in dataset. Available fields: {list(train_features.keys())}")

        # Determine max length
        if self.max_length is not None:
            max_len = self.max_length
        else:
            all_smiles = (
                train_features[smiles_field]
                + valid_features[smiles_field]
                + test_features[smiles_field]
            )
            inferred_len = max(len(smiles) for smiles in all_smiles)
            max_len = min(inferred_len + 10, 512)  # SMILES are usually not too long, limit to 512

        train_dataset = SMILESDatasetWithLabels(
            train_features[smiles_field], train_features["official_label"], self.tokenizer, max_len
        )
        valid_dataset = SMILESDatasetWithLabels(
            valid_features[smiles_field], valid_features["official_label"], self.tokenizer, max_len
        )
        test_dataset = SMILESDatasetWithLabels(
            test_features[smiles_field], test_features["official_label"], self.tokenizer, max_len
        )

        logger.info(
            f"Prepared SMILES datasets with max_len={max_len} "
            f"(Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)})"
        )
        return train_dataset, valid_dataset, test_dataset

    # ============ metrics ============
    def _get_compute_metrics_fn(self):
        if self.task_type == "binary_classification":

            def compute_metrics(pred):
                labels = pred.label_ids
                logits = pred.predictions
                preds = logits.argmax(-1)
                probs = softmax(logits, axis=-1)[:, 1]
                return evaluate_classification(y_true=labels, y_pred=preds, y_score=probs)

        elif self.task_type == "multi_class_classification":

            def compute_metrics(pred):
                labels = pred.label_ids
                logits = pred.predictions
                preds = logits.argmax(-1)
                probs = softmax(logits, axis=-1)
                return evaluate_classification(y_true=labels, y_pred=preds, y_score=probs)

        elif self.task_type == "regression":

            def compute_metrics(pred):
                labels = np.array(pred.label_ids).reshape(-1)
                preds = np.array(pred.predictions).reshape(-1)
                return evaluate_regression(labels, preds)

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return compute_metrics

    # ============ Train Model ============
    def _train_model(self) -> Dict[str, Any]:
        train_dataset, valid_dataset, _ = self._prepare_datasets()

        # Compatible with evaluation_strategy / eval_strategy
        kwargs = dict(
            output_dir=self.full_output_dir,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            seed=self.random_seed,
            load_best_model_at_end=True,
            report_to=self.report_to,
            save_total_limit=self.save_total_limit,
            logging_strategy=self.logging_strategy,
            save_strategy=self.save_strategy,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
        )
        try:
            training_args = TrainingArguments(
                evaluation_strategy=self.evaluation_strategy,
                **kwargs,
            )
        except TypeError:
            training_args = TrainingArguments(
                eval_strategy=self.evaluation_strategy,
                **kwargs,
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=self._get_compute_metrics_fn(),
            callbacks=[EarlyStoppingCallback(self.early_stopping_patience)],
        )

        logger.info(f"Training SMILES model: {self.model_name}")
        self.trainer.train()

        history = getattr(self.trainer.state, "log_history", [])
        return {"history": history}

    # ============ Prediction ============
    def _predict(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        output = self.trainer.predict(X)
        logits = output.predictions

        if self.task_type in ["binary_classification", "multi_class_classification"]:
            preds = logits.argmax(-1)
            probs = softmax(logits, axis=-1)
            if self.task_type == "binary_classification":
                probs = probs[:, 1]
        elif self.task_type == "regression":
            preds = logits.reshape(-1)
            probs = preds
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return preds, probs

    # ============ Feature Preparation (for inference) ============
    def _prepare_features_for_prediction(
        self, features: Dict[str, Any]
    ) -> Tuple[Dataset, np.ndarray]:
        # Get SMILES data
        smiles_field = self.smiles_field
        if smiles_field not in features:
            # If the specified SMILES field is not present, try common field names
            possible_fields = ['smiles', 'SMILES', 'official_smiles', 'official_fasta', 'sequence']
            smiles_field = None
            for field in possible_fields:
                if field in features:
                    smiles_field = field
                    break
            
            if smiles_field is None:
                raise ValueError(f"No suitable SMILES field found in features. Available fields: {list(features.keys())}")

        if self.max_length is not None:
            max_len = self.max_length
        else:
            inferred_len = max(len(smiles) for smiles in features[smiles_field])
            max_len = min(inferred_len + 10, 512)

        dataset = SMILESDatasetWithLabels(
            features[smiles_field],
            features.get("official_label", [0] * len(features[smiles_field])),
            self.tokenizer,
            max_len=max_len,
        )
        labels = np.array(features.get("official_label", [0] * len(features[smiles_field])))
        return dataset, labels

    # ============ Save and Load ============
    def _save_model(self):
        model_path = os.path.join(self.full_output_dir, "model")
        if self.use_lora:
            self.model.save_pretrained(model_path)
        else:
            self.trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)
        logger.info(f"Saved SMILES model to {model_path}")

    @classmethod
    def load_model(cls, model_path: str, **kwargs):
        use_lora = kwargs.get("use_lora", False)
        model_instance = cls(model_name=model_path, **kwargs)
        if use_lora:
            base_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model_instance.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model_instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model_instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_instance.is_trained = True
        return model_instance

    # ============ Training History ============
    def _get_training_history(self) -> List[Dict[str, Any]]:
        return getattr(self.trainer, "log_history", [])


# =========================
# Refactored PLM-style SMILES runner
# =========================

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Protocol, Type

from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager
from pepbenchmark.evaluator import Classification_Metric, Regression_Metric
from pepbenchmark.utils.seed import set_seed
from transformers import DataCollatorWithPadding


@dataclass
class LoRASettings:
    use_lora: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "key", "value"])


@dataclass
class RuntimeSettings:
    local_files_only: bool = False
    dataloader_num_workers: int = 0
    fp16: bool = False
    bf16: bool = False
    report_to: str = "none"
    cache_dir: Optional[str] = None
    use_pepclm_tokenizer: bool = False
    pepclm_vocab_file: Optional[str] = None
    pepclm_spe_file: Optional[str] = None


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
class SMILESConfig:
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
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

    def resolved_metrics(self) -> List[str]:
        if self.metrics is not None:
            return list(self.metrics)
        if self.task_type == "regression":
            return list(Regression_Metric)
        if self.task_type == "binary_classification":
            return list(Classification_Metric)
        raise ValueError(f"Unsupported task_type: {self.task_type}")


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


def get_smiles_and_labels(
    features: Dict[str, Any],
    smiles_key: str = "smiles",
    label_key: str = "label",
) -> Tuple[List[str], List[Any]]:
    smiles = features.get(smiles_key)
    labels = features.get(label_key)
    if smiles is None or labels is None:
        raise KeyError(f"Features must contain keys '{smiles_key}' and '{label_key}'.")
    return smiles, labels


class TokenizedSMILESDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[Any]], labels: List[Any]):
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
        smiles_key: str,
        label_key: str,
        extra_padding: int = 10,
    ) -> int:
        if self.max_length is not None:
            return self.max_length
        all_smiles: List[str] = []
        for features in features_list:
            smiles, _ = get_smiles_and_labels(features, smiles_key=smiles_key, label_key=label_key)
            all_smiles.extend(smiles)
        token_lengths = [
            len(self.tokenizer(s, add_special_tokens=True, truncation=False)["input_ids"])
            for s in all_smiles
        ]
        if not token_lengths:
            raise ValueError("Cannot infer max_length from empty SMILES.")
        return max(token_lengths) + extra_padding

    def build_dataset(self, features: Dict[str, Any], smiles_key: str, label_key: str, max_length: int) -> TokenizedSMILESDataset:
        smiles, labels = get_smiles_and_labels(features, smiles_key=smiles_key, label_key=label_key)
        encoded_labels = [self.task_adapter.encode_label(label) for label in labels]
        encodings = self.tokenizer(smiles, add_special_tokens=True, truncation=True, max_length=max_length)
        return TokenizedSMILESDataset(encodings=encodings, labels=encoded_labels)

    def build_prediction_dataset(
        self,
        features: Dict[str, Any],
        smiles_key: str,
        label_key: str,
        max_length: int,
    ) -> Tuple[TokenizedSMILESDataset, np.ndarray]:
        ds = self.build_dataset(features, smiles_key=smiles_key, label_key=label_key, max_length=max_length)
        _, labels = get_smiles_and_labels(features, smiles_key=smiles_key, label_key=label_key)
        encoded_labels = [self.task_adapter.encode_label(label) for label in labels]
        return ds, np.array(encoded_labels)


class PeptideCLMTokenizer:
    """Minimal SMILES-SPE tokenizer adapter compatible with HuggingFace Trainer collators.

    This follows PeptideCLM's required loading mechanism:
    - vocabulary file (e.g. new_vocab.txt)
    - SPE merge/split file (e.g. new_splits.txt)
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, spe_file: str):
        try:
            from SmilesPE.tokenizer import SPE_Tokenizer
        except Exception as exc:
            raise ImportError(
                "PeptideCLM tokenizer requires SmilesPE. Install with: pip install SmilesPE"
            ) from exc

        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"PeptideCLM vocab file not found: {vocab_file}")
        if not os.path.isfile(spe_file):
            raise FileNotFoundError(f"PeptideCLM SPE file not found: {spe_file}")

        self.vocab_file = vocab_file
        self.spe_file = spe_file
        self.vocab: Dict[str, int] = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                tok = line.rstrip("\n")
                if tok:
                    self.vocab[tok] = i

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, self.pad_token_id)
        self.cls_token_id = self.vocab.get(self.cls_token, self.pad_token_id)
        self.sep_token_id = self.vocab.get(self.sep_token, self.pad_token_id)

        with open(spe_file, "r", encoding="utf-8") as sf:
            self._spe = SPE_Tokenizer(sf)

    def _tokenize(self, text: str) -> List[str]:
        tokenized = self._spe.tokenize(text)
        if isinstance(tokenized, str):
            return tokenized.split(" ")
        return list(tokenized)

    def _encode_one(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        tokens = self._tokenize(text)
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": attn}

    def __call__(
        self,
        text,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        if isinstance(text, str):
            return self._encode_one(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
        encoded = [
            self._encode_one(
                t,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for t in text
        ]
        return {
            "input_ids": [e["input_ids"] for e in encoded],
            "attention_mask": [e["attention_mask"] for e in encoded],
        }

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(encoded_inputs, dict):
            features = []
            n = len(encoded_inputs["input_ids"])
            for i in range(n):
                one = {k: encoded_inputs[k][i] for k in encoded_inputs.keys()}
                features.append(one)
        else:
            features = list(encoded_inputs)

        if max_length is None:
            max_length = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of:
            if max_length % pad_to_multiple_of != 0:
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        labels = []
        has_labels = all("labels" in f for f in features)

        for f in features:
            ids = list(f["input_ids"])
            mask = list(f.get("attention_mask", [1] * len(ids)))
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + ([self.pad_token_id] * pad_len)
                mask = mask + ([0] * pad_len)
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            if has_labels:
                labels.append(f["labels"])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }
        if has_labels:
            batch["labels"] = labels

        if return_tensors == "pt":
            batch = {
                k: torch.tensor(v, dtype=torch.long if k != "labels" else None)
                if k != "labels"
                else torch.tensor(v)
                for k, v in batch.items()
            }

        return batch

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        out_vocab = os.path.join(save_directory, "vocab.txt")
        with open(out_vocab, "w", encoding="utf-8") as f:
            for token, _ in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                f.write(token + "\n")
        return (save_directory,)


class ModelFactory:
    @staticmethod
    def _resolve_local_pepclm_files() -> Tuple[Optional[str], Optional[str]]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            (os.path.join(current_dir, "new_vocab.txt"), os.path.join(current_dir, "new_splits.txt")),
            (
                os.path.join(current_dir, "pepclm_tokenizer", "new_vocab.txt"),
                os.path.join(current_dir, "pepclm_tokenizer", "new_splits.txt"),
            ),
        ]
        for vocab_file, spe_file in candidates:
            if os.path.isfile(vocab_file) and os.path.isfile(spe_file):
                return vocab_file, spe_file
        return None, None

    @staticmethod
    def build_tokenizer(config: SMILESConfig):
        use_pepclm = config.runtime.use_pepclm_tokenizer or ("peptideclm" in config.model_name.lower())
        if use_pepclm:
            vocab_file = config.runtime.pepclm_vocab_file
            spe_file = config.runtime.pepclm_spe_file
            if not vocab_file or not spe_file:
                vocab_file, spe_file = ModelFactory._resolve_local_pepclm_files()

            if not vocab_file or not spe_file:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                raise ValueError(
                    "PeptideCLM requires local tokenizer files. "
                    "Please put `new_vocab.txt` and `new_splits.txt` under the current module directory "
                    f"({current_dir}) or pass `pepclm_vocab_file`/`pepclm_spe_file`."
                )
            logger.info("Using PeptideCLM custom SMILES-SPE tokenizer.")
            return PeptideCLMTokenizer(
                vocab_file=vocab_file,
                spe_file=spe_file,
            )

        return AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.runtime.local_files_only,
            cache_dir=config.runtime.cache_dir,
            use_fast=False,
        )

    @staticmethod
    def build_model(config: SMILESConfig, task_adapter: TaskAdapter):
        if config.random_init:
            hf_config = AutoConfig.from_pretrained(
                config.model_name,
                num_labels=task_adapter.get_num_labels(),
                problem_type=task_adapter.get_problem_type(),
                local_files_only=config.runtime.local_files_only,
                cache_dir=config.runtime.cache_dir,
                **config.model_params,
            )
            model = AutoModelForSequenceClassification.from_config(hf_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=task_adapter.get_num_labels(),
                problem_type=task_adapter.get_problem_type(),
                local_files_only=config.runtime.local_files_only,
                cache_dir=config.runtime.cache_dir,
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
        return model


class TrainerFactory:
    @staticmethod
    def build_training_arguments(config: SMILESConfig, output_dir: str, seed: int) -> TrainingArguments:
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
            return TrainingArguments(evaluation_strategy=config.training.evaluation_strategy, **kwargs)
        except TypeError:
            return TrainingArguments(eval_strategy=config.training.evaluation_strategy, **kwargs)

    @staticmethod
    def build_eval_arguments(config: SMILESConfig, output_dir: str) -> TrainingArguments:
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
        return DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")


@dataclass
class RunContext:
    split_type: str
    fold_seed: int
    run_seed: int
    experiment_dir: str


class ExperimentManager:
    def __init__(self, config: SMILESConfig, dataset: Optional[SinglePeptideDatasetManager]):
        self.config = config
        self.dataset = dataset

    def get_experiment_dir(self, split_type: str, fold_seed: int, run_seed: int) -> str:
        dataset_name = self.dataset.dataset_name if self.dataset is not None else "unknown_dataset"
        safe_model_name = self.config.model_name.replace("/", "_")
        return os.path.join(self.config.base_dir, dataset_name, safe_model_name, split_type, f"fold_{fold_seed}_seed_{run_seed}")

    def create_run_context(self, split_type: str, fold_seed: int, run_seed: int) -> RunContext:
        experiment_dir = self.get_experiment_dir(split_type, fold_seed, run_seed)
        os.makedirs(experiment_dir, exist_ok=True)
        return RunContext(split_type=split_type, fold_seed=fold_seed, run_seed=run_seed, experiment_dir=experiment_dir)


class SMILESRunner:
    model_type = "smiles"

    def __init__(self, config: SMILESConfig, dataset: Optional[SinglePeptideDatasetManager] = None):
        self.config = config
        self.dataset = dataset
        self.task_adapter = build_task_adapter(config.task_type)
        self.experiment_manager = ExperimentManager(config=config, dataset=dataset)

    def _compute_metrics(self, pred: Any) -> Dict[str, float]:
        return self.task_adapter.prediction_to_metrics(pred)

    def _build_runtime(self) -> Tuple[Any, Any, DatasetBuilder]:
        tokenizer = ModelFactory.build_tokenizer(self.config)
        model = ModelFactory.build_model(self.config, self.task_adapter)
        dataset_builder = DatasetBuilder(tokenizer=tokenizer, task_adapter=self.task_adapter, max_length=self.config.max_length)
        return tokenizer, model, dataset_builder

    def _predict_with_trainer(self, model: Any, tokenizer: Any, dataset: Dataset, output_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        trainer = Trainer(
            model=model,
            args=TrainerFactory.build_eval_arguments(self.config, output_dir),
            data_collator=TrainerFactory.build_collator(tokenizer),
        )
        output = trainer.predict(dataset)
        return self.task_adapter.logits_to_predictions(np.array(output.predictions))

    def run(self, run_seed: Optional[int] = None, fold_seed: int = 0, split_type: str = "random_split") -> ModelResults:
        if self.dataset is None:
            raise ValueError("dataset cannot be None in run().")

        effective_run_seed = self.config.seed if run_seed is None else run_seed
        set_seed(effective_run_seed)

        self.dataset.set_official_split_indices(split_type=split_type, fold_seed=fold_seed)
        train_features, valid_features, test_features = self.dataset.get_train_val_test_features("dict")

        run_context = self.experiment_manager.create_run_context(split_type, fold_seed, effective_run_seed)

        tokenizer, model, dataset_builder = self._build_runtime()
        max_length = dataset_builder.infer_max_length(
            [train_features, valid_features, test_features],
            smiles_key=self.config.smiles_key,
            label_key=self.config.label_key,
        )

        train_dataset = dataset_builder.build_dataset(train_features, self.config.smiles_key, self.config.label_key, max_length)
        valid_dataset = dataset_builder.build_dataset(valid_features, self.config.smiles_key, self.config.label_key, max_length)

        trainer = Trainer(
            model=model,
            args=TrainerFactory.build_training_arguments(self.config, run_context.experiment_dir, effective_run_seed),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(self.config.training.early_stopping_patience)],
            data_collator=TrainerFactory.build_collator(tokenizer),
        )

        start = time.time()
        trainer.train()
        training_time = time.time() - start

        model_path = os.path.join(run_context.experiment_dir, "model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        def _eval_split(features: Dict[str, Any]) -> Dict[str, float]:
            ds, labels = dataset_builder.build_prediction_dataset(features, self.config.smiles_key, self.config.label_key, max_length)
            preds, probs = self._predict_with_trainer(model, tokenizer, ds, run_context.experiment_dir)
            return self.task_adapter.evaluate(y_true=labels, preds=preds, probs=probs)

        train_metrics = _eval_split(train_features)
        valid_metrics = _eval_split(valid_features)
        test_metrics = _eval_split(test_features)

        train_smiles, _ = get_smiles_and_labels(train_features, self.config.smiles_key, self.config.label_key)
        valid_smiles, _ = get_smiles_and_labels(valid_features, self.config.smiles_key, self.config.label_key)
        test_smiles, _ = get_smiles_and_labels(test_features, self.config.smiles_key, self.config.label_key)

        param_stats = count_parameters(model)
        results = ModelResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            training_time=training_time,
            train_samples=len(train_smiles),
            valid_samples=len(valid_smiles),
            test_samples=len(test_smiles),
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

        with open(os.path.join(run_context.experiment_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"smiles_config": asdict(self.config)}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_context.experiment_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False)

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
            logger.warning("[RUN_MULTI] run_seeds and fold_seeds length mismatch, truncating to min length")
            min_len = min(len(run_seeds), len(fold_seeds))
            run_seeds = run_seeds[:min_len]
            fold_seeds = fold_seeds[:min_len]

        all_results: List[ModelResults] = []
        for one_run_seed, one_fold_seed in zip(run_seeds, fold_seeds):
            logger.info(f"[RUN_MULTI] running seed={one_run_seed}, fold={one_fold_seed}")
            result = self.run(run_seed=one_run_seed, fold_seed=one_fold_seed, split_type=split_type)
            all_results.append(result)

        logger.info(f"[RUN_MULTI DONE] total runs: {len(all_results)}")
        return MultiRunResults(results=all_results)


# Keep legacy name, but now points to refactored runner style.
class SMILESModel(SMILESRunner):
    pass


MODEL_REGISTRY: Dict[str, Type[SMILESRunner]] = {
    "smiles": SMILESRunner,
}


def build_smiles_model(
    model_type: str = "smiles",
    dataset: Optional[SinglePeptideDatasetManager] = None,
    **kwargs,
) -> SMILESRunner:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_name = kwargs.pop("model_name", kwargs.pop("model", "aaronfeller/PeptideCLM-23M-all"))
    # deprecated: do not expose cache_dir in public API
    kwargs.pop("cache_dir", None)
    local_model = kwargs.pop("local_model", False)
    use_pepclm_tokenizer = kwargs.pop("use_pepclm_tokenizer", False)
    pepclm_vocab_file = kwargs.pop("pepclm_vocab_file", kwargs.pop("tokenizer_vocab_file", None))
    pepclm_spe_file = kwargs.pop("pepclm_spe_file", kwargs.pop("tokenizer_spe_file", None))

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
            use_pepclm_tokenizer=use_pepclm_tokenizer,
            pepclm_vocab_file=pepclm_vocab_file,
            pepclm_spe_file=pepclm_spe_file,
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


def count_parameters(model: Any) -> Dict[str, Any]:
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
