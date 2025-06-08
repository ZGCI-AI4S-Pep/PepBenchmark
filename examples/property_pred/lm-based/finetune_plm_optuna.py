# -*- coding: utf-8 -*-
"""
finetune_plm_auto_tune.py
=========================
Refactored to **remove ``raw_sequence`` from the batch features** (it broke Hugging Face's
``convert_to_tensors``) and to pull the raw sequences directly from the dataset object when
writing prediction files.

Other functionality (auto‑tuning, multi‑split training, metrics saving) remains unchanged.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import pandas as pd
from pepbenchmark.metadata import DATASET_MAP  # noqa: F401
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
MAX_LEN=100
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars",
)

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
class SequenceDatasetWithLabels(Dataset):
    """Tokenises sequences and keeps the originals for later logging."""

    def __init__(self, path: str, tokenizer):
        self.sequences, self.labels = self._load_data(path)
        self.tokenizer = tokenizer

    @staticmethod
    def _load_data(path: str):
        df = pd.read_csv(path)
        return df["sequence"].tolist(), df["label"].tolist()

    def __getitem__(self, idx):
        seq = " ".join(self.sequences[idx])
        features = self.tokenizer(
            seq,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        features["labels"] = self.labels[idx]
        return features

    def __len__(self):
        return len(self.sequences)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def binary_classification_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(labels, preds),
    }


def multi_class_classification_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        n_classes = int(np.max(labels)) + 1
        labels_onehot = label_binarize(labels, classes=np.arange(n_classes))
        roc_auc_ovr = roc_auc_score(labels_onehot, pred.predictions, average="macro", multi_class="ovr")
        roc_auc_ovo = roc_auc_score(labels_onehot, pred.predictions, average="macro", multi_class="ovo")
    except Exception:
        roc_auc_ovr = roc_auc_ovo = None
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "roc_auc_ovr": roc_auc_ovr,
        "roc_auc_ovo": roc_auc_ovo,
    }


def regression_compute_metrics(pred):
    labels = np.array(pred.label_ids).reshape(-1)
    preds = np.array(pred.predictions).reshape(-1)
    mse = mean_squared_error(labels, preds)
    return {
        "mse": mse,
        "mae": mean_absolute_error(labels, preds),
        "rmse": np.sqrt(mse),
        "r2": r2_score(labels, preds),
        "spearman": spearmanr(labels, preds)[0],
        "pcc": pearsonr(labels, preds)[0],
    }


# ---------------------------------------------------------------------------
# Model map (unchanged)
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "prot_bert_bfd": "Rostlab/prot_bert_bfd",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
    "dplm_150m": "airkingbd/dplm_150m",
    "dplm_650m": "airkingbd/dplm_650m",
}


# ---------------------------------------------------------------------------
# Helper: save predictions & metrics
# ---------------------------------------------------------------------------

def save_predictions_and_metrics(predictions_output, prefix: str, output_dir: str, dataset):
    preds = predictions_output.predictions
    labels = predictions_output.label_ids
    metrics = predictions_output.metrics

    # Argmax for classification tasks
    preds = np.argmax(preds, axis=1) if preds.ndim == 2 else preds.flatten()

    # Raw sequences are stored inside the dataset instance
    df_preds = pd.DataFrame({
        "sequence": dataset.sequences,
        "prediction": preds,
        "label": labels,
    })
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(Path(output_dir) / f"{prefix}_metrics.csv", index=False)
    print(f"✅ Saved {prefix} predictions & metrics ▶ {output_dir}")


# ---------------------------------------------------------------------------
# Argument parsing (unchanged except docstring)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Fine‑tune PLMs with optional hyper‑parameter tuning")
    parser.add_argument("--task", type=str, default="Nonfouling", choices=list(DATASET_MAP.keys()))
    parser.add_argument("--split_type", type=str, default="Random_split", choices=["Random_split", "Homology_based_split"])
    parser.add_argument("--split_index", type=str, default="random1", choices=["random1", "random2", "random3", "random4", "random5"])
    parser.add_argument("--model_name", type=str, default="esm2_150M", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--output_dir", type=str)
    # Training defaults (can be overridden by tuning)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="all")
    # Auto‑tune flags
    parser.add_argument("--auto_tune", action="store_true", help="Tune on random1 then apply to random2‑random5")
    parser.add_argument("--n_trials", type=int, default=20, help="Optuna trials when auto‑tuning")
    parser.add_argument("--tag", type=str, help="Optional experiment tag appended to output_dir")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hyper‑parameter search space for Optuna
# ---------------------------------------------------------------------------

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8,16,32,64]),
    }


# ---------------------------------------------------------------------------
# Build TrainingArguments from base args + tuned params
# ---------------------------------------------------------------------------

def build_training_args(base_args, hyperparams: Dict, out_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=hyperparams.get("num_train_epochs", base_args.num_train_epochs),
        learning_rate=hyperparams.get("learning_rate", base_args.learning_rate),
        per_device_train_batch_size=hyperparams.get("per_device_train_batch_size", base_args.per_device_train_batch_size),
        warmup_steps=hyperparams.get("warmup_steps", base_args.warmup_steps),
        weight_decay=hyperparams.get("weight_decay", base_args.weight_decay),
        gradient_accumulation_steps=1,
        lr_scheduler_type="constant",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=base_args.seed,
        report_to=base_args.report_to,
    )


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.task not in DATASET_MAP:
        raise ValueError(f"Unknown task {args.task!r} – choose from {list(DATASET_MAP)}")

    set_seed(args.seed)

    model_ckpt = MODEL_MAP[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Experiment root directory
    exp_root = Path(args.output_dir or Path("checkpoints") / args.task / args.split_type / args.model_name)
    if args.tag:
        exp_root /= args.tag
    exp_root.mkdir(parents=True, exist_ok=True)

    meta = DATASET_MAP[args.task]
    task_type = meta["type"]
    num_labels = 2 if task_type == "binary_classification" else meta.get("num_classes", 1)

    metric_fn = {
        "binary_classification": binary_classification_compute_metrics,
        "multi_class_classification": multi_class_classification_compute_metrics,
        "regression": regression_compute_metrics,
    }[task_type]

    # ----------------------- 1. Optional hyper‑parameter tuning -----------------------
    best_hyperparams = {}

    print("🚀 Hyper‑parameter tuning …")
    split_path = Path(meta["path"]) / args.split_type / args.split_index
    train_ds = SequenceDatasetWithLabels(split_path / "train.csv", tokenizer)
    val_ds = SequenceDatasetWithLabels(split_path / "valid.csv", tokenizer)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)

    tune_args = build_training_args(args, {}, exp_root / "random1_tuning")
    tuner = Trainer(
        args=tune_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metric_fn,
        model_init=model_init,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    best_run = tuner.hyperparameter_search(
        hp_space=hp_space,
        n_trials=args.n_trials,
        direction="minimize",
    )
    best_hyperparams = best_run.hyperparameters
    print(f"🏆 Best hyper‑params found: {best_hyperparams}")
    with open(exp_root / "best_hyperparams.json", "w") as fp:
        json.dump(best_hyperparams, fp, indent=2)
    print("🏆 Best hyper‑params:")
    print(json.dumps(best_hyperparams, indent=2))

    # Override defaults
    for k, v in best_hyperparams.items():
        setattr(args, k, v)

    # # ----------------------- 2. Train/eval on required splits ------------------------
    # splits: List[str] = ["random1", "random2", "random3", "random4", "random5"] if args.auto_tune else [args.split_index]

    # for split in splits:
    #     print(f"\n📂 Processing split {split} …")
    #     out_dir = exp_root / split
    #     out_dir.mkdir(parents=True, exist_ok=True)

    #     split_base = Path(meta["path"]) / args.split_type / split
    #     train_ds = SequenceDatasetWithLabels(split_base / "train.csv", tokenizer)
    #     val_ds = SequenceDatasetWithLabels(split_base / "valid.csv", tokenizer)
    #     test_ds = SequenceDatasetWithLabels(split_base / "test.csv", tokenizer)
    #     print(f"  • Train {len(train_ds)} | Valid {len(val_ds)} | Test {len(test_ds)}")

    #     tr_args = build_training_args(args, best_hyperparams, out_dir)
    #     model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
    #     trainer = Trainer(
    #         model=model,
    #         args=tr_args,
    #         train_dataset=train_ds,
    #         eval_dataset=val_ds,
    #         tokenizer=tokenizer,
    #         compute_metrics=metric_fn,
    #         callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    #     )

    #     trainer.train()

    #     for name, ds in {"train": train_ds, "valid": val_ds, "test": test_ds}.items():
    #         output = trainer.predict(ds, metric_key_prefix=f"eval_{name}")
    #         save_predictions_and_metrics(output, name, out_dir, ds)

    # print("\n🎉 Finished!")


if __name__ == "__main__":
    main()

# WANDB_PROJECT=ttt2 python test.py --num_train_epochs 30   --task AF_APML  --per_device_train_batch_size 16  --early_stopping_patience 5 --weight_decay 0.0 --split_type Homology_based_split --split_index random1 --model_name dplm_150m --auto_tune --n_trials 10 
