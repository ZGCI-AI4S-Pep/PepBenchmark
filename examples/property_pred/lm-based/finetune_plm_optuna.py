# -*- coding: utf-8 -*-
"""
finetune_plm_optuna.py
=====================
PLM-based property prediction with Optuna hyperparameter tuning.
Supports automatic hyperparameter optimization and evaluation across multiple splits.
"""

import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from pepbenchmark.metadata import DATASET_MAP
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager
from pepbenchmark.evaluator import evaluate_classification, evaluate_regression
from pepbenchmark.utils.seed import set_seed

MAX_LEN = 200
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars",
)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
class SequenceDatasetWithLabels(Dataset):
    """Tokenises sequences and keeps the originals for later logging."""

    def __init__(self, sequences, labels, tokenizer, max_len=200):
        self.sequences, self.labels = sequences, labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        seq = " ".join(self.sequences[idx])
        features = self.tokenizer(
            seq,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        features["labels"] = self.labels[idx]
        return features

    def __len__(self):
        return len(self.sequences)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def binary_classification_compute_metrics(pred):
    """Compute metrics for binary classification tasks."""
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    probs = softmax(logits, axis=-1)[:, 1]
    metrics = evaluate_classification(y_true=labels, y_pred=preds, y_score=probs)
    return metrics


def multi_class_classification_compute_metrics(pred):
    """Compute metrics for multi-class classification tasks."""
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    probs = softmax(logits, axis=-1)
    metrics = evaluate_classification(y_true=labels, y_pred=preds, y_score=probs)
    return metrics


def regression_compute_metrics(pred):
    """Compute metrics for regression tasks."""
    labels = pred.label_ids
    preds = pred.predictions
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)
    metrics = evaluate_regression(labels, preds)
    return metrics


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="PLM-based property prediction with Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        choices=list(DATASET_MAP.keys()),
        help="Task name from the dataset map"
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random_split",
        choices=["random_split", "mmseqs2_split", "cdhit_split"],
        help="Split type"
    )
    parser.add_argument(
        "--fold_seed",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Fold seed for split"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="facebook/esm2_t30_150M_UR50D",
        help="Pre-trained model name or path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints",
        help="Output directory for model and results"
    )
    # Training defaults (can be overridden by tuning)
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64, help="Training batch size per device")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_to", type=str, default="all", help="Report to (wandb, tensorboard, etc.)")
    # Optuna tuning flags
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials for hyperparameter tuning"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hyper‑parameter search space for Optuna
# ---------------------------------------------------------------------------

def create_objective(args, train_dataset, val_dataset, tokenizer, model_ckpt, task_type, max_len):
    """Create objective function for Optuna optimization"""
    
    def objective(trial):
        # Suggest hyperparameters
        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32, 64]
            ),
        }
        
        # Determine number of labels
        if task_type == "binary_classification":
            num_labels = 2
            compute_metrics = binary_classification_compute_metrics
        elif task_type == "multi_class_classification":
            # Get from dataset metadata
            dataset_metadata = DATASET_MAP[args.task]
            num_labels = dataset_metadata.get("num_classes", 2)
            compute_metrics = multi_class_classification_compute_metrics
        elif task_type == "regression":
            num_labels = 1
            compute_metrics = regression_compute_metrics
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Create model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, local_files_only=True
        )
        
        # Build training arguments
        training_args = TrainingArguments(
            output_dir=f"./temp_trial_{trial.number}",
            num_train_epochs=args.num_train_epochs,
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=hyperparams["per_device_train_batch_size"],
            warmup_steps=args.warmup_steps,
            weight_decay=hyperparams["weight_decay"],
            gradient_accumulation_steps=1,
            lr_scheduler_type="constant",
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="best",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=args.seed,
            report_to="none",  # Disable logging for trials
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
            ],
        )
        
        # Train and get validation loss
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(f"./temp_trial_{trial.number}"):
            shutil.rmtree(f"./temp_trial_{trial.number}")
        
        return eval_results["eval_loss"]
    
    return objective


def create_final_model(args, best_params, model_ckpt, task_type):
    """Create final model with best parameters"""
    if task_type == "binary_classification":
        num_labels = 2
    elif task_type == "multi_class_classification":
        dataset_metadata = DATASET_MAP[args.task]
        num_labels = dataset_metadata.get("num_classes", 2)
    elif task_type == "regression":
        num_labels = 1
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=num_labels, local_files_only=True
    )


# ---------------------------------------------------------------------------
# Build TrainingArguments from base args + tuned params
# ---------------------------------------------------------------------------


def build_training_args(base_args, hyperparams, out_dir) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=hyperparams.get("num_train_epochs", base_args.num_train_epochs),
        learning_rate=hyperparams.get("learning_rate", base_args.learning_rate),
        per_device_train_batch_size=hyperparams.get(
            "per_device_train_batch_size", base_args.per_device_train_batch_size
        ),
        warmup_steps=hyperparams.get("warmup_steps", base_args.warmup_steps),
        weight_decay=hyperparams.get("weight_decay", base_args.weight_decay),
        gradient_accumulation_steps=base_args.gradient_accumulation_steps,
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
    print(f"Arguments: {args}")

    if args.task not in DATASET_MAP:
        raise ValueError(
            f"Task {args.task} is not supported. Please choose from {list(DATASET_MAP.keys())}."
        )

    set_seed(args.seed)

    # Load dataset
    dataset_manager = SingleTaskDatasetManager(
        dataset_name=args.task, 
        official_feature_names=["fasta", "label"]
    )
    dataset_manager.set_official_split_indices(
        split_type=args.split_type, 
        fold_seed=args.fold_seed
    )

    train_features, valid_features, test_features = dataset_manager.get_train_val_test_features(format="dict")
    
    dataset_metadata = dataset_manager.get_dataset_metadata()
    max_len = dataset_metadata.get("max_len", 200)
    task_type = dataset_metadata["type"]

    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_dataset = SequenceDatasetWithLabels(
        train_features["official_fasta"], 
        train_features["official_label"], 
        tokenizer, 
        max_len=max_len
    )
    valid_dataset = SequenceDatasetWithLabels(
        valid_features["official_fasta"], 
        valid_features["official_label"], 
        tokenizer, 
        max_len=max_len
    )
    test_dataset = SequenceDatasetWithLabels(
        test_features["official_fasta"], 
        test_features["official_label"], 
        tokenizer, 
        max_len=max_len
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Prepare output directory
    args.output_dir = os.path.join(
        args.output_dir, 
        args.task, 
        args.split_type, 
        args.model_name.replace("/", "_"), 
        str(args.fold_seed)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if task_type not in ["binary_classification", "multi_class_classification", "regression"]:
        raise ValueError(
            f"Task type {task_type} is not supported. Please choose from 'binary_classification', 'multi_class_classification', or 'regression'."
        )

    # Hyperparameter tuning
    print(f"Starting Optuna tuning for model '{args.model_name}' with {args.n_trials} trials...")
    
    objective = create_objective(args, train_dataset, valid_dataset, tokenizer, args.model_name, task_type, max_len)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    
    best_params = study.best_params
    best_loss = study.best_value
    
    # Save best parameters
    params_path = os.path.join(args.output_dir, f"best_params_{args.model_name.replace('/', '_')}.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best params for {args.model_name}: {best_params}")
    print(f"Saved best params to {params_path}")
    
    # Save tuning results
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(args.output_dir, f"tuning_trials_{args.model_name.replace('/', '_')}.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"Tuning trials saved to {trials_path}")
    
    print("Hyperparameter tuning completed.")
    
    # Train final model with best parameters and evaluate
    print(f"\nTraining final model with best parameters...")
    final_model = create_final_model(args, best_params, args.model_name, task_type)
    
    # Get compute metrics function
    if task_type == "binary_classification":
        compute_metrics = binary_classification_compute_metrics
    elif task_type == "multi_class_classification":
        compute_metrics = multi_class_classification_compute_metrics
    elif task_type == "regression":
        compute_metrics = regression_compute_metrics
    
    # Build training arguments with best parameters
    training_args = build_training_args(args, best_params, args.output_dir)
    
    trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )
    
    trainer.train()
    print("Training complete. Evaluating...")
    
    # Save training config
    training_config = {
        "model_name": args.model_name,
        "task": args.task,
        "split_type": args.split_type,
        "fold_seed": args.fold_seed,
        "task_type": task_type,
        "max_len": max_len,
        "best_hyperparams": best_params,
        "n_trials": args.n_trials,
        "seed": args.seed,
    }
    
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Training config saved to {config_path}")

    # Evaluate on all splits
    results = []
    datasets = [
        ("Train", train_dataset),
        ("Validation", valid_dataset), 
        ("Test", test_dataset)
    ]
    
    for name, dataset in datasets:
        output = trainer.predict(dataset, metric_key_prefix=f"eval_{name.lower()}")
        metrics = {k.replace(f"eval_{name.lower()}_", ""): v for k, v in output.metrics.items()}
        
        print(f"{name} set metrics: {metrics}")
        results.append({
            "Split": name,
            "Model": args.model_name,
            **metrics,
        })

    # Save model
    model_path = os.path.join(args.output_dir, "model")
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    print("Hyperparameter tuning and final model training completed.")
    print("Done.")


if __name__ == "__main__":
    main()

# Example usage:
# WANDB_PROJECT=ttt python finetune_plm_optuna.py --task AV_APML --num_train_epochs 30 --per_device_train_batch_size 16 --early_stopping_patience 5 --weight_decay 0.0 --split_type random_split --fold_seed 0 --model_name facebook/esm2_t30_150M_UR50D --n_trials 10
