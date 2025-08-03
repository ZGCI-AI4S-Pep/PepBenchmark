import argparse
import json
import os
import warnings

import pandas as pd
from pepbenchmark.evaluator import evaluate_classification, evaluate_regression


from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager
from pepbenchmark.utils.seed import set_seed
import os
os.environ["WANDB_DISABLED"] = "true"

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars",
)
import numpy as np
from scipy.special import softmax
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from pepbenchmark.raw_data import DATASET_MAP

class SequenceDatasetWithLabels(Dataset):
    def __init__(self, sequences,labels, tokenizer, max_len=200):
        self.sequences, self.labels = sequences, labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        seq = " ".join(self.sequences[idx])
        seq_ids = self.tokenizer(
            seq,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        seq_ids["labels"] = self.labels[idx]
        seq_ids["raw_sequence"] = self.sequences[idx]
        return seq_ids

    def __len__(self):
        return len(self.sequences)


def binary_classification_compute_metrics(pred):
    """Compute metrics for binary classification tasks."""

    labels = pred.label_ids
    logits = pred.predictions  

    preds = logits.argmax(-1)
    probs = softmax(logits, axis=-1)[:, 1]
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


def parse_args():
    parser = argparse.ArgumentParser(description="PLM-based property prediction")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(DATASET_MAP.keys()),
        help="Task name from the dataset map",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random_split",
        choices=["random_split", "mmseqs2_split", "cdhit_split"],
        help="Split type",
    )
    parser.add_argument(
        "--fold_seed",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Fold seed for split",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t30_150M_UR50D",
        help="Pre-trained model name or path",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for model and results")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="best", help="Save strategy")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end")
    parser.add_argument("--report_to", type=str, default="all", help="Report to (wandb, tensorboard, etc.)")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    return parser.parse_args()





if __name__ == "__main__":
    print("Starting PLM-based property prediction...")
    args = parse_args()
    print(f"Arguments: {args}")

    if args.task not in DATASET_MAP.keys():
        raise ValueError(
            f"Task {args.task} is not supported. Please choose from {list(DATASET_MAP.keys())}."
        )

    set_seed(args.seed)

    dataset_manager = SingleTaskDatasetManager(dataset_name=args.task, official_feature_names=["fasta", "label"])

    dataset_manager.set_official_split_indices(
        split_type=args.split_type, fold_seed=args.fold_seed
    )

    train_features, valid_features, test_features = dataset_manager.get_train_val_test_features(format="dict")

    dataset_metadata = dataset_manager.get_dataset_metadata()
    max_len = dataset_metadata.get("max_len", 200)
    task_type = dataset_metadata["type"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/home/dataset-assist-0/rouyi/rouyi/.hug_cache/hub", local_files_only=True)

    train_dataset = SequenceDatasetWithLabels(train_features["official_fasta"], train_features["official_label"], tokenizer, max_len=max_len)
    valid_dataset = SequenceDatasetWithLabels(valid_features["official_fasta"], valid_features["official_label"], tokenizer, max_len=max_len)
    test_dataset = SequenceDatasetWithLabels(test_features["official_fasta"], test_features["official_label"], tokenizer, max_len=max_len)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Prepare output directory
    args.output_dir = os.path.join(
        args.output_dir, args.task, args.split_type, args.model_name.replace("/", "_"), str(args.fold_seed)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if task_type not in [
        "binary_classification",
        "multi_class_classification",
        "regression",
    ]:
        raise ValueError(
            f"Task type {task_type} is not supported. Please choose from 'binary_classification', 'multi_class_classification', or 'regression'. (current task_type: {task_type})"
        )

    if task_type == "binary_classification":
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels, cache_dir="/home/dataset-assist-0/rouyi/rouyi/.hug_cache/hub", local_files_only=True
        )
        compute_metrics = binary_classification_compute_metrics
    # TODO: multi_class_classification
    elif task_type == "multi_class_classification":
        raise NotImplementedError("Multi-class classification is not yet implemented")
    elif task_type == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=1, cache_dir="/home/dataset-assist-0/rouyi/rouyi/.hug_cache/hub", local_files_only=True
        )
        compute_metrics = regression_compute_metrics


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.per_device_train_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        load_best_model_at_end=args.load_best_model_at_end,
        report_to=args.report_to,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )
    
    print(f"Training model '{args.model_name}'...")
    trainer.train()
    print("Training complete. Evaluating...")

    # Save training arguments
    training_config = {
        "model_name": args.model_name,
        "task": args.task,
        "split_type": args.split_type,
        "fold_seed": args.fold_seed,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "task_type": task_type,
        "max_len": max_len,
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
    print("Done.")





# WANDB_PROJECT=ttt python finetune_plm.py --task AV_APML --num_train_epochs 1 --load_best_model_at_end --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 5e-5 --early_stopping_patience 5 --weight_decay 0.0 --split_type random_split --fold_seed 0 --model_name facebook/esm2_t30_150M_UR50D
