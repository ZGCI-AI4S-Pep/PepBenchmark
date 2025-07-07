import argparse
import os
from tokenize import Single
import warnings
from curses import meta

import pandas as pd
from pepbenchmark.evaluator import evaluate_classification, evaluate_regression
from pepbenchmark.single_pred.base_dataset import SingleTaskDatasetManager
import wandb
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from pepbenchmark.utils.seed import set_seed


warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars",
)
import numpy as np
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from pepbenchmark.metadata import DATASET_MAP

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
    logits = pred.predictions  # 通常为 shape = (batch_size, 2)

    # 分类预测（取最大概率类别）
    preds = logits.argmax(-1)
    probs = softmax(logits, axis=-1)[:, 1]  # 取第二列，即正类的概率
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
    parser = argparse.ArgumentParser(description="TrainingArguments parser")
    parser.add_argument(
        "--task",
        type=str,
        default="Nonfouling",
        choices=list(DATASET_MAP.keys()),
        help="Task name from the dataset map",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random_split",
        choices=["random_split", "mmseqs2_split"],
    )
    parser.add_argument(
        "--fold_seed",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t30_150M_UR50D",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="best")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the model checkpoints and outputs",
    )

    return parser.parse_args()


def save_predictions_and_metrics(predictions_output,metrics, prefix, output_dir, dataset):

    preds = predictions_output.predictions
    labels = predictions_output.label_ids

    # 分类任务取argmax
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    else:
        preds = preds.flatten()

    texts = [dataset[i]["raw_sequence"] for i in range(len(dataset))]

    df_preds = pd.DataFrame({"text": texts, "prediction": preds, "label": labels})

    df_preds.to_csv(os.path.join(output_dir, f"{prefix}_predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, f"{prefix}_metrics.csv"), index=False
    )

    print(f"✅ Saved {prefix} predictions and metrics (with text) to {output_dir}")


if __name__ == "__main__":
    args = parse_args()

    if args.task not in DATASET_MAP.keys():
        raise ValueError(
            f"Task {args.task} is not supported. Please choose from {list(DATASET_MAP.keys())}."
        )
    


    set_seed(args.seed)
    dataset_manager = SingleTaskDatasetManager(dataset_name=args.task,official_feature_names=["fasta","label"])

    dataset_manager.set_official_split_indices(
        split_type=args.split_type, fold_seed=args.fold_seed
    )

    train_features, valid_features, test_features = dataset_manager.get_train_val_test_features(format="dict")

    dataset_metadata = dataset_manager.get_dataset_metadata()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_len = dataset_metadata.get("max_len", 200)

    train_dataset = SequenceDatasetWithLabels(train_features["official_fasta"],train_features["official_label"], tokenizer, max_len=max_len)
    valid_dataset = SequenceDatasetWithLabels(valid_features["official_fasta"], valid_features["official_label"], tokenizer, max_len=max_len)
    test_dataset = SequenceDatasetWithLabels(test_features["official_fasta"], test_features["official_label"], tokenizer, max_len=max_len)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.save_dir, args.task, args.split_type, args.model_name, str(args.fold_seed)
        )

    task_type = dataset_metadata["type"]
    if task_type not in [
        "binary_classification",
        "multi_class_classification",
        "regression",
    ]:
        raise ValueError(
            f"Task type {task_type} is not supported. Please choose from 'binary_classification' or 'multi_class_classification'. (current task_type: {task_type})"
        )

    if task_type == "binary_classification":
        num_labels = 2
        compute_metrics = binary_classification_compute_metrics
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )
    elif task_type == "multi_class_classification":
        pass

    elif task_type == "regression":
        num_labels = 1
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
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
    trainer.train()

    # Predict and save train
    train_output = trainer.predict(train_dataset, metric_key_prefix="test_train")
    # 删除前缀
    metrics = {k.replace("test_train_", ""): v for k, v in train_output.metrics.items()}
    save_predictions_and_metrics(train_output, metrics,"train", args.output_dir, train_dataset)

    # Predict and save valid
    valid_output = trainer.predict(valid_dataset, metric_key_prefix="test_valid")
    # 删除前缀
    metrics = {k.replace("test_valid_", ""): v for k, v in valid_output.metrics.items()}
    save_predictions_and_metrics(valid_output, metrics,"valid", args.output_dir, valid_dataset)

    # Predict and save test
    test_output = trainer.predict(test_dataset, metric_key_prefix="test_test")
    # 删除前缀  
    metrics = {k.replace("test_test_", ""): v for k, v in test_output.metrics.items()}
    save_predictions_and_metrics(test_output, metrics,"test", args.output_dir, test_dataset)


# WANDB_PROJECT=ttt python finetune_plm.py --num_train_epochs 1  --load_best_model_at_end  --task AV_APML  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 5e-5 --early_stopping_patience 5 --weight_decay 0.0 --split_type random_split --fold_seed 0 --model_name facebook/esm2_t30_150M_UR50D
