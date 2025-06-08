import argparse
from curses import meta
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
import pandas as pd
import wandb
from pepbenchmark.metadata import get_dataset_path
from torch.utils.data import Dataset
import os 
from transformers import set_seed
import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from pepbenchmark.metadata import DATASET_MAP
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

class SequenceDatasetWithLabels(Dataset):
    def __init__(self, path,tokenizer,max_len=200):
        self.sequences,self.labels = load_data(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __getitem__(self, idx):
        seq = " ".join(self.sequences[idx])
        seq_ids = self.tokenizer(seq, add_special_tokens=True,padding='max_length', truncation=True, max_length=self.max_len)
        seq_ids["labels"] = self.labels[idx]
        seq_ids["raw_sequence"] = self.sequences[idx]
        return seq_ids
    def __len__(self):
        return len(self.sequences)

def load_data(path):
    """
    Load sequences and labels from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        tuple: A tuple containing sequences and labels.
    """

    df = pd.read_csv(path)
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    return sequences, labels



def binary_classification_compute_metrics(pred):
    """Compute metrics for binary classification tasks."""
    
    labels = pred.label_ids
    logits = pred.predictions  # 通常为 shape = (batch_size, 2)

    # 分类预测（取最大概率类别）
    preds = logits.argmax(-1)

    # 获取正类的概率（用于 ROC AUC）
    probs = softmax(logits, axis=-1)[:, 1]  # 取第二列，即正类的概率

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "micro_f1": f1_score(labels, preds, average='micro'),
        "macro_f1": f1_score(labels, preds, average='macro', zero_division=0),
        "roc_auc": roc_auc_score(labels, probs)
    }

    

def multi_class_classification_compute_metrics(pred):
    """
    Compute metrics for multi-class classification tasks, including average ROC AUC.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    # ROC AUC for multi-class: need one-hot encoding for labels
    try:
        n_classes = np.max(labels) + 1
        labels_onehot = label_binarize(labels, classes=np.arange(n_classes))
        roc_auc_ovr = roc_auc_score(labels_onehot, pred.predictions, average='macro', multi_class='ovr')
        roc_auc_ovo = roc_auc_score(labels_onehot, pred.predictions, average='macro', multi_class='ovo')
    except Exception:
        roc_auc_ovr = None
        roc_auc_ovo = None

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'roc_auc_ovr': roc_auc_ovr,
        'roc_auc_ovo': roc_auc_ovo,
    }

def regression_compute_metrics(pred):
    """Compute metrics for regression tasks."""
    
    labels = pred.label_ids
    preds = pred.predictions
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)
    pcc, _ = pearsonr(labels, preds)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman': spearman_corr,
        'pcc': pcc,
    }


model_map = {
    "prot_bert_bfd": "Rostlab/prot_bert_bfd",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
    "dplm_150m":"airkingbd/dplm_150m",
    "dplm_650m":"airkingbd/dplm_650m"
}


def parse_args():
    parser = argparse.ArgumentParser(description="TrainingArguments parser")
    parser.add_argument('--task', type=str, default="Nonfouling",choices=list(DATASET_MAP.keys()), help="Task name from the dataset map")
    parser.add_argument('--split_type', type=str, default="Random_split",choices=["Random_split", "Homology_based_split"])    
    parser.add_argument('--split_index', type=str, default="random1",choices=["random1", "random2", "random3", "random4", "random5"])
    parser.add_argument('--model_name', type=str, default="esm2_150M",choices=list(model_map.keys()), help="Model name from the model map")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--eval_strategy', type=str, default='epoch')
    parser.add_argument('--save_strategy', type=str, default='best')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_best_model_at_end', action='store_true')
    parser.add_argument('--report_to', type=str, default='all')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--tag', type=str, default=None)

    return parser.parse_args()


def save_predictions_and_metrics(predictions_output, prefix, output_dir, dataset):
    """
    保存预测结果 + 原始文本 + 标签 + 预测 + 评估指标
    """
    preds = predictions_output.predictions
    labels = predictions_output.label_ids
    metrics = predictions_output.metrics

    # 分类任务取argmax
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    else:
        preds = preds.flatten()

    # 提取原始文本（要求 dataset[i]["raw_sequence"] 存在）
    texts = [dataset[i]["raw_sequence"] for i in range(len(dataset))]

    df_preds = pd.DataFrame({
        "text": texts,
        "prediction": preds,
        "label": labels
    })

    df_preds.to_csv(os.path.join(output_dir, f"{prefix}_predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, f"{prefix}_metrics.csv"), index=False)

    print(f"✅ Saved {prefix} predictions and metrics (with text) to {output_dir}")

if __name__ == "__main__":
    args = parse_args()


    if args.task not in DATASET_MAP.keys():
        raise ValueError(f"Task {args.task} is not supported. Please choose from {list(DATASET_MAP.keys())}.")
    dataset_metadata = DATASET_MAP.get(args.task)
    train_path = os.path.join(dataset_metadata["path"], args.split_type,args.split_index,"train.csv")
    valid_path = os.path.join(dataset_metadata["path"],  args.split_type,args.split_index,"valid.csv")
    test_path = os.path.join(dataset_metadata["path"],  args.split_type,args.split_index,"test.csv")

    set_seed(args.seed)
    model_name = model_map[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len= dataset_metadata.get("max_len", 200)

    train_dataset = SequenceDatasetWithLabels(train_path, tokenizer,max_len=max_len)
    valid_dataset = SequenceDatasetWithLabels(valid_path, tokenizer,max_len=max_len)
    test_dataset = SequenceDatasetWithLabels(test_path, tokenizer,max_len=max_len)




    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    

    if args.output_dir is None:
        args.output_dir  = os.path.join("checkpoints",args.task, args.split_type, model_name, args.split_index)     
        
    if args.tag is not None:
        args.output_dir = os.path.join(args.output_dir, args.tag)
        
    task_type = dataset_metadata["type"]
    if task_type not in ["binary_classification", "multi_class_classification", "regression"]:
        
        
        raise ValueError(f"Task type {task_type} is not supported. Please choose from 'binary_classification' or 'multi_class_classification'. (current task_type: {task_type})")
    
    if task_type == "binary_classification":
        num_labels = 2
        compute_metrics = binary_classification_compute_metrics
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)     
    elif task_type == "multi_class_classification":
        num_labels = dataset_metadata["num_classes"]
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)   
        compute_metrics = multi_class_classification_compute_metrics
    elif task_type == "regression":
        num_labels = 1
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        compute_metrics = regression_compute_metrics
        
        
    # wandb.init(name=str(args.output_dir))
    # wandb.define_metric("*", step_metric="epoch")
    
    
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
        compute_metrics = compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    )
    trainer.train()
        
    # Predict and save train
    train_output = trainer.predict(train_dataset,metric_key_prefix="test_train")
    save_predictions_and_metrics(train_output, "train", args.output_dir,train_dataset)

    # Predict and save valid
    valid_output = trainer.predict(valid_dataset,metric_key_prefix="test_valid")
    save_predictions_and_metrics(valid_output, "valid", args.output_dir,valid_dataset)

    # Predict and save test
    test_output = trainer.predict(test_dataset,metric_key_prefix="test_test")
    save_predictions_and_metrics(test_output, "test", args.output_dir,test_dataset)
    
    
# WANDB_PROJECT=AF_APML python finetune_plm.py --num_train_epochs 30  --load_best_model_at_end  --task AF_APML  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 5e-5 --early_stopping_patience 5 --weight_decay 0.0 --split_type Random_split --split_index random1 --model_name esm2_150M
# WANDB_PROJECT=P.aeruginosa python finetune_plm.py --num_train_epochs 30  --load_best_model_at_end  --task P.aeruginosa  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 5e-5 --early_stopping_patience 5 --weight_decay 0.0 --split_type Random_split --split_index random1 --model_name esm2_150M



# WANDB_PROJECT=AF_APML python finetune_plm.py --num_train_epochs 100  --load_best_model_at_end  --task AF_APML  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-5 --early_stopping_patience 10 --weight_decay 0.0 --split_type Random_split --split_index random1 --model_name prot_bert_bfd --output_dir test_lr_0.00001