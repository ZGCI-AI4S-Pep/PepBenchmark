from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score
)

Classification_Metric_Map = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "micro-f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'),
    "macro-f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    "roc-auc": roc_auc_score,
    "avg-roc-auc": lambda y_true, y_score: roc_auc_score(y_true, y_score, average='macro', multi_class='ovr'),
    "pr-auc": average_precision_score,
    "kappa": cohen_kappa_score,
}

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np
from scipy.stats import pearsonr, spearmanr

Regression_Metric_Map = {
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
    "mae": mean_absolute_error,
    "r2": r2_score,
    "pcc": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
    # range_logAUC is a custom metric; not available in sklearn
}
