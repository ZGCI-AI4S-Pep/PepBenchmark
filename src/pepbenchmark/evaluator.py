"""Evaluation helpers for classification and regression benchmarks."""

# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from pepbenchmark.utils.logging import get_logger

logger = get_logger()
# ---------------------------------------------------------------------------
# Helper implementations for metrics not directly shipped with scikit‑learn
# ---------------------------------------------------------------------------


def specificity_score(
    y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
) -> float:
    """Compute specificity for binary classification predictions.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted binary labels.

    Returns:
        The true-negative rate.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("specificity_score currently supports binary tasks only")
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def g_mean_score(
    y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
) -> float:
    """Compute the geometric mean of sensitivity and specificity.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted binary labels.

    Returns:
        The geometric mean score.
    """
    sensitivity = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return np.sqrt(sensitivity * specificity)


# ---------------------------------------------------------------------------
# Metric maps
# ---------------------------------------------------------------------------
# Classification metrics dictionary – maps *public* metric names to callables.
Classification_Metric_Map: Dict[str, callable] = {
    # Basic
    "accuracy": accuracy_score,
    "balanced-accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "specificity": specificity_score,
    "f1": f1_score,
    "micro-f1": lambda yt, yp: f1_score(yt, yp, average="micro"),
    "macro-f1": lambda yt, yp: f1_score(yt, yp, average="macro"),
    "weighted-f1": lambda yt, yp: f1_score(yt, yp, average="weighted"),
    # Correlation / agreement
    "mcc": matthews_corrcoef,
    "kappa": cohen_kappa_score,
    # Composite / diagnostic
    "g-mean": g_mean_score,
    # Probability‑aware metrics
    "roc-auc": roc_auc_score,
    "avg-roc-auc": lambda yt, ys: roc_auc_score(
        yt, ys, average="macro", multi_class="ovr"
    ),
    "pr-auc": average_precision_score,
    "brier-score": brier_score_loss,
    "log-loss": log_loss,
    # Rank‑/top‑k‑based
}
Classification_Metric = Classification_Metric_Map.keys()
# Regression metrics dictionary – maps names to functions
Regression_Metric_Map: Dict[str, callable] = {
    "mse": mean_squared_error,
    "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
    "mae": mean_absolute_error,
    "r2": r2_score,
    "pcc": lambda yt, yp: pearsonr(yt, yp)[0],
    "spearman": lambda yt, yp: spearmanr(yt, yp)[0],
}
Regression_Metric=Regression_Metric_Map.keys()
# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_classification(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_score: Union[List, np.ndarray] | None = None,
    metrics: List[str] | None = None,
) -> Dict[str, float]:
    """Evaluate a classification task with one or more metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Discrete class predictions aligned with ``y_true``.
        y_score: Optional probability or score values required by
            probability-aware metrics such as ``roc-auc``.
        metrics: Optional list of metric names. When omitted, all registered
            classification metrics are attempted.

    Returns:
        A mapping from metric name to computed value.
    """
    if metrics is None:
        metrics = list(Classification_Metric_Map.keys())

    results: Dict[str, float] = {}

    for name in metrics:
        if name not in Classification_Metric_Map:
            logger.warning(f"Unknown metric '{name}' – skipped")
            continue

        fn = Classification_Metric_Map[name]
        prob_required = name in {
            "roc-auc",
            "avg-roc-auc",
            "pr-auc",
            "brier-score",
            "log-loss",
            "top-5-accuracy",
        }
        try:
            if prob_required:
                if y_score is None:
                    logger.warning(f"{name} requires probability 'y_score' – skipped")
                    continue
                results[name] = fn(y_true, y_score)
            else:
                results[name] = fn(y_true, y_pred)
        except Exception as exc:
            logger.error(f"Error computing {name}: {exc}")
    return results


def evaluate_regression(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    metrics: List[str] | None = None,
) -> Dict[str, float]:
    """Evaluate a regression task with one or more metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        metrics: Optional list of regression metric names to compute.

    Returns:
        A mapping from metric name to computed value.
    """
    if metrics is None:
        metrics = list(Regression_Metric_Map.keys())

    results: Dict[str, float] = {}
    for name in metrics:
        if name not in Regression_Metric_Map:
            logger.warning(f"Unknown metric '{name}' – skipped")
            continue
        try:
            results[name] = Regression_Metric_Map[name](y_true, y_pred)
        except Exception as exc:
            logger.error(f"Error computing {name}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Recommendations & convenience wrappers
# ---------------------------------------------------------------------------


# TODO: Recommended metrics should be dataset-dependent; we'll look at this after running the results.
def get_recommended_metrics(task_type: str) -> List[str]:
    """Return a task-specific default metric subset.

    Args:
        task_type: Task name such as ``"binary_classification"`` or
            ``"regression"``.

    Returns:
        A list of recommended metric names.
    """
    recs = {
        "binary_classification": [
            "accuracy",
            "balanced-accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "roc-auc",
            "pr-auc",
            "brier-score",
        ],
        "multiclass_classification": [
            "accuracy",
            "balanced-accuracy",
            "macro-f1",
            "weighted-f1",
            "avg-roc-auc",
            "kappa",
        ],
        "regression": [
            "mse",
            "rmse",
            "mae",
            "r2",
            "pcc",
            "spearman",
        ],
    }
    return recs.get(task_type.lower(), [])


def compute_all_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    task_type: str,
    y_score: Union[List, np.ndarray] | None = None,
) -> Dict[str, float]:
    """Compute the recommended metrics for a given task type.

    Args:
        y_true: Ground-truth labels or targets.
        y_pred: Predicted labels or targets.
        task_type: Task identifier used to choose the default metrics.
        y_score: Optional score or probability array for classification tasks.

    Returns:
        A mapping from metric name to computed value.
    """
    task_type = task_type.lower()
    if "classification" in task_type:
        return evaluate_classification(
            y_true, y_pred, y_score, get_recommended_metrics(task_type)
        )
    elif "regression" in task_type:
        return evaluate_regression(y_true, y_pred, get_recommended_metrics(task_type))
    else:
        raise ValueError(f"Unknown task type: {task_type}")
