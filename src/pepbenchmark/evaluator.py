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
from pepbenchmark.utils.logging import get_logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = get_logger()
# Classification metrics dictionary
# Maps metric names to their corresponding functions
Classification_Metric_Map = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "micro-f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
    "macro-f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "roc-auc": roc_auc_score,
    "avg-roc-auc": lambda y_true, y_score: roc_auc_score(
        y_true, y_score, average="macro", multi_class="ovr"
    ),
    "pr-auc": average_precision_score,
    "kappa": cohen_kappa_score,
}

# Regression metrics dictionary
# Maps metric names to their corresponding functions
Regression_Metric_Map = {
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
    "mae": mean_absolute_error,
    "r2": r2_score,
    "pcc": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
}


def evaluate_classification(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_score: Union[List, np.ndarray] = None,
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    Evaluate classification performance using multiple metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Prediction scores/probabilities (required for ROC-AUC, PR-AUC)
        metrics: List of metric names to compute. If None, computes all available.

    Returns:
        Dictionary mapping metric names to computed values

    Examples:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> results = evaluate_classification(y_true, y_pred, metrics=['accuracy', 'f1'])
        >>> print(f"Accuracy: {results['accuracy']:.3f}")
    """
    if metrics is None:
        metrics = list(Classification_Metric_Map.keys())

    results = {}

    for metric_name in metrics:
        if metric_name not in Classification_Metric_Map:
            logger.warning(f"Unknown metric '{metric_name}' skipped")
            continue

        metric_fn = Classification_Metric_Map[metric_name]

        try:
            # Some metrics require prediction scores instead of labels
            if metric_name in ["roc-auc", "avg-roc-auc", "pr-auc"]:
                if y_score is None:
                    logger.warning(f"{metric_name} requires y_score, skipped")
                    continue
                results[metric_name] = metric_fn(y_true, y_score)
            else:
                results[metric_name] = metric_fn(y_true, y_pred)
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")

    return results


def evaluate_regression(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    Evaluate regression performance using multiple metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metric names to compute. If None, computes all available.

    Returns:
        Dictionary mapping metric names to computed values

    Examples:
        >>> y_true = [1.2, 2.3, 3.1, 4.5]
        >>> y_pred = [1.1, 2.4, 3.0, 4.3]
        >>> results = evaluate_regression(y_true, y_pred, metrics=['mse', 'pcc'])
        >>> print(f"MSE: {results['mse']:.3f}")
        >>> print(f"PCC: {results['pcc']:.3f}")
    """
    if metrics is None:
        metrics = list(Regression_Metric_Map.keys())

    results = {}

    for metric_name in metrics:
        if metric_name not in Regression_Metric_Map:
            logger.warning(f"Unknown metric '{metric_name}' skipped")
            continue

        metric_fn = Regression_Metric_Map[metric_name]

        try:
            results[metric_name] = metric_fn(y_true, y_pred)
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")

    return results


def get_recommended_metrics(task_type: str) -> List[str]:
    """
    Get recommended metrics for different task types.

    Args:
        task_type: Type of task ('binary_classification', 'multiclass_classification', 'regression')

    Returns:
        List of recommended metric names

    Examples:
        >>> metrics = get_recommended_metrics('binary_classification')
        >>> print(metrics)
        ['accuracy', 'f1', 'roc-auc', 'pr-auc']
    """
    recommendations = {
        "binary_classification": [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "roc-auc",
            "pr-auc",
        ],
        "multiclass_classification": [
            "accuracy",
            "macro-f1",
            "micro-f1",
            "avg-roc-auc",
            "kappa",
        ],
        "regression": ["mse", "rmse", "mae", "r2", "pcc", "spearman"],
    }

    return recommendations.get(task_type, [])


def compute_all_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    task_type: str,
    y_score: Union[List, np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a given task type.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: Type of task ('binary_classification', 'multiclass_classification', 'regression')
        y_score: Prediction scores (for classification tasks requiring probabilities)

    Returns:
        Dictionary of all computed metrics

    Examples:
        >>> # Binary classification
        >>> results = compute_all_metrics([0,1,1,0], [0,1,0,0], 'binary_classification')
        >>>
        >>> # Regression
        >>> results = compute_all_metrics([1.2,2.3], [1.1,2.4], 'regression')
    """
    if "classification" in task_type:
        return evaluate_classification(
            y_true, y_pred, y_score, get_recommended_metrics(task_type)
        )
    elif task_type == "regression":
        return evaluate_regression(y_true, y_pred, get_recommended_metrics(task_type))
    else:
        raise ValueError(f"Unknown task type: {task_type}")
