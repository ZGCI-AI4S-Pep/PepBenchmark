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

"""
Outlier removal utility module
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


def remove_outliers(
    df: pd.DataFrame,
    label_column: str = "label",
    method: str = "iqr",
    threshold: float = 1.5,
    z_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Remove outliers for regression tasks

    Args:
        df (pd.DataFrame): DataFrame containing sequences and labels
        label_column (str): Label column name, default is "label"
        method (str): Outlier detection method, supports "iqr" or "zscore"
        threshold (float): Threshold multiplier for IQR method, default is 1.5
        z_threshold (float): Z-score method threshold, default is 3.0

    Returns:
        Tuple[pd.DataFrame, List[int]]: (DataFrame after outlier removal, list of outlier indices)
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    if df[label_column].dtype not in ["float64", "float32", "int64", "int32"]:
        logger.warning(
            f"Label column '{label_column}' is not numeric. Converting to float."
        )
        df = df.copy()
        df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    original_count = len(df)
    labels = np.array(df[label_column])

    logger.info(f"Removing outliers using {method} method")

    if method.lower() == "iqr":
        # IQR method
        q1 = np.percentile(labels, 25)
        q3 = np.percentile(labels, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Mark outliers
        outlier_mask = (labels < lower_bound) | (labels > upper_bound)

        logger.info("IQR outlier detection:")
        logger.info(f"  Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
        logger.info(f"  Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")

    elif method.lower() == "zscore":
        # Z-score method
        mean = np.mean(labels)
        std = np.std(labels)

        z_scores = np.abs((labels - mean) / std)
        outlier_mask = z_scores > z_threshold

        logger.info("Z-score outlier detection:")
        logger.info(f"  Mean: {mean:.4f}, Std: {std:.4f}")
        logger.info(f"  Z-score threshold: {z_threshold}")

    else:
        raise ValueError(
            f"Unknown outlier detection method: {method}. Supported methods: 'iqr', 'zscore'"
        )

    # Count outlier information
    outlier_count = np.sum(outlier_mask)
    outlier_indices = np.where(outlier_mask)[0].tolist()

    if outlier_count > 0:
        logger.info(f"Found {outlier_count} outliers:")
        for idx in outlier_indices[:10]:  # Only show first 10 outliers
            logger.info(f"  Index {idx}: label={labels[idx]:.4f}")
        if len(outlier_indices) > 10:
            logger.info(f"  ... and {len(outlier_indices) - 10} more outliers")
    else:
        logger.info("No outliers detected")

    # Remove outliers
    df_clean = df[~outlier_mask].reset_index(drop=True)

    # Record statistics
    removed_count = original_count - len(df_clean)
    logger.info("Outlier removal completed:")
    logger.info(f"  Original samples: {original_count}")
    logger.info(f"  After outlier removal: {len(df_clean)}")
    logger.info(f"  Removed: {removed_count} ({removed_count/original_count*100:.2f}%)")

    return df_clean, outlier_indices


def remove_outliers_by_sequence(
    df: pd.DataFrame,
    sequence_column: str = "sequence",
    label_column: str = "label",
    method: str = "iqr",
    threshold: float = 1.5,
    z_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Remove outliers for regression tasks and display sequence information in logs

    Args:
        df (pd.DataFrame): DataFrame containing sequences and labels
        sequence_column (str): Sequence column name, default is "sequence"
        label_column (str): Label column name, default is "label"
        method (str): Outlier detection method, supports "iqr" or "zscore"
        threshold (float): Threshold multiplier for IQR method, default is 1.5
        z_threshold (float): Z-score method threshold, default is 3.0

    Returns:
        Tuple[pd.DataFrame, List[int]]: (DataFrame after outlier removal, list of outlier indices)
    """
    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{sequence_column}' not found in DataFrame")

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    if df[label_column].dtype not in ["float64", "float32", "int64", "int32"]:
        logger.warning(
            f"Label column '{label_column}' is not numeric. Converting to float."
        )
        df = df.copy()
        df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    original_count = len(df)
    labels = np.array(df[label_column])
    sequences = df[sequence_column].tolist()

    logger.info(f"Removing outliers using {method} method")

    if method.lower() == "iqr":
        # IQR method
        q1 = np.percentile(labels, 25)
        q3 = np.percentile(labels, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Mark outliers
        outlier_mask = (labels < lower_bound) | (labels > upper_bound)

        logger.info("IQR outlier detection:")
        logger.info(f"  Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
        logger.info(f"  Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")

    elif method.lower() == "zscore":
        # Z-score method
        mean = np.mean(labels)
        std = np.std(labels)

        z_scores = np.abs((labels - mean) / std)
        outlier_mask = z_scores > z_threshold

        logger.info("Z-score outlier detection:")
        logger.info(f"  Mean: {mean:.4f}, Std: {std:.4f}")
        logger.info(f"  Z-score threshold: {z_threshold}")

    else:
        raise ValueError(
            f"Unknown outlier detection method: {method}. Supported methods: 'iqr', 'zscore'"
        )

    # Count outlier information
    outlier_count = np.sum(outlier_mask)
    outlier_indices = np.where(outlier_mask)[0].tolist()

    if outlier_count > 0:
        logger.info(f"Found {outlier_count} outliers:")
        for idx in outlier_indices[:10]:  # Only show first 10 outliers
            logger.info(
                f"  Index {idx}: sequence='{sequences[idx]}', label={labels[idx]:.4f}"
            )
        if len(outlier_indices) > 10:
            logger.info(f"  ... and {len(outlier_indices) - 10} more outliers")
    else:
        logger.info("No outliers detected")

    # Remove outliers
    df_clean = df[~outlier_mask].reset_index(drop=True)

    # Record statistics
    removed_count = original_count - len(df_clean)
    logger.info("Outlier removal completed:")
    logger.info(f"  Original samples: {original_count}")
    logger.info(f"  After outlier removal: {len(df_clean)}")
    logger.info(f"  Removed: {removed_count} ({removed_count/original_count*100:.2f}%)")

    return df_clean, outlier_indices
