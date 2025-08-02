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

from typing import Tuple

import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


def deduplicate_single(
    df: pd.DataFrame,
    task_type: str,
    sequence_col: str = "sequence",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove completely duplicate sequence values

    Args:
        df: DataFrame containing sequences and labels
        task_type: Task type, 'binary_classification' or 'regression'
        sequence_col: Sequence column name, default is 'sequence'
        label_col: Label column name, default is 'label'

    Returns:
        tuple: (Deduplicated DataFrame, deduplication statistics dictionary)
    """

    if task_type == "binary_classification":
        # Classification task: remove duplicate sequences, keep only the first occurrence
        logger.info(
            f"Processing {task_type} task - removing duplicate sequences, keeping first occurrence"
        )
        df_dedup = df.drop_duplicates(subset=[sequence_col], keep="first")

    elif task_type == "regression":
        # Regression task: remove duplicate sequences, calculate average of corresponding labels
        logger.info(
            f"Processing {task_type} task - removing duplicate sequences, averaging labels"
        )

        # Group by sequence, calculate average of labels
        df_grouped = (
            df.groupby(sequence_col)[label_col].agg(["mean", "count"]).reset_index()
        )
        df_grouped.columns = [sequence_col, label_col, "count"]

        # Record deduplication information
        duplicate_info = df_grouped[df_grouped["count"] > 1]
        if not duplicate_info.empty:
            logger.info(f"Found {len(duplicate_info)} sequences with duplicates:")

        df_dedup = df_grouped[[sequence_col, label_col]]

    else:
        raise ValueError(f"Unknown dataset type '{task_type}'")

    # Record deduplication statistics
    original_count = len(df)
    dedup_count = len(df_dedup)
    removed_count = original_count - dedup_count

    stats = {
        "original_count": original_count,
        "dedup_count": dedup_count,
        "removed_count": removed_count,
        "reduction_percentage": removed_count / original_count * 100,
    }

    logger.info("Deduplication completed:")
    logger.info(f"  Original sequences: {original_count}")
    logger.info(f"  After deduplication: {dedup_count}")
    logger.info(f"  Reduction: {stats['reduction_percentage']:.2f}%")

    return df_dedup, stats


def deduplicate_pair(
    df: pd.DataFrame,
    task_type: str,
    seq1_col: str = "protein_sequence",
    seq2_col: str = "peptide_sequence",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove duplicate data for dual instances (such as protein-peptide pairs)

    Args:
        df: DataFrame containing two sequences and labels
        task_type: Task type, 'binary_classification' or 'regression'
        seq1_col: First sequence column name, default is 'protein_sequence'
        seq2_col: Second sequence column name, default is 'peptide_sequence'
        label_col: Label column name, default is 'label'

    Returns:
        tuple: (Deduplicated DataFrame, deduplication statistics dictionary)
    """

    if task_type == "binary_classification":
        # Classification task: remove duplicate sequence pairs, keep only the first occurrence
        logger.info(
            f"Processing {task_type} task - removing duplicate sequence pairs, keeping first occurrence"
        )
        df_dedup = df.drop_duplicates(subset=[seq1_col, seq2_col], keep="first")

    elif task_type == "regression":
        # Regression task: remove duplicate sequence pairs, calculate average of corresponding labels
        logger.info(
            f"Processing {task_type} task - removing duplicate sequence pairs, averaging labels"
        )

        # Group by sequence pairs, calculate average of labels
        df_grouped = (
            df.groupby([seq1_col, seq2_col])[label_col]
            .agg(["mean", "count"])
            .reset_index()
        )
        df_grouped.columns = [seq1_col, seq2_col, label_col, "count"]

        # Record deduplication information
        duplicate_info = df_grouped[df_grouped["count"] > 1]
        if not duplicate_info.empty:
            logger.info(f"Found {len(duplicate_info)} sequence pairs with duplicates:")
            for _, row in duplicate_info.iterrows():
                logger.info(
                    f"  Pair: ({row[seq1_col][:20]}..., {row[seq2_col][:20]}...), "
                    f"Original count: {row['count']}, Averaged label: {row[label_col]:.4f}"
                )

        df_dedup = df_grouped[[seq1_col, seq2_col, label_col]]

    else:
        raise ValueError(f"Unknown dataset type '{task_type}'")

    # Record deduplication statistics
    original_count = len(df)
    dedup_count = len(df_dedup)
    removed_count = original_count - dedup_count

    stats = {
        "original_count": original_count,
        "dedup_count": dedup_count,
        "removed_count": removed_count,
        "reduction_percentage": removed_count / original_count * 100,
    }

    logger.info("Pair deduplication completed:")
    logger.info(f"  Original pairs: {original_count}")
    logger.info(f"  After deduplication: {dedup_count}")
    logger.info(f"  Reduction: {stats['reduction_percentage']:.2f}%")

    return df_dedup, stats
