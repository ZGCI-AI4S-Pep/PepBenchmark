"""Utilities for aggregating experiment outputs across repeated runs."""

from __future__ import annotations

import json
import os

import pandas as pd

from pepbenchmark.utils.logging import get_logger


logger = get_logger(__name__)


def load_results_from_dir(
    base_dir,
    dataset_name,
    model,
    split_type="mmseqs2_split",
    expected_seeds=5,
):
    """Load all seed-level metric files for one model configuration."""
    model_dir = os.path.join(base_dir, dataset_name, model, split_type)
    records = []

    if not os.path.exists(model_dir):
        logger.warning("Directory does not exist: {}", model_dir)
        return pd.DataFrame()

    for fold_seed in os.listdir(model_dir):
        fold_path = os.path.join(model_dir, fold_seed)
        if not os.path.isdir(fold_path):
            continue

        metrics_file = os.path.join(fold_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            record = {
                "fold_seed": int(fold_seed),
                **{f"train_{k}": v for k, v in metrics.get("train", {}).items()},
                **{f"valid_{k}": v for k, v in metrics.get("valid", {}).items()},
                **{f"test_{k}": v for k, v in metrics.get("test", {}).items()},
            }
            records.append(record)

    df = pd.DataFrame(records)
    if len(df) < expected_seeds:
        logger.warning(
            "{}-{}-{} has only {} seed results, fewer than {}",
            dataset_name,
            model,
            split_type,
            len(df),
            expected_seeds,
        )
    else:
        logger.info(
            "{}-{}-{} found {} seed results in total",
            dataset_name,
            model,
            split_type,
            len(df),
        )

    return df


def summarize_results_grouped(
    base_dir,
    dataset_name,
    models,
    split_type="mmseqs2_split",
    target_splits=["train", "valid", "test"],
):
    """Summarize metric means and standard deviations for several models."""
    summary_data = []

    for model in models:
        df = load_results_from_dir(base_dir, dataset_name, model, split_type)
        if df.empty:
            logger.warning("No data found for model {}", model)
            continue

        model_stats = {"model": model}

        for split in target_splits:
            split_columns = [
                column for column in df.columns if column.startswith(f"{split}_")
            ]

            if not split_columns:
                logger.warning("No {} columns found for model {}", split, model)
                continue

            split_df = df[split_columns]
            stats = split_df.describe().T[["mean", "std"]]

            for metric_col, metric_stats in stats.iterrows():
                metric_name = metric_col.replace(f"{split}_", "")
                model_stats[f"{split}_{metric_name}_mean"] = (
                    f"{metric_stats['mean']:.4f}"
                )
                model_stats[f"{split}_{metric_name}_std"] = (
                    f"{metric_stats['std']:.4f}"
                )

        summary_data.append(model_stats)

    summary_df = pd.DataFrame(summary_data)
    if summary_df.empty:
        logger.warning("No summary data generated")
        return summary_df

    return summary_df
