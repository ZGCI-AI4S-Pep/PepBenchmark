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

from typing import Dict, List, Optional

import pandas as pd
from pepbenchmark.analyze.similarity import (
    calculate_cross_dataset_similarity,
    get_max_cross_dataset_similarity_per_sample,
)


def analyze_split_class_distribution(
    split_indices: Dict[str, List[int]],
    labels: Optional[List[int]] = None,
    fold_name: str = "Single_Fold",
) -> pd.DataFrame:
    """
    Analyze the positive/negative sample distribution for a single split from JSON format.

    Args:
        split_indices: Dictionary containing train/valid/test indices (e.g., {"train": [1,2,3], "valid": [4,5,6], "test": [7,8,9]})
        labels: List of labels (0 for negative, 1 for positive). If None, will return empty DataFrame.
        fold_name: Name for this fold/split

    Returns:
        DataFrame containing the analysis results with columns:
        - fold: fold name
        - split: train/valid/test
        - total_samples: total number of samples
        - positive_samples: number of positive samples
        - negative_samples: number of negative samples
        - positive_ratio: ratio of positive samples
        - negative_ratio: ratio of negative samples
    """
    if labels is None:
        return pd.DataFrame()

    results = []

    for split_name in ["train", "valid", "test"]:
        if split_name in split_indices:
            indices = split_indices.get(split_name, [])
            split_labels = [labels[i] for i in indices]

            total_samples = len(split_labels)
            positive_samples = sum(split_labels)
            negative_samples = total_samples - positive_samples
            positive_ratio = (
                positive_samples / total_samples if total_samples > 0 else 0
            )
            negative_ratio = (
                negative_samples / total_samples if total_samples > 0 else 0
            )

            results.append(
                {
                    "fold": fold_name,
                    "split": split_name,
                    "total_samples": total_samples,
                    "positive_samples": positive_samples,
                    "negative_samples": negative_samples,
                    "positive_ratio": positive_ratio,
                    "negative_ratio": negative_ratio,
                }
            )

    return pd.DataFrame(results)


def print_split_class_distribution_summary(
    split_indices: Dict[str, List[int]],
    labels: Optional[List[int]] = None,
    fold_name: str = "Single_Fold",
) -> None:
    """
    Print a summary of the positive/negative sample distribution for a single split from JSON format.

    Args:
        split_indices: Dictionary containing train/valid/test indices
        labels: List of labels (0 for negative, 1 for positive). If None, will print error message.
        fold_name: Name for this fold/split
    """
    if labels is None:
        print("Error: labels is None. Cannot analyze class distribution.")
        return

    df = analyze_split_class_distribution(split_indices, labels, fold_name)

    if df.empty:
        print("No data to analyze.")
        return

    print("=" * 80)
    print("SPLIT CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)

    print(f"\n{fold_name}:")
    print("-" * 40)
    for _, row in df.iterrows():
        split = row["split"]
        total = row["total_samples"]
        pos = row["positive_samples"]
        neg = row["negative_samples"]
        pos_ratio = row["positive_ratio"]
        neg_ratio = row["negative_ratio"]

        print(
            f"  {split:>6}: {total:>4} samples "
            f"(Pos: {pos:>3} [{pos_ratio:>6.1%}], "
            f"Neg: {neg:>3} [{neg_ratio:>6.1%}])"
        )


def analyze_cross_dataset_similarity(
    samples1: List[str],
    samples2: List[str],
    dataset_name1: str = "Test",
    dataset_name2: str = "Train",
    processes: Optional[int] = None,
) -> pd.DataFrame:
    """
    分析测试集在训练集中的相似度分布。

    Args:
        test_samples: 测试集的序列列表
        train_samples: 训练集的序列列表
        test_dataset_name: 测试集的名称
        train_dataset_name: 训练集的名称
        processes: 并行进程数

    Returns:
        包含相似度分析结果的DataFrame
    """
    # 只计算测试集在训练集中的最大相似度
    max_sim_test = get_max_cross_dataset_similarity_per_sample(
        samples1, samples2, processes
    )[0]  # 只取第一个返回值（测试集在训练集中的最大相似度）

    # 创建分析结果
    results = []

    # 分析测试集中每个序列在训练集中的最大相似度
    if max_sim_test:
        results.append(
            {
                "dataset": dataset_name1,
                "target_dataset": dataset_name2,
                "total_samples": len(samples1),
                "avg_max_similarity": sum(max_sim_test) / len(max_sim_test),
                "max_similarity": max(max_sim_test),
                "min_similarity": min(max_sim_test),
                "high_similarity_count": sum(1 for sim in max_sim_test if sim > 0.8),
                "high_similarity_ratio": sum(1 for sim in max_sim_test if sim > 0.8)
                / len(max_sim_test),
            }
        )

    return pd.DataFrame(results)


def print_cross_dataset_similarity_summary(
    samples1: List[str],
    samples2: List[str],
    dataset_name1: str = "Test",
    dataset_name2: str = "Train",
    processes: Optional[int] = None,
) -> None:
    """
    打印测试集在训练集中相似度分析的摘要。

    Args:
        test_samples: 测试集的序列列表
        train_samples: 训练集的序列列表
        test_dataset_name: 测试集的名称
        train_dataset_name: 训练集的名称
        processes: 并行进程数
    """
    df = analyze_cross_dataset_similarity(
        samples1, samples2, dataset_name1, dataset_name2, processes
    )

    if df.empty:
        print("No data to analyze.")
        return

    print("=" * 80)
    print("CROSS-DATASET SIMILARITY ANALYSIS")
    print("=" * 80)

    for _, row in df.iterrows():
        total_samples = row["total_samples"]
        avg_sim = row["avg_max_similarity"]
        max_sim = row["max_similarity"]
        min_sim = row["min_similarity"]
        high_sim_count = row["high_similarity_count"]
        high_sim_ratio = row["high_similarity_ratio"]

        print(f"\n{dataset_name1} -> {dataset_name2}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Average max similarity: {avg_sim:.4f}")
        print(f"  Max similarity: {max_sim:.4f}")
        print(f"  Min similarity: {min_sim:.4f}")
        print(f"  High similarity (>0.8) count: {high_sim_count}")
        print(f"  High similarity ratio: {high_sim_ratio:.2%}")


if __name__ == "__main__":
    from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

    # 1. 加载数据集
    dataset_name = "bbp"  # 替换为你的数据集名
    manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta", "label"]
    )

    # 2. 设置官方划分（以random_split为例，seed=0）
    manager.set_official_split_indices(split_type="random_split", fold_seed=0)

    # 3. 获取划分后的序列
    split_indices = manager.get_split_indices()
    fasta_list = manager.get_official_feature("fasta")
    labels = manager.get_official_feature("label")

    # 4. 校验单个split的class distribution分析函数
    print("=" * 80)
    print("VALIDATING SINGLE SPLIT CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # 测试单个split分析

    # 测试单个split打印函数
    print_split_class_distribution_summary(split_indices, labels, "fold_0")

    # 5. 测试跨数据集相似度分析
    print("\n" + "=" * 80)
    print("VALIDATING CROSS-DATASET SIMILARITY ANALYSIS")
    print("=" * 80)

    # 获取不同split的序列用于测试
    train_indices = split_indices.get("train", [])
    test_indices = split_indices.get("test", [])

    if train_indices and test_indices and fasta_list is not None:
        train_sequences: List[str] = [fasta_list[i] for i in train_indices]
        test_sequences: List[str] = [fasta_list[i] for i in test_indices]

        # 测试跨数据集相似度计算
        print("\nCalculating cross-dataset similarities...")
        similarities = calculate_cross_dataset_similarity(
            train_sequences,
            test_sequences,
            processes=2,  # 使用较少的进程进行测试
        )

        # 测试最大相似度计算
        max_sim_train, max_sim_test = get_max_cross_dataset_similarity_per_sample(
            train_sequences, test_sequences, processes=2
        )

        # 测试相似度摘要打印
        print_cross_dataset_similarity_summary(
            test_sequences, train_sequences, "test", "train", processes=2
        )
    else:
        print("No train/test splits available for cross-dataset similarity testing.")
