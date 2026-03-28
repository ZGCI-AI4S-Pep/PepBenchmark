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

"""Split validation and cross-dataset analysis functions.

This module provides functions for validating dataset splits and analyzing
similarities between different datasets or splits. It focuses on ensuring
proper data separation and understanding potential data leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import similarity calculation functions
from pepbenchmark.similarity.similarity import compute_similarity_matrix
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_split_class_distribution(
    split_indices: Dict[str, List[int]],
    labels: Optional[List[int]] = None,
    fold_name: str = "Single_Fold"
) -> pd.DataFrame:
    """Analyze the positive/negative sample distribution for dataset splits.

    Args:
        split_indices: Dictionary containing train/valid/test indices 
                      (e.g., {"train": [1,2,3], "valid": [4,5,6], "test": [7,8,9]}).
        labels: List of binary labels (0 for negative, 1 for positive). 
               If None, returns empty DataFrame.
        fold_name: Name identifier for this fold/split.

    Returns:
        DataFrame containing the analysis results with columns:
        - fold: fold name
        - split: train/valid/test
        - total_samples: total number of samples
        - positive_samples: number of positive samples
        - negative_samples: number of negative samples
        - positive_ratio: ratio of positive samples
        - negative_ratio: ratio of negative samples
        
    Examples:
        >>> split_indices = {"train": [0, 1, 2], "test": [3, 4]}
        >>> labels = [1, 0, 1, 0, 1]
        >>> result_df = analyze_split_class_distribution(split_indices, labels)
    """
    if labels is None:
        return pd.DataFrame()

    results = []

    for split_name in ["train", "valid", "test"]:
        if split_name in split_indices:
            indices = split_indices.get(split_name, [])
            
            # Validate indices
            if any(idx >= len(labels) or idx < 0 for idx in indices):
                raise ValueError(f"Invalid indices in {split_name} split")
            
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
    fold_name: str = "Single_Fold"
) -> None:
    """Print a formatted summary of class distribution across splits.

    Args:
        split_indices: Dictionary containing train/valid/test indices.
        labels: List of binary labels (0 for negative, 1 for positive). 
               If None, prints error message.
        fold_name: Name identifier for this fold/split.
        
    Examples:
        >>> split_indices = {"train": [0, 1, 2], "test": [3, 4]}
        >>> labels = [1, 0, 1, 0, 1]
        >>> print_split_class_distribution_summary(split_indices, labels)
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
    samples1: Union[List[str], pd.DataFrame],
    samples2: Union[List[str], pd.DataFrame],
    dataset_name1: str = "Dataset1",
    dataset_name2: str = "Dataset2",
    similarity_metric: str = "sliding_window",
    processes: Optional[int] = None,
    sequence_col: str = "sequence"
) -> pd.DataFrame:
    """Analyze similarity distribution between two datasets.

    Args:
        samples1: First dataset sequences (list or DataFrame with sequence column).
        samples2: Second dataset sequences (list or DataFrame with sequence column).
        dataset_name1: Name for the first dataset.
        dataset_name2: Name for the second dataset.
        similarity_metric: Similarity metric to use ('sliding_window', 'levenshtein', etc.).
        processes: Number of parallel processes for similarity calculation.
        sequence_col: Column name for sequences if input is DataFrame.

    Returns:
        DataFrame containing similarity analysis results with columns:
        - dataset: source dataset name
        - target_dataset: target dataset name  
        - total_samples: number of samples
        - avg_max_similarity: average of maximum similarities
        - max_similarity: highest similarity found
        - min_similarity: lowest similarity found
        - high_similarity_count: number of samples with similarity > 0.8
        - high_similarity_ratio: ratio of high similarity samples
        
    Examples:
        >>> dataset1 = ["ACDEF", "GHIKL"] 
        >>> dataset2 = ["MNPQR", "ACDEF"]  # One identical sequence
        >>> result = analyze_cross_dataset_similarity(dataset1, dataset2)
    """
    
    # Extract sequences if DataFrame input
    def extract_sequences(data):
        if isinstance(data, pd.DataFrame):
            if sequence_col not in data.columns:
                raise ValueError(f"Column '{sequence_col}' not found in DataFrame")
            return data[sequence_col].dropna().astype(str).tolist()
        return list(data)
    
    seq_list1 = extract_sequences(samples1)
    seq_list2 = extract_sequences(samples2)
    
    if not seq_list1 or not seq_list2:
        return pd.DataFrame()

    # Calculate cross-dataset similarity matrix using unified entry
    try:
        result = compute_similarity_matrix(
            data1=seq_list1,
            data2=seq_list2,
            input_type="sequence",
            method=similarity_metric,
            processes=processes,
            show_progress=False,
            return_matrix=True,
            mode="full"
        )
        
        # Handle different return types from compute_similarity_matrix
        if isinstance(result, tuple):
            similarity_matrix, _ = result  # Unpack (matrix, result_set)
        else:
            similarity_matrix = result
        
        # Extract maximum similarities per sample
        # For each sample in seq_list1, find max similarity to any sample in seq_list2
        max_sim_1 = np.max(similarity_matrix, axis=1) if similarity_matrix.size > 0 else []
        # For each sample in seq_list2, find max similarity to any sample in seq_list1  
        max_sim_2 = np.max(similarity_matrix, axis=0) if similarity_matrix.size > 0 else []
        
    except Exception as e:
        # Fallback to empty results if similarity calculation fails
        logger.warning(f"Similarity calculation failed: {e}")
        max_sim_1, max_sim_2 = [], []
    
    results = []
    
    # Analyze samples1 vs samples2 
    if len(max_sim_1) > 0:
        results.append({
            "dataset": dataset_name1,
            "target_dataset": dataset_name2,
            "total_samples": len(seq_list1),
            "avg_max_similarity": np.mean(max_sim_1),
            "max_similarity": np.max(max_sim_1),
            "min_similarity": np.min(max_sim_1),
            "high_similarity_count": sum(1 for sim in max_sim_1 if sim > 0.8),
            "high_similarity_ratio": sum(1 for sim in max_sim_1 if sim > 0.8) / len(max_sim_1),
        })
    
    # Analyze samples2 vs samples1
    if len(max_sim_2) > 0:
        results.append({
            "dataset": dataset_name2,
            "target_dataset": dataset_name1,
            "total_samples": len(seq_list2),
            "avg_max_similarity": np.mean(max_sim_2),
            "max_similarity": np.max(max_sim_2),
            "min_similarity": np.min(max_sim_2),
            "high_similarity_count": sum(1 for sim in max_sim_2 if sim > 0.8),
            "high_similarity_ratio": sum(1 for sim in max_sim_2 if sim > 0.8) / len(max_sim_2),
        })

    return pd.DataFrame(results)


def print_cross_dataset_similarity_summary(
    samples1: Union[List[str], pd.DataFrame],
    samples2: Union[List[str], pd.DataFrame],
    dataset_name1: str = "Dataset1",
    dataset_name2: str = "Dataset2",
    similarity_metric: str = "sliding_window",
    processes: Optional[int] = None,
    sequence_col: str = "sequence"
) -> None:
    """Print formatted summary of cross-dataset similarity analysis.

    Args:
        samples1: First dataset sequences.
        samples2: Second dataset sequences.
        dataset_name1: Name for the first dataset.
        dataset_name2: Name for the second dataset.
        similarity_metric: Similarity metric to use.
        processes: Number of parallel processes.
        sequence_col: Column name for sequences if input is DataFrame.
        
    Examples:
        >>> dataset1 = ["ACDEF", "GHIKL"] 
        >>> dataset2 = ["MNPQR", "ACDEF"]
        >>> print_cross_dataset_similarity_summary(dataset1, dataset2)
    """
    df = analyze_cross_dataset_similarity(
        samples1, samples2, dataset_name1, dataset_name2, 
        similarity_metric, processes, sequence_col
    )

    if df.empty:
        print("No data to analyze.")
        return

    print("=" * 80)
    print("CROSS-DATASET SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"Similarity metric: {similarity_metric}")

    for _, row in df.iterrows():
        dataset = row["dataset"]
        target = row["target_dataset"]
        total_samples = row["total_samples"]
        avg_sim = row["avg_max_similarity"]
        max_sim = row["max_similarity"]
        min_sim = row["min_similarity"]
        high_sim_count = row["high_similarity_count"]
        high_sim_ratio = row["high_similarity_ratio"]

        print(f"\n{dataset} -> {target}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Average max similarity: {avg_sim:.4f}")
        print(f"  Max similarity: {max_sim:.4f}")
        print(f"  Min similarity: {min_sim:.4f}")
        print(f"  High similarity (>0.8) count: {high_sim_count}")
        print(f"  High similarity ratio: {high_sim_ratio:.2%}")


def detect_potential_data_leakage(
    train_sequences: Union[List[str], pd.DataFrame],
    test_sequences: Union[List[str], pd.DataFrame],
    similarity_threshold: float = 0.9,
    similarity_metric: str = "sliding_window",
    processes: Optional[int] = None,
    sequence_col: str = "sequence"
) -> Dict[str, Union[int, float, List[Tuple[int, int, float]]]]:
    """Detect potential data leakage between training and test sets.

    Args:
        train_sequences: Training set sequences.
        test_sequences: Test set sequences.
        similarity_threshold: Threshold above which sequences are considered too similar.
        similarity_metric: Similarity metric to use.
        processes: Number of parallel processes.
        sequence_col: Column name for sequences if input is DataFrame.

    Returns:
        Dictionary containing:
        - total_train_samples: number of training samples
        - total_test_samples: number of test samples
        - leakage_count: number of test samples with high similarity to training
        - leakage_ratio: ratio of test samples with potential leakage
        - max_similarity: highest similarity found
        - avg_high_similarity: average similarity among high-similarity pairs
        - leaky_pairs: list of (train_idx, test_idx, similarity) tuples above threshold
        
    Examples:
        >>> train_seqs = ["ACDEF", "GHIKL"]
        >>> test_seqs = ["MNPQR", "ACDEF"]  # Second sequence identical to first train
        >>> leakage_info = detect_potential_data_leakage(train_seqs, test_seqs)
    """
    # Validate similarity metric
    supported_metrics = ["sliding_window", "levenshtein", "jaccard_3mer"]
    if similarity_metric not in supported_metrics:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}. "
                        f"Supported: {supported_metrics}")
    
    # Extract sequences if DataFrame input
    def extract_sequences(data):
        if isinstance(data, pd.DataFrame):
            if sequence_col not in data.columns:
                raise ValueError(f"Column '{sequence_col}' not found in DataFrame")
            return data[sequence_col].dropna().astype(str).tolist()
        return list(data)
    
    train_list = extract_sequences(train_sequences)
    test_list = extract_sequences(test_sequences)
    
    if not train_list or not test_list:
        return {
            "total_train_samples": len(train_list),
            "total_test_samples": len(test_list),
            "leakage_count": 0,
            "leakage_ratio": 0.0,
            "max_similarity": 0.0,
            "avg_high_similarity": 0.0,
            "leaky_pairs": []
        }
    
    # Calculate cross-dataset similarities using unified entry
    try:
        result = compute_similarity_matrix(
            data1=train_list,
            data2=test_list,
            input_type="sequence",
            method=similarity_metric,
            processes=processes,
            show_progress=False,
            return_matrix=True,
            mode="full"
        )
        
        # Handle different return types from compute_similarity_matrix
        if isinstance(result, tuple):
            similarity_matrix, _ = result  # Unpack (matrix, result_set)
        else:
            similarity_matrix = result
        
        # Find pairs above threshold
        leaky_pairs = []
        max_similarity = 0.0
        high_similarities = []
        
        # Iterate through the similarity matrix
        for train_idx in range(len(train_list)):
            for test_idx in range(len(test_list)):
                similarity = similarity_matrix[train_idx, test_idx]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                
                if similarity > similarity_threshold:
                    leaky_pairs.append({
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "train_sequence": train_list[train_idx],
                        "test_sequence": test_list[test_idx],
                        "similarity": similarity
                    })
                    high_similarities.append(similarity)
        
    except Exception as e:
        logger.warning(f"Similarity calculation failed: {e}")
        leaky_pairs = []
        max_similarity = 0.0
        high_similarities = []
    
    # Calculate statistics
    leakage_count = len(leaky_pairs)  # Number of leaky pairs
    leakage_ratio = leakage_count / (len(train_list) * len(test_list)) if train_list and test_list else 0.0
    
    avg_high_similarity = (
        np.mean(high_similarities) if high_similarities else 0.0
    )
    
    return {
        "total_train_samples": len(train_list),
        "total_test_samples": len(test_list),
        "leakage_count": leakage_count,
        "leakage_ratio": leakage_ratio,
        "max_similarity": max_similarity,
        "avg_high_similarity": avg_high_similarity,
        "leaky_pairs": leaky_pairs
    }


def print_data_leakage_summary(
    train_sequences: Union[List[str], pd.DataFrame],
    test_sequences: Union[List[str], pd.DataFrame],
    similarity_threshold: float = 0.9,
    similarity_metric: str = "sliding_window",
    processes: Optional[int] = None,
    sequence_col: str = "sequence"
) -> None:
    """Print formatted summary of potential data leakage analysis.

    Args:
        train_sequences: Training set sequences.
        test_sequences: Test set sequences.
        similarity_threshold: Threshold for considering sequences too similar.
        similarity_metric: Similarity metric to use.
        processes: Number of parallel processes.
        sequence_col: Column name for sequences if input is DataFrame.
        
    Examples:
        >>> train_seqs = ["ACDEF", "GHIKL"]
        >>> test_seqs = ["MNPQR", "ACDEF"]
        >>> print_data_leakage_summary(train_seqs, test_seqs, threshold=0.9)
    """
    leakage_info = detect_potential_data_leakage(
        train_sequences, test_sequences, similarity_threshold,
        similarity_metric, processes, sequence_col
    )
    
    print("=" * 80)
    print("DATA LEAKAGE DETECTION ANALYSIS")
    print("=" * 80)
    print(f"Similarity metric: {similarity_metric}")
    print(f"Similarity threshold: {similarity_threshold}")
    print("")
    
    print(f"Training samples: {leakage_info['total_train_samples']}")
    print(f"Test samples: {leakage_info['total_test_samples']}")
    print(f"Max similarity found: {leakage_info['max_similarity']:.4f}")
    print("")
    
    leakage_count = leakage_info['leakage_count']
    leakage_ratio = leakage_info['leakage_ratio']
    
    if leakage_count > 0:
        print(f"⚠️  POTENTIAL DATA LEAKAGE DETECTED!")
        print(f"   Test samples with high similarity: {leakage_count}")
        print(f"   Leakage ratio: {leakage_ratio:.2%}")
        print(f"   Average high similarity: {leakage_info['avg_high_similarity']:.4f}")
        print(f"   Number of high-similarity pairs: {len(leakage_info['leaky_pairs'])}")
        
        # Show top 5 most similar pairs
        top_pairs = sorted(leakage_info['leaky_pairs'], key=lambda x: x[2], reverse=True)[:5]
        if top_pairs:
            print(f"\n   Top {len(top_pairs)} most similar pairs:")
            for train_idx, test_idx, sim in top_pairs:
                print(f"     Train[{train_idx}] vs Test[{test_idx}]: {sim:.4f}")
    else:
        print("✅ No significant data leakage detected.")
        print(f"   All similarities below threshold ({similarity_threshold})")


if __name__ == "__main__":
    # Test the split validation functions
    print("=== Testing Split Validation Functions ===\n")
    
    # Test data setup
    test_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",  # 0
        "ACDEFGHIKLMNPQRST",     # 1  
        "MNPQRSTVWY",            # 2
        "ACDEFG",                # 3
        "GHIKLMNPQRSTVWY",       # 4
        "DEFGHIKLMNPQR",         # 5
        "PQRSTVWY",              # 6
        "ACDEFGHIKL",            # 7
        "LMNPQRSTVWY",           # 8
        "GHIKLMNPQ"              # 9
    ]
    
    test_labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # Binary labels
    
    # Create test splits
    split_indices = {
        "train": [0, 1, 2, 3, 4, 5],
        "valid": [6, 7], 
        "test": [8, 9]
    }
    
    # Create DataFrames for testing
    train_df = pd.DataFrame({
        'sequence': [test_sequences[i] for i in split_indices['train']],
        'label': [test_labels[i] for i in split_indices['train']]
    })
    
    test_df = pd.DataFrame({
        'sequence': [test_sequences[i] for i in split_indices['test']],
        'label': [test_labels[i] for i in split_indices['test']]
    })
    
    print("1. Testing split class distribution analysis:")
    print("-" * 50)
    
    try:
        # Test analysis function
        distribution_df = analyze_split_class_distribution(split_indices, test_labels, "Fold_1")
        print("✓ Split class distribution analysis completed")
        print(f"  Generated {len(distribution_df)} split analysis records")
        
        # Show summary
        for _, row in distribution_df.iterrows():
            print(f"  {row['split']}: {row['total_samples']} samples "
                  f"({row['positive_ratio']:.2%} positive)")
        
        # Test print function
        print("\n✓ Testing formatted output:")
        print_split_class_distribution_summary(split_indices, test_labels, "Test_Fold")
        
    except Exception as e:
        print(f"✗ Split class distribution analysis failed: {e}")
    
    print("\n2. Testing cross-dataset similarity analysis:")
    print("-" * 50)
    
    # Create two different datasets for testing
    dataset1 = test_sequences[:5]
    dataset2 = test_sequences[5:] + ["ACDEFGHIKLMNPQRSTVWY"]  # Add duplicate for testing
    
    for metric in ["sliding_window", "levenshtein", "jaccard_3mer"]:
        try:
            similarity_df = analyze_cross_dataset_similarity(
                dataset1, dataset2, 
                dataset_name1="Dataset_A", 
                dataset_name2="Dataset_B",
                similarity_metric=metric,
                processes=1  # Use single process for testing
            )
            print(f"✓ Cross-dataset similarity analysis with {metric} completed")
            print(f"  Generated {len(similarity_df)} analysis records")
            
            if not similarity_df.empty:
                for _, row in similarity_df.iterrows():
                    print(f"  {row['dataset']} -> {row['target_dataset']}: "
                          f"avg_sim={row['avg_max_similarity']:.3f}, "
                          f"max_sim={row['max_similarity']:.3f}")
        except Exception as e:
            print(f"✗ Cross-dataset similarity with {metric} failed: {e}")
    
    print("\n3. Testing DataFrame input support:")
    print("-" * 50)
    
    try:
        # Test with DataFrame inputs
        similarity_df_pandas = analyze_cross_dataset_similarity(
            train_df, test_df,
            dataset_name1="Train_Set",
            dataset_name2="Test_Set",
            similarity_metric="sliding_window",
            processes=1
        )
        print("✓ DataFrame input support works")
        print(f"  Generated {len(similarity_df_pandas)} records from DataFrames")
    except Exception as e:
        print(f"✗ DataFrame input support failed: {e}")
    
    print("\n4. Testing data leakage detection:")
    print("-" * 50)
    
    # Create datasets with known leakage for testing
    train_seqs = test_sequences[:6]
    test_seqs_with_leakage = test_sequences[6:] + [test_sequences[0]]  # Add duplicate from train
    
    try:
        leakage_info = detect_potential_data_leakage(
            train_seqs, test_seqs_with_leakage,
            similarity_threshold=0.95,
            similarity_metric="sliding_window",
            processes=1
        )
        print("✓ Data leakage detection completed")
        print(f"  Found {leakage_info['leakage_count']} potentially leaked samples")
        print(f"  Leakage ratio: {leakage_info['leakage_ratio']:.2%}")
        print(f"  Max similarity: {leakage_info['max_similarity']:.4f}")
        
        # Test print function
        print("\n✓ Testing formatted leakage report:")
        print_data_leakage_summary(
            train_seqs, test_seqs_with_leakage,
            similarity_threshold=0.95,
            similarity_metric="sliding_window",
            processes=1
        )
        
    except Exception as e:
        print(f"✗ Data leakage detection failed: {e}")
    
    print("\n5. Testing edge cases:")
    print("-" * 50)
    
    # Empty inputs
    try:
        empty_distribution = analyze_split_class_distribution({}, [])
        print(f"✓ Empty split indices handled: {len(empty_distribution)} records")
    except Exception as e:
        print(f"✗ Empty split indices failed: {e}")
    
    # No labels
    try:
        print_split_class_distribution_summary(split_indices, None, "No_Labels")
        print("✓ None labels handled gracefully")
    except Exception as e:
        print(f"✗ None labels failed: {e}")
    
    # Empty sequences for similarity
    try:
        empty_similarity = analyze_cross_dataset_similarity([], [])
        print(f"✓ Empty sequences handled: {len(empty_similarity)} records")
    except Exception as e:
        print(f"✗ Empty sequences failed: {e}")
    
    # Invalid indices
    try:
        invalid_indices = {"train": [0, 100]}  # Index 100 doesn't exist
        analyze_split_class_distribution(invalid_indices, test_labels)
        print("✗ Should have failed with invalid indices")
    except ValueError:
        print("✓ Invalid indices correctly caught")
    except Exception as e:
        print(f"✗ Unexpected error with invalid indices: {e}")
    
    print("\n✅ All split validation function tests completed!")
    print("\n🔍 Available validation functions:")
    print("   - analyze_split_class_distribution(): Analyze class balance across splits")
    print("   - print_split_class_distribution_summary(): Formatted class distribution report") 
    print("   - analyze_cross_dataset_similarity(): Compare similarity between datasets")
    print("   - print_cross_dataset_similarity_summary(): Formatted similarity report")
    print("   - detect_potential_data_leakage(): Detect high similarity between train/test")
    print("   - print_data_leakage_summary(): Formatted data leakage report")
    print("   - Support for multiple similarity metrics and DataFrame inputs")


if __name__ == "__main__":
    from pepbenchmark.dataset_manager.single_dataset import SingleTaskDatasetManager
    from pepbenchmark.splitter.split_analyzer import SplitAnalyzer
    import numpy as np

    # 1. Load dataset
    dataset_name = "bbp"  # Replace with your dataset name
    manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta", "label"]
    )

    # 2. Set official split (e.g., random_split with seed=0)
    manager.set_official_split_indices(split_type="random_split", fold_seed=0)

    # 3. Get sequences after splitting
    split_indices = manager.get_split_indices()
    fasta_list = manager.get_official_feature("fasta")
    labels = manager.get_official_feature("label")

    # 4. Validate class distribution analysis function for a single split
    print("=" * 80)
    print("VALIDATING SINGLE SPLIT CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Test single split analysis
    class_stats = analyze_split_class_distribution(split_indices, labels)
    print(f"✓ analyze_split_class_distribution completed: {len(class_stats)} splits analyzed")

    # Test single split print function
    print_split_class_distribution_summary(split_indices, labels, "fold_0")

    # 5. Test cross-dataset similarity analysis
    print("\n" + "=" * 80)
    print("VALIDATING CROSS-DATASET SIMILARITY ANALYSIS")
    print("=" * 80)

    # Get sequences from different splits for testing
    train_indices = split_indices.get("train", [])
    test_indices = split_indices.get("test", [])

    if train_indices and test_indices and fasta_list is not None:
        train_sequences: List[str] = [fasta_list[i] for i in train_indices]
        test_sequences: List[str] = [fasta_list[i] for i in test_indices]

        # Test cross-dataset similarity calculation
        print("\nCalculating cross-dataset similarities...")
        similarities = analyze_cross_dataset_similarity(
            train_sequences,
            test_sequences,
            processes=2,  # Use fewer processes for testing
        )
        print(f"✓ analyze_cross_dataset_similarity completed: {len(similarities)} similarities calculated")

        # Test similarity summary printing
        print_cross_dataset_similarity_summary(
            test_sequences, train_sequences, "test", "train", processes=2
        )

        # 6. Demonstrate new SplitAnalyzer functionality
        print("\n" + "=" * 80)
        print("DEMONSTRATING NEW SPLITANALYZER FUNCTIONALITY")
        print("=" * 80)

        # Generate simple mock embeddings for demonstration
        def generate_simple_embeddings(sequences):
            embeddings = []
            for seq in sequences:
                # Generate simple features based on sequence properties
                features = [
                    len(seq),
                    seq.count('A') / len(seq),
                    seq.count('G') / len(seq),
                    seq.count('P') / len(seq),
                    seq.count('C') / len(seq)
                ]
                # Pad to a fixed dimension
                while len(features) < 10:
                    features.append(0.0)
                embeddings.append(features[:10])
            return np.array(embeddings)

        try:
            embeddings = generate_simple_embeddings(fasta_list)
            
            # Initialize SplitAnalyzer
            analyzer = SplitAnalyzer(
                sequences=fasta_list,
                labels=labels,
                embeddings=embeddings
            )
            
            # Comprehensive quality evaluation
            quality_report = analyzer.evaluate_split_quality(split_indices)
            print("\nSplit Quality Report:")
            print(quality_report)
            
            # Class distribution analysis
            if labels is not None:
                class_analysis = analyzer.analyze_class_distribution(split_indices)
                print(f"\n✓ Class distribution analysis completed")
                print(f"  Balance score: {class_analysis.get('balance_score', 'N/A'):.4f}")
            
            # Cross-split similarity analysis
            cross_sim = analyzer.analyze_cross_split_similarity(split_indices)
            print(f"\n✓ Cross-split similarity analysis completed")
            for pair, data in cross_sim.items():
                print(f"  {pair}: mean={data['mean_similarity']:.3f}, max={data['max_similarity']:.3f}")
            
            # Data leakage detection
            leakage = analyzer.detect_data_leakage(split_indices)
            print(f"\n✓ Data leakage detection completed")
            for pair, data in leakage.items():
                ratio = data.get('high_similarity_ratio', 0)
                print(f"  {pair}: high similarity ratio = {ratio:.3f}")
            
            print("\n✅ SplitAnalyzer demonstration completed successfully!")
            
        except Exception as e:
            print(f"\n❌ SplitAnalyzer demonstration failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No train/test splits available for cross-dataset similarity testing.")
