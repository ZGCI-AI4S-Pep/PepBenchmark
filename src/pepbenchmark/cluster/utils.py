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
Unified utilities for clustering algorithms.

This module contains all shared functions used by clustering implementations,
merging functionality from common_utils.py and utils.py to eliminate duplication.
"""

import os
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=None, tablefmt="grid"):
        # Fallback implementation
        return "\n".join([str(row) for row in data])

from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# File I/O Operations
# =============================================================================

def save_fasta(sequences: List[str], path: str) -> None:
    """
    Save sequences to a FASTA file.
    
    Args:
        sequences: List of sequence strings
        path: Output FASTA file path
    """
    with open(path, "w", encoding="utf-8") as file:
        for index, sequence in enumerate(sequences):
            file.write(f">seq{index}\n{sequence}\n")


def load_fasta(path: str) -> List[str]:
    """
    Load sequences from a FASTA file.
    
    Args:
        path: FASTA file path
        
    Returns:
        List of sequence strings
    """
    sequences: List[str] = []
    chunks: List[str] = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    sequences.append("".join(chunks))
                    chunks = []
                continue
            chunks.append(line)

    if chunks:
        sequences.append("".join(chunks))

    return sequences


# =============================================================================
# Command Line Utilities
# =============================================================================

def dict_to_cli_args(params: Dict[str, Any], prefix: str = "-") -> List[str]:
    """
    Convert Python dict to command line arguments.
    
    Args:
        params: Dictionary of parameters
        prefix: Prefix for arguments (- for short, -- for long)
        
    Returns:
        List of command line arguments
    """
    args = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                args.append(f"{prefix}{key}")
        else:
            args.extend([f"{prefix}{key}", str(value)])
    return args


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command and handle errors consistently.
    
    Args:
        cmd: Command as list of strings
        check: Whether to raise exception on non-zero exit code
        
    Returns:
        CompletedProcess instance
        
    Raises:
        RuntimeError: If command fails and check=True
    """
    logger.info("Running command: " + " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {' '.join(cmd)}\nstderr: {e.stderr}\nstdout: {e.stdout}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# =============================================================================
# Parameter Validation
# =============================================================================

def validate_parameter_range(
    params: Dict[str, Any],
    key: str,
    valid_range: Any,
    message: str,
) -> Optional[str]:
    """
    Validate a single parameter against a valid range.
    
    Args:
        params: Dictionary of parameters
        key: Parameter key to validate
        valid_range: Either a tuple (min, max) or list of valid values
        message: Error message template
        
    Returns:
        Warning message if validation fails, None otherwise
    """
    if key in params:
        value = params[key]
        if isinstance(valid_range, tuple):
            if not (valid_range[0] <= value <= valid_range[1]):
                return message.format(value)
        elif value not in valid_range:
            return message.format(value)
    return None


def validate_parameters(
    params: Dict[str, Any],
    validations: Dict[str, Tuple[Any, str]]
) -> List[str]:
    """
    Validate multiple parameters against their ranges.
    
    Args:
        params: Dictionary of parameters
        validations: Dict mapping parameter names to (valid_range, message) tuples
        
    Returns:
        List of warning messages
    """
    warnings = []
    for key, (valid_range, message) in validations.items():
        warning = validate_parameter_range(params, key, valid_range, message)
        if warning:
            warnings.append(warning)
    
    if warnings:
        logger.warning(f"Parameter validation warnings:\n" + "\n".join(f"- {w}" for w in warnings))
    
    return warnings


# =============================================================================
# Representative Selection
# =============================================================================

def select_representative_sequences(
    sequences: List[str], 
    cluster_map: Dict[str, List[int]], 
    strategy: str = "first"
) -> List[Tuple[str, int, str]]:
    """
    Select representative sequences from clustering results.
    
    Args:
        sequences: Original list of sequences
        cluster_map: Dict mapping cluster_id to list of sequence indices
        strategy: Strategy for selecting representatives ("first", "longest", "shortest", "middle")
        
    Returns:
        List of (cluster_id, representative_index, representative_sequence) tuples
    """
    representatives = []
    
    for cluster_id, seq_indices in cluster_map.items():
        if not seq_indices:
            continue
            
        if strategy == "first":
            rep_idx = seq_indices[0]
        elif strategy == "longest":
            rep_idx = max(seq_indices, key=lambda i: len(sequences[i]))
        elif strategy == "shortest":
            rep_idx = min(seq_indices, key=lambda i: len(sequences[i]))
        elif strategy == "middle":
            rep_idx = seq_indices[len(seq_indices) // 2]
        elif strategy == "random":
            rep_idx = random.choice(seq_indices)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using 'first'")
            rep_idx = seq_indices[0]
            
        rep_seq = sequences[rep_idx]
        representatives.append((cluster_id, rep_idx, rep_seq))
    
    return representatives


def get_cluster_representatives(
    data: Union[List[str], np.ndarray, List[np.ndarray]],
    cluster_labels: List[int],
    strategy: str = "first",
    similarity_matrix: Optional[np.ndarray] = None,
) -> Dict[int, int]:
    """
    Select representative item per cluster from cluster labels.
    
    Args:
        data: List of sequences or other data items
        cluster_labels: List of cluster labels for each item
        strategy: Strategy for representative selection
        similarity_matrix: Optional similarity matrix (required for "most_similar")

    Returns:
        Mapping from cluster ID to item index
    """
    # Group indices by cluster
    cluster_indices = {}
    for i, label in enumerate(cluster_labels):
        if label != -1:  # Ignore noise points
            cluster_indices.setdefault(label, []).append(i)
    
    representatives = {}
    for cluster_id, indices in cluster_indices.items():
        if strategy == "first":
            representatives[cluster_id] = indices[0]
        elif strategy == "longest":
            representatives[cluster_id] = max(indices, key=lambda i: len(str(data[i])))
        elif strategy == "shortest":
            representatives[cluster_id] = min(indices, key=lambda i: len(str(data[i])))
        elif strategy == "random":
            representatives[cluster_id] = random.choice(indices)
        elif strategy == "most_similar":
            if similarity_matrix is None:
                raise ValueError("similarity_matrix required for 'most_similar' strategy")
            best_score, best_idx = -1.0, indices[0]
            for idx in indices:
                # Calculate average similarity to other cluster members
                similarities = [similarity_matrix[idx, j] for j in indices if j != idx]
                if similarities:
                    avg_sim = np.mean(similarities)
                    if avg_sim > best_score:
                        best_score, best_idx = avg_sim, idx
            representatives[cluster_id] = best_idx
        else:
            logger.warning(f"Unknown strategy '{strategy}', using 'first'")
            representatives[cluster_id] = indices[0]
    
    return representatives


# =============================================================================
# Redundancy Removal
# =============================================================================

def remove_redundancy(
    sequences: List[str], 
    cluster_map: Dict[str, List[int]], 
    strategy: str = "first"
) -> Tuple[List[str], Dict[int, int]]:
    """
    Remove redundant sequences based on clustering results.
    
    Args:
        sequences: Original list of sequences
        cluster_map: Dict mapping cluster_id to list of sequence indices
        strategy: Strategy for selecting representatives
        
    Returns:
        Tuple of (deduplicated_sequences, old_to_new_index_mapping)
    """
    representatives = select_representative_sequences(sequences, cluster_map, strategy)
    
    # Extract deduplicated sequences
    dedup_sequences = [rep_seq for _, _, rep_seq in representatives]
    
    # Build index mapping
    old_to_new_idx = {}
    for new_idx, (_, old_idx, _) in enumerate(representatives):
        old_to_new_idx[old_idx] = new_idx
    
    return dedup_sequences, old_to_new_idx


def calculate_redundancy_reduction(
    original_count: int, 
    clustered_count: int
) -> Dict[str, float]:
    """
    Calculate redundancy reduction statistics.
    
    Args:
        original_count: Number of original sequences
        clustered_count: Number of clusters/representatives
        
    Returns:
        Dictionary with reduction statistics
    """
    if original_count == 0:
        return {"reduction_ratio": 0.0, "redundancy_removed": 0.0, "compression_ratio": 1.0}
    
    reduction_ratio = clustered_count / original_count
    redundancy_removed = (original_count - clustered_count) / original_count
    compression_ratio = original_count / clustered_count if clustered_count > 0 else 1.0
    
    return {
        "reduction_ratio": reduction_ratio,
        "redundancy_removed": redundancy_removed,
        "compression_ratio": compression_ratio,
        "original_count": original_count,
        "final_count": clustered_count,
        "removed_count": original_count - clustered_count
    }





# =============================================================================
# Statistics and Reporting
# =============================================================================

def print_cluster_statistics(
    cluster_map: Dict[str, List[Any]],
    algorithm_name: str = "Clustering",
    data_type: str = "sequences",
    tablefmt: str = "grid",
    sort_desc: bool = True
) -> None:
    """
    Print comprehensive cluster statistics in a tabulated format.

    Args:
        cluster_map: Dictionary mapping cluster IDs to member lists
        algorithm_name: Name of the clustering algorithm
        data_type: Type of data being clustered (sequences, molecules, etc.)
        tablefmt: Tabulate table format (e.g., "plain", "grid", "fancy_grid")
        sort_desc: Whether to sort clusters by size (largest first)
    """
    if not cluster_map:
        logger.warning(f"No clusters found in {algorithm_name} results.")
        return

    cluster_sizes = [len(members) for members in cluster_map.values()]

    # Statistical information
    total_items = sum(cluster_sizes)
    avg_cluster_size = np.mean(cluster_sizes)
    median_cluster_size = np.median(cluster_sizes)
    max_cluster_size = max(cluster_sizes)
    min_cluster_size = min(cluster_sizes)

    # Summary table
    summary_table = [
        ["Total clusters", len(cluster_map)],
        [f"Total {data_type}", total_items],
        ["Average cluster size", f"{avg_cluster_size:.2f}"],
        ["Median cluster size", f"{median_cluster_size:.1f}"],
        ["Min cluster size", min_cluster_size],
        ["Max cluster size", max_cluster_size],
    ]

    # Format output
    log_message = f"\n[{algorithm_name}] Clustering Statistics:\n"
    log_message += tabulate(summary_table, headers=["Metric", "Value"], tablefmt=tablefmt) + "\n"

    logger.info(log_message)


def calculate_cluster_quality_metrics(
    cluster_labels: List[int],
    similarity_matrix: Optional[np.ndarray] = None,
    true_labels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate various cluster quality metrics.
    
    Args:
        cluster_labels: List of cluster assignments
        similarity_matrix: Optional similarity matrix for internal metrics
        true_labels: Optional true labels for external validation
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Basic statistics
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:  # Remove noise label if present
        unique_labels.remove(-1)
    
    metrics['n_clusters'] = len(unique_labels)
    metrics['n_items'] = len(cluster_labels)
    metrics['n_noise'] = cluster_labels.count(-1)
    
    # Cluster size statistics
    cluster_sizes = []
    for label in unique_labels:
        size = cluster_labels.count(label)
        cluster_sizes.append(size)
    
    if cluster_sizes:
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['min_cluster_size'] = min(cluster_sizes)
        metrics['max_cluster_size'] = max(cluster_sizes)
        metrics['std_cluster_size'] = np.std(cluster_sizes)
    
    # Purity (if true labels provided)
    if true_labels is not None:
        metrics['purity'] = calculate_purity(cluster_labels, true_labels)
    
    return metrics


def calculate_purity(cluster_labels: List[int], true_labels: List[int]) -> float:
    """
    Calculate cluster purity score.
    
    Args:
        cluster_labels: Predicted cluster labels
        true_labels: True class labels
        
    Returns:
        Purity score (0-1, higher is better)
    """
    if len(cluster_labels) != len(true_labels):
        raise ValueError("cluster_labels and true_labels must have same length")
    
    # Group by cluster
    cluster_to_true_labels = {}
    for i, (cluster_id, true_label) in enumerate(zip(cluster_labels, true_labels)):
        if cluster_id != -1:  # Ignore noise
            cluster_to_true_labels.setdefault(cluster_id, []).append(true_label)
    
    total_correct = 0
    total_items = sum(len(labels) for labels in cluster_to_true_labels.values())
    
    # For each cluster, find the most common true label
    for cluster_id, labels in cluster_to_true_labels.items():
        if labels:
            # Count occurrences of each true label
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            # Add the count of the most common label
            total_correct += max(label_counts.values())
    
    return total_correct / total_items if total_items > 0 else 0.0


# =============================================================================
# Conversion Utilities
# =============================================================================

def cluster_map_to_labels(
    cluster_map: Dict[str, List[int]], 
    n_items: int
) -> List[int]:
    """
    Convert cluster map to label array.
    
    Args:
        cluster_map: Dict mapping cluster_id to list of item indices
        n_items: Total number of items
        
    Returns:
        List of cluster labels (-1 for unassigned items)
    """
    labels = [-1] * n_items
    
    for cluster_id, indices in cluster_map.items():
        # Convert cluster_id to integer if it's a string
        cluster_label = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
        for idx in indices:
            if 0 <= idx < n_items:
                labels[idx] = cluster_label
    
    return labels


def labels_to_cluster_map(cluster_labels: List[int]) -> Dict[str, List[int]]:
    """
    Convert label array to cluster map.
    
    Args:
        cluster_labels: List of cluster labels
        
    Returns:
        Dict mapping cluster_id to list of item indices
    """
    cluster_map = {}
    
    for idx, label in enumerate(cluster_labels):
        if label != -1:  # Ignore noise/unassigned
            cluster_id = str(label)
            cluster_map.setdefault(cluster_id, []).append(idx)
    
    return cluster_map


# =============================================================================
# Cluster Result Analysis
# =============================================================================

def analyze_cluster_result(cluster_result, sequences: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze clustering result and provide comprehensive statistics.
    
    Args:
        cluster_result: UnifiedClusterResult object or cluster assignments dict
        sequences: Optional list of original sequences for length analysis
        
    Returns:
        Dictionary containing detailed cluster analysis
    """
    # Extract cluster assignments
    if hasattr(cluster_result, 'cluster_assignments'):
        cluster_assignments = cluster_result.cluster_assignments
        total_clusters = cluster_result.total_clusters
        total_sequences = cluster_result.total_sequences
        algorithm = getattr(cluster_result, 'algorithm', 'unknown')
        parameters = getattr(cluster_result, 'parameters', {})
    else:
        # Legacy format - assume it's already a dict
        cluster_assignments = cluster_result
        total_clusters = len(cluster_assignments)
        total_sequences = sum(len(members) for members in cluster_assignments.values())
        algorithm = 'legacy'
        parameters = {}
    
    # Calculate cluster size statistics
    cluster_sizes = [len(members) for members in cluster_assignments.values()]
    
    if not cluster_sizes:
        return {
            "total_clusters": 0,
            "total_sequences": 0,
            "algorithm": algorithm,
            "parameters": parameters,
            "cluster_size_stats": {},
            "size_distribution": {},
            "quality_metrics": {}
        }
    
    # Basic statistics
    avg_cluster_size = np.mean(cluster_sizes)
    median_cluster_size = np.median(cluster_sizes)
    min_cluster_size = min(cluster_sizes)
    max_cluster_size = max(cluster_sizes)
    std_cluster_size = np.std(cluster_sizes)
    
    # Size distribution
    size_bins = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
    for size in cluster_sizes:
        if size == 1:
            size_bins["1"] += 1
        elif 2 <= size <= 5:
            size_bins["2-5"] += 1
        elif 6 <= size <= 10:
            size_bins["6-10"] += 1
        elif 11 <= size <= 20:
            size_bins["11-20"] += 1
        else:
            size_bins["21+"] += 1
    
    # Calculate percentages for size distribution
    size_distribution = {}
    for size_range, count in size_bins.items():
        percentage = (count / total_clusters) * 100 if total_clusters > 0 else 0
        size_distribution[size_range] = {
            "count": count,
            "percentage": percentage
        }
    
    # Quality metrics
    quality_metrics = {
        "singleton_ratio": size_bins["1"] / total_clusters if total_clusters > 0 else 0,
        "large_cluster_ratio": size_bins["21+"] / total_clusters if total_clusters > 0 else 0,
        "size_variance": std_cluster_size ** 2,
        "compression_ratio": total_sequences / total_clusters if total_clusters > 0 else 1.0
    }
    
    # Sequence length analysis if sequences provided
    length_analysis = {}
    if sequences is not None:
        cluster_length_stats = {}
        for cluster_id, indices in cluster_assignments.items():
            cluster_lengths = [len(sequences[i]) for i in indices if i < len(sequences)]
            if cluster_lengths:
                cluster_length_stats[cluster_id] = {
                    "avg_length": np.mean(cluster_lengths),
                    "min_length": min(cluster_lengths),
                    "max_length": max(cluster_lengths),
                    "std_length": np.std(cluster_lengths)
                }
        
        # Overall length statistics
        all_cluster_lengths = []
        for stats in cluster_length_stats.values():
            all_cluster_lengths.append(stats["avg_length"])
        
        if all_cluster_lengths:
            length_analysis = {
                "cluster_avg_lengths": cluster_length_stats,
                "overall_avg_cluster_length": np.mean(all_cluster_lengths),
                "length_variation_across_clusters": np.std(all_cluster_lengths)
            }
    
    return {
        "total_clusters": total_clusters,
        "total_sequences": total_sequences,
        "algorithm": algorithm,
        "parameters": parameters,
        "cluster_size_stats": {
            "average": avg_cluster_size,
            "median": median_cluster_size,
            "minimum": min_cluster_size,
            "maximum": max_cluster_size,
            "std_deviation": std_cluster_size
        },
        "size_distribution": size_distribution,
        "quality_metrics": quality_metrics,
        "length_analysis": length_analysis
    }


def log_cluster_analysis(cluster_result, sequences: Optional[List[str]] = None, 
                        logger_instance=None) -> None:
    """
    Log detailed cluster analysis results.
    
    Args:
        cluster_result: UnifiedClusterResult object or cluster assignments dict
        sequences: Optional list of original sequences
        logger_instance: Optional logger instance, uses module logger if None
    """
    log = logger_instance or logger
    
    analysis = analyze_cluster_result(cluster_result, sequences)
    
    log.info("=" * 80)
    log.info("🧬 CLUSTERING RESULTS - DETAILED ANALYSIS")
    log.info("=" * 80)
    log.info(f"📊 Cluster Overview:")
    log.info(f"   • Algorithm: {analysis['algorithm']}")
    log.info(f"   • Total clusters created: {analysis['total_clusters']}")
    log.info(f"   • Total sequences clustered: {analysis['total_sequences']}")
    
    if analysis['cluster_size_stats']:
        stats = analysis['cluster_size_stats']
        log.info(f"")
        log.info(f"📈 Cluster Size Statistics:")
        log.info(f"   • Average cluster size: {stats['average']:.2f}")
        log.info(f"   • Median cluster size: {stats['median']}")
        log.info(f"   • Smallest cluster: {stats['minimum']} sequences")
        log.info(f"   • Largest cluster: {stats['maximum']} sequences")
        log.info(f"   • Standard deviation: {stats['std_deviation']:.2f}")
    
    if analysis['size_distribution']:
        log.info(f"")
        log.info(f"📊 Cluster Size Distribution:")
        for size_range, data in analysis['size_distribution'].items():
            count = data['count']
            percentage = data['percentage']
            log.info(f"   • Size {size_range:>5}: {count:>4} clusters ({percentage:>5.1f}%)")
    
    if analysis['quality_metrics']:
        quality = analysis['quality_metrics']
        log.info(f"")
        log.info(f"🎯 Quality Metrics:")
        log.info(f"   • Singleton ratio: {quality['singleton_ratio']:.3f}")
        log.info(f"   • Large cluster ratio: {quality['large_cluster_ratio']:.3f}")
        log.info(f"   • Compression ratio: {quality['compression_ratio']:.2f}x")
        log.info(f"   • Size variance: {quality['size_variance']:.2f}")
    
    if analysis['parameters']:
        log.info(f"")
        log.info(f"⚙️  Clustering Parameters:")
        for key, value in analysis['parameters'].items():
            log.info(f"   • {key}: {value}")
    
    log.info("=" * 80)


def get_cluster_info_dict(cluster_result, sequences: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive information about clustering result.
    
    Args:
        cluster_result: UnifiedClusterResult object or cluster assignments dict
        sequences: Optional list of original sequences
        
    Returns:
        Dictionary containing cluster statistics and parameters,
        or None if no clustering result provided
    """
    if cluster_result is None:
        return None
    
    return analyze_cluster_result(cluster_result, sequences)


def print_cluster_statistics_from_map(
    cluster_map: Dict[str, List[str]], 
    data_type: str = "sequences",
    logger_instance=None
) -> None:
    """
    Print comprehensive cluster statistics from cluster map.
    
    Args:
        cluster_map: Dictionary mapping cluster IDs to sequence ID lists
        data_type: Type of data being clustered (sequences, molecules, etc.)
        logger_instance: Optional logger instance, uses module logger if None
    """
    log = logger_instance or logger
    
    cluster_sizes = [len(members) for members in cluster_map.values()]

    # Statistical information
    total_items = sum(cluster_sizes)
    avg_cluster_size = np.mean(cluster_sizes)
    median_cluster_size = np.median(cluster_sizes)
    max_cluster_size = max(cluster_sizes)
    min_cluster_size = min(cluster_sizes)

    bins = {
        "size = 1": 0,
        "size 2–4": 0,
        "size 5–9": 0,
        "size 10–19": 0,
        "size 20+": 0,
    }

    for size in cluster_sizes:
        if size == 1:
            bins["size = 1"] += 1
        elif 2 <= size <= 4:
            bins["size 2–4"] += 1
        elif 5 <= size <= 9:
            bins["size 5–9"] += 1
        elif 10 <= size <= 19:
            bins["size 10–19"] += 1
        else:
            bins["size 20+"] += 1

    log_message = (
        f"Clustering Statistics:\n"
        f"  Total clusters: {len(cluster_map)}\n"
        f"  Total {data_type}: {total_items}\n"
        f"  Average cluster size: {avg_cluster_size:.2f}\n"
        f"  Median cluster size: {median_cluster_size:.1f}\n"
        f"  Min cluster size: {min_cluster_size}\n"
        f"  Max cluster size: {max_cluster_size}\n"
        f"  Cluster size distribution:\n"
    )

    for label, count in bins.items():
        percentage = (count / len(cluster_map)) * 100
        log_message += f"    {label}: {count} clusters ({percentage:.1f}%)\n"

    log.info(log_message)
