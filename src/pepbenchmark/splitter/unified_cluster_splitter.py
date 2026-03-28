"""
Enhanced cluster-based splitter that works directly with UnifiedClusterResult.

This module provides a splitter that can work with pre-existing clustering results
from UnifiedClusterResult, enabling flexible separation of clustering and splitting phases.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import random
from collections import defaultdict

from pepbenchmark.splitter.base_splitter import AbstractSplitter
from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedClusterSplitter(AbstractSplitter):
    """
    A splitter that works directly with UnifiedClusterResult objects.
    
    This class separates clustering from splitting, allowing you to:
    1. Use pre-computed clustering results
    2. Apply different splitting strategies to the same clustering
    3. Analyze clustering quality before splitting
    4. Cache clustering results for multiple split experiments
    """
    
    def __init__(self, cluster_result: Optional[UnifiedClusterResult] = None):
        """
        Initialize the unified cluster splitter.
        
        Args:
            cluster_result: Pre-computed clustering result (optional)
        """
        super().__init__()
        self.cluster_result = cluster_result
        self._last_split_info = None
    
    def set_cluster_result(self, cluster_result: UnifiedClusterResult) -> None:
        """
        Set or update the clustering result.
        
        Args:
            cluster_result: Clustering result to use for splitting
        """
        self.cluster_result = cluster_result
        logger.info(f"Set clustering result: {cluster_result.total_clusters} clusters, "
                   f"{cluster_result.total_sequences} sequences, "
                   f"method: {cluster_result.algorithm}")
    
    def get_split_indices(
        self,
        data: Union[List[str], np.ndarray],
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = 42,
        strategy: str = "size_aware",
        preserve_cluster_integrity: bool = True,
        balance_classes: bool = False,
        labels: Optional[List[int]] = None,
        cluster_result: Optional[UnifiedClusterResult] = None,
        **kwargs: Any,
    ) -> Dict[str, Union[List[int], np.ndarray]]:
        """
        Generate split indices from clustering result.
        
        Args:
            data: Input data (used for size validation)
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed for reproducibility
            strategy: Cluster distribution strategy:
                - "size_aware": Distribute largest clusters first
                - "random": Random cluster assignment
                - "balanced": Balance cluster sizes across splits
                - "round_robin": Assign clusters in round-robin fashion
            preserve_cluster_integrity: Keep sequences from same cluster together
            balance_classes: Balance class distribution (requires labels)
            labels: Optional class labels for balanced splitting
            cluster_result: Optional clustering result (overrides instance result)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with train/valid/test split indices
        """
        # Use provided cluster_result or instance cluster_result
        if cluster_result is not None:
            working_result = cluster_result
        elif self.cluster_result is not None:
            working_result = self.cluster_result
        else:
            raise ValueError("No clustering result available. Provide cluster_result parameter or call set_cluster_result()")
        
        # Validate inputs
        self.validate_fractions(frac_train, frac_valid, frac_test)
        
        if len(data) != working_result.total_sequences:
            logger.warning(f"Data size ({len(data)}) doesn't match clustering result size "
                         f"({working_result.total_sequences})")
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        logger.info(f"Starting unified cluster splitting: {working_result.total_clusters} clusters, "
                   f"strategy: {strategy}, preserve_integrity: {preserve_cluster_integrity}")
        
        # Generate splits
        if preserve_cluster_integrity:
            if balance_classes and labels is not None:
                split_result = self._split_with_class_balance(
                    working_result, frac_train, frac_valid, frac_test, 
                    strategy, labels, seed
                )
            else:
                split_result = self._split_by_clusters(
                    working_result, frac_train, frac_valid, frac_test,
                    strategy, seed
                )
        else:
            split_result = self._split_within_clusters(
                working_result, frac_train, frac_valid, frac_test, seed
            )
        
        # Store split information
        self._last_split_info = {
            "strategy": strategy,
            "preserve_integrity": preserve_cluster_integrity,
            "balance_classes": balance_classes,
            "cluster_result": working_result,
            "parameters": kwargs
        }
        
        # Log results
        self._log_split_statistics(split_result, working_result)
        
        return split_result
    
    def _split_by_clusters(
        self,
        cluster_result: UnifiedClusterResult,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        strategy: str,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """
        Split by assigning entire clusters to splits.
        
        Args:
            cluster_result: Clustering result
            frac_train, frac_valid, frac_test: Split fractions
            strategy: Distribution strategy
            seed: Random seed
            
        Returns:
            Split indices dictionary
        """
        cluster_assignments = cluster_result.cluster_assignments
        
        if len(cluster_assignments) < 3:
            logger.warning(f"Only {len(cluster_assignments)} clusters available for 3-way split")
        
        # Get cluster items sorted by strategy
        cluster_items = list(cluster_assignments.items())
        
        if strategy == "size_aware":
            # Sort by cluster size (largest first)
            cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
        elif strategy == "random":
            # Shuffle randomly
            random.shuffle(cluster_items)
        elif strategy in ["balanced", "round_robin"]:
            # For balanced/round_robin, sort by size then add randomness
            cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
            if strategy == "balanced":
                # Add some randomness for balanced distribution
                random.shuffle(cluster_items[:len(cluster_items)//2])
        
        # Initialize splits
        splits = {"train": [], "valid": [], "test": []}
        split_sizes = {"train": 0, "valid": 0, "test": 0}
        
        # Target sizes
        target_sizes = {
            "train": int(frac_train * cluster_result.total_sequences),
            "valid": int(frac_valid * cluster_result.total_sequences),
            "test": int(frac_test * cluster_result.total_sequences)
        }
        
        if strategy == "round_robin":
            # Simple round-robin assignment
            split_names = ["train", "valid", "test"]
            for i, (cluster_id, seq_indices) in enumerate(cluster_items):
                split_name = split_names[i % 3]
                splits[split_name].extend(seq_indices)
                split_sizes[split_name] += len(seq_indices)
        
        elif strategy in ["balanced", "size_aware"]:
            # Greedy assignment to maintain target proportions
            for cluster_id, seq_indices in cluster_items:
                cluster_size = len(seq_indices)
                
                # Find split that would be closest to target after adding this cluster
                best_split = None
                best_score = float('inf')
                
                for split_name in ["train", "valid", "test"]:
                    new_size = split_sizes[split_name] + cluster_size
                    target_size = target_sizes[split_name]
                    
                    # Score based on deviation from target ratio
                    current_ratio = new_size / cluster_result.total_sequences
                    target_ratio = target_size / cluster_result.total_sequences
                    score = abs(current_ratio - target_ratio)
                    
                    if score < best_score:
                        best_score = score
                        best_split = split_name
                
                # Assign to best split
                splits[best_split].extend(seq_indices)
                split_sizes[best_split] += cluster_size
        
        elif strategy == "random":
            # Random assignment maintaining roughly correct proportions
            split_names = ["train", "valid", "test"]
            split_weights = [frac_train, frac_valid, frac_test]
            
            for cluster_id, seq_indices in cluster_items:
                # Weighted random selection
                split_name = np.random.choice(split_names, p=split_weights)
                splits[split_name].extend(seq_indices)
                split_sizes[split_name] += len(seq_indices)
        
        return splits
    
    def _split_with_class_balance(
        self,
        cluster_result: UnifiedClusterResult,
        frac_train: float,
        frac_valid: float, 
        frac_test: float,
        strategy: str,
        labels: List[int],
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """
        Split clusters while balancing class distribution.
        
        Args:
            cluster_result: Clustering result
            frac_train, frac_valid, frac_test: Split fractions
            strategy: Distribution strategy
            labels: Class labels for sequences
            seed: Random seed
            
        Returns:
            Split indices dictionary
        """
        # Analyze cluster class composition
        cluster_class_info = {}
        unique_labels = sorted(set(labels))
        
        for cluster_id, seq_indices in cluster_result.cluster_assignments.items():
            class_counts = {}
            for label in unique_labels:
                class_counts[label] = sum(1 for idx in seq_indices if labels[idx] == label)
            
            total_count = len(seq_indices)
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            cluster_class_info[cluster_id] = {
                'seq_indices': seq_indices,
                'class_counts': class_counts,
                'total_count': total_count,
                'dominant_class': dominant_class
            }
        
        # Group clusters by dominant class
        class_clusters = defaultdict(list)
        for cluster_id, info in cluster_class_info.items():
            class_clusters[info['dominant_class']].append((cluster_id, info))
        
        # Initialize splits
        splits = {"train": [], "valid": [], "test": []}
        split_class_counts = {
            split: {label: 0 for label in unique_labels}
            for split in ["train", "valid", "test"]
        }
        
        # Calculate target class distribution for each split
        total_class_counts = {label: sum(1 for l in labels if l == label) for label in unique_labels}
        target_class_counts = {
            "train": {label: int(count * frac_train) for label, count in total_class_counts.items()},
            "valid": {label: int(count * frac_valid) for label, count in total_class_counts.items()},
            "test": {label: int(count * frac_test) for label, count in total_class_counts.items()}
        }
        
        # Distribute clusters by class
        for label in unique_labels:
            label_clusters = class_clusters[label]
            
            # Sort clusters within class by strategy
            if strategy == "size_aware":
                label_clusters.sort(key=lambda x: x[1]['total_count'], reverse=True)
            elif strategy == "random":
                random.shuffle(label_clusters)
            
            # Assign clusters to splits to balance class distribution
            for cluster_id, cluster_info in label_clusters:
                # Find split that needs more of this class
                best_split = None
                best_score = float('inf')
                
                for split_name in ["train", "valid", "test"]:
                    current_count = split_class_counts[split_name][label]
                    target_count = target_class_counts[split_name][label]
                    cluster_contribution = cluster_info['class_counts'][label]
                    
                    # Score: distance from target after adding cluster
                    new_count = current_count + cluster_contribution
                    score = abs(new_count - target_count)
                    
                    if score < best_score:
                        best_score = score
                        best_split = split_name
                
                # Assign cluster to best split
                splits[best_split].extend(cluster_info['seq_indices'])
                for class_label in unique_labels:
                    split_class_counts[best_split][class_label] += cluster_info['class_counts'][class_label]
        
        # Log final class distribution
        logger.info("Final class distribution:")
        for split_name in ["train", "valid", "test"]:
            class_info = [f"class_{label}: {split_class_counts[split_name][label]}" 
                         for label in unique_labels]
            logger.info(f"  {split_name}: {', '.join(class_info)}")
        
        return splits
    
    def _split_within_clusters(
        self,
        cluster_result: UnifiedClusterResult,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """
        Split sequences within clusters (breaks cluster integrity).
        
        Args:
            cluster_result: Clustering result
            frac_train, frac_valid, frac_test: Split fractions
            seed: Random seed
            
        Returns:
            Split indices dictionary
        """
        splits = {"train": [], "valid": [], "test": []}
        
        for cluster_id, seq_indices in cluster_result.cluster_assignments.items():
            # Shuffle sequences within cluster
            shuffled_indices = seq_indices.copy()
            random.shuffle(shuffled_indices)
            
            # Calculate split sizes for this cluster
            n_seqs = len(shuffled_indices)
            n_train = int(n_seqs * frac_train)
            n_valid = int(n_seqs * frac_valid)
            
            # Assign sequences to splits
            splits["train"].extend(shuffled_indices[:n_train])
            splits["valid"].extend(shuffled_indices[n_train:n_train + n_valid])
            splits["test"].extend(shuffled_indices[n_train + n_valid:])
        
        return splits
    
    def _log_split_statistics(
        self,
        split_result: Dict[str, List[int]],
        cluster_result: UnifiedClusterResult
    ) -> None:
        """Log detailed statistics about the split."""
        total_seqs = sum(len(indices) for indices in split_result.values())
        
        logger.info("Split Statistics:")
        for split_name, indices in split_result.items():
            percentage = len(indices) / total_seqs * 100 if total_seqs > 0 else 0
            logger.info(f"  {split_name}: {len(indices)} sequences ({percentage:.1f}%)")
        
        # Analyze cluster distribution
        if hasattr(cluster_result, 'get_sequence_to_cluster_map'):
            seq_to_cluster = cluster_result.get_sequence_to_cluster_map()
            
            # Count unique clusters per split
            clusters_per_split = {}
            for split_name, seq_indices in split_result.items():
                unique_clusters = set()
                for seq_idx in seq_indices:
                    if seq_idx in seq_to_cluster:
                        unique_clusters.add(seq_to_cluster[seq_idx])
                clusters_per_split[split_name] = len(unique_clusters)
            
            logger.info("Cluster Distribution:")
            for split_name, cluster_count in clusters_per_split.items():
                logger.info(f"  {split_name}: {cluster_count} unique clusters")
    
    def analyze_clustering_before_split(
        self,
        sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze clustering quality before splitting.
        
        Args:
            sequences: Original sequences for similarity analysis
            
        Returns:
            Detailed clustering analysis
        """
        if self.cluster_result is None:
            raise ValueError("No clustering result available")
        
        result = self.cluster_result
        analysis = {
            "basic_stats": result.get_statistics(),
            "cluster_distribution": result.get_cluster_distribution(),
            "largest_clusters": result.get_largest_clusters(10),
            "balance_metrics": result.analyze_cluster_balance()
        }
        
        # Add validation results
        validation = result.validate_clustering()
        analysis["validation"] = validation
        
        # Generate summary
        analysis["summary"] = result.get_summary_table()
        
        return analysis
    
    def get_split_quality_metrics(
        self,
        split_result: Dict[str, List[int]],
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for the split.
        
        Args:
            split_result: Split indices
            labels: Optional labels for class distribution analysis
            
        Returns:
            Quality metrics dictionary
        """
        if self.cluster_result is None:
            raise ValueError("No clustering result available")
        
        metrics = {}
        
        # Basic split metrics
        total_seqs = sum(len(indices) for indices in split_result.values())
        for split_name, indices in split_result.items():
            metrics[f"{split_name}_size"] = len(indices)
            metrics[f"{split_name}_fraction"] = len(indices) / total_seqs if total_seqs > 0 else 0
        
        # Cluster integrity metrics
        if hasattr(self.cluster_result, 'get_sequence_to_cluster_map'):
            seq_to_cluster = self.cluster_result.get_sequence_to_cluster_map()
            
            # Calculate cluster fragmentation
            cluster_splits = defaultdict(set)
            for split_name, seq_indices in split_result.items():
                for seq_idx in seq_indices:
                    if seq_idx in seq_to_cluster:
                        cluster_id = seq_to_cluster[seq_idx]
                        cluster_splits[cluster_id].add(split_name)
            
            fragmented_clusters = sum(1 for splits in cluster_splits.values() if len(splits) > 1)
            total_clusters = len(cluster_splits)
            
            metrics["cluster_integrity"] = {
                "total_clusters": total_clusters,
                "fragmented_clusters": fragmented_clusters,
                "integrity_ratio": 1.0 - (fragmented_clusters / total_clusters) if total_clusters > 0 else 1.0
            }
        
        # Class balance metrics (if labels provided)
        if labels is not None:
            class_distribution = {}
            unique_labels = sorted(set(labels))
            
            for split_name, seq_indices in split_result.items():
                split_labels = [labels[i] for i in seq_indices if i < len(labels)]
                class_counts = {label: split_labels.count(label) for label in unique_labels}
                total_split_size = len(split_labels)
                
                class_distribution[split_name] = {
                    "counts": class_counts,
                    "ratios": {label: count / total_split_size if total_split_size > 0 else 0 
                              for label, count in class_counts.items()}
                }
            
            metrics["class_distribution"] = class_distribution
        
        return metrics
    
    def export_split_results(
        self,
        split_result: Dict[str, List[int]],
        filepath: str,
        include_cluster_info: bool = True,
        format: str = "json"
    ) -> None:
        """
        Export split results with clustering information.
        
        Args:
            split_result: Split indices to export
            filepath: Output file path
            include_cluster_info: Include clustering metadata
            format: Export format ("json" or "numpy")
        """
        export_data = {
            "splits": split_result,
            "split_metadata": {
                "total_sequences": sum(len(indices) for indices in split_result.values()),
                "split_fractions": {
                    name: len(indices) / sum(len(idx) for idx in split_result.values())
                    for name, indices in split_result.items()
                }
            }
        }
        
        if include_cluster_info and self.cluster_result is not None:
            export_data["clustering_info"] = {
                "algorithm": self.cluster_result.algorithm,
                "total_clusters": self.cluster_result.total_clusters,
                "parameters": self.cluster_result.parameters,
                "cluster_assignments": self.cluster_result.cluster_assignments
            }
        
        if self._last_split_info is not None:
            export_data["split_parameters"] = {
                k: v for k, v in self._last_split_info.items() 
                if k != "cluster_result"  # Don't serialize cluster_result
            }
        
        # Save using parent class method
        self.save_split_results(export_data, filepath, format)
        logger.info(f"Split results exported to {filepath}")


def create_unified_cluster_splitter(
    cluster_result: UnifiedClusterResult
) -> UnifiedClusterSplitter:
    """
    Factory function to create a UnifiedClusterSplitter.
    
    Args:
        cluster_result: Pre-computed clustering result
        
    Returns:
        Configured UnifiedClusterSplitter instance
    """
    return UnifiedClusterSplitter(cluster_result)


def split_from_clustering_result(
    cluster_result: UnifiedClusterResult,
    data: Union[List[str], np.ndarray],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    strategy: str = "size_aware",
    seed: Optional[int] = 42,
    **kwargs
) -> Dict[str, List[int]]:
    """
    Convenience function to directly split from a clustering result.
    
    Args:
        cluster_result: Pre-computed clustering result
        data: Original data
        frac_train, frac_valid, frac_test: Split fractions
        strategy: Distribution strategy
        seed: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Split indices dictionary
    """
    splitter = UnifiedClusterSplitter(cluster_result)
    return splitter.get_split_indices(
        data=data,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        strategy=strategy,
        seed=seed,
        **kwargs
    )
