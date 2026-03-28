"""
Unified Cluster Result Splitter.

This module provides a splitter that can work directly with UnifiedClusterResult objects,
allowing users to use pre-computed clustering results from any algorithm.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import random
from collections import defaultdict

from pepbenchmark.splitter.base_splitter import AbstractSplitter
from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedResultSplitter(AbstractSplitter):
    """
    Splitter that works directly with UnifiedClusterResult objects.
    
    This class enables splitting based on pre-computed clustering results,
    regardless of which algorithm was used to generate them. It provides
    flexibility to use cached clustering results or results from external tools.
    
    Key features:
    - Direct integration with UnifiedClusterResult
    - Multiple distribution strategies
    - Cluster integrity preservation options
    - Support for both cluster-level and sequence-level splitting
    - Comprehensive validation and statistics
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize the unified result splitter.
        
        Args:
            random_seed: Default random seed for reproducibility
        """
        super().__init__()
        self.random_seed = random_seed
        self._last_cluster_result: Optional[UnifiedClusterResult] = None
        self._last_split_result: Optional[Dict[str, List[int]]] = None
    
    def split_from_cluster_result(
        self,
        cluster_result: UnifiedClusterResult,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        cluster_distribution_strategy: str = "balanced",
        preserve_cluster_integrity: bool = True,
        min_cluster_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """
        Generate splits directly from a UnifiedClusterResult.
        
        Args:
            cluster_result: Pre-computed clustering result
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set  
            frac_test: Fraction for test set
            seed: Random seed for reproducibility (overrides default)
            cluster_distribution_strategy: How to distribute clusters across splits
                - "balanced": Try to balance sequence count across splits
                - "random": Randomly assign clusters to splits  
                - "size_aware": Assign largest clusters first to balance sizes
                - "stratified": Maintain cluster size distribution across splits
            preserve_cluster_integrity: Keep sequences from same cluster together
            min_cluster_size: Minimum cluster size to consider (filter small clusters)
            **kwargs: Additional parameters for future extensions
            
        Returns:
            Dictionary with train/valid/test split indices
        """
        logger.info(f"Starting split from cluster result: {cluster_result.total_clusters} clusters, {cluster_result.total_sequences} sequences")
        
        # Validate fractions
        self.validate_fractions(frac_train, frac_valid, frac_test)
        
        # Set random seed
        if seed is None:
            seed = self.random_seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Store cluster result
        self._last_cluster_result = cluster_result
        
        # Filter clusters by size if specified
        filtered_cluster_result = self._filter_clusters_by_size(cluster_result, min_cluster_size)
        
        # Generate splits based on strategy
        if preserve_cluster_integrity:
            split_result = self._split_by_clusters(
                filtered_cluster_result, frac_train, frac_valid, frac_test, 
                cluster_distribution_strategy, seed
            )
        else:
            split_result = self._split_within_clusters(
                filtered_cluster_result, frac_train, frac_valid, frac_test, seed
            )
        
        self._last_split_result = split_result
        
        # Log split statistics
        self._log_split_statistics(split_result, filtered_cluster_result)
        
        return split_result
    
    def get_split_indices(
        self,
        data: Union[List[str], np.ndarray],
        cluster_result: UnifiedClusterResult,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """
        Generate split indices using provided cluster result.
        
        This method provides compatibility with the AbstractSplitter interface
        while using a pre-computed clustering result.
        
        Args:
            data: Input data (used for validation only)
            cluster_result: Pre-computed clustering result
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed for reproducibility
            **kwargs: Additional parameters passed to split_from_cluster_result
            
        Returns:
            Dictionary with train/valid/test split indices
        """
        # Validate data size matches cluster result
        data_size = len(data) if hasattr(data, '__len__') else cluster_result.total_sequences
        if data_size != cluster_result.total_sequences:
            logger.warning(f"Data size ({data_size}) doesn't match cluster result sequences ({cluster_result.total_sequences})")
        
        return self.split_from_cluster_result(
            cluster_result=cluster_result,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed,
            **kwargs
        )
    
    def _filter_clusters_by_size(
        self, 
        cluster_result: UnifiedClusterResult, 
        min_size: Optional[int]
    ) -> UnifiedClusterResult:
        """Filter clusters by minimum size."""
        if min_size is None or min_size <= 1:
            return cluster_result
        
        filtered_assignments = {}
        total_filtered_sequences = 0
        
        for cluster_id, seq_indices in cluster_result.cluster_assignments.items():
            if len(seq_indices) >= min_size:
                filtered_assignments[cluster_id] = seq_indices
                total_filtered_sequences += len(seq_indices)
        
        logger.info(f"Filtered clusters: {len(filtered_assignments)}/{cluster_result.total_clusters} clusters kept, "
                   f"{total_filtered_sequences}/{cluster_result.total_sequences} sequences")
        
        # Create new cluster result with filtered data
        return UnifiedClusterResult(
            cluster_assignments=filtered_assignments,
            total_clusters=len(filtered_assignments),
            total_sequences=total_filtered_sequences,
            algorithm=cluster_result.algorithm,
            parameters={**cluster_result.parameters, "min_cluster_size_filter": min_size},
            metadata=cluster_result.metadata
        )
    
    def _split_by_clusters(
        self,
        cluster_result: UnifiedClusterResult,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        strategy: str,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """Split by assigning entire clusters to splits."""
        cluster_assignments = cluster_result.cluster_assignments
        
        if len(cluster_assignments) < 3:
            logger.warning(f"Only {len(cluster_assignments)} clusters available, may not achieve desired split ratios")
        
        # Get cluster items (cluster_id, sequence_indices)
        cluster_items = list(cluster_assignments.items())
        
        # Initialize splits
        splits = {"train": [], "valid": [], "test": []}
        split_sizes = {"train": 0, "valid": 0, "test": 0}
        target_sizes = {
            "train": int(frac_train * cluster_result.total_sequences),
            "valid": int(frac_valid * cluster_result.total_sequences),
            "test": int(frac_test * cluster_result.total_sequences)
        }
        
        if strategy == "size_aware":
            # Sort by cluster size (largest first) for better distribution
            cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
            
            for cluster_id, seq_indices in cluster_items:
                cluster_size = len(seq_indices)
                
                # Find the split that needs more sequences to reach target ratio
                best_split = min(splits.keys(), 
                               key=lambda s: abs(split_sizes[s] + cluster_size - target_sizes[s]))
                
                # Assign cluster to best split
                splits[best_split].extend(seq_indices)
                split_sizes[best_split] += cluster_size
                
        elif strategy == "random":
            # Completely random assignment
            if seed is not None:
                random.seed(seed)
            
            split_names = ["train", "valid", "test"]
            for cluster_id, seq_indices in cluster_items:
                # Randomly choose a split
                split_name = random.choice(split_names)
                splits[split_name].extend(seq_indices)
                split_sizes[split_name] += len(seq_indices)
                
        elif strategy == "balanced":
            # Balanced assignment - try to maintain target proportions
            if seed is not None:
                random.seed(seed)
            
            # Shuffle clusters for randomness
            random.shuffle(cluster_items)
            
            for cluster_id, seq_indices in cluster_items:
                cluster_size = len(seq_indices)
                
                # Calculate how far each split is from its target
                split_deficits = {}
                for split_name in ["train", "valid", "test"]:
                    current_ratio = split_sizes[split_name] / cluster_result.total_sequences if cluster_result.total_sequences > 0 else 0
                    if split_name == "train":
                        target_ratio = frac_train
                    elif split_name == "valid":
                        target_ratio = frac_valid
                    else:
                        target_ratio = frac_test
                    
                    split_deficits[split_name] = target_ratio - current_ratio
                
                # Assign to the split with largest deficit
                best_split = max(split_deficits.keys(), key=split_deficits.get)
                splits[best_split].extend(seq_indices)
                split_sizes[best_split] += cluster_size
                
        elif strategy == "stratified":
            # Stratified assignment - maintain cluster size distribution
            if seed is not None:
                random.seed(seed)
            
            # Group clusters by size
            size_groups = defaultdict(list)
            for cluster_id, seq_indices in cluster_items:
                size = len(seq_indices)
                size_groups[size].append((cluster_id, seq_indices))
            
            # For each size group, distribute proportionally across splits
            for size, clusters in size_groups.items():
                random.shuffle(clusters)
                
                n_clusters = len(clusters)
                n_train = max(1, int(n_clusters * frac_train))
                n_valid = max(1, int(n_clusters * frac_valid)) if n_clusters > 2 else 0
                
                # Assign clusters to splits
                for i, (cluster_id, seq_indices) in enumerate(clusters):
                    if i < n_train:
                        splits["train"].extend(seq_indices)
                        split_sizes["train"] += len(seq_indices)
                    elif i < n_train + n_valid:
                        splits["valid"].extend(seq_indices)
                        split_sizes["valid"] += len(seq_indices)
                    else:
                        splits["test"].extend(seq_indices)
                        split_sizes["test"] += len(seq_indices)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Supported: balanced, random, size_aware, stratified")
        
        # Sort indices for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        return splits
    
    def _split_within_clusters(
        self,
        cluster_result: UnifiedClusterResult,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """Split sequences within clusters (break cluster integrity)."""
        splits = {"train": [], "valid": [], "test": []}
        
        if seed is not None:
            random.seed(seed)
        
        for cluster_id, seq_indices in cluster_result.cluster_assignments.items():
            # Shuffle sequences within cluster
            shuffled_indices = seq_indices.copy()
            random.shuffle(shuffled_indices)
            
            # Split within cluster
            n_seqs = len(shuffled_indices)
            n_train = int(n_seqs * frac_train)
            n_valid = int(n_seqs * frac_valid)
            
            splits["train"].extend(shuffled_indices[:n_train])
            splits["valid"].extend(shuffled_indices[n_train:n_train + n_valid])
            splits["test"].extend(shuffled_indices[n_train + n_valid:])
        
        # Sort indices for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        return splits
    
    def generate_multiple_splits(
        self,
        cluster_result: UnifiedClusterResult,
        n_splits: int = 5,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Union[List[int], int, None] = None,
        **kwargs: Any
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate multiple random splits from the same cluster result.
        
        Args:
            cluster_result: Pre-computed clustering result
            n_splits: Number of splits to generate
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed or list of seeds
            **kwargs: Additional parameters for split_from_cluster_result
            
        Returns:
            Dictionary with keys "split_0", "split_1", ... containing split indices
        """
        logger.info(f"Generating {n_splits} splits from cluster result")
        
        # Prepare seeds
        if seed is None:
            seeds = [self.random_seed + i for i in range(n_splits)] if self.random_seed is not None else [42 + i for i in range(n_splits)]
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(n_splits)]
        elif isinstance(seed, list):
            if len(seed) != n_splits:
                raise ValueError(f"Expected {n_splits} seeds, got {len(seed)}")
            seeds = seed
        else:
            raise ValueError("Seed must be int, list of ints, or None")
        
        # Generate multiple splits
        splits = {}
        for i in range(n_splits):
            current_seed = seeds[i]
            split_indices = self.split_from_cluster_result(
                cluster_result=cluster_result,
                frac_train=frac_train,
                frac_valid=frac_valid,
                frac_test=frac_test,
                seed=current_seed,
                **kwargs
            )
            splits[f"split_{i}"] = split_indices
            
        return splits
    
    def generate_kfold_splits(
        self,
        cluster_result: UnifiedClusterResult,
        k_folds: int = 5,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate k-fold cross-validation splits from cluster result.
        
        Args:
            cluster_result: Pre-computed clustering result
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with keys "fold_0", "fold_1", ... each containing
            'train' and 'test' indices
        """
        logger.info(f"Generating {k_folds}-fold splits from cluster result")
        
        if seed is None:
            seed = self.random_seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Store cluster result
        self._last_cluster_result = cluster_result
        
        # Assign clusters to folds
        cluster_ids = list(cluster_result.cluster_assignments.keys())
        random.shuffle(cluster_ids)
        
        # Distribute clusters across folds as evenly as possible
        folds = defaultdict(list)
        for i, cluster_id in enumerate(cluster_ids):
            fold_id = i % k_folds
            folds[fold_id].extend(cluster_result.cluster_assignments[cluster_id])
        
        # Create train/test splits for each fold
        fold_splits = {}
        for test_fold in range(k_folds):
            test_indices = sorted(folds[test_fold])
            train_indices = []
            
            for train_fold in range(k_folds):
                if train_fold != test_fold:
                    train_indices.extend(folds[train_fold])
                    
            fold_splits[f"fold_{test_fold}"] = {
                'train': sorted(train_indices),
                'test': test_indices
            }
            
        logger.info(f"{k_folds}-fold splits completed")
        return fold_splits
    
    def analyze_split_quality(
        self,
        split_result: Optional[Dict[str, List[int]]] = None,
        cluster_result: Optional[UnifiedClusterResult] = None
    ) -> Dict[str, Any]:
        """
        Analyze the quality of generated splits.
        
        Args:
            split_result: Split result to analyze (uses last split if None)
            cluster_result: Cluster result to analyze (uses last result if None)
            
        Returns:
            Dictionary with quality metrics
        """
        if split_result is None:
            split_result = self._last_split_result
        if cluster_result is None:
            cluster_result = self._last_cluster_result
            
        if split_result is None or cluster_result is None:
            raise ValueError("No split or cluster result available for analysis")
        
        # Basic split statistics
        total_sequences = sum(len(indices) for indices in split_result.values())
        split_ratios = {name: len(indices) / total_sequences for name, indices in split_result.items()}
        
        # Cluster distribution analysis
        seq_to_cluster = cluster_result.get_sequence_to_cluster_map()
        cluster_split_distribution = defaultdict(lambda: {"train": 0, "valid": 0, "test": 0})
        
        for split_name, seq_indices in split_result.items():
            for seq_idx in seq_indices:
                if seq_idx in seq_to_cluster:
                    cluster_id = seq_to_cluster[seq_idx]
                    cluster_split_distribution[cluster_id][split_name] += 1
        
        # Calculate cluster integrity metrics
        intact_clusters = 0
        partially_split_clusters = 0
        
        for cluster_id, split_counts in cluster_split_distribution.items():
            non_zero_splits = sum(1 for count in split_counts.values() if count > 0)
            if non_zero_splits == 1:
                intact_clusters += 1
            else:
                partially_split_clusters += 1
        
        cluster_integrity_ratio = intact_clusters / len(cluster_split_distribution) if cluster_split_distribution else 0
        
        # Cluster size distribution across splits
        split_cluster_sizes = {}
        for split_name in split_result.keys():
            clusters_in_split = set()
            for seq_idx in split_result[split_name]:
                if seq_idx in seq_to_cluster:
                    clusters_in_split.add(seq_to_cluster[seq_idx])
            
            split_cluster_sizes[split_name] = {
                "num_clusters": len(clusters_in_split),
                "avg_cluster_size": np.mean([len(cluster_result.cluster_assignments[cid]) 
                                           for cid in clusters_in_split]) if clusters_in_split else 0
            }
        
        return {
            "split_ratios": split_ratios,
            "total_sequences": total_sequences,
            "total_clusters": cluster_result.total_clusters,
            "cluster_integrity": {
                "intact_clusters": intact_clusters,
                "partially_split_clusters": partially_split_clusters,
                "integrity_ratio": cluster_integrity_ratio
            },
            "cluster_distribution": dict(split_cluster_sizes),
            "algorithm_info": {
                "algorithm": cluster_result.algorithm,
                "parameters": cluster_result.parameters
            }
        }
    
    def _log_split_statistics(
        self,
        split_result: Dict[str, List[int]],
        cluster_result: UnifiedClusterResult
    ):
        """Log statistics about the split."""
        total_seqs = sum(len(indices) for indices in split_result.values())
        
        logger.info(f"Split completed:")
        for split_name, indices in split_result.items():
            percentage = len(indices) / total_seqs * 100 if total_seqs > 0 else 0
            logger.info(f"  {split_name}: {len(indices)} sequences ({percentage:.1f}%)")
        
        # Analyze cluster distribution across splits
        cluster_distribution = defaultdict(lambda: {"train": 0, "valid": 0, "test": 0})
        
        seq_to_cluster = cluster_result.get_sequence_to_cluster_map()
        
        for split_name, seq_indices in split_result.items():
            for seq_idx in seq_indices:
                if seq_idx in seq_to_cluster:
                    cluster_id = seq_to_cluster[seq_idx]
                    cluster_distribution[cluster_id][split_name] += 1
        
        # Count clusters per split
        clusters_per_split = {"train": set(), "valid": set(), "test": set()}
        for cluster_id, split_counts in cluster_distribution.items():
            for split_name, count in split_counts.items():
                if count > 0:
                    clusters_per_split[split_name].add(cluster_id)
        
        logger.info(f"Cluster distribution:")
        for split_name, clusters in clusters_per_split.items():
            logger.info(f"  {split_name}: {len(clusters)} clusters")
    
    def get_cluster_result(self) -> Optional[UnifiedClusterResult]:
        """Get the last clustering result."""
        return self._last_cluster_result
    
    def get_last_split_result(self) -> Optional[Dict[str, List[int]]]:
        """Get the last split result."""
        return self._last_split_result
