#!/usr/bin/env python3
"""
Motif-based sequence splitting for homology-aware data splitting.

This module provides MotifSplitter, a splitter that groups sequences based on
shared sequence motifs to prevent data leakage in machine learning evaluation.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np

from ..cluster.interfaces import AbstractClusterer, UnifiedClusterResult
from ..cluster.kmer_cluster import MotifClusterConfig, MotifClusterer
from .base_splitter import AbstractClusteringSplitter
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MotifSplitterConfig(MotifClusterConfig):
    """
    Configuration class for MotifSplitter that extends MotifClusterConfig with splitting-specific parameters.
    
    This inherits all motif clustering parameters from MotifClusterConfig and adds splitter-specific ones.
    """
    # Splitting-specific parameters (not part of clustering config)
    balance_strategy: str = "cluster_aware"
    preserve_ratio: bool = True
    
    def to_motif_cluster_config(self) -> MotifClusterConfig:
        """Convert to MotifClusterConfig by filtering out splitting-specific parameters."""
        return MotifClusterConfig(
            ks=self.ks,
            topM=self.topM,
            top_fraction=self.top_fraction,
            mode=self.mode,
            count_mode=self.count_mode,
            test_method=self.test_method,
            alternative=self.alternative,
            min_pos_count=self.min_pos_count,
            min_neg_count=self.min_neg_count,
            min_cluster_size=self.min_cluster_size,
            strict=self.strict,
            pval_threshold=self.pval_threshold,
            min_score=self.min_score,
            min_pos=self.min_pos,
            min_neg=self.min_neg,
            filter_fn=self.filter_fn,
            fdr_correct=self.fdr_correct,
            min_count=self.min_count,
            min_support=self.min_support,
            min_jaccard=self.min_jaccard,
            conflict_strategy=self.conflict_strategy,
            merge_strategy=self.merge_strategy,
            use_generic_clustering=self.use_generic_clustering,
            clustering_method=self.clustering_method,
            similarity_threshold=self.similarity_threshold
        )


class MotifSplitter(AbstractClusteringSplitter):
    """
    Motif-based sequence splitter for homology-aware splitting.

    This splitter uses motif-based clustering to group sequences that share
    common sequence motifs, preventing data leakage in evaluation by ensuring
    similar sequences are placed in the same split.

    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys

    This implementation inherits from AbstractClusteringSplitter and uses the unified clustering interface.
    """

    def __init__(
        self,
        config: Optional[MotifSplitterConfig] = None,
        **kwargs
    ):
        """
        Initialize MotifSplitter with motif-specific parameters.
        
        Args:
            config: MotifSplitterConfig object with all parameters
            **kwargs: Individual parameters that override config values
        """
        # Create config if not provided
        if config is None:
            config = MotifSplitterConfig(**kwargs)
        else:
            # Override config values with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Store config
        self.config = config
        
        # Store splitting-specific parameters for easy access
        self.balance_strategy = config.balance_strategy
        self.preserve_ratio = config.preserve_ratio
        
        # Initialize parent
        super().__init__(random_seed=getattr(config, 'random_seed', 42))
        
        # Log initialization
        self.logger.info("MotifSplitter initialized")
        
        # Create clusterer with proper clustering configuration
        cluster_config = config.to_motif_cluster_config()
        self.clusterer = MotifClusterer(cluster_config)
        
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Get clustering result using motif-based clustering.
        
        Args:
            sequences: List of sequences to cluster
            labels: List of labels (0/1) for each sequence - can be None if passed via kwargs
            **kwargs: Clustering parameters that may also include 'labels'
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        logger.info(f"Performing motif-based clustering on {len(sequences)} sequences")
        
        # Use labels from parameter, or fall back to kwargs if not provided
        if labels is None:
            labels = kwargs.get('labels', None)
        
        # Create motif configuration from config, allowing kwargs to override
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)  # Override with any runtime parameters
        
        # Remove splitting-specific parameters that don't belong in MotifClusterConfig
        splitting_params = ['balance_strategy', 'preserve_ratio']
        for param in splitting_params:
            config_dict.pop(param, None)
            
        if labels is None:
            logger.info(f"Using unsupervised motif clustering: k={config_dict.get('k')}, top_m={config_dict.get('top_m')}, min_support={config_dict.get('min_support')}")
        else:
            logger.info(f"Using supervised motif clustering: k={config_dict.get('k')}, top_m={config_dict.get('top_m')}, min_support={config_dict.get('min_support')}")
        
        # Create clusterer with configuration
        cluster_config = MotifClusterConfig(**config_dict)
        clusterer = MotifClusterer(cluster_config)
        
        # Perform clustering
        result = clusterer.cluster_sequences(sequences, labels=labels)
        
        logger.info(f"Motif clustering completed: {result.total_clusters} clusters")
        return result
    
    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about the current motif clustering.
        
        Returns:
            Dictionary containing cluster statistics and motif information
        """
        info = super().get_cluster_info()
        if info is not None and hasattr(self, '_last_cluster_result') and self._last_cluster_result:
            # Add motif-specific metadata
            if hasattr(self._last_cluster_result, 'metadata') and self._last_cluster_result.metadata:
                info.update({
                    "motif_dict": self._last_cluster_result.metadata.get("motif_dict", {}),
                    "merge_pairs": self._last_cluster_result.metadata.get("merge_pairs", []),
                    "cluster_stats": self._last_cluster_result.metadata.get("cluster_stats", {})
                })
        return info


# Convenience function for direct use
def get_split_indices(
    sequences: List[str],
    labels: List[int],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: Optional[int] = 42,
    # Basic motif extraction parameters
    k: int = 5,
    top_m: int = 50,
    top_fraction: Optional[float] = None,
    # Supervised motif extraction parameters (extract_motifs_enriched)
    mode: str = "pos",
    count_mode: str = "presence",
    test_method: str = "fisher",
    alternative: str = "greater",
    min_pos_count: int = 3,
    min_neg_count: int = 3,
    min_cluster_size: int = 2,
    strict: bool = False,
    pval_threshold: Optional[float] = None,
    min_score: Optional[float] = None,
    min_pos: Optional[int] = None,
    min_neg: Optional[int] = None,
    filter_fn: Optional[callable] = None,
    fdr_correct: bool = False,
    # Unsupervised motif extraction parameters (extract_motifs_frequency)
    min_count: int = 1,
    # Clustering parameters
    min_support: int = 3,
    min_jaccard: float = 0.5,
    conflict_strategy: str = "max_support",
    merge_strategy: str = "jaccard",
    # Advanced clustering options
    use_generic_clustering: bool = False,
    clustering_method: str = "connected",
    similarity_threshold: float = 0.5,
    # Splitting-specific parameters
    balance_strategy: str = "cluster_aware",
    preserve_ratio: bool = True,
    **kwargs: Any,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """
    Convenience function for motif-based splitting without creating a MotifSplitter instance.
    
    This is a wrapper around MotifSplitter.get_split_indices() for easier direct usage.
    
    Args:
        sequences: List of peptide sequences
        labels: List of binary labels (0/1) corresponding to sequences  
        frac_train: Fraction of data for training (default: 0.8)
        frac_valid: Fraction of data for validation (default: 0.1)
        frac_test: Fraction of data for testing (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
        k: K-mer size for motif extraction (default: 5)
        top_m: Number of top motifs to consider (default: 50)
        min_support: Minimum support for motif merging (default: 3)
        min_jaccard: Minimum Jaccard similarity for motif merging (default: 0.5)
        conflict_strategy: Strategy for resolving conflicts (default: "max_support")
        mode: Motif counting mode (default: "presence") 
        balance_strategy: Class balancing strategy (default: "cluster_aware")
        preserve_ratio: Whether to preserve original positive/negative ratio (default: True)
        use_unified_clustering: Whether to use the unified clustering interface (default: False)
        clustering_method: Method for unified clustering (default: "connected") 
        similarity_threshold: Threshold for similarity-based clustering (default: 0.5)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing train/valid/test split indices
        
    Example:
        >>> sequences = ["ACDEFG", "GHIKLM", "NOPQRST", ...]
        >>> labels = [1, 0, 1, ...]
        >>> splits = get_split_indices(sequences, labels, k=5, top_m=20)
        >>> train_seqs = [sequences[i] for i in splits['train']]
    """
    # Create DataFrame from sequences and labels
    data = pd.DataFrame({'sequence': sequences, 'label': labels})
    
    # Create splitter and perform split
    splitter = MotifSplitter(
        k=k,
        top_m=top_m,
        top_fraction=top_fraction,
        mode=mode,
        count_mode=count_mode,
        test_method=test_method,
        alternative=alternative,
        min_pos_count=min_pos_count,
        min_neg_count=min_neg_count,
        min_cluster_size=min_cluster_size,
        strict=strict,
        pval_threshold=pval_threshold,
        min_score=min_score,
        min_pos=min_pos,
        min_neg=min_neg,
        filter_fn=filter_fn,
        fdr_correct=fdr_correct,
        min_count=min_count,
        min_support=min_support,
        min_jaccard=min_jaccard,
        conflict_strategy=conflict_strategy,
        merge_strategy=merge_strategy,
        use_generic_clustering=use_generic_clustering,
        clustering_method=clustering_method,
        similarity_threshold=similarity_threshold,
        balance_strategy=balance_strategy,
        preserve_ratio=preserve_ratio
    )
    
    return splitter.get_split_indices(
        data=data,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
        **kwargs
    )


def create_motif_splitter(
    # Basic motif extraction parameters
    k: int = 5,
    top_m: int = 50,
    top_fraction: Optional[float] = None,
    # Supervised motif extraction parameters
    mode: str = "pos",
    count_mode: str = "presence",
    test_method: str = "fisher",
    alternative: str = "greater",
    min_pos_count: int = 3,
    min_neg_count: int = 3,
    min_cluster_size: int = 2,
    strict: bool = False,
    pval_threshold: Optional[float] = None,
    min_score: Optional[float] = None,
    min_pos: Optional[int] = None,
    min_neg: Optional[int] = None,
    filter_fn: Optional[callable] = None,
    fdr_correct: bool = False,
    # Unsupervised motif extraction parameters
    min_count: int = 1,
    # Clustering parameters
    min_support: int = 3,
    min_jaccard: float = 0.5,
    conflict_strategy: str = "max_support",
    merge_strategy: str = "jaccard",
    # Advanced clustering options
    use_generic_clustering: bool = False,
    clustering_method: str = "connected",
    similarity_threshold: float = 0.5,
    # Splitting-specific parameters
    balance_strategy: str = "cluster_aware",
    preserve_ratio: bool = True,
    random_seed: Optional[int] = 42,
    verbose: bool = False
) -> MotifSplitter:
    """
    Create a MotifSplitter with specified parameters.
    
    This is a convenience function that maintains backward compatibility
    for existing code that uses individual parameters.
    
    Args:
        # Basic motif extraction parameters
        k: K-mer size for motif extraction (default: 5)
        top_m: Number of top motifs to consider (default: 50)
        top_fraction: Alternative to top_m as fraction (0-1) of total motifs
        # Supervised motif extraction parameters (extract_motifs_enriched)
        mode: Motif extraction mode ("pos", "neg", "both") (default: "pos")
        count_mode: Motif counting mode ("presence", "count") (default: "presence")
        test_method: Statistical test method ("fisher", "chi2", "ratio", etc.) (default: "fisher")
        alternative: Test alternative ("greater", "less", "two-sided") (default: "greater")
        min_pos_count: Minimum positive count for motifs (default: 3)
        min_neg_count: Minimum negative count for motifs (default: 3)
        min_cluster_size: Minimum cluster size for motifs (default: 2)
        strict: Use strict filtering (default: False)
        pval_threshold: P-value threshold for filtering (default: None)
        min_score: Minimum score threshold for filtering (default: None)
        min_pos: Minimum positive samples for filtering (default: None)
        min_neg: Minimum negative samples for filtering (default: None)
        filter_fn: Custom filter function (default: None)
        fdr_correct: Apply FDR correction (default: False)
        # Unsupervised motif extraction parameters (extract_motifs_frequency)
        min_count: Minimum count for frequency-based extraction (default: 1)
        # Clustering parameters
        min_support: Minimum support for motif merging (default: 3)
        min_jaccard: Minimum Jaccard similarity for motif merging (default: 0.5)
        conflict_strategy: Strategy for resolving conflicts ("max_support", "first", "random")
        merge_strategy: Strategy for motif merging ("jaccard", "overlap", "union")
        # Advanced clustering options
        use_generic_clustering: Whether to use unified clustering interface
        clustering_method: Method for unified clustering ("connected", "hierarchical", "kmeans")
        similarity_threshold: Threshold for similarity-based clustering
        # Splitting-specific parameters
        balance_strategy: Class balancing strategy ("cluster_aware", "global", "none")
        preserve_ratio: Whether to preserve original positive/negative ratio
        random_seed: Random seed for reproducibility
        verbose: Enable verbose logging
        
    Returns:
        Configured MotifSplitter instance
    """
    config = MotifSplitterConfig(
        k=k,
        top_m=top_m,
        top_fraction=top_fraction,
        mode=mode,
        count_mode=count_mode,
        test_method=test_method,
        alternative=alternative,
        min_pos_count=min_pos_count,
        min_neg_count=min_neg_count,
        min_cluster_size=min_cluster_size,
        strict=strict,
        pval_threshold=pval_threshold,
        min_score=min_score,
        min_pos=min_pos,
        min_neg=min_neg,
        filter_fn=filter_fn,
        fdr_correct=fdr_correct,
        min_count=min_count,
        min_support=min_support,
        min_jaccard=min_jaccard,
        conflict_strategy=conflict_strategy,
        merge_strategy=merge_strategy,
        use_generic_clustering=use_generic_clustering,
        clustering_method=clustering_method,
        similarity_threshold=similarity_threshold,
        balance_strategy=balance_strategy,
        preserve_ratio=preserve_ratio,
        random_seed=random_seed,
        verbose=verbose
    )
    return MotifSplitter(config)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate some test sequences and labels
    sequences = [
        "ACDEFGHIKLM",  # motif: ACDEF
        "ACDEFGHIKLN",  # motif: ACDEF (similar to above)
        "NOPQRSTUVWX",  # different motif
        "NOPQRSTUVWY",  # similar to above
        "YZABCDEFGHI"   # another motif
    ]
    labels = [1, 1, 0, 0, 1]
    
    # Create splitter
    splitter = MotifSplitter(k=5, top_m=20, min_support=2)
    
    # Generate splits
    splits = splitter.get_split_indices(
        pd.DataFrame({'sequence': sequences, 'label': labels})
    )
    
    print("Split results:")
    for split_name, indices in splits.items():
        print(f"{split_name}: {indices}")
        print(f"Sequences: {[sequences[i] for i in indices]}")
    
    # Get cluster information
    cluster_info = splitter.get_cluster_info()
    if cluster_info:
        print(f"\nCluster info: {cluster_info}")
        
    # Generate multiple splits
    multi_splits = splitter.get_split_indices_n(
        pd.DataFrame({'sequence': sequences, 'label': labels}), 
        n_splits=3
    )
    print(f"\nMultiple splits generated: {list(multi_splits.keys())}")
    
    # Generate k-fold splits
    kfold_splits = splitter.get_split_kfold_indices(
        sequences, 
        k_folds=3, 
        labels=labels
    )
    print(f"\nK-fold splits generated: {list(kfold_splits.keys())}")
