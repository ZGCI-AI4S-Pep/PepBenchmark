
"""
Hybrid clustering-based sequence splitting for homology-aware data splitting.

This module provides HybridSplitter, a splitter that uses the HybridClusterer 
to combine motif extraction with MMseqs2 clustering for comprehensive data splitting.
"""

import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np

from ..cluster.interfaces import UnifiedClusterResult
from ..cluster.hybrid_cluster import HybridClusterer, HybridClusterConfig
from .base_splitter import AbstractClusteringSplitter
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HybridSplitterConfig(HybridClusterConfig):
    """
    Configuration class for HybridSplitter that extends HybridClusterConfig with splitting-specific parameters.
    
    This inherits all hybrid clustering parameters from HybridClusterConfig and adds splitter-specific ones.
    """
    # Splitting-specific parameters (not part of clustering config)
    cluster_distribution_strategy: str = "balanced"  # "balanced", "random", "size_aware"
    preserve_cluster_integrity: bool = True
    balance_labels: bool = False
    
    def to_hybrid_cluster_config(self) -> HybridClusterConfig:
        """Convert to HybridClusterConfig by filtering out splitting-specific parameters."""
        # Get all attributes from parent class
        cluster_params = {}
        for field in HybridClusterConfig.__dataclass_fields__:
            if hasattr(self, field):
                cluster_params[field] = getattr(self, field)
        
        return HybridClusterConfig(**cluster_params)


class HybridSplitter(AbstractClusteringSplitter):
    """
    Hybrid clustering-based sequence splitter for homology-aware splitting.

    This splitter uses the HybridClusterer which combines motif extraction with MMseqs2 clustering
    to create comprehensive data splits that respect both functional motifs and sequence similarity.

    The splitter workflow:
    1. Use motif extraction to identify sequences with significant k-mers (motif enriched cluster)
    2. Apply MMseqs2 clustering to remaining sequences
    3. Distribute clusters across train/valid/test splits while preserving cluster integrity

    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys
    """

    def __init__(
        self,
        config: Optional[HybridSplitterConfig] = None,
        **kwargs
    ):
        """
        Initialize HybridSplitter with hybrid clustering parameters.
        
        Args:
            config: HybridSplitterConfig object with all parameters
            **kwargs: Individual parameters that override config values
        """
        # Create config if not provided
        if config is None:
            config = HybridSplitterConfig(**kwargs)
        else:
            # Override config values with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Store config
        self.config = config
        
        # Store splitting-specific parameters for easy access
        self.cluster_distribution_strategy = config.cluster_distribution_strategy
        self.preserve_cluster_integrity = config.preserve_cluster_integrity
        self.balance_labels = config.balance_labels
        
        # Initialize parent with random seed
        super().__init__(random_seed=getattr(config, 'random_seed', 42))
        
        # Log initialization
        logger.info("HybridSplitter initialized")
        
        # Create clusterer with proper clustering configuration
        cluster_config = config.to_hybrid_cluster_config()
        self.clusterer = HybridClusterer(cluster_config)
        
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Get clustering result using hybrid clustering (motif + MMseqs2).
        
        Args:
            sequences: List of sequences to cluster
            labels: List of labels (0/1) for each sequence - can be None
            **kwargs: Clustering parameters
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        logger.info(f"Performing hybrid clustering on {len(sequences)} sequences")
        
        # Use labels from positional argument, or fall back to kwargs
        labels = labels if labels is not None else kwargs.pop('labels', None)
        
        # Create hybrid configuration from config, allowing kwargs to override
        config_dict = asdict(self.config)
        config_dict.update(kwargs)  # Override with any runtime parameters
        
        # Remove splitting-specific parameters that don't belong in HybridClusterConfig
        splitting_params = ['cluster_distribution_strategy', 'preserve_cluster_integrity', 'balance_labels']
        for param in splitting_params:
            config_dict.pop(param, None)
            
        if labels is None:
            logger.info(f"Using unsupervised hybrid clustering: ks={config_dict.get('ks')}, topM={config_dict.get('topM')}")
        else:
            logger.info(f"Using supervised hybrid clustering: ks={config_dict.get('ks')}, topM={config_dict.get('topM')}")
        
        # Create clusterer with configuration
        cluster_config = HybridClusterConfig(**config_dict)
        clusterer = HybridClusterer(cluster_config)
        
        # Perform clustering
        result = clusterer.cluster_sequences(sequences, labels=labels)
        
        logger.info(f"Hybrid clustering completed: {result.total_clusters} clusters")
        logger.info(f"Motif enriched sequences: {result.metadata.get('motif_enriched_count', 0)}")
        logger.info(f"MMseqs2 clustered sequences: {result.metadata.get('mmseqs_clustered_count', 0)}")
        logger.info(f"Singleton sequences: {result.metadata.get('singleton_count', 0)}")
        
        return result
    
    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about the current hybrid clustering.
        
        Returns:
            Dictionary containing cluster statistics and hybrid-specific information
        """
        base_info = super().get_cluster_info()
        if base_info is not None and self._last_cluster_result:
            # Add hybrid-specific metadata
            if hasattr(self._last_cluster_result, 'metadata') and self._last_cluster_result.metadata:
                metadata = self._last_cluster_result.metadata
                base_info.update({
                    "motif_enriched_count": metadata.get("motif_enriched_count", 0),
                    "mmseqs_clustered_count": metadata.get("mmseqs_clustered_count", 0),
                    "singleton_count": metadata.get("singleton_count", 0),
                    "motif_extraction_method": metadata.get("motif_extraction_method", "unknown"),
                    "cluster_metadata": metadata.get("cluster_metadata", {}),
                })
                
                # Add cluster type distribution
                cluster_metadata = metadata.get("cluster_metadata", {})
                cluster_types = {}
                for cluster_id, cluster_meta in cluster_metadata.items():
                    cluster_type = cluster_meta.get('type', 'unknown')
                    cluster_types[cluster_type] = cluster_types.get(cluster_type, 0) + 1
                
                base_info["cluster_type_distribution"] = cluster_types
                
        return base_info


# Convenience function for direct use
def get_split_indices(
    sequences: List[str],
    labels: Optional[List[int]] = None,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: Optional[int] = 42,
    # Motif extraction parameters
    ks: int = 5,
    topM: Optional[int] = None,
    mode: str = "pos",
    test_method: str = "fisher",
    min_pos_count: int = 3,
    min_neg_count: int = 3,
    # MMseqs2 parameters
    min_seq_id: float = 0.8,
    c: float = 0.8,
    s: float = 8.0,
    # Hybrid control parameters
    use_motif_clustering: bool = True,
    use_mmseqs_clustering: bool = True,
    motif_cluster_name: str = "motif_enriched",
    # Splitting parameters
    cluster_distribution_strategy: str = "balanced",
    preserve_cluster_integrity: bool = True,
    balance_labels: bool = False,
    **kwargs: Any,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """
    Convenience function for hybrid clustering-based splitting without creating a HybridSplitter instance.
    
    This is a wrapper around HybridSplitter.get_split_indices() for easier direct usage.
    
    Args:
        sequences: List of peptide sequences
        labels: List of binary labels (0/1) corresponding to sequences (optional)
        frac_train: Fraction of data for training (default: 0.8)
        frac_valid: Fraction of data for validation (default: 0.1)
        frac_test: Fraction of data for testing (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
        
        # Motif extraction parameters
        ks: K-mer size for motif extraction (default: 5)
        topM: Number of top motifs to consider (default: None)
        mode: Motif extraction mode ("pos", "neg", "both") (default: "pos")
        test_method: Statistical test method ("fisher", "chi2", "ratio") (default: "fisher")
        min_pos_count: Minimum positive count for motifs (default: 3)
        min_neg_count: Minimum negative count for motifs (default: 3)
        
        # MMseqs2 parameters
        min_seq_id: Sequence identity threshold (0.0-1.0) (default: 0.8)
        c: Coverage threshold (0.0-1.0) (default: 0.8)
        s: Sensitivity (1.0-20.0) (default: 8.0)
        
        # Hybrid control parameters
        use_motif_clustering: Whether to enable motif-based clustering (default: True)
        use_mmseqs_clustering: Whether to enable MMseqs2 clustering (default: True)
        motif_cluster_name: Name for the motif enriched cluster (default: "motif_enriched")
        
        # Splitting parameters
        cluster_distribution_strategy: How to distribute clusters ("balanced", "random", "size_aware")
        preserve_cluster_integrity: Keep sequences from same cluster together (default: True)
        balance_labels: Whether to balance class labels across splits (default: False)
        
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing train/valid/test split indices
        
    Example:
        >>> sequences = ["ACDEFG", "GHIKLM", "NOPQRST", ...]
        >>> labels = [1, 0, 1, ...]
        >>> splits = get_split_indices(sequences, labels, ks=5, topM=20)
        >>> train_seqs = [sequences[i] for i in splits['train']]
    """
    # Prepare data - handle both with and without labels
    if labels is not None:
        data = pd.DataFrame({'sequence': sequences, 'label': labels})
    else:
        data = sequences
    
    # Create splitter
    splitter = HybridSplitter(
        ks=ks,
        topM=topM,
        mode=mode,
        test_method=test_method,
        min_pos_count=min_pos_count,
        min_neg_count=min_neg_count,
        min_seq_id=min_seq_id,
        c=c,
        s=s,
        use_motif_clustering=use_motif_clustering,
        use_mmseqs_clustering=use_mmseqs_clustering,
        motif_cluster_name=motif_cluster_name,
        cluster_distribution_strategy=cluster_distribution_strategy,
        preserve_cluster_integrity=preserve_cluster_integrity,
        balance_labels=balance_labels,
        **kwargs
    )
    
    return splitter.get_split_indices(
        data=data,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
        **kwargs
    )


def create_hybrid_splitter(
    # Motif extraction parameters
    ks: int = 5,
    topM: Optional[int] = None,
    top_fraction: Optional[float] = None,
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
    # MMseqs2 parameters
    min_seq_id: float = 0.8,
    c: float = 0.8,
    cov_mode: int = 2,
    alignment_mode: int = 3,
    seq_id_mode: int = 2,
    s: float = 8.0,
    kmer_per_seq: int = 50,
    cluster_mode: int = 2,
    threads: Optional[int] = None,
    # Hybrid control parameters
    use_motif_clustering: bool = True,
    use_mmseqs_clustering: bool = True,
    motif_cluster_name: str = "motif_enriched",
    # Splitting-specific parameters
    cluster_distribution_strategy: str = "balanced",
    preserve_cluster_integrity: bool = True,
    balance_labels: bool = False,
    random_seed: Optional[int] = 42,
    verbose: bool = False
) -> HybridSplitter:
    """
    Create a HybridSplitter with specified parameters.
    
    This is a convenience function that maintains backward compatibility
    for existing code that uses individual parameters.
    
    Args:
        # Motif extraction parameters
        ks: K-mer size for motif extraction (default: 5)
        topM: Number of top motifs to consider (default: None)
        top_fraction: Alternative to topM as fraction (0-1) of total motifs
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
        # Unsupervised motif extraction parameters
        min_count: Minimum count for frequency-based extraction (default: 1)
        # MMseqs2 parameters
        min_seq_id: Sequence identity threshold (0.0-1.0) (default: 0.8)
        c: Coverage threshold (0.0-1.0) (default: 0.8)
        cov_mode: Coverage mode (0: query, 1: target, 2: shorter) (default: 2)
        alignment_mode: Alignment mode (0: auto, 1: score, 2: coverage, 3: score+coverage) (default: 3)
        seq_id_mode: Sequence identity mode (0: alignment, 1: shorter, 2: longer) (default: 2)
        s: Sensitivity (1.0-20.0) (default: 8.0)
        kmer_per_seq: K-mer per sequence (default: 50)
        cluster_mode: Clustering mode (0: greedy set cover, 1: connected component, 2: greedy incremental) (default: 2)
        threads: Number of threads (default: None)
        # Hybrid control parameters
        use_motif_clustering: Whether to enable motif-based clustering (default: True)
        use_mmseqs_clustering: Whether to enable MMseqs2 clustering (default: True)
        motif_cluster_name: Name for the motif enriched cluster (default: "motif_enriched")
        # Splitting-specific parameters
        cluster_distribution_strategy: How to distribute clusters ("balanced", "random", "size_aware")
        preserve_cluster_integrity: Keep sequences from same cluster together (default: True)
        balance_labels: Whether to balance class labels across splits (default: False)
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Enable verbose logging (default: False)
        
    Returns:
        Configured HybridSplitter instance
    """
    config = HybridSplitterConfig(
        ks=ks,
        topM=topM,
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
        min_seq_id=min_seq_id,
        c=c,
        cov_mode=cov_mode,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        s=s,
        kmer_per_seq=kmer_per_seq,
        cluster_mode=cluster_mode,
        threads=threads,
        use_motif_clustering=use_motif_clustering,
        use_mmseqs_clustering=use_mmseqs_clustering,
        motif_cluster_name=motif_cluster_name,
        cluster_distribution_strategy=cluster_distribution_strategy,
        preserve_cluster_integrity=preserve_cluster_integrity,
        balance_labels=balance_labels,
        random_seed=random_seed,
        verbose=verbose
    )
    return HybridSplitter(config)


def test_hybrid_splitter():
    """
    Test hybrid splitter functionality with various scenarios.
    """
    print("=" * 60)
    print("Testing HybridSplitter")
    print("=" * 60)
    
    # Test data with diverse sequences
    test_sequences = [
        'ARRGPGPG',   # Contains ARR, RRG, GPG motifs
        'KKKLGFFF',   # Contains KKK, KLG, GFF motifs  
        'ARRRRGPG',   # Contains ARR, RRR, RGP motifs (similar to seq 0)
        'KKLGFFFF',   # Contains KKL, KLG, GFF motifs (similar to seq 1)
        'WWWLLLGG',   # Contains WWW, WLL, LLG motifs (distinct)
        'YYYQQQAA',   # Contains YYY, YQQ, QAA motifs (distinct)
        'TTTDDDSS',   # Contains TTT, TDD, DSS motifs (distinct)
        'HHHEEECC',   # Contains HHH, HEE, ECC motifs (distinct)
        'ARRWWWPG',   # Contains ARR and WWW motifs (hybrid)
        'KKKYYYLG',   # Contains KKK and YYY motifs (hybrid)
    ]
    test_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Alternating labels
    
    print(f"Test data: {len(test_sequences)} sequences")
    for i, (seq, label) in enumerate(zip(test_sequences, test_labels)):
        print(f"  {i}: {seq} (label={label})")
    
    # Test 1: Basic hybrid splitting (supervised)
    print(f"\n--- Test 1: Basic hybrid splitting (supervised) ---")
    try:
        splitter = create_hybrid_splitter(
            ks=3, 
            topM=10, 
            min_pos_count=2, 
            min_neg_count=2,
            min_seq_id=0.7,
            c=0.7,
            cluster_distribution_strategy="balanced",
            preserve_cluster_integrity=True,
            verbose=False
        )
        
        # Create DataFrame with sequences and labels
        data = pd.DataFrame({'sequence': test_sequences, 'label': test_labels})
        
        result = splitter.get_split_indices(data)
        
        print(f"✅ Hybrid splitting successful!")
        print(f"Split sizes: train={len(result['train'])}, valid={len(result['valid'])}, test={len(result['test'])}")
        
        print(f"\nSplit assignments:")
        for split_name, indices in result.items():
            sequences_in_split = [test_sequences[i] for i in indices]
            labels_in_split = [test_labels[i] for i in indices]
            print(f"  {split_name}: {indices}")
            print(f"    sequences: {sequences_in_split}")
            print(f"    labels: {labels_in_split}")
        
        # Show cluster information
        cluster_info = splitter.get_cluster_info()
        if cluster_info:
            print(f"\nCluster info:")
            print(f"  Total clusters: {cluster_info.get('total_clusters', 'N/A')}")
            print(f"  Motif enriched count: {cluster_info.get('motif_enriched_count', 'N/A')}")
            print(f"  MMseqs2 clustered count: {cluster_info.get('mmseqs_clustered_count', 'N/A')}")
            print(f"  Cluster type distribution: {cluster_info.get('cluster_type_distribution', {})}")
                
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Unsupervised hybrid splitting
    print(f"\n--- Test 2: Unsupervised hybrid splitting ---")
    try:
        splitter = create_hybrid_splitter(
            ks=3, 
            topM=5,
            min_count=2,
            min_seq_id=0.8,
            cluster_distribution_strategy="random",
            verbose=False
        )
        
        result = splitter.get_split_indices(test_sequences)  # No labels
        
        print(f"✅ Unsupervised hybrid splitting successful!")
        print(f"Split sizes: train={len(result['train'])}, valid={len(result['valid'])}, test={len(result['test'])}")
        
        print(f"\nSplit assignments:")
        for split_name, indices in result.items():
            sequences_in_split = [test_sequences[i] for i in indices]
            print(f"  {split_name}: {indices} -> {sequences_in_split}")
                
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Multiple splits
    print(f"\n--- Test 3: Multiple splits ---")
    try:
        splitter = create_hybrid_splitter(ks=3, verbose=False)
        data = pd.DataFrame({'sequence': test_sequences, 'label': test_labels})
        
        multi_splits = splitter.get_split_indices_n(data, n_splits=3)
        
        print(f"✅ Multiple splits successful!")
        print(f"Generated splits: {list(multi_splits.keys())}")
        
        for split_key, split_result in multi_splits.items():
            print(f"  {split_key}: train={len(split_result['train'])}, valid={len(split_result['valid'])}, test={len(split_result['test'])}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: K-fold splits
    print(f"\n--- Test 4: K-fold splits ---")
    try:
        splitter = create_hybrid_splitter(ks=3, verbose=False)
        data = pd.DataFrame({'sequence': test_sequences, 'label': test_labels})
        
        kfold_splits = splitter.get_split_kfold_indices(data, k_folds=3)
        
        print(f"✅ K-fold splits successful!")
        print(f"Generated folds: {list(kfold_splits.keys())}")
        
        for fold_key, fold_result in kfold_splits.items():
            print(f"  {fold_key}: train={len(fold_result['train'])}, valid={len(fold_result['valid'])}, test={len(fold_result['test'])}")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test 5: Convenience function
    print(f"\n--- Test 5: Convenience function ---")
    try:
        result = get_split_indices(
            sequences=test_sequences,
            labels=test_labels,
            ks=3,
            topM=5,
            min_pos_count=2,
            min_seq_id=0.7
        )
        
        print(f"✅ Convenience function successful!")
        print(f"Split sizes: train={len(result['train'])}, valid={len(result['valid'])}, test={len(result['test'])}")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("HybridSplitter testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_hybrid_splitter()
