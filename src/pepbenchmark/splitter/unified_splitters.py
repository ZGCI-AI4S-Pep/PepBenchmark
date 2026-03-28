"""
Unified cluster-based splitter implementations.

This module provides concrete implementations of cluster-based splitters
that use the unified clustering interface.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pepbenchmark.splitter.base_splitter import AbstractClusteringSplitter
from pepbenchmark.cluster.interfaces import AbstractClusterer, UnifiedClusterResult
from pepbenchmark.cluster.factory import create_cdhit_clusterer, create_mmseqs2_clusterer
from pepbenchmark.cluster.kmer_cluster import KmerClusterer, KmerClusterConfig
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class CDHitSplitter(AbstractClusteringSplitter):
    """
    CDHit-based sequence splitter using unified clustering interface.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        min_cluster_size: int = 1,
        **cdhit_kwargs
    ):
        """
        Initialize CDHit splitter.
        
        Args:
            similarity_threshold: CDHit sequence identity threshold
            min_cluster_size: Minimum cluster size
            **cdhit_kwargs: Additional CDHit parameters
        """
        clusterer = create_cdhit_clusterer(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            **cdhit_kwargs
        )
        super().__init__(clusterer)
        
    def get_split_name(self) -> str:
        """Get descriptive name for this split method."""
        threshold = self.clusterer.config.similarity_threshold
        return f"cdhit_similarity_{threshold:.2f}"


class MMseqs2Splitter(AbstractClusteringSplitter):
    """
    MMseqs2-based sequence splitter using unified clustering interface.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        min_cluster_size: int = 1,
        **mmseqs_kwargs
    ):
        """
        Initialize MMseqs2 splitter.
        
        Args:
            similarity_threshold: MMseqs2 sequence identity threshold
            min_cluster_size: Minimum cluster size
            **mmseqs_kwargs: Additional MMseqs2 parameters
        """
        clusterer = create_mmseqs2_clusterer(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            **mmseqs_kwargs
        )
        super().__init__(clusterer)
        
    def get_split_name(self) -> str:
        """Get descriptive name for this split method."""
        threshold = self.clusterer.config.similarity_threshold
        return f"mmseqs2_similarity_{threshold:.2f}"


class KmerBasedSplitter(AbstractClusteringSplitter):
    """
    K-mer based sequence splitter using unified clustering interface.
    """
    
    def __init__(
        self,
        k_values: List[int],
        min_cluster_size: int = 5,
        max_clusters: Optional[int] = None,
        **kmer_kwargs
    ):
        """
        Initialize k-mer based splitter.
        
        Args:
            k_values: List of k-mer sizes to use for clustering
            min_cluster_size: Minimum cluster size
            max_clusters: Maximum number of clusters
            **kmer_kwargs: Additional k-mer clustering parameters
        """
        # Create k-mer cluster configuration
        if not k_values:
            raise ValueError("k_values must contain at least one k-mer size")
        self.k_values = list(k_values)

        config_kwargs = dict(kmer_kwargs)
        config_kwargs["ks"] = self.k_values[0]

        if max_clusters is not None:
            logger.warning("`max_clusters` is currently ignored by KmerClusterConfig")

        config = KmerClusterConfig(
            min_cluster_size=min_cluster_size,
            **config_kwargs
        )
        
        # Create k-mer clusterer
        clusterer = KmerClusterer(config)
        super().__init__(clusterer)
        
    def get_split_name(self) -> str:
        """Get descriptive name for this split method."""
        k_str = "_".join(map(str, self.k_values))
        min_size = self.clusterer.config.min_cluster_size
        return f"kmer_k{k_str}_minsize{min_size}"


# Factory functions for easy creation
def create_cdhit_splitter(
    similarity_threshold: float = 0.9,
    **kwargs
) -> CDHitSplitter:
    """
    Create a CDHit-based splitter with specified parameters.
    
    Args:
        similarity_threshold: CDHit sequence identity threshold
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        **kwargs: Additional parameters
        
    Returns:
        Configured CDHitSplitter instance
    """
    splitter = CDHitSplitter(similarity_threshold=similarity_threshold, **kwargs)
    return splitter


def create_mmseqs2_splitter(
    similarity_threshold: float = 0.9,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    **kwargs
) -> MMseqs2Splitter:
    """
    Create an MMseqs2-based splitter with specified parameters.
    
    Args:
        similarity_threshold: MMseqs2 sequence identity threshold
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        **kwargs: Additional parameters
        
    Returns:
        Configured MMseqs2Splitter instance
    """
    splitter = MMseqs2Splitter(similarity_threshold=similarity_threshold, **kwargs)
    return splitter


def create_kmer_splitter(
    k_values: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_cluster_size: int = 5,
    **kwargs
) -> KmerBasedSplitter:
    """
    Create a k-mer based splitter with specified parameters.
    
    Args:
        k_values: List of k-mer sizes for clustering
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        min_cluster_size: Minimum cluster size
        **kwargs: Additional parameters
        
    Returns:
        Configured KmerBasedSplitter instance
    """
    splitter = KmerBasedSplitter(
        k_values=k_values,
        min_cluster_size=min_cluster_size,
        **kwargs
    )
    return splitter


class MultiMethodSplitter(AbstractClusteringSplitter):
    """
    A splitter that can use multiple clustering methods and compare results.
    """
    
    def __init__(self, clusterers: Dict[str, AbstractClusterer]):
        """
        Initialize multi-method splitter.
        
        Args:
            clusterers: Dictionary of clusterer name -> clusterer instance
        """
        # Use the first clusterer as primary
        primary_name = next(iter(clusterers.keys()))
        super().__init__(clusterers[primary_name])
        
        self.clusterers = clusterers
        self.clustering_results: Dict[str, UnifiedClusterResult] = {}
        
    def split_sequences(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        **kwargs
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split sequences using all clustering methods and return results comparison.
        
        Args:
            sequences: List of sequences to split
            labels: Optional labels for sequences
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices) using primary clusterer
        """
        logger.info(f"Running multi-method clustering with {len(self.clusterers)} methods")
        
        # Run clustering with all methods
        for name, clusterer in self.clusterers.items():
            try:
                logger.info(f"Clustering with method: {name}")
                result = clusterer.cluster_sequences(sequences, labels, **kwargs)
                self.clustering_results[name] = result
                logger.info(f"Method {name}: {result.total_clusters} clusters from {result.total_sequences} sequences")
            except Exception as e:
                logger.error(f"Clustering failed for method {name}: {e}")
                continue
        
        # Use primary clusterer result for actual splitting
        if self.clustering_results:
            primary_name = next(iter(self.clustering_results.keys()))
            self.last_cluster_result = self.clustering_results[primary_name]
            
            # Perform cluster-based splitting
            return self._split_by_clusters(
                self.last_cluster_result,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                **kwargs
            )
        else:
            raise RuntimeError("All clustering methods failed")
    
    def get_clustering_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison of clustering results across all methods.
        
        Returns:
            Dictionary with comparison metrics for each clustering method
        """
        comparison = {}
        
        for name, result in self.clustering_results.items():
            # Basic metrics
            metrics = {
                "total_clusters": result.total_clusters,
                "total_sequences": result.total_sequences,
                "avg_cluster_size": result.total_sequences / result.total_clusters if result.total_clusters > 0 else 0,
                "method": result.clustering_method
            }
            
            # Cluster size distribution
            cluster_sizes = [len(indices) for indices in result.cluster_assignments.values()]
            if cluster_sizes:
                metrics.update({
                    "min_cluster_size": min(cluster_sizes),
                    "max_cluster_size": max(cluster_sizes),
                    "median_cluster_size": np.median(cluster_sizes),
                    "cluster_size_std": np.std(cluster_sizes)
                })
            
            comparison[name] = metrics
            
        return comparison
    
    def get_split_name(self) -> str:
        """Get descriptive name for this split method."""
        method_names = list(self.clusterers.keys())
        return f"multi_method_{'_'.join(method_names[:3])}"  # Limit to first 3 methods


def create_multi_method_splitter(
    methods: List[str],
    similarity_threshold: float = 0.9,
    k_values: Optional[List[int]] = None,
    **kwargs
) -> MultiMethodSplitter:
    """
    Create a multi-method splitter that compares different clustering approaches.
    
    Args:
        methods: List of method names ('cdhit', 'mmseqs2', 'kmer')
        similarity_threshold: Similarity threshold for sequence-based methods
        k_values: K-mer values for k-mer based method
        **kwargs: Additional parameters
        
    Returns:
        Configured MultiMethodSplitter instance
    """
    clusterers = {}
    normalized_methods = {m.lower() for m in methods}

    if 'cdhit' in normalized_methods:
        clusterers['cdhit'] = create_cdhit_clusterer(
            similarity_threshold=similarity_threshold,
            **kwargs
        )
    
    if 'mmseqs2' in normalized_methods:
        clusterers['mmseqs2'] = create_mmseqs2_clusterer(
            similarity_threshold=similarity_threshold,
            **kwargs
        )

    if 'kmer' in normalized_methods:
        if k_values is None:
            k_values = [3, 4, 5]  # Default k values
        if not k_values:
            raise ValueError("k_values must contain at least one k-mer size")
        
        config = KmerClusterConfig(ks=k_values[0], **kwargs)
        clusterers['kmer'] = KmerClusterer(config)
    
    if not clusterers:
        raise ValueError(f"No valid clustering methods specified: {methods}")
    
    return MultiMethodSplitter(clusterers)
