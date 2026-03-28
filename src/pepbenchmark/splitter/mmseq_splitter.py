#!/usr/bin/env python3
"""
MMseqs2-based sequence splitter for homology-aware splitting.

This module provides the MMseqs2Splitter class that uses MMseqs2 clustering
to ensure that similar sequences are placed in the same split, preventing
data leakage in evaluation.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from pepbenchmark.cluster.mmseqs_cluster import MMseqs2Clusterer, MMseqs2Config
from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.splitter.base_splitter import AbstractClusteringSplitter
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class MMseqs2Splitter(AbstractClusteringSplitter):
    """
    MMseqs2-based sequence splitter for homology-aware splitting.

    This splitter uses MMseqs2 clustering to ensure that similar sequences
    are placed in the same split, preventing data leakage in evaluation.

    The splitter calls MMseqs2 clusterer to get UnifiedClusterResult,
    then performs cluster-aware splitting based on the clustering results.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize MMseqs2 splitter with minimal parameters.
        
        All MMseqs2 clustering parameters are specified at split time
        for maximum flexibility.
        
        Args:
            random_seed: Random seed for reproducibility (default: 42)
        """
        super().__init__(random_seed=random_seed)
        self.logger.info("MMseqs2Splitter initialized")
        
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Get clustering result using MMseqs2 clustering.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels (not used for MMseqs2 clustering but passed through)
            **kwargs: MMseqs2 clustering parameters including:
                - min_seq_id: Minimum sequence identity threshold (0.0-1.0, default: 0.8)
                - c: Coverage threshold (0.0-1.0, default: 0.8)
                - cov_mode: Coverage mode (0: query, 1: target, 2: shorter, default: 2)
                - alignment_mode: Alignment mode (0: auto, 1: score, 2: coverage, 3: score+coverage, default: 3)
                - seq_id_mode: Sequence identity mode (0: alignment, 1: shorter, 2: longer, default: 2)
                - s: Search sensitivity (1.0-20.0, default: 8.0)
                - kmer_per_seq: K-mer per sequence (default: 50)
                - cluster_mode: Clustering mode (0: greedy set cover, 1: connected component, 2: greedy incremental, default: 2)
                - threads: Number of threads (None for auto-detection)
                - min_cluster_size: Minimum cluster size for post-processing (default: 1)
                - verbose: Enable verbose logging (default: False)
                
                # Legacy aliases for backward compatibility:
                - identity: Alias for min_seq_id
                - coverage: Alias for c
                - sensitivity: Alias for s
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        logger.info(f"Performing MMseqs2 clustering on {len(sequences)} sequences")
        
        # Extract MMseqs2 parameters with defaults
        min_seq_id = kwargs.get('min_seq_id', kwargs.get('identity', 0.8))
        c = kwargs.get('c', kwargs.get('coverage', 0.8))
        cov_mode = kwargs.get('cov_mode', 2)
        alignment_mode = kwargs.get('alignment_mode', 3)
        seq_id_mode = kwargs.get('seq_id_mode', 2)
        s = kwargs.get('s', kwargs.get('sensitivity', 8.0))
        kmer_per_seq = kwargs.get('kmer_per_seq', 50)
        cluster_mode = kwargs.get('cluster_mode', 2)
        threads = kwargs.get('threads', None)
        min_cluster_size = kwargs.get('min_cluster_size', 1)
        verbose = kwargs.get('verbose', False)
        
        # Warn about legacy parameter usage
        if 'identity' in kwargs:
            logger.warning("Parameter 'identity' is deprecated, use 'min_seq_id' instead")
        if 'coverage' in kwargs:
            logger.warning("Parameter 'coverage' is deprecated, use 'c' instead")
        if 'sensitivity' in kwargs:
            logger.warning("Parameter 'sensitivity' is deprecated, use 's' instead")
        
        # Create MMseqs2 configuration
        config = MMseqs2Config(
            min_seq_id=min_seq_id,
            c=c,
            cov_mode=cov_mode,
            alignment_mode=alignment_mode,
            seq_id_mode=seq_id_mode,
            s=s,
            kmer_per_seq=kmer_per_seq,
            cluster_mode=cluster_mode,
            threads=threads,
            min_cluster_size=min_cluster_size,
            random_seed=self.random_seed,
            verbose=verbose
        )
        
        logger.info(f"MMseqs2 parameters: min_seq_id={min_seq_id}, c={c}, s={s}")
        
        # Create clusterer with configuration
        clusterer = MMseqs2Clusterer(config)
        
        # Perform clustering
        result = clusterer.cluster_sequences(sequences, labels=labels)
        
        logger.info(f"MMseqs2 clustering completed: {result.total_clusters} clusters")
        return result
