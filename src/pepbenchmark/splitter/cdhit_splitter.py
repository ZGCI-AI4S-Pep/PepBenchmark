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
CD-HIT-based sequence splitter for homology-aware splitting.

This module provides CDHitSplitter, which uses CD-HIT clustering to ensure
that similar sequences are placed in the same split, preventing data leakage.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from pepbenchmark.cluster.cdhit_cluster import CDHitClusterer, CDHitConfig
from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.splitter.base_splitter import AbstractClusteringSplitter
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class CDHitSplitter(AbstractClusteringSplitter):
    """
    CD-HIT-based sequence splitter for homology-aware splitting.

    This splitter uses CD-HIT clustering to ensure that similar sequences
    are placed in the same split, preventing data leakage in evaluation.

    The splitter calls CD-HIT clusterer to get UnifiedClusterResult,
    then performs cluster-aware splitting based on the clustering results.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize CD-HIT splitter with minimal parameters.
        
        All CD-HIT clustering parameters are specified at split time
        for maximum flexibility.
        
        Args:
            random_seed: Random seed for reproducibility (default: 42)
        """
        super().__init__(random_seed=random_seed)
        self.logger.info("CDHitSplitter initialized")
        
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Get clustering result using CD-HIT clustering.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels (not used for CD-HIT clustering but passed through)
            **kwargs: CD-HIT clustering parameters including:
                
                # Primary CD-HIT parameters:
                - c: Sequence identity threshold (0.4-1.0, default: 0.9)
                - aL: Alignment coverage for longer sequence (0.0-1.0, default: 0.7)
                - aS: Alignment coverage for shorter sequence (0.0-1.0, default: 0.0)
                - G: Global/local alignment (0: global, 1: local, default: 0)
                - t: Tolerance for redundancy (0-5, default: 0)
                - n: Word length (2-5, default: 2)
                - d: Length of description in .clstr file (default: 0)
                - l: Minimum length of thrown away sequences (default: 2)
                - g: Accurate mode (0: fast, 1: accurate, default: 1)
                - T: Number of threads (None for auto-detection)
                - M: Memory limit in MB (0: unlimited, default: 0)
                - min_cluster_size: Minimum cluster size for post-processing (default: 1)
                
                # Legacy aliases for backward compatibility:
                - identity: Alias for c
                - local_alignment: If True, sets G=1
                - aln_coverage: Alias for aL
                - tolerant: If True, sets t=1
                - verbose: Enable verbose logging (default: False)
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        logger.info(f"Performing CD-HIT clustering on {len(sequences)} sequences")
        
        # Extract CD-HIT parameters with defaults
        c = kwargs.get('c', kwargs.get('identity', 0.9))
        aL = kwargs.get('aL', kwargs.get('aln_coverage', 0.7))
        aS = kwargs.get('aS', 0.0)
        G = kwargs.get('G', 1 if kwargs.get('local_alignment', False) else 0)
        t = kwargs.get('t', 1 if kwargs.get('tolerant', False) else 0)
        n = kwargs.get('n', 2)
        d = kwargs.get('d', 0)
        l = kwargs.get('l', 2)
        g = kwargs.get('g', 1)
        T = kwargs.get('T', None)
        M = kwargs.get('M', 0)
        min_cluster_size = kwargs.get('min_cluster_size', 1)
        verbose = kwargs.get('verbose', False)
        
        # Warn about legacy parameter usage
        if 'identity' in kwargs:
            logger.warning("Parameter 'identity' is deprecated, use 'c' instead")
        if 'local_alignment' in kwargs:
            logger.warning("Parameter 'local_alignment' is deprecated, use 'G=1' instead")
        if 'aln_coverage' in kwargs:
            logger.warning("Parameter 'aln_coverage' is deprecated, use 'aL' instead")
        if 'tolerant' in kwargs:
            logger.warning("Parameter 'tolerant' is deprecated, use 't=1' instead")
        
        # Create CD-HIT configuration
        config = CDHitConfig(
            c=c,
            aL=aL,
            aS=aS,
            G=G,
            t=t,
            n=n,
            d=d,
            l=l,
            g=g,
            T=T,
            M=M,
            min_cluster_size=min_cluster_size,
            random_seed=self.random_seed,
            verbose=verbose
        )
        
        logger.info(f"CD-HIT parameters: c={c}, aL={aL}, G={G}, t={t}")
        
        # Create clusterer with configuration
        clusterer = CDHitClusterer(config)
        
        # Perform clustering
        result = clusterer.cluster_sequences(sequences, labels=labels)
        
        logger.info(f"CD-HIT clustering completed: {result.total_clusters} clusters")
        return result


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate some test sequences
    sequences = [
        "ACDEFGHIKLM",
        "ACDEFGHIKLN", 
        "NOPQRSTUVWX",
        "NOPQRSTUVWY",
        "YZABCDEFGHI"
    ]
    
    # Create splitter
    splitter = CDHitSplitter()
    
    # Generate splits
    splits = splitter.get_split_indices(sequences, c=0.8)
    
    print("Split results:")
    for split_name, indices in splits.items():
        print(f"{split_name}: {indices}")
        print(f"Sequences: {[sequences[i] for i in indices]}")
    
    # Get cluster information
    cluster_info = splitter.get_cluster_info()
    if cluster_info:
        print(f"\nCluster info: {cluster_info}")
        
    # Generate multiple splits
    multi_splits = splitter.get_split_indices_n(sequences, n_splits=3, c=0.8)
    print(f"\nMultiple splits generated: {list(multi_splits.keys())}")
    
    # Generate k-fold splits
    kfold_splits = splitter.get_split_kfold_indices(sequences, k_folds=3, c=0.8)
    print(f"\nK-fold splits generated: {list(kfold_splits.keys())}")