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
Hybrid clustering module combining motif extraction with MMseqs2 clustering.

This module implements a two-stage clustering approach:
1. Use motif extraction (extract_motifs_enriched/extract_motifs_frequency) to identify sequences with significant k-mers
2. Apply MMseqs2 clustering to remaining sequences
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np

from pepbenchmark.utils.logging import get_logger
from pepbenchmark.cluster.interfaces import AbstractClusterer, ClusterConfig, UnifiedClusterResult
from pepbenchmark.cluster.topk_mer import extract_motifs_enriched, extract_motifs_frequency
from pepbenchmark.cluster.mmseqs_cluster import MMseqs2Clusterer, MMseqs2Config
from pepbenchmark.cluster.utils import save_fasta, print_cluster_statistics

logger = get_logger(__name__)


def get_enriched_seq_ids(results):
    """
    Extract the sequence indices associated with enriched motifs.

    Args:
        results: Output returned by ``extract_motifs_enriched`` or
            ``extract_motifs_frequency``.

    Returns:
        A tuple containing the sequence-index set and its size.
    """
    all_seq_ids = set()
    
    # Handle both single-k and multi-k return formats.
    if isinstance(results, OrderedDict) or (isinstance(results, dict) and 
                                           all(isinstance(v, dict) and "seq_ids" in v for v in results.values())):
        # Single-k results use the shape: {motif: {info}}.
        motifs = results
        for motif, info in motifs.items():
            if isinstance(info, dict) and "seq_ids" in info:
                all_seq_ids.update(info["seq_ids"])
    else:
        # Multi-k results use the shape: {k: {motif: {info}}}.
        for k_results in results.values():
            if isinstance(k_results, (dict, OrderedDict)):
                for motif, info in k_results.items():
                    if isinstance(info, dict) and "seq_ids" in info:
                        all_seq_ids.update(info["seq_ids"])
    
    return all_seq_ids, len(all_seq_ids)


@dataclass
class HybridClusterConfig(ClusterConfig):
    """
    Configuration for the hybrid motif-plus-MMseqs2 clusterer.

    The same dataclass stores parameters for both motif extraction and
    MMseqs2-based clustering.
    """
    # === Motif extraction parameters ===
    # Shared parameters used by both motif extraction functions.
    ks: int = 5                   # k-mer length
    topM: Optional[int] = None    # Select the top-N motifs
    top_fraction: Optional[float] = None  # Select the top fraction of motifs
    count_mode: str = "presence"  # Counting mode: "presence" or "count"
    
    # Parameters specific to ``extract_motifs_enriched`` (supervised mode).
    mode: str = "pos"            # "pos", "neg", or "both"
    test_method: str = "fisher"  # Statistical test: "fisher", "chi2", or "ratio"
    alternative: str = "greater" # Hypothesis direction: "greater", "less", or "two-sided"
    min_pos_count: int = 3       # Minimum positive count
    min_neg_count: int = 3       # Minimum negative count
    min_cluster_size: int = 2    # Minimum cluster size
    strict: bool = False         # Strict filtering mode
    pval_threshold: Optional[float] = None  # P-value threshold
    min_score: Optional[float] = None       # Minimum motif score
    min_pos: Optional[int] = None           # Minimum positive support
    min_neg: Optional[int] = None           # Minimum negative support
    filter_fn: Optional[callable] = None    # Custom filter callback
    fdr_correct: bool = False               # Apply FDR multiple-testing correction
    
    # Parameters specific to ``extract_motifs_frequency`` (unsupervised mode).
    min_count: int = 1           # Minimum motif frequency
    
    # === MMseqs2 clustering parameters ===
    # Basic thresholds
    min_seq_id: float = 0.8      # --min-seq-id minimum sequence identity
    c: float = 0.8               # -c coverage threshold
    
    # Clustering parameters
    cov_mode: int = 2            # --cov-mode coverage mode (0: query, 1: target, 2: shorter)
    alignment_mode: int = 3      # --alignment-mode (0: auto, 1: score, 2: coverage, 3: score+coverage)
    seq_id_mode: int = 2         # --seq-id-mode sequence identity mode (0: alignment, 1: shorter, 2: longer)
    
    # Sensitivity and performance
    s: float = 8.0               # -s sensitivity (1.0-20.0)
    kmer_per_seq: int = 50       # --kmer-per-seq k-mer per sequence
    cluster_mode: int = 2        # --cluster-mode (0: greedy set cover, 1: connected component, 2: greedy incremental)
    
    # Performance parameters
    threads: Optional[int] = None  # --threads number of threads
    
    # === Hybrid pipeline controls ===
    use_motif_clustering: bool = True    # Enable the motif stage
    use_mmseqs_clustering: bool = True   # Run MMseqs2 on remaining sequences
    motif_cluster_name: str = "motif_enriched"  # Cluster name for motif hits
    
    def get_motif_params(self, for_enriched: bool = True) -> Dict[str, Any]:
        """Return the motif-extraction subset of the configuration."""
        if for_enriched:
            # Parameters for extract_motifs_enriched (supervised)
            motif_keys = [
                'ks', 'topM', 'top_fraction', 'count_mode', 'mode', 'test_method', 
                'alternative', 'min_pos_count', 'min_neg_count', 'min_cluster_size',
                'strict', 'pval_threshold', 'min_score', 'min_pos', 'min_neg',
                'filter_fn', 'fdr_correct'
            ]
        else:
            # Parameters for extract_motifs_frequency (unsupervised)
            motif_keys = [
                'ks', 'topM', 'top_fraction', 'count_mode', 'min_count'
            ]
        
        params = {}
        for key in motif_keys:
            if hasattr(self, key):
                value = getattr(self, key)
                if value is not None:
                    params[key] = value
        return params
    
    def get_mmseqs2_params(self) -> Dict[str, Any]:
        """Return the MMseqs2-specific subset of the configuration."""
        mmseqs_keys = [
            'min_seq_id', 'c', 'cov_mode', 'alignment_mode', 'seq_id_mode',
            's', 'kmer_per_seq', 'cluster_mode', 'threads'
        ]
        
        params = {}
        for key in mmseqs_keys:
            if hasattr(self, key):
                value = getattr(self, key)
                if value is not None:
                    params[key] = value
        return params


class HybridClusterer(AbstractClusterer):
    """
    Hybrid clusterer that combines motif extraction with MMseqs2.

    Workflow:
    1. Identify sequences containing significant motifs and group them into a
       dedicated motif-enriched cluster.
    2. Cluster the remaining sequences with MMseqs2.
    3. Merge both outputs into a unified clustering result.
    """
    
    def __init__(self, config: Optional[HybridClusterConfig] = None):
        """Initialize the hybrid clusterer."""
        if config is None:
            config = HybridClusterConfig()
        super().__init__(config)
        self._validate_config()
        
        # Initialize the MMseqs2 clusterer used for the second stage.
        mmseqs_config = MMseqs2Config(**config.get_mmseqs2_params())
        self.mmseqs_clusterer = MMseqs2Clusterer(mmseqs_config)
    
    def _validate_config(self):
        """Validate hybrid-specific configuration."""
        config = self.config
        if config.ks <= 0:
            logger.warning(f"k-mer size should be positive, got {config.ks}")
        if config.topM is not None and config.topM <= 0:
            logger.warning(f"topM should be positive, got {config.topM}")
        if not config.use_motif_clustering and not config.use_mmseqs_clustering:
            raise ValueError("At least one of motif_clustering or mmseqs_clustering should be enabled")
    
    def cluster_sequences(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> UnifiedClusterResult:
        """
        Perform hybrid clustering on input sequences.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels (0/1) for each sequence
                   If None, uses frequency-based motif extraction
                   If provided, uses enrichment-based motif extraction
            **kwargs: Additional parameters
            
        Returns:
            UnifiedClusterResult containing clustering results
        """
        if not sequences:
            return UnifiedClusterResult(
                clusters={},
                total_clusters=0,
                total_sequences=0,
                algorithm="hybrid",
                parameters=self.config.to_dict()
            )
        
        config = self.config
        total_sequences = len(sequences)
        
        # Merge configuration defaults with runtime keyword arguments.
        if labels is None:
            # Unsupervised mode uses frequency-based motif extraction.
            motif_params = config.get_motif_params(for_enriched=False)
        else:
            # Supervised mode uses enrichment-based motif extraction.
            motif_params = config.get_motif_params(for_enriched=True)
        
        # Runtime keyword arguments take precedence when they are applicable.
        applicable_params = {k: v for k, v in kwargs.items() if k in motif_params}
        motif_params.update(applicable_params)
        
        logger.info(f"Starting hybrid clustering: {total_sequences} sequences")
        logger.info(f"Motif enabled: {config.use_motif_clustering}, MMseqs2 enabled: {config.use_mmseqs_clustering}")
        
        try:
            cluster_assignments = {}
            cluster_metadata = {}
            motif_enriched_ids = set()
            
            # === Step 1: Motif-based clustering ===
            if config.use_motif_clustering:
                logger.info("Step 1: Extracting motifs and identifying enriched sequences")
                
                if labels is None:
                    # Frequency-based motif extraction for unsupervised mode.
                    logger.info("Using frequency-based motif extraction (unsupervised)")
                    motif_results = extract_motifs_frequency(sequences, **motif_params)
                else:
                    # Enrichment-based motif extraction for supervised mode.
                    logger.info("Using enrichment-based motif extraction (supervised)")
                    motif_results = extract_motifs_enriched(sequences, labels, **motif_params)
                
                # Extract sequences that contain significant k-mers.
                motif_enriched_ids, enriched_count = get_enriched_seq_ids(motif_results)
                logger.info(f"Found {enriched_count} sequences with significant motifs")
                
                if motif_enriched_ids:
                    # Create a dedicated motif-enriched cluster.
                    cluster_name = config.motif_cluster_name
                    cluster_assignments[cluster_name] = sorted(list(motif_enriched_ids))
                    cluster_metadata[cluster_name] = {
                        'type': 'motif_enriched',
                        'size': len(motif_enriched_ids),
                        'motif_results': motif_results,
                        'extraction_method': 'enriched' if labels is not None else 'frequency'
                    }
                    
                    logger.info(f"Created motif enriched cluster: {len(motif_enriched_ids)} sequences")
            
            # === Step 2: MMseqs2 clustering for remaining sequences ===
            remaining_ids = set(range(total_sequences)) - motif_enriched_ids
            logger.info(f"Remaining sequences for MMseqs2 clustering: {len(remaining_ids)}")
            
            if config.use_mmseqs_clustering and remaining_ids:
                logger.info("Step 2: Running MMseqs2 clustering on remaining sequences")
                
                # Prepare the remaining sequences for MMseqs2.
                remaining_sequences = [sequences[i] for i in sorted(remaining_ids)]
                remaining_id_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted(remaining_ids))}
                
                # Run MMseqs2 clustering on the remaining sequences.
                mmseqs_result = self.mmseqs_clusterer.cluster_sequences(remaining_sequences)
                
                # Map MMseqs2 cluster members back to the original indices.
                mmseqs_cluster_id = 0
                for mmseqs_cluster_name, mmseqs_seq_indices in mmseqs_result.cluster_assignments.items():
                    # Convert local MMseqs2 indices to original dataset indices.
                    original_indices = [remaining_id_mapping[idx] for idx in mmseqs_seq_indices]
                    
                    cluster_name = f"mmseqs_{mmseqs_cluster_id}"
                    cluster_assignments[cluster_name] = sorted(original_indices)
                    cluster_metadata[cluster_name] = {
                        'type': 'mmseqs2',
                        'size': len(original_indices),
                        'original_mmseqs_cluster': mmseqs_cluster_name
                    }
                    mmseqs_cluster_id += 1
                
                logger.info(f"MMseqs2 created {len(mmseqs_result.cluster_assignments)} clusters")
            
            # === Step 3: Handle sequences that weren't clustered ===
            all_clustered = set()
            for seq_indices in cluster_assignments.values():
                all_clustered.update(seq_indices)
            
            unclustered_ids = set(range(total_sequences)) - all_clustered
            if unclustered_ids:
                logger.info(f"Creating singleton clusters for {len(unclustered_ids)} unclustered sequences")
                for seq_idx in sorted(unclustered_ids):
                    cluster_name = f"singleton_{seq_idx}"
                    cluster_assignments[cluster_name] = [seq_idx]
                    cluster_metadata[cluster_name] = {
                        'type': 'singleton',
                        'size': 1,
                        'reason': 'not_clustered_by_either_method'
                    }
            
            # === Step 4: Create unified result ===
            result = UnifiedClusterResult(
                cluster_assignments=cluster_assignments,
                total_clusters=len(cluster_assignments),
                total_sequences=total_sequences,
                algorithm="hybrid",
                parameters=config.to_dict(),
                metadata={
                    'cluster_metadata': cluster_metadata,
                    'motif_enriched_count': len(motif_enriched_ids),
                    'mmseqs_clustered_count': len(remaining_ids) if config.use_mmseqs_clustering else 0,
                    'singleton_count': len(unclustered_ids),
                    'motif_extraction_method': 'enriched' if labels is not None else 'frequency'
                }
            )
            
            self._last_result = result
            
            if config.verbose:
                print_cluster_statistics(
                    cluster_assignments,
                    algorithm_name="Hybrid (Motif + MMseqs2)",
                    data_type="sequences"
                )
            
            logger.info(f"Hybrid clustering completed: {result.total_clusters} clusters")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid clustering failed: {e}")
            raise
    
    def cluster_sequences_simple(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, List[int]]:
        """
        Simple clustering interface returning cluster assignments.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels for motif extraction
            **kwargs: Additional parameters
            
        Returns:
            Dict mapping cluster IDs to lists of sequence indices
        """
        result = self.cluster_sequences(sequences, labels, **kwargs)
        return result.cluster_assignments


def create_hybrid_clusterer(
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
    
    **kwargs
) -> HybridClusterer:
    """
    Create a hybrid clusterer with specified parameters.
    
    Args:
        # Motif parameters
        ks: K-mer size for motif extraction
        topM: Number of top motifs to extract
        mode: Motif extraction mode ("pos", "neg", "both")
        test_method: Statistical test method ("fisher", "chi2", "ratio")
        min_pos_count: Minimum positive count for motif
        min_neg_count: Minimum negative count for motif
        
        # MMseqs2 parameters
        min_seq_id: Sequence identity threshold (0.0-1.0)
        c: Coverage threshold (0.0-1.0)
        s: Sensitivity (1.0-20.0)
        
        # Hybrid control
        use_motif_clustering: Whether to enable motif-based clustering
        use_mmseqs_clustering: Whether to enable MMseqs2 clustering
        motif_cluster_name: Name for the motif enriched cluster
        
        **kwargs: Additional parameters
        
    Returns:
        Configured HybridClusterer instance
    """
    config = HybridClusterConfig(
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
        **kwargs
    )
    return HybridClusterer(config)


def test_hybrid_clustering():
    """
    Test hybrid clustering functionality with various scenarios.
    """
    print("=" * 60)
    print("Testing HybridClusterer")
    print("=" * 60)
    
    # Test data with diverse sequences
    test_sequences = [
        'ARRGPGPG',  # Contains ARR, RRG, GPG motifs
        'KKKLGFFF',  # Contains KKK, KLG, GFF motifs  
        'ARRRRGPG',  # Contains ARR, RRR, RGP motifs (similar to seq 0)
        'KKLGFFFF',  # Contains KKL, KLG, GFF motifs (similar to seq 1)
        'WWWLLLGG',  # Contains WWW, WLL, LLG motifs (distinct)
        'YYYQQQAA',  # Contains YYY, YQQ, QAA motifs (distinct)
        'TTTDDDSS',  # Contains TTT, TDD, DSS motifs (distinct)
        'HHHEEECC',  # Contains HHH, HEE, ECC motifs (distinct)
        'ARRWWWPG',  # Contains ARR and WWW motifs (hybrid)
        'KKKYYYLG',  # Contains KKK and YYY motifs (hybrid)
    ]
    test_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Alternating labels
    
    print(f"Test data: {len(test_sequences)} sequences")
    for i, (seq, label) in enumerate(zip(test_sequences, test_labels)):
        print(f"  {i}: {seq} (label={label})")
    
    # Test 1: Basic hybrid clustering (supervised)
    print(f"\n--- Test 1: Basic hybrid clustering (supervised) ---")
    try:
        clusterer = create_hybrid_clusterer(
            ks=3, 
            topM=10, 
            min_pos_count=2, 
            min_neg_count=2,
            min_seq_id=0.7,
            c=0.7,
            verbose=True
        )
        
        result = clusterer.cluster_sequences(test_sequences, test_labels)
        
        print(f"✅ Hybrid clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        print(f"Total sequences: {result.total_sequences}")
        print(f"Algorithm: {result.algorithm}")
        
        print(f"\nCluster assignments:")
        for cluster_id, seq_indices in result.cluster_assignments.items():
            sequences_in_cluster = [test_sequences[i] for i in seq_indices]
            print(f"  {cluster_id}: {seq_indices} -> {sequences_in_cluster}")
        
        # Show cluster metadata
        if 'cluster_metadata' in result.metadata:
            print(f"\nCluster metadata:")
            for cluster_id, metadata in result.metadata['cluster_metadata'].items():
                print(f"  {cluster_id}: {metadata}")
                
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Unsupervised hybrid clustering
    print(f"\n--- Test 2: Unsupervised hybrid clustering ---")
    try:
        clusterer = create_hybrid_clusterer(
            ks=3, 
            topM=5,
            min_count=2,
            min_seq_id=0.8,
            verbose=True
        )
        
        result = clusterer.cluster_sequences(test_sequences)  # No labels
        
        print(f"✅ Unsupervised hybrid clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        
        print(f"\nCluster assignments:")
        for cluster_id, seq_indices in result.cluster_assignments.items():
            sequences_in_cluster = [test_sequences[i] for i in seq_indices]
            print(f"  {cluster_id}: {seq_indices} -> {sequences_in_cluster}")
                
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Only motif clustering
    print(f"\n--- Test 3: Only motif clustering ---")
    try:
        clusterer = create_hybrid_clusterer(
            use_motif_clustering=True,
            use_mmseqs_clustering=False,
            ks=3,
            topM=10,
            verbose=True
        )
        
        result = clusterer.cluster_sequences(test_sequences, test_labels)
        
        print(f"✅ Motif-only clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: Only MMseqs2 clustering
    print(f"\n--- Test 4: Only MMseqs2 clustering ---")
    try:
        clusterer = create_hybrid_clusterer(
            use_motif_clustering=False,
            use_mmseqs_clustering=True,
            min_seq_id=0.6,
            verbose=True
        )
        
        result = clusterer.cluster_sequences(test_sequences)
        
        print(f"✅ MMseqs2-only clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test 5: Simple interface
    print(f"\n--- Test 5: Simple clustering interface ---")
    try:
        clusterer = create_hybrid_clusterer(ks=3, verbose=False)
        cluster_assignments = clusterer.cluster_sequences_simple(test_sequences, test_labels)
        
        print(f"✅ Simple interface successful!")
        print(f"Clusters: {len(cluster_assignments)}")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("HybridClusterer testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_hybrid_clustering()
