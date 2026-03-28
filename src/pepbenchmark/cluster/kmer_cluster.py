"""
Kmer-based clustering module for peptide sequences.

This module provides unified clustering interfaces that combine k-mer extraction
with various clustering algorithms from the cluster package.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

from pepbenchmark.cluster.topk_mer import extract_motifs_enriched, extract_motifs_frequency
from pepbenchmark.cluster.smilarity_cluster import SimilarityClusterer, SimilarityClusterConfig, create_similarity_clusterer
from pepbenchmark.cluster.interfaces import AbstractClusterer, ClusterConfig, UnifiedClusterResult
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KmerClusterConfig(ClusterConfig):
    """Configuration for motif clustering.

    This configuration supports all parameters used by the two main functions
    in `topk_mer.py`:
    - `extract_motifs_frequency`: frequency-based motif extraction
    - `extract_motifs_enriched`: statistically enriched motif extraction

    Parameter names and default values are kept consistent with `topk_mer.py`.
    """
    # Basic parameters shared by both functions; names match `topk_mer.py`.
    ks: int = 5                   # k-mer length (matches the default in `extract_motifs_enriched`)
    topM: Optional[int] = None    # Select the top N motifs
    top_fraction: Optional[float] = None  # Select the top N% of motifs
    count_mode: str = "presence"  # Counting mode: "presence" or "count"
    
    # Parameters specific to `extract_motifs_enriched`; defaults match the function.
    mode: str = "pos"            # Mode: "pos", "neg", or "both"
    test_method: str = "fisher"  # Statistical test method: "fisher", "chi2", "ratio"
    alternative: str = "greater" # Hypothesis direction: "greater", "less", "two-sided"
    min_pos_count: int = 3       # Minimum positive count
    min_neg_count: int = 3       # Minimum negative count
    min_cluster_size: int = 2    # Minimum cluster size
    strict: bool = False         # Strict mode
    pval_threshold: Optional[float] = None  # p-value threshold
    min_score: Optional[float] = None       # Minimum score
    min_pos: Optional[int] = None           # Minimum positive sample count
    min_neg: Optional[int] = None           # Minimum negative sample count
    filter_fn: Optional[callable] = None    # Custom filter function
    fdr_correct: bool = False               # FDR multiple-testing correction
    
    # Parameters specific to `extract_motifs_frequency`; defaults match the function.
    min_count: int = 1           # Minimum occurrence count
    
    # Kmer clustering algorithm parameters.
    min_support: int = 3         # Minimum support required to merge motifs
    min_jaccard: float = 0.5     # Minimum Jaccard similarity required to merge motifs
    conflict_strategy: str = "max_support"  # Conflict resolution strategy: "max_support", "first", "random"
    merge_strategy: str = "jaccard"         # Merge strategy: "jaccard", "overlap", "union"
    use_generic_clustering: bool = False    # Whether to use the generic clustering algorithm
    clustering_method: str = "connected"    # Clustering method: "connected", "hierarchical", "kmeans"
    
    # Generic clustering parameters.
    similarity_threshold: float = 0.8  # Similarity threshold
    
    def get_kmer_params(self) -> Dict[str, Any]:
        """Get motif-specific parameters as a dictionary with names matching topk_mer.py functions."""
        # Parameters for extract_motifs_enriched (supervised), using the same names as the function.
        enriched_keys = [
            'ks', 'topM', 'top_fraction', 'mode', 'count_mode', 'test_method', 
            'alternative', 'min_pos_count', 'min_neg_count', 'min_cluster_size', 
            'strict', 'pval_threshold', 'min_score', 'min_pos', 'min_neg', 
            'filter_fn', 'fdr_correct'
        ]
        
        # Parameters for extract_motifs_frequency (unsupervised), using the same names as the function.
        frequency_keys = [
            'ks', 'topM', 'top_fraction', 'count_mode', 'min_count'
        ]
        
        # Clustering parameters
        cluster_keys = [
            'min_support', 'min_jaccard', 'conflict_strategy', 'merge_strategy', 
            'use_generic_clustering', 'clustering_method', 'similarity_threshold'
        ]
        
        # Combine all possible keys
        all_keys = set(enriched_keys + frequency_keys + cluster_keys)
        
        params = {}
        for key in all_keys:
            if hasattr(self, key):
                value = getattr(self, key)
                if value is not None:
                    params[key] = value
        return params


class KmerClusterer(AbstractClusterer):
    """
    Unified motif-based clustering interface.
    
    This class combines motif extraction with clustering algorithms to provide
    a comprehensive solution for sequence clustering based on motif similarity.
    """
    
    def __init__(self, config: Optional[KmerClusterConfig] = None):
        """Initialize the motif clusterer."""
        if config is None:
            config = KmerClusterConfig()
        super().__init__(config)
        self._validate_config()
    
    def _validate_config(self):
        """Validate motif-specific configuration."""
        config = self.config
        if config.ks <= 0:
            logger.warning(f"k-mer size should be positive, got {config.ks}")
        if config.topM is not None and config.topM <= 0:
            logger.warning(f"topM should be positive, got {config.topM}")
    
    def cluster_sequences(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Cluster sequences based on shared motifs.
        
        Args:
            sequences: List of sequences to cluster
            labels: List of labels (0/1) for each sequence (optional)
                   If None, uses frequency-based motif extraction instead of enrichment-based
            **kwargs: Additional motif-specific parameters
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        config = self.config
        
        if labels is None:
            logger.info(f"Starting unsupervised kmer clustering: {len(sequences)} sequences, ks={config.ks}, topM={config.topM}")
        else:
            logger.info(f"Starting supervised kmer clustering: {len(sequences)} sequences, ks={config.ks}, topM={config.topM}")
        
        try:
            # Merge config parameters with runtime kwargs
            kmer_params = config.get_kmer_params()
            
            # Runtime parameters take highest priority.
            kmer_params.update(kwargs)
            
            # Print parameter information for debugging
            self._print_parameters_info(kmer_params, labels is None)
            
            # Perform clustering
            result = self._cluster_sequences_internal(sequences, labels, kmer_params)
            
            self._last_result = result
            logger.info(f"Kmer clustering completed: {result.total_clusters} clusters")
            
            return result
            
        except Exception as e:
            logger.error(f"Kmer clustering failed: {e}")
            raise
    
    def _print_parameters_info(self, params: Dict[str, Any], is_unsupervised: bool):
        """Print detailed parameter information for debugging."""
        logger.info("=" * 60)
        logger.info("KMER CLUSTERING PARAMETERS")
        logger.info("=" * 60)
        
        # Basic clustering mode
        mode_str = "Unsupervised (frequency-based)" if is_unsupervised else "Supervised (enrichment-based)"
        logger.info(f"Clustering Mode: {mode_str}")
        
        # Kmer extraction parameters
        logger.info("Kmer Extraction Parameters:")
        logger.info(f"  ks (k-mer size): {params.get('ks', 5)}")
        logger.info(f"  topM (max motifs): {params.get('topM', None)}")
        logger.info(f"  count_mode: {params.get('count_mode', 'presence')}")
        
        if not is_unsupervised:
            # Supervised-specific parameters
            logger.info("Supervised Mode Parameters:")
            logger.info(f"  mode: {params.get('mode', 'pos')}")
            logger.info(f"  test_method: {params.get('test_method', 'fisher')}")
            logger.info(f"  min_pos_count: {params.get('min_pos_count', 3)}")
            logger.info(f"  min_neg_count: {params.get('min_neg_count', 3)}")
        else:
            # Unsupervised-specific parameters
            logger.info("Unsupervised Mode Parameters:")
            logger.info(f"  min_count: {params.get('min_count', 1)}")
        
        # Clustering parameters
        logger.info("Clustering Parameters:")
        logger.info(f"  min_support: {params.get('min_support', 3)}")
        logger.info(f"  min_jaccard: {params.get('min_jaccard', 0.5)}")
        logger.info(f"  conflict_strategy: {params.get('conflict_strategy', 'max_support')}")
        logger.info(f"  merge_strategy: {params.get('merge_strategy', 'jaccard')}")
        
        # Advanced clustering options
        logger.info("Advanced Options:")
        logger.info(f"  use_generic_clustering: {params.get('use_generic_clustering', False)}")
        if params.get('use_generic_clustering', False):
            logger.info(f"  clustering_method: {params.get('clustering_method', 'connected')}")
            logger.info(f"  similarity_threshold: {params.get('similarity_threshold', 0.5)}")
        
        # Print additional parameters that may have been passed via kwargs
        extra_params = {k: v for k, v in params.items() 
                       if k not in ['k', 'top_m', 'mode', 'count_mode', 'test_method', 
                                   'min_pos_count', 'min_neg_count', 'min_count', 'min_support',
                                   'min_jaccard', 'conflict_strategy', 'merge_strategy',
                                   'use_generic_clustering', 'clustering_method', 'similarity_threshold']}
        
        if extra_params:
            logger.info("Additional Parameters:")
            for key, value in extra_params.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
    
    def _cluster_sequences_internal(
        self,
        sequences: List[str],
        labels: Optional[List[int]],
        params: Dict[str, Any]
    ) -> UnifiedClusterResult:
        """Internal clustering implementation."""
        ks = params.get('ks', 5)  # Use the updated parameter name `ks`.
        topM = params.get('topM', None)  # Use the updated parameter name `topM`.
        
        if labels is None:
            # Step 1a: Extract motifs using frequency-based method (unsupervised)
            logger.info("Using frequency-based motif extraction (unsupervised)")
            
            # Prepare parameters for extract_motifs_frequency
            freq_params = {
                'ks': ks,
                'topM': topM,
                'top_fraction': params.get('top_fraction'),
                'count_mode': params.get('count_mode', 'presence'),
                'min_count': params.get('min_count', 1)
            }
            
            # Filter out None values
            freq_params = {k: v for k, v in freq_params.items() if v is not None}
            
            logger.info(f"Calling extract_motifs_frequency with params: {freq_params}")
            
            motif_results = extract_motifs_frequency(sequences, **freq_params)
            
            # Convert frequency results to enrichment-like format for compatibility
            if isinstance(motif_results, OrderedDict):
                # Single k result from extract_motifs_frequency
                motif_dict = {}
                for motif, info in motif_results.items():
                    motif_dict[motif] = {
                        'seq_ids': info['seq_ids'],
                        'pos_count': len(info['seq_ids']),  # Treat all as 'positive' for compatibility
                        'neg_count': 0,
                        'score': info['count'],
                        'method': 'frequency'
                    }
            else:
                # Multi-k result, extract the specific k
                ks_results = motif_results.get(ks, {}) if isinstance(motif_results, dict) else {}
                motif_dict = {}
                for motif, info in ks_results.items():
                    motif_dict[motif] = {
                        'seq_ids': info['seq_ids'],
                        'pos_count': len(info['seq_ids']),
                        'neg_count': 0,
                        'score': info['count'],
                        'method': 'frequency'
                    }
        else:
            # Step 1b: Extract motifs using enrichment-based method (supervised)
            logger.info("Using enrichment-based motif extraction (supervised)")
            
            # Prepare parameters for extract_motifs_enriched  
            enrich_params = {
                'ks': ks,
                'topM': topM,
                'top_fraction': params.get('top_fraction'),
                'mode': params.get('mode', 'pos'),
                'count_mode': params.get('count_mode', 'presence'),
                'test_method': params.get('test_method', 'fisher'),
                'alternative': params.get('alternative', 'greater'),
                'min_pos_count': params.get('min_pos_count', 3),
                'min_neg_count': params.get('min_neg_count', 3),
                'min_cluster_size': params.get('min_cluster_size', 2),
                'strict': params.get('strict', False),
                'pval_threshold': params.get('pval_threshold'),
                'min_score': params.get('min_score'),
                'min_pos': params.get('min_pos'),
                'min_neg': params.get('min_neg'),
                'filter_fn': params.get('filter_fn'),
                'fdr_correct': params.get('fdr_correct', False)
            }
            
            # Filter out None values
            enrich_params = {k: v for k, v in enrich_params.items() if v is not None}
            
            logger.info(f"Calling extract_motifs_enriched with params: {enrich_params}")
            
            motif_results = extract_motifs_enriched(sequences, labels, **enrich_params)
            
            if isinstance(motif_results, OrderedDict):
                # Single k result
                motif_dict = dict(motif_results)
            else:
                # Multi-k result
                if ks not in motif_results:
                    logger.warning(f"No motifs extracted for ks={ks}")
                    motif_dict = {}
                else:
                    motif_dict = dict(motif_results[ks])
            
        logger.info(f"Extracted {len(motif_dict)} motifs")
        
        # Step 2: Find motif pairs to merge
        merge_pairs = self._find_merge_pairs(motif_dict, params)
        logger.info(f"Found {len(merge_pairs)} motif pairs to merge")
        
        # Step 3: Build clusters
        if params.get('use_generic_clustering', False):
            cluster_map = self._build_clusters_generic(motif_dict, merge_pairs, params)
        else:
            cluster_map = self._build_clusters_traditional(motif_dict, merge_pairs, params)
            
        logger.info(f"Created {len(cluster_map)} clusters")
        
        # Step 4: Resolve conflicts
        cluster_map = self._resolve_conflicts(cluster_map, params.get('conflict_strategy', 'max_support'))
        logger.info(f"Final clusters after conflict resolution: {len(cluster_map)}")
        
        # Step 5: Convert to unified format
        result = self._convert_to_kmer_result(cluster_map, motif_dict, merge_pairs, len(sequences))
        
        return result
    
    def cluster_sequences_simple(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, List[int]]:
        """
        Simple clustering interface that returns cluster_id -> sequence_indices mapping.
        
        Args:
            sequences: List of sequences to cluster
            labels: List of labels for motif extraction (optional)
                   If None, uses frequency-based motif extraction
            **kwargs: Kmer-specific parameters
            
        Returns:
            Dictionary mapping cluster IDs to lists of sequence indices
        """
        result = self.cluster_sequences(sequences, labels, **kwargs)
        return result.cluster_assignments
    
    def _convert_to_kmer_result(
        self,
        cluster_map: Dict[str, Dict],
        motif_dict: Dict[str, Dict],
        merge_pairs: List[Tuple[str, str, float]],
        total_sequences: int
    ) -> UnifiedClusterResult:
        """Convert internal cluster map to UnifiedClusterResult format."""
        cluster_assignments = {}
        cluster_representatives = {}
        cluster_metadata = {}
        
        cluster_id = 0
        clustered_sequences = set()
        
        # Process existing clusters
        for cluster_name, cluster_info in cluster_map.items():
            seq_indices = cluster_info.get('seq_ids', [])
            if seq_indices:
                unified_cluster_name = f"kmer_cluster_{cluster_id}"
                cluster_assignments[unified_cluster_name] = seq_indices
                
                # Track which sequences have been clustered
                clustered_sequences.update(seq_indices)
                
                # Representative is the first sequence (could be improved)
                cluster_representatives[unified_cluster_name] = seq_indices[0] if seq_indices else 0
                
                cluster_metadata[unified_cluster_name] = {
                    "size": len(seq_indices),
                    "motifs": cluster_info.get('motifs', []),
                    "support": cluster_info.get('support', len(seq_indices)),
                    "representative_index": seq_indices[0] if seq_indices else 0,
                    "original_cluster_name": cluster_name
                }
                
                cluster_id += 1
        
        # Create singleton clusters for unclustered sequences
        all_sequences = set(range(total_sequences))
        unclustered_sequences = all_sequences - clustered_sequences
        
        logger.info(f"Found {len(unclustered_sequences)} unclustered sequences, creating singleton clusters")
        
        for seq_idx in sorted(unclustered_sequences):
            singleton_cluster_name = f"kmer_cluster_{cluster_id}"
            cluster_assignments[singleton_cluster_name] = [seq_idx]
            cluster_representatives[singleton_cluster_name] = seq_idx
            
            cluster_metadata[singleton_cluster_name] = {
                "size": 1,
                "motifs": [],  # No motifs for singleton clusters
                "support": 1,
                "representative_index": seq_idx,
                "original_cluster_name": f"singleton_{seq_idx}",
                "is_singleton": True
            }
            
            cluster_id += 1
        
        return UnifiedClusterResult(
            cluster_assignments=cluster_assignments,
            total_clusters=len(cluster_assignments),
            total_sequences=total_sequences,
            algorithm="kmer",
            parameters=self.config.to_dict(),
            metadata={
                "cluster_representatives": cluster_representatives,
                "cluster_metadata": cluster_metadata,
                "motif_dict": motif_dict,
                "merge_pairs": merge_pairs,
                "raw_cluster_map": cluster_map,
                "singleton_clusters": len(unclustered_sequences),
                "motif_clusters": len(cluster_map)
            }
        )
    
    def _find_merge_pairs(
        self, 
        motif_dict: Dict[str, Dict], 
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of motifs that should be merged based on sequence overlap."""
        motifs = list(motif_dict.keys())
        merge_pairs = []
        
        min_support = params.get('min_support', 3)
        min_jaccard = params.get('min_jaccard', 0.5)
        merge_strategy = params.get('merge_strategy', 'jaccard')
        
        for i in range(len(motifs)):
            for j in range(i + 1, len(motifs)):
                motif1, motif2 = motifs[i], motifs[j]
                
                seq_ids1 = set(motif_dict[motif1]['seq_ids'])
                seq_ids2 = set(motif_dict[motif2]['seq_ids'])
                
                # Calculate similarity based on strategy
                intersection = len(seq_ids1 & seq_ids2)
                
                if merge_strategy == "jaccard":
                    union = len(seq_ids1 | seq_ids2)
                    similarity = intersection / union if union > 0 else 0
                elif merge_strategy == "overlap":
                    min_size = min(len(seq_ids1), len(seq_ids2))
                    similarity = intersection / min_size if min_size > 0 else 0
                elif merge_strategy == "union":
                    similarity = intersection / min_support if min_support > 0 else 0
                else:
                    raise ValueError(f"Unknown merge strategy: {merge_strategy}")
                
                # Check if they should be merged
                if intersection >= min_support and similarity >= min_jaccard:
                    merge_pairs.append((motif1, motif2, similarity))
                    
        return merge_pairs
    
    def _build_clusters_traditional(
        self,
        motif_dict: Dict[str, Dict],
        merge_pairs: List[Tuple[str, str, float]],
        params: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Build clusters using traditional greedy merging approach."""
        # Start with individual clusters
        cluster_map = {}
        
        for motif, info in motif_dict.items():
            cluster_map[f"cluster_{motif}"] = {
                'motifs': [motif],
                'seq_ids': info['seq_ids'].copy(),
                'support': len(info['seq_ids'])
            }
        
        # Merge based on pairs (greedy approach)
        motif_to_cluster = {motif: f"cluster_{motif}" for motif in motif_dict.keys()}
        
        # Sort merge pairs by similarity (descending)
        sorted_pairs = sorted(merge_pairs, key=lambda x: x[2], reverse=True)
        
        for motif1, motif2, similarity in sorted_pairs:
            cluster1_name = motif_to_cluster.get(motif1)
            cluster2_name = motif_to_cluster.get(motif2)
            
            if (cluster1_name and cluster2_name and 
                cluster1_name != cluster2_name and 
                cluster1_name in cluster_map and 
                cluster2_name in cluster_map):
                
                # Merge cluster2 into cluster1
                cluster1 = cluster_map[cluster1_name]
                cluster2 = cluster_map[cluster2_name]
                
                # Update cluster1
                cluster1['motifs'].extend(cluster2['motifs'])
                cluster1['seq_ids'].extend(cluster2['seq_ids'])
                cluster1['seq_ids'] = list(set(cluster1['seq_ids']))  # Remove duplicates
                cluster1['support'] = len(cluster1['seq_ids'])
                
                # Update motif mapping
                for motif in cluster2['motifs']:
                    motif_to_cluster[motif] = cluster1_name
                    
                # Remove cluster2
                del cluster_map[cluster2_name]
        
        return cluster_map
    
    def _build_clusters_generic(
        self,
        motif_dict: Dict[str, Dict],
        merge_pairs: List[Tuple[str, str, float]],
        params: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Build clusters using generic clustering algorithms."""
        if not motif_dict:
            return {}
            
        # Build similarity matrix from motifs
        motifs = list(motif_dict.keys())
        n_motifs = len(motifs)
        similarity_matrix = np.zeros((n_motifs, n_motifs))
        
        # Fill similarity matrix
        motif_to_idx = {motif: i for i, motif in enumerate(motifs)}
        
        for motif1, motif2, similarity in merge_pairs:
            if motif1 in motif_to_idx and motif2 in motif_to_idx:
                i, j = motif_to_idx[motif1], motif_to_idx[motif2]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Set diagonal to 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Use generic clustering
        try:
            # Create similarity clusterer
            clusterer_config = SimilarityClusterConfig(
                similarity_threshold=params.get('similarity_threshold', 0.5),
                method=params.get('clustering_method', 'connected')
            )
            clusterer = SimilarityClusterer(clusterer_config)
            
            # Prepare sequence list for clustering (motif strings)
            motif_sequences = [motif for motif in motifs]
            
            cluster_result = clusterer.cluster_sequences(
                sequences=motif_sequences,
                precomputed_similarity=similarity_matrix
            )
            
            # Convert generic result to our format
            cluster_map = {}
            for cluster_id, motif_indices in cluster_result.cluster_assignments.items():
                if not motif_indices:  # Skip empty clusters
                    continue
                    
                # Get motifs in this cluster
                cluster_motifs = [motifs[i] for i in motif_indices]
                
                # Collect all sequence IDs from these motifs
                all_seq_ids = []
                for motif in cluster_motifs:
                    all_seq_ids.extend(motif_dict[motif]['seq_ids'])
                
                all_seq_ids = list(set(all_seq_ids))  # Remove duplicates
                
                cluster_map[f"generic_cluster_{cluster_id}"] = {
                    'motifs': cluster_motifs,
                    'seq_ids': all_seq_ids,
                    'support': len(all_seq_ids)
                }
                
            return cluster_map
            
        except Exception as e:
            logger.warning(f"Generic clustering failed: {e}, falling back to traditional method")
            return self._build_clusters_traditional(motif_dict, merge_pairs, params)
    
    def _resolve_conflicts(
        self, 
        cluster_map: Dict[str, Dict], 
        strategy: str = "max_support"
    ) -> Dict[str, Dict]:
        """Resolve conflicts when sequences belong to multiple clusters."""
        # Find sequences that appear in multiple clusters
        seq_to_clusters = defaultdict(list)
        
        for cluster_name, cluster_info in cluster_map.items():
            for seq_id in cluster_info['seq_ids']:
                seq_to_clusters[seq_id].append(cluster_name)
        
        # Find conflicts
        conflicts = {seq_id: clusters for seq_id, clusters in seq_to_clusters.items() if len(clusters) > 1}
        
        if not conflicts:
            return cluster_map  # No conflicts
        
        logger.info(f"Resolving {len(conflicts)} sequence conflicts using strategy: {strategy}")
        
        # Resolve conflicts
        for seq_id, cluster_names in conflicts.items():
            if strategy == "max_support":
                # Assign to cluster with highest support
                best_cluster = max(cluster_names, key=lambda c: cluster_map[c]['support'])
            elif strategy == "first":
                best_cluster = cluster_names[0]
            elif strategy == "random":
                import random
                best_cluster = random.choice(cluster_names)
            else:
                best_cluster = cluster_names[0]  # Default to first
                
            # Remove from other clusters
            for cluster_name in cluster_names:
                if cluster_name != best_cluster and seq_id in cluster_map[cluster_name]['seq_ids']:
                    cluster_map[cluster_name]['seq_ids'].remove(seq_id)
                    cluster_map[cluster_name]['support'] = len(cluster_map[cluster_name]['seq_ids'])
        
        # Remove empty clusters
        empty_clusters = [name for name, info in cluster_map.items() if len(info['seq_ids']) == 0]
        for name in empty_clusters:
            del cluster_map[name]
            
        logger.info(f"Conflict resolution completed. Removed {len(empty_clusters)} empty clusters")
        
        return cluster_map
    
def create_kmer_clusterer(
    ks: int = 5,
    topM: int = None,
    **kwargs
) -> KmerClusterer:
    """
    Create a motif clusterer with specified parameters.
    
    Args:
        ks: K-mer size for motif extraction (name matches `topk_mer.py`)
        topM: Number of top motifs to extract (name matches `topk_mer.py`)
        **kwargs: Additional motif clustering parameters
        
    Returns:
        Configured KmerClusterer instance
    """
    config = KmerClusterConfig(
        ks=ks,
        topM=topM,
        **kwargs
    )
    return KmerClusterer(config)


def create_motif_clusterer(
    ks: int = 5,
    topM: int = None,
    **kwargs
) -> KmerClusterer:
    """Backward-compatible alias for `create_kmer_clusterer`."""
    return create_kmer_clusterer(ks=ks, topM=topM, **kwargs)


MotifClusterConfig = KmerClusterConfig
MotifClusterer = KmerClusterer


def test_kmer_clustering():
    """
    Test motif clustering functionality with various scenarios.
    """
    print("=" * 60)
    print("Testing KmerClusterer")
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
    
    # Test 1: Basic motif clustering
    print(f"\n--- Test 1: Basic motif clustering ---")
    try:
        clusterer = create_kmer_clusterer(ks=3, topM=10, min_support=2, min_jaccard=0.3)
        
        result = clusterer.cluster_sequences(test_sequences, test_labels)
        
        print(f"✅ Clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        print(f"Total sequences: {result.total_sequences}")
        print(f"Algorithm: {result.algorithm}")
        
        print(f"\nCluster assignments:")
        for cluster_id, seq_indices in result.cluster_assignments.items():
            print(f"  {cluster_id}: sequences {seq_indices}")
            for idx in seq_indices:
                print(f"    {idx}: {test_sequences[idx]} (label={test_labels[idx]})")
        
        # Show cluster metadata
        if 'cluster_metadata' in result.metadata:
            print(f"\nCluster metadata:")
            for cluster_id, metadata in result.metadata['cluster_metadata'].items():
                motifs = metadata.get('motifs', [])
                size = metadata.get('size', 0)
                print(f"  {cluster_id}: {size} sequences, motifs={motifs}")
                
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Generic clustering method
    print(f"\n--- Test 2: Generic clustering method ---")
    try:
        config = KmerClusterConfig(
            k=3, 
            top_m=8, 
            min_support=2, 
            min_jaccard=0.4,
            use_generic_clustering=True,
            clustering_method='connected',
            similarity_threshold=0.4
        )
        clusterer = KmerClusterer(config)
        
        result = clusterer.cluster_sequences(test_sequences, test_labels)
        
        print(f"✅ Generic clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        
        print(f"\nCluster assignments:")
        for cluster_id, seq_indices in result.cluster_assignments.items():
            print(f"  {cluster_id}: sequences {seq_indices}")
            
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Simple clustering interface
    print(f"\n--- Test 3: Simple clustering interface ---")
    try:
        clusterer = create_kmer_clusterer(ks=4, topM=5, min_support=1, min_jaccard=0.2)
        
        simple_result = clusterer.cluster_sequences_simple(test_sequences, test_labels)
        
        print(f"✅ Simple clustering successful!")
        print(f"Cluster map: {simple_result}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Unsupervised clustering (no labels)
    print(f"\n--- Test 4: Unsupervised clustering (no labels) ---")
    try:
        clusterer = create_kmer_clusterer(ks=3, topM=10, min_support=2, min_jaccard=0.3, min_count=2)
        
        result = clusterer.cluster_sequences(test_sequences)  # No labels provided
        
        print(f"✅ Unsupervised clustering successful!")
        print(f"Total clusters: {result.total_clusters}")
        print(f"Total sequences: {result.total_sequences}")
        print(f"Algorithm: {result.algorithm}")
        
        print(f"\nCluster assignments:")
        for cluster_id, seq_indices in result.cluster_assignments.items():
            print(f"  {cluster_id}: sequences {seq_indices}")
            for idx in seq_indices:
                print(f"    {idx}: {test_sequences[idx]}")
        
        # Show cluster metadata
        if 'cluster_metadata' in result.metadata:
            print(f"\nCluster metadata:")
            for cluster_id, metadata in result.metadata['cluster_metadata'].items():
                motifs = metadata.get('motifs', [])
                size = metadata.get('size', 0)
                print(f"  {cluster_id}: {size} sequences, motifs={motifs}")
                
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Different conflict strategies
    print(f"\n--- Test 5: Different conflict strategies ---")
    strategies = ['max_support', 'first', 'random']
    
    for strategy in strategies:
        try:
            config = KmerClusterConfig(
                ks=3, 
                topM=8, 
                min_support=1, 
                min_jaccard=0.1,  # Low threshold to create conflicts
                conflict_strategy=strategy
            )
            clusterer = KmerClusterer(config)
            
            result = clusterer.cluster_sequences(test_sequences, test_labels)
            
            print(f"✅ Strategy '{strategy}': {result.total_clusters} clusters")
            
        except Exception as e:
            print(f"❌ Strategy '{strategy}' failed: {e}")
    
    # Test 6: Edge cases
    print(f"\n--- Test 6: Edge cases ---")
    
    # Empty sequences
    try:
        clusterer = create_kmer_clusterer()
        result = clusterer.cluster_sequences([], [])
        print(f"❌ Empty sequences should raise an error")
    except Exception as e:
        print(f"✅ Empty sequences correctly handled: {type(e).__name__}")
    
    # Test unsupervised with single sequence
    try:
        clusterer = create_kmer_clusterer(ks=2, topM=1, min_support=1, min_count=1)
        result = clusterer.cluster_sequences(['AAAA'])  # No labels
        print(f"✅ Single sequence (unsupervised): {result.total_clusters} clusters")
    except Exception as e:
        print(f"❌ Single sequence (unsupervised) failed: {e}")
    
    # Single sequence with labels
    try:
        clusterer = create_kmer_clusterer(ks=2, topM=1, min_support=1)
        result = clusterer.cluster_sequences(['AAAA'], [1])
        print(f"✅ Single sequence (supervised): {result.total_clusters} clusters")
    except Exception as e:
        print(f"❌ Single sequence (supervised) failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("KmerClusterer testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_kmer_clustering()


