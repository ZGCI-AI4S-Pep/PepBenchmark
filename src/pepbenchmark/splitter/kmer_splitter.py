#!/usr/bin/env python3
"""
K-mer based sequence splitter for similarity-aware splitting.

This module provides the KmerSplitter class that uses k-mer frequency analysis
to ensure that sequences with similar k-mer patterns are placed in the same split,
preventing data leakage in evaluation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from pepbenchmark.splitter.base_splitter import AbstractClusteringSplitter
from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class KmerSplitter(AbstractClusteringSplitter):
    """
    K-mer based sequence splitter for similarity-aware splitting.

    This splitter uses k-mer frequency analysis to cluster sequences
    and ensure that similar sequences are placed in the same split.
    
    The splitter workflow:
    1. Extract k-mers from all sequences
    2. Create k-mer frequency vectors for each sequence
    3. Calculate similarity matrix using cosine similarity
    4. Perform hierarchical clustering based on similarity
    5. Distribute clusters across train/valid/test splits
    """

    def __init__(
        self, 
        k: int = 3,
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 2,
        random_seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize K-mer splitter.
        
        Args:
            k: K-mer size (default: 3)
            similarity_threshold: Similarity threshold for clustering (default: 0.8)
            min_cluster_size: Minimum cluster size (default: 2)
            random_seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters
        """
        super().__init__(random_seed=random_seed)
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        
        logger.info(f"KmerSplitter initialized with k={k}, similarity_threshold={similarity_threshold}")
        
    def _extract_kmers(self, sequence: str) -> List[str]:
        """Extract k-mers from a sequence."""
        if len(sequence) < self.k:
            return [sequence]  # Return the sequence itself if shorter than k
        
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmers.append(sequence[i:i + self.k])
        return kmers
    
    def _create_kmer_vectors(self, sequences: List[str]) -> np.ndarray:
        """
        Create k-mer frequency vectors for sequences.
        
        Args:
            sequences: List of sequences
            
        Returns:
            2D numpy array where each row is a k-mer frequency vector
        """
        # Extract all k-mers from all sequences
        all_kmers = set()
        sequence_kmers = []
        
        for seq in sequences:
            kmers = self._extract_kmers(seq)
            sequence_kmers.append(kmers)
            all_kmers.update(kmers)
        
        # Create k-mer vocabulary
        kmer_vocab = sorted(list(all_kmers))
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmer_vocab)}
        
        # Create frequency vectors
        vectors = np.zeros((len(sequences), len(kmer_vocab)))
        
        for seq_idx, kmers in enumerate(sequence_kmers):
            kmer_counts = Counter(kmers)
            for kmer, count in kmer_counts.items():
                kmer_idx = kmer_to_idx[kmer]
                vectors[seq_idx, kmer_idx] = count
        
        # Normalize vectors
        row_sums = vectors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        vectors = vectors / row_sums
        
        logger.info(f"Created k-mer vectors: {vectors.shape[0]} sequences, {vectors.shape[1]} unique {self.k}-mers")
        
        return vectors
    
    def _cluster_sequences(self, vectors: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Cluster sequences based on k-mer similarity.
        
        Args:
            vectors: K-mer frequency vectors
            
        Returns:
            Tuple of (cluster_labels, n_clusters)
        """
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.similarity_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        n_clusters = len(np.unique(cluster_labels))
        
        logger.info(f"K-mer clustering completed: {n_clusters} clusters")
        
        # Log cluster size distribution
        cluster_sizes = Counter(cluster_labels)
        size_dist = Counter(cluster_sizes.values())
        logger.info(f"Cluster size distribution: {dict(size_dist)}")
        
        return cluster_labels, n_clusters
    
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Get clustering result using k-mer analysis.
        
        Args:
            sequences: List of sequences to cluster
            labels: List of labels (optional, not used in clustering)
            **kwargs: Additional clustering parameters
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        logger.info(f"Performing k-mer clustering on {len(sequences)} sequences")
        
        # Create k-mer vectors
        vectors = self._create_kmer_vectors(sequences)
        
        # Cluster sequences
        cluster_labels, n_clusters = self._cluster_sequences(vectors)
        
        # Create cluster assignments
        cluster_assignments = {}
        for seq_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in cluster_assignments:
                cluster_assignments[cluster_id] = []
            cluster_assignments[cluster_id].append(seq_idx)
        
        # Create metadata
        metadata = {
            'k': self.k,
            'similarity_threshold': self.similarity_threshold,
            'min_cluster_size': self.min_cluster_size,
            'n_unique_kmers': vectors.shape[1],
            'clustering_method': 'hierarchical_cosine'
        }
        
        # Create UnifiedClusterResult
        result = UnifiedClusterResult(
            cluster_assignments=cluster_assignments,
            total_clusters=n_clusters,
            total_sequences=len(sequences),
            algorithm="kmer_clustering",
            parameters={
                'k': self.k,
                'similarity_threshold': self.similarity_threshold,
                'min_cluster_size': self.min_cluster_size,
                'random_seed': self.random_seed
            },
            metadata=metadata
        )
        
        logger.info(f"K-mer clustering completed: {n_clusters} clusters")
        
        return result


def get_split_indices(
    sequences: List[str],
    labels: Optional[List[int]] = None,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: Optional[int] = 42,
    k: int = 3,
    similarity_threshold: float = 0.8,
    min_cluster_size: int = 2,
    **kwargs: Any,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """
    Convenience function for k-mer clustering-based splitting.
    
    Args:
        sequences: List of sequences
        labels: List of labels (optional)
        frac_train: Fraction for training set (default: 0.8)
        frac_valid: Fraction for validation set (default: 0.1)
        frac_test: Fraction for test set (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
        k: K-mer size (default: 3)
        similarity_threshold: Similarity threshold for clustering (default: 0.8)
        min_cluster_size: Minimum cluster size (default: 2)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing train/valid/test split indices
        
    Example:
        >>> sequences = ["ACDEFG", "GHIKLM", "NOPQRST", ...]
        >>> splits = get_split_indices(sequences, k=3, similarity_threshold=0.8)
        >>> train_seqs = [sequences[i] for i in splits['train']]
    """
    # Prepare data
    if labels is not None:
        data = pd.DataFrame({'sequence': sequences, 'label': labels})
    else:
        data = sequences
    
    # Create splitter
    splitter = KmerSplitter(
        k=k,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        random_seed=seed,
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


def test_kmer_splitter():
    """Test k-mer splitter functionality."""
    print("=" * 60)
    print("Testing KmerSplitter")
    print("=" * 60)
    
    # Test data with similar k-mer patterns
    test_sequences = [
        'ARRGPGPG',   # Contains ARR, RRG, GPG
        'KKKLGFFF',   # Contains KKK, KLG, GFF
        'ARRRRGPG',   # Similar to seq 0 (ARR, RRR, RGP)
        'KKLGFFFF',   # Similar to seq 1 (KKL, KLG, GFF)
        'WWWLLLGG',   # Distinct k-mers
        'YYYQQQAA',   # Distinct k-mers
        'TTTDDDSS',   # Distinct k-mers
        'HHHEEECC',   # Distinct k-mers
    ]
    test_labels = [0, 1, 0, 1, 0, 1, 0, 1]
    
    print(f"Test data: {len(test_sequences)} sequences")
    for i, (seq, label) in enumerate(zip(test_sequences, test_labels)):
        print(f"  {i}: {seq} (label={label})")
    
    # Test basic k-mer splitting
    print(f"\n--- Test: Basic k-mer splitting ---")
    try:
        splitter = KmerSplitter(k=3, similarity_threshold=0.7)
        
        data = pd.DataFrame({'sequence': test_sequences, 'label': test_labels})
        result = splitter.get_split_indices(data)
        
        print(f"✅ K-mer splitting successful!")
        print(f"Split sizes: train={len(result['train'])}, valid={len(result['valid'])}, test={len(result['test'])}")
        
        # Show cluster information
        cluster_info = splitter.get_cluster_info()
        if cluster_info:
            print(f"\nCluster info: {cluster_info.get('total_clusters', 'N/A')} clusters")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test convenience function
    print(f"\n--- Test: Convenience function ---")
    try:
        result = get_split_indices(
            sequences=test_sequences,
            labels=test_labels,
            k=3,
            similarity_threshold=0.8
        )
        
        print(f"✅ Convenience function successful!")
        print(f"Split sizes: train={len(result['train'])}, valid={len(result['valid'])}, test={len(result['test'])}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("KmerSplitter testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_kmer_splitter()
