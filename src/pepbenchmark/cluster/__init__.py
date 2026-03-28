# Import unified interfaces and utilities
from pepbenchmark.cluster.interfaces import (
    AbstractClusterer,
    ClusterConfig,
    UnifiedClusterResult
)

from pepbenchmark.cluster.factory import (
    ClusterFactory,
    create_clusterer,
    create_cdhit_clusterer,
    create_mmseqs2_clusterer,
    create_motif_clusterer,
    create_kmer_clusterer,
    create_similarity_clusterer,
    list_available_methods
)

# Import clean implementations
from pepbenchmark.cluster.cdhit_cluster import CDHitClusterer, CDHitConfig
from pepbenchmark.cluster.mmseqs_cluster import MMseqs2Clusterer, MMseqs2Config

# Register clustering algorithms
ClusterFactory.register_clusterer("cdhit", CDHitClusterer, CDHitConfig)
ClusterFactory.register_clusterer("mmseqs2", MMseqs2Clusterer, MMseqs2Config)

# Try to register other algorithms if available
from pepbenchmark.cluster.kmer_cluster import (
    KmerClusterer,
    KmerClusterConfig,
    MotifClusterer,
    MotifClusterConfig,
)
ClusterFactory.register_clusterer("motif", MotifClusterer, MotifClusterConfig)
ClusterFactory.register_clusterer("kmer", KmerClusterer, KmerClusterConfig)

from pepbenchmark.cluster.smilarity_cluster import SimilarityClusterer, SimilarityClusterConfig
ClusterFactory.register_clusterer("similarity", SimilarityClusterer, SimilarityClusterConfig)

from pepbenchmark.cluster.molecular_cluster import MolecularClusterer, MolecularClusterConfig, create_molecular_clusterer
ClusterFactory.register_clusterer("molecular", MolecularClusterer, MolecularClusterConfig)

# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------
__all__ = [
    # Core interfaces
    'AbstractClusterer',
    'ClusterConfig',
    'UnifiedClusterResult',

    # Factory functions
    'ClusterFactory',
    'create_clusterer',
    'create_cdhit_clusterer',
    'create_mmseqs2_clusterer',
    'create_motif_clusterer',
    'create_kmer_clusterer',
    'create_similarity_clusterer',
    'create_molecular_clusterer',
    'list_available_methods',

    # Specific implementations
    'CDHitClusterer',
    'CDHitConfig',
    'MMseqs2Clusterer',
    'MMseqs2Config',
    'MotifClusterer',
    'MotifClusterConfig',
    'KmerClusterer',
    'KmerClusterConfig',

    # Primary clustering entry point
    'cluster_sequences',
]


def cluster_sequences(
    sequences,
    method="cdhit",
    **kwargs
):
    """
    Main clustering function with unified interface.

    This is the primary entry point for **clustering** sequences with any
    supported algorithm.  For deduplication (picking one representative per
    cluster), see :func:`pepbenchmark.redundancy.remove_redundancy`.

    Args:
        sequences: List of sequences to cluster
        method: Clustering method ("cdhit", "mmseqs2", "motif", "kmer", "similarity")
        **kwargs: Method-specific parameters

    Returns:
        UnifiedClusterResult containing clustering information

    Examples:
        # Basic CD-HIT clustering
        result = cluster_sequences(sequences, method="cdhit", c=0.9)

        # MMseqs2 clustering
        result = cluster_sequences(sequences, method="mmseqs2", identity=0.8)

        # Kmer-based clustering
        result = cluster_sequences(sequences, method="kmer", ks=5)

        # Inspect cluster assignments
        clusters = result.cluster_assignments
        print(f"Found {result.total_clusters} clusters")
    """
    clusterer = create_clusterer(method, **kwargs)
    return clusterer.cluster_sequences(sequences, **kwargs)


