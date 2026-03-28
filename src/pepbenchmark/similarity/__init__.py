
"""Public exports for sequence, fingerprint, embedding, and molecular similarity helpers."""

from pepbenchmark.similarity.analyse import SimilarityAnalyse, SimilarityAnalyzer
from pepbenchmark.similarity.schemas import (
    MetricsConfig,
    Pair,
    SimilarityMetrics,
)
from pepbenchmark.similarity.similarity import (
    InputType,
    MatrixMode,
    build_sparse_matrix_from_topk,
    compute_similarity_matrix,
    compute_topk,
)

from pepbenchmark.similarity.fasta import SimilarityMethod, fasta_similarity
from pepbenchmark.similarity.embedding import embedding_similarity
from pepbenchmark.similarity.molecular import (
    FingerprintType,
    MolecularSimilarityMethod,
    compute_maccs_similarity,
    compute_molecular_fingerprints,
    compute_molecular_similarity_matrix,
    compute_morgan_similarity,
    compute_rdkit_similarity,
    compute_tanimoto_similarity,
    find_similar_molecules,
    fingerprint_similarity,
)


__all__ = [
    "FingerprintType",
    "InputType",
    "MatrixMode",
    "MetricsConfig",
    "MolecularSimilarityMethod",
    "Pair",
    "SimilarityAnalyse",
    "SimilarityAnalyzer",
    "SimilarityMethod",
    "SimilarityMetrics",
    "build_sparse_matrix_from_topk",
    "compute_maccs_similarity",
    "compute_molecular_fingerprints",
    "compute_molecular_similarity_matrix",
    "compute_morgan_similarity",
    "compute_rdkit_similarity",
    "compute_similarity_matrix",
    "compute_tanimoto_similarity",
    "compute_topk",
    "embedding_similarity",
    "fasta_similarity",
    "find_similar_molecules",
    "fingerprint_similarity",
]
