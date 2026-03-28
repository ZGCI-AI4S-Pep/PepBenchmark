import numpy as np

from pepbenchmark.cluster import create_clusterer
from pepbenchmark.cluster.utils import load_fasta, save_fasta


def test_similarity_clusterer_groups_sequences_by_threshold():
    sequences = ["ACDE", "ACDF", "WXYZ"]
    similarity_matrix = np.array(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]], dtype=float
    )

    clusterer = create_clusterer("similarity", similarity_threshold=0.8)
    result = clusterer.cluster_sequences(sequences, similarity_matrix=similarity_matrix)

    assert result.total_clusters == 2
    assert result.cluster_assignments["0"] == [0, 1]
    assert result.cluster_assignments["1"] == [2]


def test_cluster_fasta_helpers_roundtrip(tmp_path):
    sequences = ["ACDE", "AAAA"]
    fasta_path = tmp_path / "demo.fasta"

    save_fasta(sequences, str(fasta_path))
    loaded = load_fasta(str(fasta_path))

    assert loaded == sequences
