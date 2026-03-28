import numpy as np
import pytest

from pepbenchmark.similarity import MetricsConfig, SimilarityAnalyzer, compute_similarity_matrix


def test_sequence_similarity_matrix_and_single_field_analysis():
    sequences = ["ACDE", "ACDF", "AAAA"]
    matrix = compute_similarity_matrix(
        sequences,
        input_type="sequence",
        method="levenshtein",
        show_progress=False,
        processes=1,
    )
    metrics = SimilarityAnalyzer({"custom": matrix}, topk=2).compute_metrics(
        config=MetricsConfig(threshold=0.5, k=2)
    )

    assert matrix.shape == (3, 3)
    assert matrix[0, 1] > matrix[0, 2]
    assert not metrics.to_dataframe("global_topk").empty


def test_embedding_similarity_matrix_uses_cosine_metric():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    matrix = compute_similarity_matrix(
        embeddings,
        input_type="embedding",
        method="cosine",
        show_progress=False,
    )

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.5)
