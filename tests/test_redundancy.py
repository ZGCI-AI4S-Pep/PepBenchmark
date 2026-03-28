import numpy as np
import pytest

from pepbenchmark.redundancy import RedundancyAnalyzer


def test_redundancy_analyzer_reports_duplicate_structure():
    sequences = ["ACDE", "ACDE", "AAAA"]
    similarity_matrix = np.array(
        [[1.0, 1.0, 0.25], [1.0, 1.0, 0.25], [0.25, 0.25, 1.0]], dtype=float
    )

    report = RedundancyAnalyzer(sequences, similarity_matrix).compute_metrics(
        thresholds=(0.8,),
        compute_kmer=False,
        compute_graph=False,
    )

    assert report.basic.n_total == 3
    assert report.basic.n_unique == 2
    assert report.redundancy.redundancy_rates[0.8] == pytest.approx(1 / 3)
    assert report.redundancy.neff[0.8] == pytest.approx(2.0)
