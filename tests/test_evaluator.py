import pytest

from pepbenchmark.evaluator import (
    evaluate_classification,
    evaluate_regression,
    g_mean_score,
    specificity_score,
)


def test_binary_classification_metrics_cover_expected_keys():
    metrics = evaluate_classification(
        [0, 1, 1],
        [0, 1, 0],
        metrics=["accuracy", "f1"],
    )

    assert metrics == {"accuracy": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3)}
    assert specificity_score([0, 1, 1], [0, 1, 0]) == pytest.approx(1.0)
    assert g_mean_score([0, 1, 1], [0, 1, 0]) > 0


def test_regression_metrics_cover_expected_keys():
    metrics = evaluate_regression([1.0, 2.0], [1.5, 1.5], metrics=["mae", "rmse"])

    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.5)
