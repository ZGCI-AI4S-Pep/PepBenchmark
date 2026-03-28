import json

from pepbenchmark.model.utils import load_results_from_dir, summarize_results_grouped


def test_model_result_loaders_aggregate_seed_directories(tmp_path):
    base_dir = tmp_path / "results"
    run_dir = base_dir / "demo" / "rf" / "random_split"
    for seed, score in [(0, 0.80), (1, 0.90)]:
        seed_dir = run_dir / str(seed)
        seed_dir.mkdir(parents=True)
        payload = {
            "train": {"accuracy": score},
            "valid": {"accuracy": score - 0.1},
            "test": {"accuracy": score - 0.2},
        }
        (seed_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    df = load_results_from_dir(str(base_dir), "demo", "rf", split_type="random_split", expected_seeds=2)
    summary = summarize_results_grouped(
        str(base_dir),
        "demo",
        ["rf"],
        split_type="random_split",
        target_splits=["test"],
    )

    assert len(df) == 2
    assert list(summary["model"]) == ["rf"]
    assert summary.iloc[0]["test_accuracy_mean"] == "0.6500"
