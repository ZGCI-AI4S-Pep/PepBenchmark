from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.splitter import (
    RandomSplitter,
    UnifiedResultSplitter,
    analyze_split_class_distribution,
)


def test_random_splitter_returns_complete_partition():
    split = RandomSplitter().get_split_indices(["A", "B", "C", "D", "E"], seed=0)

    assert sum(len(indices) for indices in split.values()) == 5
    assert set(split.keys()) == {"train", "valid", "test"}


def test_unified_result_splitter_preserves_cluster_membership():
    cluster_result = UnifiedClusterResult(
        cluster_assignments={"0": [0, 1], "1": [2], "2": [3, 4]},
        total_clusters=3,
        total_sequences=5,
        algorithm="demo",
        parameters={},
    )

    split = UnifiedResultSplitter(random_seed=0).split_from_cluster_result(
        cluster_result,
        frac_train=0.6,
        frac_valid=0.2,
        frac_test=0.2,
    )

    split_members = [set(indices) for indices in split.values()]
    assert {0, 1} in split_members


def test_split_class_distribution_returns_summary_frame():
    result = analyze_split_class_distribution(
        {"train": [0, 1], "test": [2]},
        labels=[1, 0, 1],
    )

    assert set(result["split"]) == {"train", "test"}
