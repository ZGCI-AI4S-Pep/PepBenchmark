from pepbenchmark.neg_sampler import (
    NegSampler,
    SamplingPoolManager,
    list_available_sampling_strategies,
)


def test_sampling_pool_manager_adds_and_filters_sequences():
    manager = SamplingPoolManager(include_sequences=["AAA", "BBBB", "CCCCC"])
    manager.filter_by_length(min_length=4)

    assert set(manager.get_sampling_pool()) == {"BBBB", "CCCCC"}


def test_random_negative_sampling_returns_expected_count():
    pool = SamplingPoolManager(include_sequences=["AAAAAA", "CCCCCC", "DDDDDD", "EEEEEE"])
    sampler = NegSampler(pool.get_sampling_pool(), ["ACDEFG", "ACDEFA"])

    negatives = sampler.sample_negatives(method="random", ratio=1.0, seed=42)

    assert len(negatives) == 2
    assert "random" in list_available_sampling_strategies()
    assert set(negatives).isdisjoint({"ACDEFG", "ACDEFA"})
