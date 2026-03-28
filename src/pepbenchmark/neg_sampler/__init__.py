"""Public API for the negative-sampling toolkit.

The package intentionally uses lazy imports so lightweight metadata helpers
remain importable even when optional analysis dependencies are unavailable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

from pepbenchmark.neg_sampler.neg_meta import (
    EXCLUSIVE_MAP,
    INCLUSIVE_MAP,
    NEG_POOL_MAP,
    get_dataset_quantity_stats,
    get_hierarchical_stats,
    read_dataset_sequences,
)


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DEFAULT_PROPERTIES": ("pepbenchmark.neg_sampler.neg_sample", "DEFAULT_PROPERTIES"),
    "NegSampler": ("pepbenchmark.neg_sampler.neg_sample", "NegSampler"),
    "NegativeSampler": ("pepbenchmark.neg_sampler.neg_sample", "NegSampler"),
    "StrategySelector": ("pepbenchmark.neg_sampler.neg_sample", "StrategySelector"),
    "choose_best_strategy_with_run": (
        "pepbenchmark.neg_sampler.neg_sample",
        "choose_best_strategy_with_run",
    ),
    "SamplingPoolManager": (
        "pepbenchmark.neg_sampler.sampling_pool_manager",
        "SamplingPoolManager",
    ),
    "SAMPLING_STRATEGY_NAMES": (
        "pepbenchmark.neg_sampler.sampling_strategies",
        "SAMPLING_STRATEGY_NAMES",
    ),
    "BaseSampler": ("pepbenchmark.neg_sampler.sampling_strategies", "BaseSampler"),
    "SamplerRegistry": (
        "pepbenchmark.neg_sampler.sampling_strategies",
        "SamplerRegistry",
    ),
    "SamplingContext": (
        "pepbenchmark.neg_sampler.sampling_strategies",
        "SamplingContext",
    ),
    "DistributionValidator": (
        "pepbenchmark.neg_sampler.distribution_validator",
        "DistributionValidator",
    ),
    "calculate_ks_critical_value": (
        "pepbenchmark.neg_sampler.distribution_validator",
        "calculate_ks_critical_value",
    ),
    "compare_properties_distribution": (
        "pepbenchmark.neg_sampler.distribution_validator",
        "compare_properties_distribution",
    ),
    "interpret_ks_test_result": (
        "pepbenchmark.neg_sampler.distribution_validator",
        "interpret_ks_test_result",
    ),
}


def __getattr__(name: str):
    """Resolve lazily exposed public symbols on demand."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def list_available_sampling_strategies() -> list[str]:
    """Return all registered sampling strategy names."""
    return __getattr__("SamplerRegistry")().list_available()


__all__ = [
    "DEFAULT_PROPERTIES",
    "EXCLUSIVE_MAP",
    "INCLUSIVE_MAP",
    "NEG_POOL_MAP",
    "BaseSampler",
    "DistributionValidator",
    "NegSampler",
    "NegativeSampler",
    "SamplerRegistry",
    "SamplingContext",
    "SamplingPoolManager",
    "SAMPLING_STRATEGY_NAMES",
    "StrategySelector",
    "calculate_ks_critical_value",
    "choose_best_strategy_with_run",
    "compare_properties_distribution",
    "get_dataset_quantity_stats",
    "get_hierarchical_stats",
    "interpret_ks_test_result",
    "list_available_sampling_strategies",
    "read_dataset_sequences",
]
