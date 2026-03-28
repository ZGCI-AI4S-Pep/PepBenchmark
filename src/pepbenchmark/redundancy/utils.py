"""Utility helpers for the redundancy module.

Intentionally kept small: heavy metric logic lives in ``metrics.py`` and
deduplication logic lives in ``deduplicator.py``.
"""
from __future__ import annotations

from typing import Dict


def calculate_reduction_stats(original_count: int, final_count: int) -> Dict[str, float]:
    """Return a summary of how much redundancy was removed.

    Parameters
    ----------
    original_count:
        Number of sequences before deduplication.
    final_count:
        Number of sequences after deduplication (i.e. number of clusters /
        representatives kept).

    Returns
    -------
    dict with keys:
        * ``original_count``   – input count
        * ``final_count``      – output count
        * ``removed_count``    – sequences removed
        * ``reduction_ratio``  – final / original  (fraction kept)
        * ``redundancy_removed`` – (original − final) / original
        * ``compression_ratio`` – original / final  (>1 means data was compressed)
    """
    if original_count == 0:
        return {
            "original_count": 0,
            "final_count": 0,
            "removed_count": 0,
            "reduction_ratio": 1.0,
            "redundancy_removed": 0.0,
            "compression_ratio": 1.0,
        }

    removed = original_count - final_count
    return {
        "original_count": original_count,
        "final_count": final_count,
        "removed_count": removed,
        "reduction_ratio": final_count / original_count,
        "redundancy_removed": removed / original_count,
        "compression_ratio": original_count / final_count if final_count > 0 else float("inf"),
    }
