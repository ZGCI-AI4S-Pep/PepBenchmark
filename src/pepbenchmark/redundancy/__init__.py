"""Redundancy analysis and cluster-based deduplication module.

This module provides two complementary capabilities:

**Redundancy Analysis** – quantify and characterise sequence redundancy:
- Basic statistics: uniqueness, duplication rates
- Sequence properties: length distribution, amino acid composition
- Pairwise similarity analysis: correlation matrices and graph connectivity
- Redundancy quantification: rates and effective counts at multiple thresholds
- Quality recommendations: automated threshold and severity assessment

**Cluster-based Deduplication** – remove redundant sequences by leveraging any
clustering backend from ``pepbenchmark.cluster`` (CD-HIT, MMseqs2, k-mer, …):
- :func:`remove_redundancy` – full pipeline (cluster → pick representative)
- :func:`deduplicate`       – convenience alias returning sequences only
- :func:`select_representative` – pure function that works on any cluster map

Key Components
--------------
- :class:`RedundancyAnalyzer`  – main entry point for redundancy metrics
- :func:`remove_redundancy`    – cluster-based deduplication
- :func:`deduplicate`          – shorthand for deduplication
- :func:`select_representative` – representative selection from a cluster map
- Schemas                      – unified data containers for analysis results

Example – analysis
------------------
>>> from pepbenchmark.redundancy import RedundancyAnalyzer
>>> import numpy as np
>>> sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY", "MVHLTSQ"]
>>> sim_matrix = np.array([
...     [1.0, 0.95, 0.2],
...     [0.95, 1.0, 0.15],
...     [0.2, 0.15, 1.0]
... ])
>>> analyzer = RedundancyAnalyzer(sequences, sim_matrix)
>>> report = analyzer.compute_metrics(thresholds=(0.7, 0.8, 0.9))
>>> print(f"Uniqueness: {report.basic.n_unique}/{report.basic.n_total}")
>>> print(f"Recommended threshold: {report.recommendation.recommended_threshold}")

Example – deduplication
-----------------------
>>> from pepbenchmark.redundancy import remove_redundancy, deduplicate
>>> sequences = ["ACDEF", "ACDEF", "MVHLT", "MVHLS"]
>>> # one-step (cluster + pick representative)
>>> dedup, result = remove_redundancy(sequences, method="cdhit", c=0.9)
>>> # or – convenience alias
>>> dedup = deduplicate(sequences, method="mmseqs2", identity=0.9)
"""

# ---------------------------------------------------------------------------
# Redundancy analysis
# ---------------------------------------------------------------------------
from pepbenchmark.redundancy.analysis import RedundancyAnalyzer, RedundancyAnalyse

# Schemas (data containers)
from pepbenchmark.redundancy.schemas import (
    BasicStats,
    LengthStats,
    AAStats,
    KmerStats,
    SimilarityStats,
    TopKStats,
    RedundancyStats,
    GraphStats,
    Recommendation,
    RedundancyReport,
)

# ---------------------------------------------------------------------------
# Cluster-based deduplication
# ---------------------------------------------------------------------------
from pepbenchmark.redundancy.deduplicator import (
    select_representative,
    remove_redundancy,
    deduplicate,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
from pepbenchmark.redundancy.utils import calculate_reduction_stats

__all__ = [
    # ── Redundancy analysis ──────────────────────────────────────────────
    "RedundancyAnalyzer",
    "RedundancyAnalyse",       # backward-compat alias

    # ── Schemas ──────────────────────────────────────────────────────────
    "BasicStats",
    "LengthStats",
    "AAStats",
    "KmerStats",
    "SimilarityStats",
    "TopKStats",
    "RedundancyStats",
    "GraphStats",
    "Recommendation",
    "RedundancyReport",

    # ── Deduplication ────────────────────────────────────────────────────
    "remove_redundancy",
    "deduplicate",
    "select_representative",

    # ── Utilities ────────────────────────────────────────────────────────
    "calculate_reduction_stats",
]
