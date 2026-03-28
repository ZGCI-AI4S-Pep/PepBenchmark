"""Cluster-based deduplication for peptide sequence datasets.

This module provides the high-level deduplication API that sits on top of
``pepbenchmark.cluster``.  The cluster module is responsible **only** for
grouping sequences; this module decides *which* sequence to keep from each
cluster and exposes a clean, user-facing interface.

Design notes
------------
* **No circular imports** – cluster is imported lazily inside the functions
  so that ``redundancy`` can be imported without triggering the full cluster
  initialisation (which registers external tools like CD-HIT / MMseqs2).
* ``select_representative`` is a pure function: it works on any
  ``Dict[str, List[int]]`` cluster map without touching the cluster module.
* ``remove_redundancy`` is the primary entry point.  It accepts either a
  pre-computed ``UnifiedClusterResult`` (zero extra clustering cost) or runs
  clustering internally.
* ``deduplicate`` is a convenience alias that returns only the de-duplicated
  sequences (without the cluster result).

Example
-------
>>> from pepbenchmark.redundancy import remove_redundancy, deduplicate
>>>
>>> sequences = ["ACDEF", "ACDEF", "MVHLT", "MVHLS"]
>>>
>>> # One-step: cluster + deduplicate
>>> dedup, result = remove_redundancy(sequences, method="cdhit", c=0.9)
>>>
>>> # Two-step: reuse a pre-computed clustering result
>>> from pepbenchmark.cluster import cluster_sequences
>>> result = cluster_sequences(sequences, method="mmseqs2", identity=0.9)
>>> dedup, _ = remove_redundancy(sequences, cluster_result=result)
>>>
>>> # Convenience alias (sequences only)
>>> dedup = deduplicate(sequences, method="cdhit", c=0.9)
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type-checking; not executed at runtime to avoid
    # circular / heavyweight imports when the redundancy module is loaded.
    from pepbenchmark.cluster.interfaces import UnifiedClusterResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_representative(
    sequences: List[str],
    cluster_map: Dict[str, List[int]],
    strategy: str = "first",
) -> List[Tuple[str, int, str]]:
    """Select one representative sequence from each cluster.

    Parameters
    ----------
    sequences:
        Full sequence list (same order used during clustering).
    cluster_map:
        Mapping ``{cluster_id: [seq_index, ...]}``, as returned by
        ``UnifiedClusterResult.cluster_assignments``.
    strategy:
        How to pick the representative from each cluster.
        Choices: ``"first"`` (default), ``"longest"``, ``"shortest"``,
        ``"middle"``, ``"random"``.

    Returns
    -------
    List of ``(cluster_id, representative_index, representative_sequence)``
    tuples, one per cluster.
    """
    result: List[Tuple[str, int, str]] = []

    for cluster_id, indices in cluster_map.items():
        if not indices:
            continue

        if strategy == "first":
            rep_idx = indices[0]
        elif strategy == "longest":
            rep_idx = max(indices, key=lambda i: len(sequences[i]))
        elif strategy == "shortest":
            rep_idx = min(indices, key=lambda i: len(sequences[i]))
        elif strategy == "middle":
            rep_idx = indices[len(indices) // 2]
        elif strategy == "random":
            rep_idx = random.choice(indices)
        else:
            raise ValueError(
                f"Unknown representative-selection strategy '{strategy}'. "
                "Valid choices: 'first', 'longest', 'shortest', 'middle', 'random'."
            )

        result.append((cluster_id, rep_idx, sequences[rep_idx]))

    return result


def remove_redundancy(
    sequences: List[str],
    method: str = "cdhit",
    cluster_result: "Optional[UnifiedClusterResult]" = None,
    strategy: str = "first",
    **kwargs,
) -> Tuple[List[str], "UnifiedClusterResult"]:
    """Remove redundant sequences using cluster-based deduplication.

    Parameters
    ----------
    sequences:
        Input sequence list.
    method:
        Clustering algorithm to use when *cluster_result* is not provided.
        Any method registered with ``ClusterFactory`` is accepted
        (e.g. ``"cdhit"``, ``"mmseqs2"``, ``"kmer"``, ``"similarity"``).
    cluster_result:
        A pre-computed :class:`~pepbenchmark.cluster.interfaces.UnifiedClusterResult`.
        When provided, *method* and *kwargs* are ignored and no additional
        clustering is performed.
    strategy:
        Representative-selection strategy passed to :func:`select_representative`.
    **kwargs:
        Method-specific clustering parameters (e.g. ``c=0.9`` for CD-HIT,
        ``identity=0.9`` for MMseqs2).  Ignored when *cluster_result* is given.

    Returns
    -------
    (deduplicated_sequences, cluster_result)
        * ``deduplicated_sequences`` – one sequence per cluster (ordered by
          cluster insertion order).
        * ``cluster_result`` – the :class:`~pepbenchmark.cluster.interfaces.UnifiedClusterResult`
          that was used (either the one passed in or the newly computed one).

    Examples
    --------
    >>> # Method 1 – clustering + deduplication in one call
    >>> dedup, result = remove_redundancy(sequences, method="mmseqs2", identity=0.9)
    >>>
    >>> # Method 2 – supply a pre-computed result (efficient in loops)
    >>> from pepbenchmark.cluster import cluster_sequences
    >>> result = cluster_sequences(sequences, method="cdhit", c=0.9)
    >>> dedup, _ = remove_redundancy(sequences, cluster_result=result)
    >>>
    >>> # Method 3 – longest-sequence representative
    >>> dedup, result = remove_redundancy(sequences, method="cdhit",
    ...                                   strategy="longest", c=0.9)
    """
    if cluster_result is not None:
        if cluster_result.total_sequences != len(sequences):
            raise ValueError(
                f"cluster_result.total_sequences ({cluster_result.total_sequences}) "
                f"does not match len(sequences) ({len(sequences)})."
            )
        result = cluster_result
    else:
        # Lazy import to avoid heavyweight initialisation on module load.
        from pepbenchmark.cluster import cluster_sequences as _cluster_sequences
        result = _cluster_sequences(sequences, method=method, **kwargs)

    representatives = select_representative(sequences, result.cluster_assignments, strategy)
    dedup_sequences = [seq for _, _, seq in representatives]

    return dedup_sequences, result


def deduplicate(
    sequences: List[str],
    method: str = "cdhit",
    cluster_result: "Optional[UnifiedClusterResult]" = None,
    strategy: str = "first",
    **kwargs,
) -> List[str]:
    """Convenience wrapper around :func:`remove_redundancy`.

    Returns only the deduplicated sequence list (the cluster result is
    discarded).  Use :func:`remove_redundancy` if you need the cluster result
    for downstream analysis.

    Parameters
    ----------
    sequences, method, cluster_result, strategy, **kwargs:
        Forwarded to :func:`remove_redundancy`.  See that function for full
        documentation.

    Returns
    -------
    List[str]
        Deduplicated sequences.
    """
    dedup, _ = remove_redundancy(
        sequences,
        method=method,
        cluster_result=cluster_result,
        strategy=strategy,
        **kwargs,
    )
    return dedup
