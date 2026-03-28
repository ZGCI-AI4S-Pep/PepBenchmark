from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def _format_table(rows: List[Tuple[object, ...]], headers: List[str]) -> str:
    """Format preview tables with ``tabulate`` when available."""
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="pretty")

    lines = ["\t".join(str(header) for header in headers)]
    lines.extend("\t".join(str(value) for value in row) for row in rows)
    return "\n".join(lines)


@dataclass(frozen=True)
class Pair:
    """Simple pair container for matrix-level similarity analysis."""

    i: int
    j: int
    sim: float

    def resolve(self, sequences: Sequence[str]) -> Tuple[Optional[str], Optional[str], float]:
        seq_i = sequences[self.i] if 0 <= self.i < len(sequences) else None
        seq_j = sequences[self.j] if 0 <= self.j < len(sequences) else None
        return seq_i, seq_j, self.sim


@dataclass
class SimilarityMetrics:
    """Structured output for matrix-level similarity summaries."""

    global_topk_pairs: List[Pair] = field(default_factory=list)
    global_topk_mean: Optional[float] = None
    per_sample_topk: Dict[int, List[Pair]] = field(default_factory=dict)
    per_sample_topk_mean: Dict[int, Optional[float]] = field(default_factory=dict)
    global_mean: Optional[float] = None
    per_sample_mean: Dict[int, Optional[float]] = field(default_factory=dict)
    global_threshold_pairs: List[Pair] = field(default_factory=list)
    per_sample_threshold_pairs: Dict[int, List[Pair]] = field(default_factory=dict)

    def summary(self, topn: int = 5) -> str:
        lines: List[str] = []
        lines.append(
            f"Global mean similarity: {self.global_mean:.4f}"
            if self.global_mean is not None
            else "Global mean similarity: None"
        )

        if self.global_topk_pairs:
            preview = [(pair.i, pair.j, pair.sim) for pair in self.global_topk_pairs[:topn]]
            lines.append(
                "Global Top-k (preview):\n"
                + _format_table(preview, headers=["i", "j", "sim"])
            )

        if self.global_threshold_pairs:
            preview = [(pair.i, pair.j, pair.sim) for pair in self.global_threshold_pairs[:topn]]
            lines.append(
                "Global Threshold> (preview):\n"
                + _format_table(preview, headers=["i", "j", "sim"])
            )

        if self.per_sample_mean:
            means_preview = list(self.per_sample_mean.items())[:topn]
            lines.append(
                "Per-sample mean (preview):\n"
                + _format_table(means_preview, headers=["sample", "mean"])
            )

        return "\n".join(lines)

    def to_dataframe(self, which: str = "global_topk") -> pd.DataFrame:
        if which == "global_topk":
            rows = [(pair.i, pair.j, pair.sim) for pair in self.global_topk_pairs]
            return pd.DataFrame(rows, columns=["i", "j", "similarity"])

        if which == "per_sample_topk":
            rows = [
                (sample_id, pair.i, pair.j, pair.sim)
                for sample_id, pairs in self.per_sample_topk.items()
                for pair in pairs
            ]
            return pd.DataFrame(rows, columns=["sample_id", "i", "j", "similarity"])

        if which == "global_threshold":
            rows = [(pair.i, pair.j, pair.sim) for pair in self.global_threshold_pairs]
            return pd.DataFrame(rows, columns=["i", "j", "similarity"])

        if which == "per_sample_threshold":
            rows = [
                (sample_id, pair.i, pair.j, pair.sim)
                for sample_id, pairs in self.per_sample_threshold_pairs.items()
                for pair in pairs
            ]
            return pd.DataFrame(rows, columns=["sample_id", "i", "j", "similarity"])

        raise ValueError(
            "Unknown which value. Expected one of: "
            "['global_topk', 'per_sample_topk', 'global_threshold', 'per_sample_threshold']"
        )

    def __str__(self) -> str:
        return self.summary(topn=10)


@dataclass
class MetricsConfig:
    """Config for selecting which matrix-level metrics to compute."""

    global_topk: bool = True
    per_sample_topk: bool = True
    global_mean: bool = True
    per_sample_mean: bool = True
    threshold: Optional[float] = None
    k: Optional[int] = None
