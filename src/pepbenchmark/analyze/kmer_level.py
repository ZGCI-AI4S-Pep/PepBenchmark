"""Sequence-level k-mer analysis utilities for peptide datasets.

This module provides reusable data structures and analysis helpers to compute
occurrence, coverage, and sequence membership statistics for peptide k-mers.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import pandas as pd

from pepbenchmark.analyze.utils import filter_to_natural


@dataclass
class KmerStats:
    """Container for aggregated k-mer statistics.

    Attributes:
        k: Size of each k-mer.
        freq_per_seq_ratio: Mapping from k-mer to the fraction of eligible
            sequences containing that k-mer at least once.
        count_in_sequences: Mapping from k-mer to the number of eligible
            sequences containing that k-mer.
        total_occurrences: Mapping from k-mer to its total number of
            occurrences across all eligible sequences.
        kmer_to_sequences: Mapping from k-mer to sorted sequence indices where
            the k-mer appears.
        sequences_with_kmers: Sorted indices of sequences that contributed at
            least one valid k-mer.
        raw_total_kmers: Total number of extracted k-mer windows across all
            eligible sequences.
        raw_unique_kmers: Number of unique k-mers observed.
        n_sequences_total: Total number of input sequences.
        n_sequences_eligible: Number of sequences with length greater than or
            equal to ``k``.
    """

    k: int
    freq_per_seq_ratio: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    count_in_sequences: "OrderedDict[str, int]" = field(default_factory=OrderedDict)
    total_occurrences: "OrderedDict[str, int]" = field(default_factory=OrderedDict)
    kmer_to_sequences: Dict[str, List[int]] = field(default_factory=dict)
    sequences_with_kmers: List[int] = field(default_factory=list)
    raw_total_kmers: int = 0
    raw_unique_kmers: int = 0
    n_sequences_total: int = 0
    n_sequences_eligible: int = 0

    def _top_items_str(self, topn: int = 10) -> str:
        """Return a compact string representation of the most frequent k-mers.

        Args:
            topn: Maximum number of entries to include.

        Returns:
            A comma-separated string of ``kmer:count`` pairs.
        """
        if not self.total_occurrences:
            return ""
        top_items = list(self.total_occurrences.items())[:topn]
        return ", ".join(f"{kmer}:{count}" for kmer, count in top_items)

    def summary(self, topn: int = 10) -> str:
        """Create a human-readable summary of the analysis result.

        Args:
            topn: Number of top k-mers to display.

        Returns:
            A multi-line summary string.
        """
        lines = [
            (
                f"k={self.k}, unique_kmers={self.raw_unique_kmers}, "
                f"total_kmers={self.raw_total_kmers}"
            ),
            (
                f"n_sequences_total={self.n_sequences_total}, "
                f"n_sequences_eligible={self.n_sequences_eligible}"
            ),
        ]
        if self.total_occurrences:
            lines.append(f"Top kmers: {self._top_items_str(topn)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the default text representation."""
        return self.summary(topn=10)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the k-mer statistics into a tabular representation.

        Returns:
            A :class:`pandas.DataFrame` with one row per k-mer.
        """
        return pd.DataFrame(
            [
                {
                    "kmer": kmer,
                    "freq_ratio": self.freq_per_seq_ratio.get(kmer, 0.0),
                    "count": self.count_in_sequences.get(kmer, 0),
                    "occurrence": self.total_occurrences.get(kmer, 0),
                }
                for kmer in self.total_occurrences
            ]
        )


class KmerAnalyzer:
    """Compute peptide k-mer statistics for a sequence collection.

    Args:
        sequences: Iterable of peptide sequences.
        k: Length of the k-mers to analyze.
        drop_non_canonical: Whether to remove non-canonical amino acid symbols
            before analysis.
    """

    def __init__(
        self,
        sequences: Iterable[str],
        k: int,
        drop_non_canonical: bool = True,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        normalized_sequences = ["" if seq is None else str(seq) for seq in sequences]
        if drop_non_canonical:
            normalized_sequences = [
                filter_to_natural(sequence) for sequence in normalized_sequences
            ]

        self.k = k
        self.sequences = normalized_sequences

    def compute_stats(self) -> KmerStats:
        """Compute aggregate k-mer statistics across all sequences.

        Returns:
            A populated :class:`KmerStats` object.
        """
        sequence_list = self.sequences
        n_total = len(sequence_list)
        if n_total == 0:
            return KmerStats(k=self.k, n_sequences_total=0, n_sequences_eligible=0)

        sequence_contains: Counter[str] = Counter()
        total_occurrences: Counter[str] = Counter()
        kmer_to_sequences: Dict[str, set[int]] = defaultdict(set)
        raw_total_kmers = 0
        eligible_indices: List[int] = []

        for index, sequence in enumerate(sequence_list):
            sequence_length = len(sequence)
            if sequence_length < self.k:
                continue

            eligible_indices.append(index)
            n_windows = sequence_length - self.k + 1
            raw_total_kmers += n_windows

            kmers_in_sequence: set[str] = set()
            for start in range(n_windows):
                kmer = sequence[start : start + self.k]
                total_occurrences[kmer] += 1
                kmers_in_sequence.add(kmer)

            for kmer in kmers_in_sequence:
                sequence_contains[kmer] += 1
                kmer_to_sequences[kmer].add(index)

        raw_unique_kmers = len(total_occurrences)
        n_eligible = len(eligible_indices)
        if n_eligible == 0:
            return KmerStats(
                k=self.k,
                raw_total_kmers=0,
                raw_unique_kmers=0,
                n_sequences_total=n_total,
                n_sequences_eligible=0,
            )

        freq_per_seq_ratio = {
            kmer: sequence_contains[kmer] / n_eligible
            for kmer in total_occurrences
        }

        sorted_total_occurrences = OrderedDict(
            sorted(total_occurrences.items(), key=lambda item: item[1], reverse=True)
        )
        sorted_count_in_sequences = OrderedDict(
            sorted(sequence_contains.items(), key=lambda item: item[1], reverse=True)
        )
        sorted_freq_per_seq_ratio = OrderedDict(
            sorted(freq_per_seq_ratio.items(), key=lambda item: item[1], reverse=True)
        )

        kmer_to_sequences_index = {
            kmer: sorted(indices) for kmer, indices in kmer_to_sequences.items()
        }
        sequences_with_kmers = sorted(
            {index for indices in kmer_to_sequences_index.values() for index in indices}
        )

        return KmerStats(
            k=self.k,
            freq_per_seq_ratio=sorted_freq_per_seq_ratio,
            count_in_sequences=sorted_count_in_sequences,
            total_occurrences=sorted_total_occurrences,
            kmer_to_sequences=kmer_to_sequences_index,
            sequences_with_kmers=sequences_with_kmers,
            raw_total_kmers=raw_total_kmers,
            raw_unique_kmers=raw_unique_kmers,
            n_sequences_total=n_total,
            n_sequences_eligible=n_eligible,
        )

    def compute_metrics(self) -> KmerStats:
        """Backward-compatible alias for :meth:`compute_stats`."""
        return self.compute_stats()

    @staticmethod
    def top_kmers(
        stats: KmerStats,
        n: int = 10,
        by: str = "count",
    ) -> OrderedDict[str, float | int]:
        """Return the top ranked k-mers from a statistics object.

        Args:
            stats: A previously computed :class:`KmerStats` object.
            n: Number of entries to return.
            by: Ranking metric. Must be one of ``"count"``, ``"occurrence"``,
                or ``"freq"``.

        Returns:
            An ordered mapping of the selected top k-mers.

        Raises:
            ValueError: If *by* is not a supported ranking mode.
        """
        if by == "count":
            items = list(stats.count_in_sequences.items())[:n]
        elif by == "occurrence":
            items = list(stats.total_occurrences.items())[:n]
        elif by == "freq":
            items = list(stats.freq_per_seq_ratio.items())[:n]
        else:
            raise ValueError("by must be 'count', 'occurrence', or 'freq'")
        return OrderedDict(items)

    @staticmethod
    def count_kmers_in_sequences(
        kmers: List[str],
        sequences: List[str],
        drop_non_canonical: bool = True,
    ) -> Dict[str, int]:
        """Count how many sequences contain each target k-mer.

        Each sequence contributes at most once to the count of a given k-mer.

        Args:
            kmers: List of k-mers to query.
            sequences: List of peptide sequences to scan.
            drop_non_canonical: Whether to remove non-canonical amino acid
                symbols before scanning.

        Returns:
            A dictionary mapping each input k-mer to the number of sequences in
            which it appears.
        """
        if drop_non_canonical:
            normalized_sequences = [filter_to_natural(str(seq)) for seq in sequences]
        else:
            normalized_sequences = [str(seq) for seq in sequences]

        return {
            kmer: sum(1 for sequence in normalized_sequences if kmer in sequence)
            for kmer in kmers
        }


def get_kmer_stats(
    sequences: Iterable[str],
    k: int,
    drop_non_canonical: bool = True,
) -> KmerStats:
    """Compute k-mer statistics through the functional compatibility API.

    Args:
        sequences: Input peptide sequences.
        k: K-mer size.
        drop_non_canonical: Whether to remove non-canonical residues before
            analysis.

    Returns:
        A populated :class:`KmerStats` object.
    """
    return KmerAnalyzer(
        sequences=sequences,
        k=k,
        drop_non_canonical=drop_non_canonical,
    ).compute_stats()


KmerAnalyse = KmerAnalyzer


if __name__ == "__main__":
    example_sequences_1 = ["ACDEFGHIK", "LMNPQRST", "ACDACD"]
    example_sequences_2 = ["CDEFGHIK", "QRSTACD", "ACD"]

    analyzer_1 = KmerAnalyzer(example_sequences_1, k=3)
    analyzer_2 = KmerAnalyzer(example_sequences_2, k=3)

    stats_1 = analyzer_1.compute_stats()
    stats_2 = analyzer_2.compute_stats()

    print(stats_1)
    print(stats_2)
    print("\nTop-5 by count:", KmerAnalyzer.top_kmers(stats_1, n=5, by="count"))
    print(
        "Top-5 by occurrence:",
        KmerAnalyzer.top_kmers(stats_1, n=5, by="occurrence"),
    )
    print("Top-5 by freq:", KmerAnalyzer.top_kmers(stats_1, n=5, by="freq"))
