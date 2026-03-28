"""Unified schemas for redundancy analysis results.

This module defines all data containers used for redundancy analysis,
grouped by analysis type: basic statistics, similarity metrics, redundancy measures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# =============================================================================
# Individual Metric Groups (Statistics)
# =============================================================================

@dataclass
class BasicStats:
    """Basic sequence set statistics: count, uniqueness, duplication rate."""
    n_total: int = 0
    n_unique: int = 0
    exact_duplicate_rate: float = 0.0


@dataclass
class LengthStats:
    """Length distribution statistics across sequences."""
    mean_len: float = 0.0
    std_len: float = 0.0
    min_len: int = 0
    max_len: int = 0
    len_range: int = 0
    len_cv: float = 0.0


@dataclass
class AAStats:
    """Amino acid composition and entropy statistics."""
    aa_entropy: float = 0.0
    most_freq_aa: str = ""
    most_freq_aa_freq: float = 0.0
    aa_kl_to_uniform: float = 0.0
    unique_residues: int = 0


@dataclass
class KmerStats:
    """K-mer diversity and coverage metrics."""
    kmer_entropy: Dict[int, float] = field(default_factory=dict)
    kmer_gini: Dict[int, float] = field(default_factory=dict)
    kmer_coverage_max: Dict[int, float] = field(default_factory=dict)
    kmer_distinct: Dict[int, int] = field(default_factory=dict)


@dataclass
class SimilarityStats:
    """Pairwise similarity statistics (mean, median, std, range)."""
    pair_mean: float = 0.0
    pair_median: float = 0.0
    pair_std: float = 0.0
    pair_min: float = 0.0
    pair_max: float = 0.0


@dataclass
class TopKStats:
    """Top-K nearest neighbor similarity statistics."""
    topk_mean_sim: float = 0.0
    topk_median: float = 0.0
    topk_q75: float = 0.0


@dataclass
class RedundancyStats:
    """Redundancy rates and effective sequence counts by threshold."""
    redundancy_rates: Dict[float, float] = field(default_factory=dict)
    neff: Dict[float, float] = field(default_factory=dict)


@dataclass
class GraphStats:
    """Graph connectivity statistics at different similarity thresholds."""
    graph_metrics: Dict[float, Dict[str, float]] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Recommendations for dataset quality and threshold selection."""
    recommended_threshold: float = 0.9
    data_quality_score: float = 0.0
    redundancy_severity: str = "unknown"


# =============================================================================
# Unified Report Container
# =============================================================================

@dataclass
class RedundancyReport:
    """
    Complete redundancy analysis report.
    
    Aggregates all computed metrics: basic, length, aa, kmer, similarity,
    redundancy, and graph statistics. Serves as primary output of RedundancyAnalyzer.
    """
    basic: BasicStats = field(default_factory=BasicStats)
    length: LengthStats = field(default_factory=LengthStats)
    aa: AAStats = field(default_factory=AAStats)
    kmer: KmerStats = field(default_factory=KmerStats)
    similarity: SimilarityStats = field(default_factory=SimilarityStats)
    topk: TopKStats = field(default_factory=TopKStats)
    redundancy: RedundancyStats = field(default_factory=RedundancyStats)
    graph: GraphStats = field(default_factory=GraphStats)
    recommendation: Recommendation = field(default_factory=Recommendation)
    
    # Metadata
    sequences: List[str] = field(default_factory=list)
    sim_matrix: Optional[np.ndarray] = None
    sim_details: Optional[Dict[str, Any]] = None
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary (excludes large arrays and sequences)."""
        return {
            'basic': self.basic.__dict__,
            'length': self.length.__dict__,
            'aa': self.aa.__dict__,
            'kmer': self.kmer.__dict__,
            'similarity': self.similarity.__dict__,
            'topk': self.topk.__dict__,
            'redundancy': self.redundancy.__dict__,
            'graph': self.graph.__dict__,
            'recommendation': self.recommendation.__dict__,
            'raw_metrics': self.raw_metrics,
        }
