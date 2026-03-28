"""Primary analyzer for redundancy analysis.

This module provides the RedundancyAnalyzer class - the main entry point for
computing comprehensive redundancy metrics and generating quality reports.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

import numpy as np

from pepbenchmark.redundancy.schemas import (
    BasicStats, LengthStats, AAStats, KmerStats, SimilarityStats, 
    TopKStats, RedundancyStats, GraphStats, Recommendation, RedundancyReport
)

try:
    from pepbenchmark.similarity.schemas import SimilarityMetrics
    _SIM_METRICS_AVAILABLE = True
except ImportError:
    _SIM_METRICS_AVAILABLE = False
    SimilarityMetrics = None


# =============================================================================
# RedundancyAnalyzer - Main Class
# =============================================================================

class RedundancyAnalyzer:
    """
    Comprehensive redundancy analyzer for sequence datasets.
    
    Computes multiple redundancy metrics:
    - Basic: total sequences, uniqueness, duplicate rate
    - Length: distribution statistics
    - AA composition: entropy, Kullback-Leibler divergence
    - Similarity: pairwise statistics and graph connectivity
    - Redundancy: rates and effective counts at multiple thresholds
    
    Parameters:
        sequences: List of sequence strings
        similarity_matrix: N×N similarity/distance matrix (e.g., from Levenshtein, BLOSUM)
    """
    
    def __init__(self, sequences: List[str], similarity_matrix: np.ndarray):
        """
        Initialize analyzer.
        
        Args:
            sequences: List of sequence strings
            similarity_matrix: N×N matrix where element [i,j] is similarity between seq i and j.
                             Values typically in [0,1] or distance metrics.
        """
        if similarity_matrix.shape[0] != len(sequences):
            raise ValueError(
                f"similarity_matrix rows ({similarity_matrix.shape[0]}) "
                f"must match sequence count ({len(sequences)})"
            )
        self.sequences = sequences
        self.sim_matrix = np.asarray(similarity_matrix, dtype=float)

    def compute_metrics(
        self,
        thresholds: Tuple[float, ...] = (0.7, 0.8, 0.9),
        topk: int = 5,
        compute_kmer: bool = True,
        compute_graph: bool = True
    ) -> RedundancyReport:
        """
        Compute comprehensive redundancy report.
        
        Args:
            thresholds: Similarity thresholds for redundancy rate & Neff computation
            topk: Number of nearest neighbors to use for TopK statistics
            compute_kmer: Whether to compute k-mer diversity metrics
            compute_graph: Whether to compute graph connectivity metrics
            
        Returns:
            RedundancyReport with all computed metrics
        """
        report = RedundancyReport(sequences=self.sequences, sim_matrix=self.sim_matrix)
        
        # Basic statistics
        report.basic = self._compute_basic()
        
        # Length distribution
        report.length = self._compute_length()
        
        # Amino acid composition
        report.aa = self._compute_aa()
        
        # Similarity statistics
        report.similarity, sim_details = self._compute_similarity_stats()
        if sim_details:
            report.sim_details = sim_details
        
        # Redundancy at multiple thresholds
        report.redundancy = self._compute_redundancy(thresholds)
        
        # Top-K statistics
        report.topk = self._compute_topk(topk)
        
        # K-mer diversity (optional, can be expensive)
        if compute_kmer:
            report.kmer = self._compute_kmer()
        
        # Graph connectivity (optional)
        if compute_graph:
            report.graph = self._compute_graph(thresholds)
        
        # Recommendation
        report.recommendation = self._compute_recommendation(report)
        
        return report

    # =========================================================================
    # Metric Computation Methods
    # =========================================================================

    def _compute_basic(self) -> BasicStats:
        """Compute basic set statistics."""
        n = len(self.sequences)
        u = len(set(self.sequences))
        return BasicStats(
            n_total=n,
            n_unique=u,
            exact_duplicate_rate=(n - u) / n if n else 0.0
        )

    def _compute_length(self) -> LengthStats:
        """Compute sequence length distribution."""
        lengths = np.array([len(s) for s in self.sequences], dtype=float)
        if not lengths.size:
            return LengthStats()
        
        mean = float(lengths.mean())
        std = float(lengths.std())
        
        return LengthStats(
            mean_len=mean,
            std_len=std,
            min_len=int(lengths.min()),
            max_len=int(lengths.max()),
            len_range=int(lengths.max() - lengths.min()),
            len_cv=std / (mean + 1e-12) if mean > 0 else 0.0
        )

    def _compute_aa(self) -> AAStats:
        """Compute amino acid composition and entropy."""
        counter = Counter("".join(self.sequences))
        total = sum(counter.values())
        
        if total == 0:
            return AAStats()
        
        # Frequencies
        freqs = np.array(list(counter.values())) / total
        
        # Entropy
        entropy = -np.sum(freqs * np.log(freqs + 1e-12))
        
        # Most frequent AA
        most_freq_aa = max(counter, key=counter.get)
        most_freq_aa_freq = counter[most_freq_aa] / total
        
        # KL divergence from uniform
        uniform = np.ones_like(freqs) / len(freqs)
        kl = float(np.sum(freqs * np.log((freqs + 1e-12) / (uniform + 1e-12))))
        
        return AAStats(
            aa_entropy=float(entropy),
            most_freq_aa=most_freq_aa,
            most_freq_aa_freq=float(most_freq_aa_freq),
            aa_kl_to_uniform=kl,
            unique_residues=len(counter)
        )

    def _compute_similarity_stats(self) -> Tuple[SimilarityStats, Optional[Dict[str, Any]]]:
        """Compute pairwise similarity statistics."""
        n = len(self.sequences)
        # Upper triangle (excluding diagonal)
        vals = self.sim_matrix[np.triu_indices(n, k=1)]
        
        if not vals.size:
            return SimilarityStats(), None
        
        stats = SimilarityStats(
            pair_mean=float(vals.mean()),
            pair_median=float(np.median(vals)),
            pair_std=float(vals.std()),
            pair_min=float(vals.min()),
            pair_max=float(vals.max())
        )
        
        details = {'global_mean': float(vals.mean())} if _SIM_METRICS_AVAILABLE else None
        return stats, details

    def _compute_redundancy(self, thresholds: Tuple[float, ...]) -> RedundancyStats:
        """Compute redundancy rates and Neff at multiple thresholds."""
        n = len(self.sequences)
        vals = self.sim_matrix[np.triu_indices(n, k=1)]
        
        rates = {}
        neffs = {}
        
        for thr in thresholds:
            # Redundancy rate: fraction of pairs above threshold
            rates[thr] = float(np.sum(vals >= thr) / len(vals)) if vals.size else 0.0
            
            # Neff (effective sequence count): sum of inverse clustering sizes
            weights = []
            for i in range(n):
                # How many sequences are similar to sequence i?
                similar_count = int(np.sum(self.sim_matrix[i] >= thr))
                weights.append(1.0 / max(1, similar_count))
            neffs[thr] = float(np.sum(weights))
        
        return RedundancyStats(redundancy_rates=rates, neff=neffs)

    def _compute_topk(self, k: int = 5) -> TopKStats:
        """Compute Top-K nearest neighbor statistics."""
        n = len(self.sequences)
        topk_sims = []
        
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            neighbors = np.argsort(self.sim_matrix[i])[::-1][1:k+1]
            if neighbors.size > 0:
                topk_sims.extend(self.sim_matrix[i, neighbors])
        
        if not topk_sims:
            return TopKStats()
        
        topk_arr = np.array(topk_sims)
        return TopKStats(
            topk_mean_sim=float(topk_arr.mean()),
            topk_median=float(np.median(topk_arr)),
            topk_q75=float(np.percentile(topk_arr, 75))
        )

    def _compute_kmer(self) -> KmerStats:
        """
        Compute k-mer diversity metrics.
        
        This is a placeholder for now. Can be enhanced with actual k-mer statistics
        from pepbenchmark.analyze.kmer_level if needed.
        """
        return KmerStats()

    def _compute_graph(self, thresholds: Tuple[float, ...]) -> GraphStats:
        """Compute graph connectivity metrics at multiple thresholds."""
        graph_metrics = {}
        
        for thr in thresholds:
            # Build adjacency matrix at threshold
            A = (self.sim_matrix >= thr).astype(int)
            np.fill_diagonal(A, 0)  # Remove self-loops
            
            # Degree statistics
            degrees = A.sum(axis=1)
            graph_metrics[thr] = {
                'avg_degree': float(degrees.mean()),
                'max_degree': int(degrees.max()),
                'avg_neighbor_degree': float(degrees.mean()) if degrees.sum() > 0 else 0.0
            }
        
        return GraphStats(graph_metrics=graph_metrics)

    def _compute_recommendation(self, report: RedundancyReport) -> Recommendation:
        """
        Generate recommendation based on analysis results.
        
        Simple heuristic: if redundancy > 30% at 0.9, recommend 0.9 as threshold
        and flag as "high" severity.
        """
        if not report.redundancy.redundancy_rates:
            return Recommendation()
        
        # Find threshold with ~30-40% redundancy
        recommended_thr = 0.9
        for thr in sorted(report.redundancy.redundancy_rates.keys(), reverse=True):
            if report.redundancy.redundancy_rates[thr] <= 0.4:
                recommended_thr = thr
                break
        
        # Quality score based on duplicate rate
        duplicate_rate = report.basic.exact_duplicate_rate
        quality_score = max(0.0, 1.0 - duplicate_rate)
        
        # Severity based on redundancy
        max_redundancy = max(report.redundancy.redundancy_rates.values()) if report.redundancy.redundancy_rates else 0.0
        if max_redundancy > 0.5:
            severity = "high"
        elif max_redundancy > 0.3:
            severity = "medium"
        else:
            severity = "low"
        
        return Recommendation(
            recommended_threshold=recommended_thr,
            data_quality_score=quality_score,
            redundancy_severity=severity
        )


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

RedundancyAnalyse = RedundancyAnalyzer  # Legacy name support
