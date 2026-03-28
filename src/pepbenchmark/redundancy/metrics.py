# -*- coding: utf-8 -*-
"""Core metrics computation for redundancy analysis (with scalable options).

This module contains the fundamental metric calculation classes:
- BasicMetrics: Set statistics, length distribution, amino acid composition
- KmerMetrics: k-mer based diversity and redundancy metrics
- SimilarityMetrics: Pairwise similarity statistics and graph connectivity
- ClusterMetrics: Clustering structure and quality evaluation

Enhancements:
- MetricOptions to control heavy computations & sampling on large inputs.
- Functions accept `options: MetricOptions | None`, auto-scaling when N large.

License: Apache-2.0
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from loguru import logger

from pepbenchmark.analyze.kmer_level import get_kmer_stats


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------
@dataclass
class MetricOptions:
    # Pairwise stats
    pairwise_full_max_n: int = 2000            # Skip full upper-triangular scans when n exceeds this value.
    pairwise_sample_pairs: int = 200_000       # Number of sampled upper-triangle pairs (skip pairwise stats if 0).
    # Graph
    graph_full_max_n: int = 2000               # Sample nodes for connectivity evaluation when n exceeds this value.
    graph_sample_nodes: int = 1200             # Number of sampled nodes for graph metrics.
    # Top-k
    topk_row_sample: Optional[int] = None      # If set and n exceeds this value, compute top-k stats on sampled rows only.
    # Adaptive / Advanced
    adaptive_full_max_n: int = 1200            # adaptive_topk switches to sampled/lightweight mode above this value.
    advanced_full_max_n: int = 1000            # advanced_similarity switches to sampled mode above this value.
    # Cluster internal indices
    silhouette_sample_points: int = 2000       # Maximum total sample size for silhouette.
    dunn_sample_points: int = 2000             # Total sampling budget for the Dunn index.
    dbi_sample_points: int = 2000              # Maximum total sample size for DBI.
    # Global
    heavy_enabled: bool = True                 # Skip expensive metrics when False (silhouette/dunn/dbi/large graphs).
    seed: Optional[int] = 42                   # Fixed random seed; use None for non-deterministic behavior.


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def _ensure_np(a):
    return a if isinstance(a, np.ndarray) else np.asarray(a)


def _upper_vals(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _gini(v: np.ndarray) -> float:
    x = np.sort(v.flatten())
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    csum = np.cumsum(x)
    return float((n + 1 - 2 * (csum / csum[-1]).sum()) / n)


def _neff(sim: np.ndarray, thr: float) -> float:
    N = sim.shape[0]
    w = []
    for i in range(N):
        cnt = int((sim[i] >= thr).sum())
        w.append(1.0 / max(1, cnt))
    return float(np.sum(w))


# ---------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------
class BasicMetrics:
    @staticmethod
    def set_metrics(seqs: List[str]) -> Dict[str, float]:
        n = len(seqs)
        u = len(set(seqs))
        return dict(n_total=n, n_unique=u, exact_duplicate=n - u,
                    exact_duplicate_rate=(n - u) / n if n else 0.0)

    @staticmethod
    def length_metrics(seqs: List[str]) -> Dict[str, float]:
        L = np.array([len(s) for s in seqs], dtype=float)
        if not L.size:
            return {k: 0.0 for k in ("mean_len", "std_len", "min_len", "max_len", "len_range", "len_cv")}
        return dict(
            mean_len=float(L.mean()),
            std_len=float(L.std()),
            min_len=int(L.min()),
            max_len=int(L.max()),
            len_range=int(L.max() - L.min()),
            len_cv=float(L.std() / (L.mean() + 1e-12))
        )

    @staticmethod
    def aa_composition_metrics(seqs: List[str]) -> Dict[str, float]:
        cnt = Counter()
        tot = 0
        for s in seqs:
            cnt.update(s); tot += len(s)
        if tot == 0:
            return {"aa_entropy": 0.0, "most_freq_aa": None, "most_freq_aa_freq": 0.0, "unique_residues": 0.0}
        freqs = np.array([c / tot for c in cnt.values()])
        ent = _entropy(freqs)
        most = max(freqs)
        most_aa = max(cnt, key=cnt.get)
        k = len(cnt)
        uniform = np.full(k, 1.0 / k)
        kl = float((freqs * np.log((freqs + 1e-12) / (uniform + 1e-12))).sum())
        return dict(
            aa_entropy=ent, most_freq_aa=most_aa, most_freq_aa_freq=float(most),
            aa_kl_to_uniform=kl, unique_residues=float(k)
        )


# ---------------------------------------------------------------------
# K-mer metrics
# ---------------------------------------------------------------------
class KmerMetrics:
    @staticmethod
    def compute_kmer_redundancy(seqs: List[str], k: int = 3, topn: int = 10) -> Dict[str, Any]:
        stats = get_kmer_stats(seqs, k=k, drop_non_canonical=True)
        if stats.raw_total_kmers == 0:
            return dict(k=k, kmer_entropy=0.0, kmer_gini=0.0, top_kmers=[],
                        n_distinct_kmers=0, kmer_coverage_max=0.0)
        freqs = np.array(list(stats.total_occurrences.values())) / stats.raw_total_kmers
        ent = _entropy(freqs); g = _gini(freqs)

        sorted_by_count = sorted(stats.count_in_sequences.items(), key=lambda x: x[1], reverse=True)[:topn]
        n_seqs = len(seqs)
        top_info = [{"kmer": km, "count": stats.total_occurrences[km], "seq_coverage": c / n_seqs if n_seqs else 0.0}
                    for km, c in sorted_by_count]
        max_cov = max(stats.count_in_sequences.values()) / n_seqs if n_seqs and stats.count_in_sequences else 0.0
        return dict(
            k=k, kmer_entropy=ent, kmer_gini=g, top_kmers=top_info,
            n_distinct_kmers=stats.raw_unique_kmers, kmer_coverage_max=max_cov, kmer_stats=stats
        )

    @staticmethod
    def compare_kmer_profiles(seqs1: List[str], seqs2: List[str], k: int = 3) -> Dict[str, Any]:
        s1 = get_kmer_stats(seqs1, k=k, drop_non_canonical=True)
        s2 = get_kmer_stats(seqs2, k=k, drop_non_canonical=True)
        km1, km2 = set(s1.total_occurrences.keys()), set(s2.total_occurrences.keys())
        common = km1 & km2
        jacc = len(common) / len(km1 | km2) if (km1 | km2) else 0.0
        return {
            'jaccard_similarity': jacc,
            'common_kmers': len(common),
            'unique_to_group1': len(km1 - km2),
            'unique_to_group2': len(km2 - km1),
            'common_kmer_list': sorted(list(common)),
            'coverage_correlation': _calculate_coverage_correlation(s1, s2, common)
        }

    @staticmethod
    def kmer_diversity_profile(seqs: List[str], k_range: Tuple[int, int] = (2, 6)) -> Dict[str, Any]:
        results = {}; ents = []
        for k in range(k_range[0], k_range[1] + 1):
            try:
                s = get_kmer_stats(seqs, k=k, drop_non_canonical=True)
                if s.raw_total_kmers > 0:
                    freqs = np.array(list(s.total_occurrences.values())) / s.raw_total_kmers
                    results[f'k{k}'] = {'entropy': _entropy(freqs), 'gini': _gini(freqs),
                                        'unique_kmers': s.raw_unique_kmers, 'total_kmers': s.raw_total_kmers}
                else:
                    results[f'k{k}'] = {'entropy': 0.0, 'gini': 0.0, 'unique_kmers': 0, 'total_kmers': 0}
            except Exception:
                results[f'k{k}'] = {'entropy': 0.0, 'gini': 0.0, 'unique_kmers': 0, 'total_kmers': 0}
            ents.append(results[f'k{k}']['entropy'])
        optimal_k = k_range[0] + (int(np.argmax(ents)) if ents else 0)
        return {'k_profiles': results, 'optimal_k': optimal_k, 'entropy_trend': ents}


def _calculate_coverage_correlation(stats1, stats2, common_kmers):
    if not common_kmers:
        return 0.0
    cov1 = [stats1.freq_per_seq_ratio.get(km, 0) for km in common_kmers]
    cov2 = [stats2.freq_per_seq_ratio.get(km, 0) for km in common_kmers]
    if len(cov1) < 2:
        return 0.0
    cov1, cov2 = np.array(cov1), np.array(cov2)
    if cov1.std() == 0 or cov2.std() == 0:
        return 1.0 if np.array_equal(cov1, cov2) else 0.0
    return float(np.corrcoef(cov1, cov2)[0, 1])


# ---------------------------------------------------------------------
# Similarity metrics (scalable)
# ---------------------------------------------------------------------

def _graph_metrics_from_adj(A: np.ndarray, sampled: bool) -> Dict[str, float]:
    n = A.shape[0]
    deg = A.sum(1)
    visited = np.zeros(n, bool)
    comp_sizes = []
    for i in range(n):
        if visited[i]:
            continue
        q = [i]; visited[i] = True; c = 0
        while q:
            u = q.pop(); c += 1
            for v in np.where(A[u] > 0)[0]:
                if not visited[v]:
                    visited[v] = True; q.append(v)
        comp_sizes.append(c)
    comp = np.array(comp_sizes, int)
    out = dict(
        graph_avg_degree=float(deg.mean()),
        graph_degree_std=float(deg.std()),
        n_components=int(comp.size),
        largest_component_frac=float(comp.max() / n) if n else 0.0,
        singleton_components=int((comp == 1).sum())
    )
    if sampled:
        out.update({'graph_sampled': True})
    return out


