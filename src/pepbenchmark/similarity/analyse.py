from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .schemas import MetricsConfig, Pair, SimilarityMetrics


class SimilarityAnalyzer:
    """Analyse one or more similarity matrices with a notebook-friendly API."""

    def __init__(
        self,
        matrices: Dict[str, np.ndarray],
        topk: int = 5,
        exclude_diag: bool = True,
    ):
        """Initialize the analyzer.

        Args:
            matrices: Named similarity matrices to analyze.
            topk: Default number of neighbors used by summary methods.
            exclude_diag: Whether diagonal values should be masked for square
                matrices.
        """
        if not matrices:
            raise ValueError("matrices must not be empty")
        self.matrices = {
            name: np.asarray(matrix, dtype=float)
            for name, matrix in matrices.items()
        }
        self.topk = topk
        self.exclude_diag = exclude_diag

    def fields(self) -> List[str]:
        """Return the available matrix field names."""
        return list(self.matrices.keys())

    def _prepare_matrix(self, field: str) -> np.ndarray:
        if field not in self.matrices:
            if len(self.matrices) == 1:
                field = next(iter(self.matrices))
            else:
                raise KeyError(f"Unknown matrix field: {field}")
        matrix = self.matrices[field].copy()
        if self.exclude_diag and matrix.shape[0] == matrix.shape[1]:
            np.fill_diagonal(matrix, -np.inf)
        return matrix

    def global_topk(self, k: int, field: str = "similarity") -> List[Pair]:
        """Return the strongest global similarity pairs for one matrix field."""
        matrix = self._prepare_matrix(field)
        if k <= 0 or matrix.size == 0:
            return []

        actual_k = min(k, matrix.size)
        flat = matrix.ravel()
        top_indices = np.argpartition(flat, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
        row_indices, col_indices = np.unravel_index(top_indices, matrix.shape)

        return [
            Pair(int(row), int(col), float(matrix[row, col]))
            for row, col in zip(row_indices, col_indices)
            if np.isfinite(matrix[row, col])
        ]

    def nearest_neighbors_all(
        self,
        k: int,
        field: str = "similarity",
    ) -> Dict[int, List[Pair]]:
        """Return per-row top-k nearest neighbors for one matrix field."""
        matrix = self._prepare_matrix(field)
        if k <= 0:
            return {row_index: [] for row_index in range(matrix.shape[0])}

        results: Dict[int, List[Pair]] = {}
        for row_index in range(matrix.shape[0]):
            row = matrix[row_index]
            actual_k = min(k, row.shape[0])
            top_indices = np.argpartition(row, -actual_k)[-actual_k:]
            top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
            results[row_index] = [
                Pair(row_index, int(col_index), float(row[col_index]))
                for col_index in top_indices
                if np.isfinite(row[col_index])
            ]
        return results

    def threshold_filter(
        self,
        threshold: float,
        field: str = "similarity",
    ) -> Tuple[List[Pair], Dict[int, List[Pair]]]:
        """Return all pairs with similarity above a threshold."""
        matrix = self._prepare_matrix(field)
        global_pairs: List[Pair] = []
        per_sample_pairs: Dict[int, List[Pair]] = {}

        for row_index in range(matrix.shape[0]):
            row_pairs = [
                Pair(row_index, col_index, float(matrix[row_index, col_index]))
                for col_index in range(matrix.shape[1])
                if np.isfinite(matrix[row_index, col_index])
                and matrix[row_index, col_index] > threshold
            ]
            per_sample_pairs[row_index] = row_pairs
            global_pairs.extend(row_pairs)

        global_pairs.sort(key=lambda pair: pair.sim, reverse=True)
        return global_pairs, per_sample_pairs

    def compute_metrics(
        self,
        field: str = "similarity",
        config: Optional[MetricsConfig] = None,
    ) -> SimilarityMetrics:
        """Build a structured summary for one similarity matrix.

        Args:
            field: Matrix field name to analyze.
            config: Optional metric-selection configuration.

        Returns:
            A populated :class:`SimilarityMetrics` object.
        """
        cfg = config or MetricsConfig()
        k = cfg.k or self.topk
        matrix = self._prepare_matrix(field)

        metrics = SimilarityMetrics()

        if cfg.global_topk:
            metrics.global_topk_pairs = self.global_topk(k, field=field)
            metrics.global_topk_mean = (
                float(np.mean([pair.sim for pair in metrics.global_topk_pairs]))
                if metrics.global_topk_pairs
                else None
            )

        if cfg.per_sample_topk:
            metrics.per_sample_topk = self.nearest_neighbors_all(k, field=field)
            metrics.per_sample_topk_mean = {
                row_index: (float(np.mean([pair.sim for pair in pairs])) if pairs else None)
                for row_index, pairs in metrics.per_sample_topk.items()
            }

        if cfg.global_mean:
            values = matrix[np.isfinite(matrix)]
            metrics.global_mean = float(np.mean(values)) if values.size else None

        if cfg.per_sample_mean:
            for row_index in range(matrix.shape[0]):
                row_values = matrix[row_index][np.isfinite(matrix[row_index])]
                metrics.per_sample_mean[row_index] = float(np.mean(row_values)) if row_values.size else None

        if cfg.threshold is not None:
            global_pairs, per_sample_pairs = self.threshold_filter(cfg.threshold, field=field)
            metrics.global_threshold_pairs = global_pairs
            metrics.per_sample_threshold_pairs = per_sample_pairs

        return metrics


SimilarityAnalyse = SimilarityAnalyzer
