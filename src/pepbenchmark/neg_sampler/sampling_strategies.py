# Copyright ZGCA
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Sampling strategies for negative peptide sequence generation.

This module contains various sampling strategies that can be used by the
NegSampler to generate negative samples with different characteristics
and distribution properties.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict

from pepbenchmark.analyze.fasta_level import compute_peptide_properties
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


# ===========================
# Sampling Strategy Names
# ===========================

# Available sampling strategy names for easy reference
SAMPLING_STRATEGY_NAMES = [
    "random",              # Uniform random selection
    "kde",                 # Kernel Density Estimation importance sampling
    "mmd",                 # Maximum Mean Discrepancy herding
    "nn",                  # Nearest neighbor matching
    "ot",                  # Optimal Transport (Sinkhorn)
    "moment",              # Moment matching via ridge regression
    "bin",                 # Histogram/quantile bin matching
    "hybrid_property_kmer", # Property + K-mer hybrid sampling
    "chunk_shuffle",       # Chunk shuffling of positive samples
    "klet_shuffle",        # K-let preserving shuffling
]


# ===========================
# Data Models & Context
# ===========================

@dataclass
class SamplingContext:
    """Immutable context shared by sampling strategies.
    
    This class encapsulates all the data needed by sampling strategies to
    perform negative sample selection, including preprocessed positive and
    pool data with computed features.

    Attributes:
        pos_sequences: List of positive peptide sequences.
        pool_df: Candidate pool dataframe with columns ["sequence", *properties].
        pos_df: Positive dataframe with columns ["sequence", *properties].
        Z_pos: Z-scored features for positive sequences.
        Z_pool: Z-scored features for pool candidates (aligned to properties).
        properties: Ordered feature names used to build Z-scored feature space.
        
    Examples:
        >>> context = SamplingContext(
        ...     pos_sequences=["PEPTIDE1", "PEPTIDE2"],
        ...     pool_df=pool_dataframe,
        ...     pos_df=pos_dataframe, 
        ...     Z_pos=np.array([[0.1, -0.5], [0.2, 0.3]]),
        ...     Z_pool=np.array([[-0.1, 0.2], [0.5, -0.3]]),
        ...     properties=["length", "charge"]
        ... )
    """
    pos_sequences: List[str]
    pool_df: pd.DataFrame
    pos_df: pd.DataFrame
    Z_pos: np.ndarray
    Z_pool: np.ndarray
    properties: List[str]


# ===========================
# Strategy Interface
# ===========================

class BaseSampler(ABC):
    """Abstract base class for negative sampling strategies.

    All sampling strategies must inherit from this class and implement the
    `sample` method. Each strategy receives a SamplingContext and returns
    a list of selected sequence strings.
    
    Attributes:
        name: Unique identifier for this sampling strategy.
        
    Examples:
        >>> class CustomSampler(BaseSampler):
        ...     name = "custom"
        ...     
        ...     def sample(self, context, target, rng, **kwargs):
        ...         # Custom sampling logic here
        ...         return selected_sequences
    """

    name: str = "base"

    @abstractmethod
    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        **kwargs,
    ) -> List[str]:
        """Sample negative sequences from the candidate pool.
        
        Args:
            context: Sampling context containing positive/pool data and features.
            target: Number of negative sequences to sample.
            rng: Random number generator for reproducible sampling.
            **kwargs: Strategy-specific parameters.
            
        Returns:
            List of selected negative sequence strings.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError


# ===========================
# Utility Functions for Samplers
# ===========================

def _median_pairwise_distance(X: np.ndarray) -> float:
    """Compute median pairwise distance for bandwidth estimation.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        
    Returns:
        Median pairwise Euclidean distance between samples.
    """
    if len(X) < 2:
        return 1.0
    m = int(min(len(X), 500))  # Subsample for efficiency
    rng = np.random.default_rng(123)
    idx = rng.choice(len(X), size=m, replace=False)
    Xs = X[idx]
    D2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
    tri = D2[np.triu_indices_from(D2, k=1)]
    med = float(np.median(tri)) if len(tri) > 0 else 1.0
    return max(np.sqrt(med), 1e-3)


def _rff_features(
    X: np.ndarray,
    D: int,
    gamma: float,
    rng: np.random.Generator,
    W: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Random Fourier Features for kernel approximation.
    
    Args:
        X: Input feature matrix of shape (n_samples, n_features).
        D: Number of random features to generate.
        gamma: RBF kernel bandwidth parameter.
        rng: Random number generator.
        W: Pre-computed weight matrix (optional).
        b: Pre-computed bias vector (optional).
        
    Returns:
        Tuple of (features, weight_matrix, bias_vector).
    """
    d = X.shape[1]
    if W is None:
        W = rng.normal(0, np.sqrt(2 * gamma), size=(d, D))
    if b is None:
        b = rng.uniform(0, 2 * np.pi, size=(D,))
    Z = np.sqrt(2.0 / D) * np.cos(X @ W + b)
    return Z, W, b


def _silverman_bw(X: np.ndarray) -> float:
    """Compute Silverman's rule-of-thumb bandwidth for KDE.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        
    Returns:
        Bandwidth value for kernel density estimation.
    """
    n, d = X.shape
    std = X.std(axis=0, ddof=1)
    sigma = float(np.mean(std[std > 0])) if np.any(std > 0) else 1.0
    bw = ((4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))) * (n ** (-1.0 / (d + 4.0))) * sigma
    return max(bw, 1e-3)


def _kde_iso_gaussian(X_ref: np.ndarray, X_eval: np.ndarray, bw: float) -> np.ndarray:
    """Evaluate isotropic Gaussian KDE at evaluation points.
    
    Args:
        X_ref: Reference points for KDE of shape (n_ref, n_features).
        X_eval: Evaluation points of shape (n_eval, n_features).
        bw: Bandwidth parameter.
        
    Returns:
        KDE values at evaluation points of shape (n_eval,).
    """
    n_ref, d = X_ref.shape
    Xr2 = np.sum(X_ref ** 2, axis=1, keepdims=True)
    Xe2 = np.sum(X_eval ** 2, axis=1, keepdims=True).T
    G = Xr2 + Xe2 - 2.0 * (X_ref @ X_eval.T)
    norm = ((2 * np.pi) ** (d / 2)) * (bw ** d)
    K = np.exp(-0.5 * G / (bw ** 2)) / norm
    return np.mean(K, axis=0) + 1e-12


def _sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float = 0.1, n_iter: int = 200) -> np.ndarray:
    """Sinkhorn algorithm for entropy-regularized optimal transport.
    
    Args:
        a: Source distribution weights of shape (n,).
        b: Target distribution weights of shape (m,).
        C: Cost matrix of shape (n, m).
        eps: Entropy regularization parameter.
        n_iter: Maximum number of iterations.
        
    Returns:
        Transport plan matrix of shape (n, m).
    """
    K = np.exp(-C / max(eps, 1e-6))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = (u[:, None] * K) * v[None, :]
    return P


# ===========================
# Concrete Sampling Strategies
# ===========================

class RandomSampler(BaseSampler):
    """Uniform random selection from the candidate pool.
    
    This is the simplest sampling strategy that selects negative samples
    uniformly at random from the available pool without considering any
    distributional properties.
    
    Examples:
        >>> sampler = RandomSampler()
        >>> negatives = sampler.sample(context, target=100, rng=rng)
    """

    name = SAMPLING_STRATEGY_NAMES[0]  # "random"

    def sample(
        self, 
        context: SamplingContext, 
        target: int, 
        rng: np.random.Generator, 
        **kwargs
    ) -> List[str]:
        """Sample sequences uniformly at random from pool.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            **kwargs: Unused for this strategy.
            
        Returns:
            List of randomly selected sequence strings.
        """
        k = min(target, len(context.pool_df))
        if k <= 0:
            return []
        idx = rng.choice(len(context.pool_df), size=k, replace=False)
        return context.pool_df.iloc[idx]["sequence"].astype(str).tolist()


class KDESampler(BaseSampler):
    """Kernel Density Estimation importance sampling.

    This strategy uses KDE to estimate the density ratio between positive
    and pool distributions, then performs importance sampling to select
    negatives that better match the positive distribution.
    
    For large pools, switches to Random Fourier Features approximation
    to maintain computational efficiency.
    
    Args (via kwargs):
        weight_clip: Tuple of (min, max) values to clip importance weights.
        rff_dim_large: RFF dimension for large pool approximation.
        large_pool_threshold: Pool size threshold to switch to RFF.
        
    Examples:
        >>> sampler = KDESampler()
        >>> negatives = sampler.sample(
        ...     context, target=100, rng=rng,
        ...     weight_clip=(1e-3, 50.0), rff_dim_large=1024
        ... )
    """

    name = SAMPLING_STRATEGY_NAMES[1]  # "kde"

    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        weight_clip: Tuple[float, float] = (1e-3, 50.0),
        rff_dim_large: int = 1024,
        large_pool_threshold: int = 20_000,
        **kwargs,
    ) -> List[str]:
        """Sample using KDE density ratio importance sampling.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            weight_clip: Min/max values to clip importance weights.
            rff_dim_large: RFF dimension for large pools.
            large_pool_threshold: Threshold to switch to RFF approximation.
            **kwargs: Additional unused parameters.
            
        Returns:
            List of selected sequence strings.
        """
        if target <= 0 or len(context.pool_df) == 0:
            return []

        Z_pos, Z_pool = context.Z_pos, context.Z_pool
        
        if len(Z_pool) > large_pool_threshold:
            # Use RFF approximation for large pools
            allZ = np.vstack([Z_pos, Z_pool])
            bw = _median_pairwise_distance(allZ)
            gamma = 1.0 / max(bw ** 2, 1e-6)
            Phi_pos, W, b = _rff_features(Z_pos, rff_dim_large, gamma, rng)
            Phi_pool, _, _ = _rff_features(Z_pool, rff_dim_large, gamma, rng, W=W, b=b)
            mu_pos = Phi_pos.mean(axis=0)
            mu_pool = Phi_pool.mean(axis=0)
            p_pos_approx = Phi_pool @ mu_pos
            p_pool_approx = Phi_pool @ mu_pool
            w = (p_pos_approx + 1e-12) / (p_pool_approx + 1e-12)
        else:
            # Exact KDE for smaller pools
            bw_pos = _silverman_bw(Z_pos)
            bw_pool = _silverman_bw(Z_pool)
            p_pos = _kde_iso_gaussian(Z_pos, Z_pool, bw_pos)
            p_pool = _kde_iso_gaussian(Z_pool, Z_pool, bw_pool)
            w = p_pos / p_pool

        # Clip weights and normalize
        w = np.clip(np.where(np.isfinite(w), w, weight_clip[0]), weight_clip[0], weight_clip[1])
        p = w / w.sum()
        k = min(target, len(context.pool_df))
        idx = rng.choice(len(context.pool_df), size=k, replace=False, p=p)
        return context.pool_df.iloc[idx]["sequence"].astype(str).tolist()


class MMDHerdingSampler(BaseSampler):
    """Maximum Mean Discrepancy (MMD) herding sampler.

    This strategy uses kernel herding in Random Fourier Feature space to
    progressively select samples that minimize the MMD between positive
    and negative distributions. Each selection greedily reduces the
    discrepancy with the positive distribution.
    
    Args (via kwargs):
        rff_dim: Dimension of Random Fourier Features.
        
    Examples:
        >>> sampler = MMDHerdingSampler()
        >>> negatives = sampler.sample(
        ...     context, target=100, rng=rng, rff_dim=512
        ... )
    """

    name = SAMPLING_STRATEGY_NAMES[2]  # "mmd"

    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        rff_dim: int = 512,
        **kwargs,
    ) -> List[str]:
        """Sample using MMD herding in RFF space.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            rff_dim: Dimension of Random Fourier Features.
            **kwargs: Additional unused parameters.
            
        Returns:
            List of selected sequence strings.
        """
        if target <= 0 or len(context.pool_df) == 0:
            return []
            
        Z_pos, Z_pool = context.Z_pos, context.Z_pool
        allZ = np.vstack([Z_pos, Z_pool])
        bw = _median_pairwise_distance(allZ)
        gamma = 1.0 / max(bw ** 2, 1e-6)
        
        # Generate RFF features
        Phi_pos, W, b = _rff_features(Z_pos, rff_dim, gamma, rng)
        Phi_pool, _, _ = _rff_features(Z_pool, rff_dim, gamma, rng, W=W, b=b)
        mu_pos = Phi_pos.mean(axis=0)

        # Greedy herding selection
        selected: List[int] = []
        sum_sel = np.zeros(rff_dim, dtype=float)
        used = np.zeros(len(context.pool_df), dtype=bool)
        steps = min(target, len(context.pool_df))
        
        for t in range(steps):
            # Compute target: positive mean - current selected mean
            delta = mu_pos if t == 0 else (mu_pos - sum_sel / t)
            score = Phi_pool @ delta
            score[used] = -np.inf
            
            j = int(np.argmax(score))
            if not np.isfinite(score[j]):
                break
                
            used[j] = True
            selected.append(j)
            sum_sel += Phi_pool[j]
            
        return context.pool_df.iloc[selected]["sequence"].astype(str).tolist()


class NNMatcherSampler(BaseSampler):
    """Nearest neighbor matching sampler.

    This strategy performs nearest neighbor matching between positive samples
    and pool candidates. For each positive sample, it finds the k closest
    pool samples and selects them as negatives.
    
    Supports FAISS acceleration for large-scale matching when available.
    
    Args (via kwargs):
        k_per_pos: Number of neighbors to match per positive sample.
        caliper: Maximum distance threshold for valid matches.
        use_faiss_if_available: Whether to use FAISS if available.
        
    Examples:
        >>> sampler = NNMatcherSampler()
        >>> negatives = sampler.sample(
        ...     context, target=100, rng=rng,
        ...     k_per_pos=2, caliper=1.5
        ... )
    """

    name = SAMPLING_STRATEGY_NAMES[3]  # "nn"
    NN_CHUNK_SIZE = 50_000

    def _pick_for_one(
        self,
        z: np.ndarray,
        Z_pool: np.ndarray,
        cand_idx: np.ndarray,
        k_per_pos: int,
        caliper: Optional[float],
    ) -> List[int]:
        """Find k nearest neighbors for a single positive sample.
        
        Args:
            z: Single positive sample features.
            Z_pool: Pool sample features.
            cand_idx: Indices of available candidates.
            k_per_pos: Number of neighbors to find.
            caliper: Maximum distance threshold.
            
        Returns:
            List of selected pool indices.
        """
        best_j, best_d = [], []
        
        for start in range(0, len(cand_idx), self.NN_CHUNK_SIZE):
            chunk = cand_idx[start : start + self.NN_CHUNK_SIZE]
            M = Z_pool[chunk]
            d2 = np.sum((M - z) ** 2, axis=1)
            
            if caliper is not None:
                mask = d2 <= (caliper ** 2)
                if not np.any(mask):
                    continue
                d2 = d2[mask]
                chunk = chunk[mask]
                
            best_j.append(chunk)
            best_d.append(d2)
            
        if not best_j:
            return []
            
        all_idx = np.concatenate(best_j)
        all_d2 = np.concatenate(best_d)
        
        if len(all_idx) <= k_per_pos:
            order = np.argsort(all_d2)
            return all_idx[order].tolist()
            
        kth = np.argpartition(all_d2, k_per_pos - 1)[:k_per_pos]
        order = kth[np.argsort(all_d2[kth])]
        return all_idx[order].tolist()

    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        k_per_pos: int = 1,
        caliper: Optional[float] = None,
        use_faiss_if_available: bool = True,
        **kwargs,
    ) -> List[str]:
        """Sample using nearest neighbor matching.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            k_per_pos: Number of neighbors per positive sample.
            caliper: Maximum distance for valid matches.
            use_faiss_if_available: Whether to use FAISS acceleration.
            **kwargs: Additional unused parameters.
            
        Returns:
            List of selected sequence strings.
        """
        if target <= 0 or len(context.pool_df) == 0:
            return []
            
        Z_pos, Z_pool = context.Z_pos, context.Z_pool
        remaining = set(range(len(context.pool_df)))
        chosen: List[int] = []

        # Try FAISS acceleration if requested and available
        if use_faiss_if_available and caliper is None:
            try:
                import faiss  # type: ignore

                index = faiss.IndexFlatL2(Z_pool.shape[1])
                index.add(Z_pool.astype(np.float32))
                need = min(target, len(Z_pool))
                quota = max(1, int(np.ceil(need / max(1, len(Z_pos)))))
                
                for i in range(len(Z_pos)):
                    if len(chosen) >= need:
                        break
                    _, I = index.search(Z_pos[i : i + 1].astype(np.float32), quota)
                    for j in I[0].tolist():
                        if j in remaining:
                            chosen.append(j)
                            remaining.remove(j)
                        if len(chosen) >= need:
                            break
                            
                return context.pool_df.iloc[sorted(set(chosen))]["sequence"].astype(str).tolist()
            except Exception:
                logger.info("FAISS not available; falling back to numpy-based NN.")

        # Fallback to numpy-based nearest neighbor search
        cand_idx = np.array(sorted(list(remaining)))
        for i in range(len(Z_pos)):
            if len(chosen) >= target or not len(cand_idx):
                break
            picks = self._pick_for_one(Z_pos[i], Z_pool, cand_idx, k_per_pos, caliper)
            for j in picks:
                if j in remaining:
                    chosen.append(j)
                    remaining.remove(j)
            cand_idx = np.array(sorted(list(remaining)))

        # Fill remaining slots randomly if needed
        if len(chosen) < target and len(remaining):
            need = target - len(chosen)
            extra = rng.choice(list(remaining), size=min(need, len(remaining)), replace=False)
            chosen.extend(extra.tolist())

        return context.pool_df.iloc[sorted(set(chosen))]["sequence"].astype(str).tolist()


class OTSinkhornSampler(BaseSampler):
    """Optimal Transport (Sinkhorn) sampler.

    This strategy uses entropy-regularized optimal transport to find the
    optimal mapping between positive and pool distributions. Samples with
    the highest transport mass are selected as negatives.
    
    Args (via kwargs):
        epsilon: Entropy regularization parameter for Sinkhorn algorithm.
        max_iter: Maximum iterations for Sinkhorn algorithm.
        n_pool_cap: Maximum pool size (subsampled if larger).
        
    Examples:
        >>> sampler = OTSinkhornSampler()
        >>> negatives = sampler.sample(
        ...     context, target=100, rng=rng,
        ...     epsilon=0.1, max_iter=500
        ... )
    """

    name = SAMPLING_STRATEGY_NAMES[4]  # "ot"

    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        epsilon: Optional[float] = None,
        max_iter: int = 500,
        n_pool_cap: int = 10_000,
        **kwargs,
    ) -> List[str]:
        """Sample using optimal transport with Sinkhorn algorithm.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            epsilon: Entropy regularization parameter.
            max_iter: Maximum Sinkhorn iterations.
            n_pool_cap: Maximum pool size for computational efficiency.
            **kwargs: Additional unused parameters.
            
        Returns:
            List of selected sequence strings.
        """
        if target <= 0 or len(context.pool_df) == 0:
            return []

        pool_df_full = context.pool_df
        Z_pos, Z_pool_full = context.Z_pos, context.Z_pool

        # Subsample pool if too large
        if len(pool_df_full) > n_pool_cap:
            idx_sub = rng.choice(len(pool_df_full), size=n_pool_cap, replace=False)
            pool_df = pool_df_full.iloc[idx_sub].reset_index(drop=True)
            Z_pool = Z_pool_full[idx_sub]
        else:
            pool_df = pool_df_full
            Z_pool = Z_pool_full

        # Compute cost matrix (squared Euclidean distance)
        X, Y = Z_pos, Z_pool
        X2 = np.sum(X ** 2, axis=1, keepdims=True)
        Y2 = np.sum(Y ** 2, axis=1, keepdims=True).T
        C = X2 + Y2 - 2 * (X @ Y.T)
        
        # Set regularization parameter
        med = np.median(C) if C.size > 0 else 1.0
        eps = epsilon if epsilon is not None else max(0.05 * med, 1e-3)

        # Solve optimal transport
        a = np.ones(len(X)) / max(1, len(X))
        b = np.ones(len(Y)) / max(1, len(Y))
        P = _sinkhorn(a, b, C, eps=eps, n_iter=max_iter)
        
        # Select samples with highest transport mass
        mass = P.sum(axis=0)
        order = np.argsort(-mass)
        k = min(target, len(pool_df))
        idx = order[:k]
        
        return pool_df.iloc[idx]["sequence"].astype(str).tolist()


class MomentMatchSampler(BaseSampler):
    """Moment matching sampler.

    This strategy matches the first and second moments (mean and variance)
    between positive and negative distributions by solving a ridge regression
    problem in augmented feature space [Z, Z^2].
    
    Args (via kwargs):
        l2_reg: L2 regularization parameter for ridge regression.
        approx_threshold: Threshold to switch to approximation method.
        
    Examples:
        >>> sampler = MomentMatchSampler()
        >>> negatives = sampler.sample(
        ...     context, target=100, rng=rng,
        ...     l2_reg=1e-6, approx_threshold=10000
        ... )
    """

    name = SAMPLING_STRATEGY_NAMES[5]  # "moment"

    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        l2_reg: float = 1e-6,
        approx_threshold: int = 10_000,
        **kwargs,
    ) -> List[str]:
        """Sample using moment matching via ridge regression.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            l2_reg: L2 regularization parameter.
            approx_threshold: Pool size threshold for approximation.
            **kwargs: Additional unused parameters.
            
        Returns:
            List of selected sequence strings.
        """
        if target <= 0 or len(context.pool_df) == 0:
            return []
            
        Z_pos, Z_pool = context.Z_pos, context.Z_pool
        
        # Compute target moments from positive samples
        mu_pos = Z_pos.mean(axis=0)
        var_pos = Z_pos.var(axis=0)
        t = np.concatenate([mu_pos, var_pos])
        
        # Augmented feature matrix [Z, Z^2]
        Phi = np.concatenate([Z_pool, Z_pool ** 2], axis=1)
        k = min(target, len(context.pool_df))
        if k <= 0:
            return []

        if len(Phi) > approx_threshold:
            # Approximation method for large pools
            raw = Phi @ t
            w = np.maximum(raw, 0.0)
            if not np.any(w > 0):
                w = np.ones(len(Phi))
            p = w / w.sum()
            idx = rng.choice(len(context.pool_df), size=k, replace=False, p=p)
        else:
            # Exact ridge regression for smaller pools
            G = Phi @ Phi.T + l2_reg * np.eye(Phi.shape[0])
            rhs = Phi @ t
            try:
                w = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(G, rhs, rcond=None)[0]
            w = np.clip(w, 0, None)
            p = (w / w.sum()) if w.sum() > 0 else np.ones_like(w) / len(w)
            idx = rng.choice(len(context.pool_df), size=k, replace=False, p=p)
            
        return context.pool_df.iloc[idx]["sequence"].astype(str).tolist()


class BinSampler(BaseSampler):
    """Histogram/quantile bin matching sampler.

    This strategy divides the positive distribution into bins based on
    peptide properties and samples from the pool to match the positive
    distribution within each bin. Supports both single-property and
    joint multi-property binning.
    
    Args (via kwargs):
        n_bins: Number of bins to use for histogram matching.
        
    Examples:
        >>> sampler = BinSampler()
        >>> negatives = sampler.sample(
        ...     context, target=100, rng=rng, n_bins=10
        ... )
    """

    name = SAMPLING_STRATEGY_NAMES[6]  # "bin"

    @staticmethod
    def _calc_bins_df(sequences: List[str], prop: str, n_bins: int) -> Tuple[pd.DataFrame, Union[List[float], np.ndarray]]:
        """Calculate bin edges for a property and assign bin labels.
        
        Args:
            sequences: List of peptide sequences.
            prop: Property name to bin.
            n_bins: Number of bins.
            
        Returns:
            Tuple of (dataframe_with_bins, bin_edges).
        """
        assert prop in ["mw", "hydrophobicity", "charge", "isoelectricpoint", "length"], \
            f"Unsupported property: {prop}"
            
        df = compute_peptide_properties(list(map(str, sequences)))
        prop_values = df[prop].dropna()
        unique_count = prop_values.nunique()
        
        if unique_count <= n_bins:
            # Use unique values as bins if few unique values
            bins = sorted(prop_values.unique().tolist() + [prop_values.max() + 1])
            df[f"{prop}_bin"] = pd.cut(df[prop], bins=bins, include_lowest=True)
            return df, bins
        else:
            # Use quantile-based binning
            df[f"{prop}_bin"], bin_edges = pd.qcut(
                prop_values, q=n_bins, duplicates="drop", retbins=True
            )
            return df, bin_edges

    @staticmethod
    def _assign_bins(
        pool_df: pd.DataFrame, 
        properties: List[str], 
        bins_dict: Dict[str, Union[List[float], np.ndarray]]
    ) -> pd.DataFrame:
        """Assign bin labels to pool samples.
        
        Args:
            pool_df: Pool dataframe.
            properties: List of property names.
            bins_dict: Dictionary mapping property names to bin edges.
            
        Returns:
            Pool dataframe with added bin columns.
        """
        out = pool_df.copy()
        for p in properties:
            edges = bins_dict[p]
            out[f"{p}_bin"] = pd.cut(out[p], bins=edges, include_lowest=True)
        return out

    def sample(
        self,
        context: SamplingContext,
        target: int,
        rng: np.random.Generator,
        n_bins: int = 10,
        **kwargs,
    ) -> List[str]:
        """Sample using histogram/bin matching.
        
        Args:
            context: Sampling context.
            target: Number of sequences to sample.
            rng: Random number generator.
            n_bins: Number of bins for histogram matching.
            **kwargs: Additional unused parameters.
            
        Returns:
            List of selected sequence strings.
        """
        if target <= 0 or len(context.pool_df) == 0:
            return []

        props = context.properties
        pos_seq = context.pos_sequences
        pool_df = context.pool_df.copy()

        if len(props) == 1:
            # Single property binning
            prop = props[0]
            pos_df, edges = self._calc_bins_df(pos_seq, prop, n_bins)
            pool_binned = self._assign_bins(pool_df, [prop], {prop: edges})
            counts = pos_df[f"{prop}_bin"].value_counts().sort_index()
            pool_binned["used"] = False
            picked = []
            pos_set = set(pos_df["sequence"])
            total = 0
            
            for i, b in enumerate(counts.index.tolist()):
                if hasattr(b, "left") and hasattr(b, "right"):
                    low, high = b.left, b.right
                else:
                    low, high = b, b
                    
                need = int(np.floor(counts[b] * target / max(1, len(pos_df))))
                cand = pool_binned[
                    (~pool_binned["used"]) & 
                    (pool_binned[prop] > low) & 
                    (pool_binned[prop] <= high)
                ]
                cand = cand[~cand["sequence"].isin(pos_set)]
                
                if len(cand) > 0 and need > 0:
                    take = min(need, len(cand))
                    sampled = cand.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
                    pool_binned.loc[sampled.index, "used"] = True
                    picked.append(sampled)
                    total += take
                    
            # Fill remaining with random selection
            if total < target:
                rest = pool_binned[~pool_binned["used"]]
                rest = rest[~rest["sequence"].isin(pos_set)]
                take = min(target - total, len(rest))
                if take > 0:
                    picked.append(rest.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
                    
            out = pd.concat(picked) if picked else pd.DataFrame(columns=["sequence"])
            return out["sequence"].astype(str).tolist()

        # Multi-property joint binning
        bins_dict: Dict[str, Union[List[float], np.ndarray]] = {}
        pos_bin_df = None
        
        for p in props:
            cur, edges = self._calc_bins_df(pos_seq, p, n_bins)
            bins_dict[p] = edges
            pos_bin_df = cur if pos_bin_df is None else pos_bin_df.merge(
                cur[["sequence", f"{p}_bin"]], on="sequence"
            )
            
        pool_binned = self._assign_bins(pool_df, props, bins_dict)
        keys = [f"{p}_bin" for p in props]
        counts = pos_bin_df.groupby(keys).size().sort_index()
        pool_binned["used"] = False
        pos_set = set(pos_bin_df["sequence"])
        picked = []
        total = 0
        
        for j, (joint_bin, cnt) in enumerate(counts.items()):
            if not isinstance(joint_bin, tuple):
                joint_bin = (joint_bin,)
                
            need = int(np.floor(int(cnt) * target / max(1, len(pos_bin_df))))
            mask = ~pool_binned["used"]
            for k, b in zip(keys, joint_bin):
                mask &= (pool_binned[k] == b)
                
            cand = pool_binned[mask]
            cand = cand[~cand["sequence"].isin(pos_set)]
            
            if len(cand) > 0 and need > 0:
                take = min(need, len(cand))
                sampled = cand.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
                pool_binned.loc[sampled.index, "used"] = True
                picked.append(sampled)
                total += take
                
        # Fill remaining with random selection
        if total < target:
            rest = pool_binned[~pool_binned["used"]]
            rest = rest[~rest["sequence"].isin(pos_set)]
            take = min(target - total, len(rest))
            if take > 0:
                picked.append(rest.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
                
        out = pd.concat(picked) if picked else pd.DataFrame(columns=["sequence"])
        return out["sequence"].astype(str).tolist()


# ===========================
# Sampler Registry
# ===========================

class SamplerRegistry:
    """Registry for managing sampling strategies.

    This class provides a centralized way to register and create sampling
    strategies. New custom strategies can be registered at runtime.
    
    Examples:
        >>> registry = SamplerRegistry()
        >>> sampler = registry.create("kde")
        >>> 
        >>> # Register custom strategy
        >>> registry.register(MyCustomSampler)
        >>> custom_sampler = registry.create("my_custom")
    """

    def __init__(self):
        """Initialize registry with default samplers."""
        self._by_name: Dict[str, Type[BaseSampler]] = {}
        
        # Register all default samplers
        default_samplers = [
            RandomSampler, KDESampler, MMDHerdingSampler, 
            NNMatcherSampler, OTSinkhornSampler, MomentMatchSampler, BinSampler,
            HybridPropertyKmerSampler, ChunkShuffleSampler, KletShuffleSampler
        ]
        
        for cls in default_samplers:
            self._by_name[cls.name] = cls

    def register(self, sampler_cls: Type[BaseSampler]) -> None:
        """Register a new sampling strategy.
        
        Args:
            sampler_cls: Sampler class that inherits from BaseSampler.
            
        Examples:
            >>> registry = SamplerRegistry()
            >>> registry.register(MyCustomSampler)
        """
        self._by_name[sampler_cls.name] = sampler_cls

    def create(self, name: str) -> BaseSampler:
        """Create a sampler instance by name.
        
        Args:
            name: Name of the sampling strategy.
            
        Returns:
            Instance of the requested sampler.
            
        Raises:
            ValueError: If sampler name is not registered.
            
        Examples:
            >>> registry = SamplerRegistry()
            >>> sampler = registry.create("kde")
        """
        key = (name or "").lower()
        if key not in self._by_name:
            raise ValueError(f"Unknown sampler '{name}'. Available: {sorted(self._by_name)}")
        return self._by_name[key]()
        
    def list_available(self) -> List[str]:
        """List all available sampler names.
        
        Returns:
            List of registered sampler names.
            
        Examples:
            >>> registry = SamplerRegistry()
            >>> print(registry.list_available())
            ['random', 'kde', 'mmd', 'nn', 'ot', 'moment', 'bin']
        """
        return sorted(self._by_name.keys())


# ===========================
# Utility functions (from sampling_strategies_ext.py)
# ===========================

_AA = "ACDEFGHIKLMNPQRSTVWY"

def _vocab_k(k: int):
    """Generate a k-mer vocabulary."""
    if k < 1:
        raise ValueError(f"k must be positive, got: {k}")
    if k > 10:  # Prevent generating an overly large vocabulary.
        raise ValueError(f"k too large (max 10), got: {k}")
    return list(_AA) if k == 1 else [''.join(p) for p in product(_AA, repeat=k)]

def _kmer_counts(seq: str, k: int, idx: Dict[str,int]) -> np.ndarray:
    """Compute the k-mer count vector for a sequence."""
    if k < 1:
        raise ValueError(f"k must be positive, got: {k}")
    if len(seq) < k:
        # Return a zero vector if the sequence is shorter than k.
        return np.zeros(len(idx), dtype=np.float64)
    
    v = np.zeros(len(idx), dtype=np.float64)
    L = len(seq)
    for i in range(L-k+1):
        kmer = seq[i:i+k]
        # Check whether the k-mer contains only valid amino acids.
        if all(aa in _AA for aa in kmer):
            j = idx.get(kmer)
            if j is not None: 
                v[j] += 1.0
        # Skip this k-mer if it contains invalid characters.
    return v

def _make_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0,1,n_bins+1)
    edges = np.quantile(values, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(values.min(), values.max()+1e-8, n_bins+1)
    return edges


# ===========================
# Generic HybridPropertyKmerSampler
# ===========================

class HybridPropertyKmerSampler(BaseSampler):
    """
    Generic sampler combining properties and k-mers.
    - Preserve the distribution by binning on the specified properties.
    - Within each bin, greedily pick samples based on the k-mer gap for k_list.
    """
    name = SAMPLING_STRATEGY_NAMES[7]  # "hybrid_property_kmer"

    def sample(self, context, target: int, rng,
               properties: List[str] = ["length","charge"],
               n_bins: int = 6,
               laplace: float = 0.5,
               k_list: List[int] = [1],
               weights: Optional[Dict[int, float]] = None) -> List[str]:

        # Validate the k_list parameter.
        if not k_list or not all(isinstance(k, int) and k >= 1 for k in k_list):
            raise ValueError(f"k_list must contain positive integers, got: {k_list}")
        
        # Check whether sequence lengths support the largest k.
        max_k = max(k_list)
        min_seq_len = min(len(seq) for seq in context.pos_sequences) if context.pos_sequences else 0
        if min_seq_len < max_k:
            raise ValueError(f"Minimum sequence length ({min_seq_len}) is less than max k-mer size ({max_k})")

        if weights is None:
            weights = {k: 1.0 for k in k_list}
            if 2 in k_list: weights[2] = 1.5

        pos_df = context.pos_df.copy()
        pool_df = context.pool_df.copy()
        if target <= 0 or len(pool_df) == 0:
            return []

        # Step 1: Multi-dimensional binning
        edges = {}
        for p in properties:
            vals = pos_df[p].astype(float).to_numpy()
            edges[p] = _make_bins(vals, n_bins)

        def assign_bin(row):
            return tuple(
                int(np.clip(np.searchsorted(edges[p], float(row[p]), side="right")-1,
                            0, len(edges[p])-2))
                for p in properties
            )

        pos_bins = {}
        for _, row in pos_df.iterrows():
            key = assign_bin(row)
            pos_bins[key] = pos_bins.get(key, 0) + 1

        quota = {k: int(round(target * cnt / len(pos_df))) for k, cnt in pos_bins.items()}
        while sum(quota.values()) < target:
            kmax = max(pos_bins, key=lambda k: pos_bins[k])
            quota[kmax] += 1
        while sum(quota.values()) > target:
            kmin = min(pos_bins, key=lambda k: quota[k])
            if quota[kmin] > 0:
                quota[kmin] -= 1
            else:
                break

        pool_df["bin"] = pool_df.apply(assign_bin, axis=1)

        # Step 2: Target k-mer distribution
        target_vecs = {}
        for k in k_list:
            vocab = _vocab_k(k)
            idx = {m: i for i,m in enumerate(vocab)}
            c = np.zeros(len(idx))
            for s in pos_df["sequence"].astype(str):
                c += _kmer_counts(s,k,idx)
            target_vecs[k] = (idx, (c+laplace)/(c.sum()+laplace*len(idx)))

        target_counts = {k: vec for k, (_, vec) in target_vecs.items()}
        gap = {k: v.copy() for k,v in target_counts.items()}

        # Step 3: Sample within each bin
        picked = []
        for key, need in sorted(quota.items(), key=lambda kv: -kv[1]):
            if need <= 0: continue
            group = pool_df[pool_df["bin"]==key]["sequence"].astype(str).tolist()
            rng.shuffle(group)
            used=set()
            for _ in range(need):
                if not group: break
                best, best_gain = None, -1e18
                for s in group:
                    if s in used: continue
                    feats = {k: _kmer_counts(s,k,target_vecs[k][0]) for k in k_list}
                    g = sum(weights[k]*float(np.dot(feats[k], np.maximum(gap[k],0.0))) for k in k_list)
                    if g > best_gain:
                        best_gain, best = g, (s,feats)
                if best is None: break
                s,feats = best
                picked.append(s)
                used.add(s)
                for k in k_list:
                    gap[k] = np.maximum(gap[k]-feats[k],0.0)
                group.remove(s)

        # Step 4: Fill the remaining slots
        if len(picked) < target:
            remain = [s for s in pool_df["sequence"].astype(str) if s not in set(picked)]
            rng.shuffle(remain)
            picked += remain[:(target-len(picked))]

        return picked[:target]


# ===========================
# Post-processor: KmerPostProcessor
# ===========================

class KmerPostProcessor:
    """
    Post-process by greedily replacing samples using the k-mer gap while keeping the property distribution unchanged.
    """

    def balance_kmer(
        self,
        positives: List[str],
        base_negatives: List[str],
        pool_df: pd.DataFrame,
        *,
        properties: List[str] = ["length","charge"],
        n_bins: int = 6,
        k_list: List[int] = [1,2],
        weights: Optional[Dict[int,float]] = None,
        laplace: float = 0.5,
        max_rounds: int = 2,
        candidates_per_bin: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        # Validate the k_list parameter.
        if not k_list or not all(isinstance(k, int) and k >= 1 for k in k_list):
            raise ValueError(f"k_list must contain positive integers, got: {k_list}")
        
        # Check whether sequence lengths support the largest k.
        max_k = max(k_list)
        all_sequences = positives + base_negatives
        if all_sequences:
            min_seq_len = min(len(seq) for seq in all_sequences)
            if min_seq_len < max_k:
                raise ValueError(f"Minimum sequence length ({min_seq_len}) is less than max k-mer size ({max_k})")
        
        # Validate the weights parameter.
        if weights is not None:
            missing_weights = set(k_list) - set(weights.keys())
            if missing_weights:
                raise ValueError(f"weights missing for k-mer sizes: {sorted(missing_weights)}")
        
        if rng is None:
            rng = np.random.default_rng(123)
        if weights is None:
            weights = {k: 1.0 for k in k_list}
            if 2 in k_list: weights[2] = 1.5

        # Bin data using boundaries computed from the positive property distribution.
        edges={}
        
        # Compute properties for positive samples.
        from pepbenchmark.analyze.fasta_level import compute_peptide_properties
        pos_properties = []
        for seq in positives:
            props = compute_peptide_properties(seq)
            pos_properties.append(props)
        
        for p in properties:
            # Extract property values from positive samples.
            vals = np.array([props[p] for props in pos_properties])
            if len(vals) == 0:
                # If there are no positive samples, fall back to values from pool_df.
                vals = pool_df[p].astype(float).to_numpy()
            edges[p] = _make_bins(vals, n_bins)

        def assign_bin(row):
            return tuple(
                int(np.clip(np.searchsorted(edges[p], float(row[p]), side="right")-1,
                            0, len(edges[p])-2))
                for p in properties
            )

        pool_df = pool_df.copy()
        pool_df["bin"] = pool_df.apply(assign_bin, axis=1)
        base_set=set(base_negatives)
        bdf=pool_df[pool_df["sequence"].isin(base_set)]
        quota: Dict[Tuple,int] = bdf.groupby("bin")["sequence"].count().to_dict()

        # Target k-mers
        target_vecs={}
        for k in k_list:
            vocab=_vocab_k(k); idx={m:i for i,m in enumerate(vocab)}
            c=np.zeros(len(idx))
            for s in positives:
                c+=_kmer_counts(s,k,idx)
            target_vecs[k]=(idx,(c+laplace)/(c.sum()+laplace*len(idx)))

        sel=list(base_negatives)
        not_sel=pool_df[~pool_df["sequence"].isin(base_set)]
        cand_by_bin={}
        for key,grp in not_sel.groupby("bin"):
            arr=grp["sequence"].astype(str).tolist()
            rng.shuffle(arr)
            cand_by_bin[key]=arr[:candidates_per_bin]

        # Create a sequence-to-bin mapping for faster lookup.
        seq_to_bin = {}
        for _, row in pool_df.iterrows():
            seq_to_bin[row["sequence"]] = row["bin"]

        # Iterative replacement
        for _ in range(max_rounds):
            improved=False
            for key,need in quota.items():
                if need<=0: continue
                S_bin=[s for s in sel if seq_to_bin.get(s)==key]
                if not S_bin: continue
                C_bin=cand_by_bin.get(key,[])
                if not C_bin: continue
                # deficit
                deficit={}
                for k,(idx,p_tar) in target_vecs.items():
                    c_cur=np.zeros(len(idx))
                    for s in sel:
                        c_cur+=_kmer_counts(s,k,idx)
                    q_cur=(c_cur+laplace)/(c_cur.sum()+laplace*len(idx))
                    deficit[k]=p_tar-q_cur
                # worst
                worst_s=None; best_c=None; best_gain=-1e18
                for s in S_bin:
                    g=0.0
                    for k,(idx,_) in target_vecs.items():
                        v=_kmer_counts(s,k,idx)
                        g+=-weights.get(k,1.0)*float(np.dot(v,deficit[k]))
                    if g>best_gain:
                        best_gain=g; worst_s=s
                # best candidate
                best_c=None; best_gain2=-1e18
                for c in C_bin:
                    g=0.0
                    for k,(idx,_) in target_vecs.items():
                        v=_kmer_counts(c,k,idx)
                        g+=weights.get(k,1.0)*float(np.dot(v,deficit[k]))
                    if g>best_gain2:
                        best_gain2=g; best_c=c
                if worst_s and best_c and best_gain2+best_gain>0:
                    sel.remove(worst_s); sel.append(best_c)
                    C_bin.remove(best_c); improved=True
            if not improved: break
        return sel


# ===========================
# Shuffling strategies
# ===========================

class ChunkShuffleSampler(BaseSampler):
    """Shuffle positive samples after splitting them into 1~4-mer chunks."""
    name = SAMPLING_STRATEGY_NAMES[8]  # "chunk_shuffle"
    def sample(self, context, target: int, rng, kmin: int = 1, kmax: int = 4, trials_per_pos: int = 3):
        pos = list(map(str, context.pos_sequences))
        if target <= 0 or not pos: return []
        out, seen = [], set()
        order=list(range(len(pos))); rng.shuffle(order); i=0
        while len(out)<target and i<trials_per_pos*len(pos):
            s=pos[order[i%len(pos)]]
            L=len(s); cuts=[0]
            while cuts[-1]<L:
                step=int(rng.integers(kmin,kmax+1)); cuts.append(min(L,cuts[-1]+step))
            chunks=[s[cuts[j]:cuts[j+1]] for j in range(len(cuts)-1)]
            rng.shuffle(chunks); t="".join(chunks)
            if t!=s and t not in seen: seen.add(t); out.append(t)
            i+=1
        return out[:target]

def _eulerian_shuffle(seq: str, k:int, rng) -> str:
    if k<=1 or len(seq)<=k:
        s=list(seq); rng.shuffle(s); return "".join(s)
    edges=defaultdict(list)
    for i in range(len(seq)-k+1):
        a=seq[i:i+k-1]; b=seq[i+1:i+k]
        edges[a].append(b)
    start=seq[:k-1]; stack=[start]; path=[]
    while stack:
        v=stack[-1]
        if edges[v]:
            j=int(rng.integers(0,len(edges[v]))); u=edges[v].pop(j); stack.append(u)
        else:
            path.append(stack.pop())
    path=path[::-1]; out=[path[0]]
    for node in path[1:]: out.append(node[-1])
    return "".join(out)

class KletShuffleSampler(BaseSampler):
    """Shuffle while preserving k-let statistics (default k=2)."""
    name = SAMPLING_STRATEGY_NAMES[9]  # "klet_shuffle"
    def sample(self, context, target: int, rng, k: int = 2):
        pos=list(map(str, context.pos_sequences))
        if target<=0 or not pos: return []
        out, seen=[],set()
        order=list(range(len(pos))); rng.shuffle(order); i=0
        while len(out)<target and i<5*len(pos):
            s=pos[order[i%len(pos)]]
            t=_eulerian_shuffle(s,k,rng)
            if t!=s and t not in seen: seen.add(t); out.append(t)
            i+=1
        return out[:target]
