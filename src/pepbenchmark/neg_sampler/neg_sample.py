"""Negative sampling module for peptide datasets.

This module provides a comprehensive negative sampling system for peptide
datasets, including various sampling strategies and distribution validation
capabilities.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pepbenchmark.neg_sampler.distribution_validator import DistributionValidator
from pepbenchmark.analyze.fasta_level import compute_peptide_properties
from pepbenchmark.neg_sampler.neg_meta import EXCLUSIVE_MAP, INCLUSIVE_MAP,read_dataset_sequences
from pepbenchmark.neg_sampler.sampling_pool_manager import SamplingPoolManager
from pepbenchmark.neg_sampler.sampling_strategies import SamplerRegistry, SamplingContext
from pepbenchmark.utils.logging import get_logger
import numpy as np
logger = get_logger()


# ===========================
# Constants & Utilities
# ===========================

DEFAULT_PROPERTIES = ["length", "charge","hydrophobicity"]


def _default_properties(props: Optional[Sequence[str]]) -> List[str]:
    """Get default properties list if none provided.
    
    Args:
        props: Optional sequence of property names.
        
    Returns:
        List of property names, either from input or defaults.
    """
    return list(props) if props else list(DEFAULT_PROPERTIES)


def _validate_properties(columns: Sequence[str], properties: Sequence[str]) -> None:
    """Validate that required properties are available in data.
    
    Args:
        columns: Available column names in the data.
        properties: Required property names.
        
    Raises:
        ValueError: If any required properties are missing.
    """
    miss = [p for p in properties if p not in columns]
    if miss:
        raise ValueError(f"Missing properties {miss}. Available: {list(columns)}")


def _safe_zscore(df: pd.DataFrame, cols: List[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Compute z-score robustly (handles std==0 case).
    
    Args:
        df: Input dataframe.
        cols: Columns to z-score normalize.
        
    Returns:
        Tuple of (z_scored_data, means, standard_deviations).
    """
    mu = df[cols].mean(numeric_only=True)
    sd = df[cols].std(ddof=0, numeric_only=True).replace(0, 1.0)
    Z = (df[cols].apply(pd.to_numeric, errors="coerce") - mu) / sd
    return Z, mu, sd


class NegSampler:
    """Comprehensive negative sampling system for peptide datasets.

    This class orchestrates the complete negative sampling pipeline, providing
    a unified interface for different sampling strategies and distribution
    validation capabilities. It manages the sampling pool, applies various
    sampling algorithms, and validates the quality of generated negative samples.

    The NegSampler handles:
    - Pool management with automatic property computation and deduplication
    - Context building with aligned feature spaces and z-score normalization  
    - Initial negative sample injection and optional pool extension
    - Strategy dispatch via pluggable sampling algorithms
    - Post-processing including deduplication and random top-up
    - Distribution validation and visualization capabilities

    Attributes:
        pos_sequences: List of positive peptide sequences.
        neg_sequences: List of final negative sequences (None until sampling).
        sampled_neg_sequences: Subset of negatives from sampling strategy only.
        initial_neg_sequences: User-provided initial negative sequences.
        sampling_pool: Dataframe containing candidate pool with computed properties.
        validator: Distribution validator for quality assessment.

    Examples:
        Basic usage with different sampling strategies:
        
        >>> # Initialize with positive sequences and candidate pool
        >>> pos_seqs = ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"]
        >>> pool_seqs = ["NEGPEP1", "NEGPEP2", "NEGPEP3", ...]
        >>> sampler = NegSampler(pool_seqs, pos_seqs)
        >>> 
        >>> # Sample negatives using KDE strategy
        >>> negatives = sampler.sample_negatives(
        ...     method="kde",
        ...     properties=["length", "charge"],
        ...     ratio=1.0,
        ...     seed=42
        ... )
        >>> 
        >>> # Validate distribution similarity
        >>> all_pass, details, passed = sampler.check_similarity("length.ks_stat<0.2")
        >>> 
        >>> # Visualize distributions
        >>> sampler.visualize_distributions(["length", "charge"])

        Advanced usage with initial negatives:
        
        >>> # Provide initial negatives that will be used first
        >>> initial_negs = ["INITIAL1", "INITIAL2"]
        >>> negatives = sampler.sample_negatives(
        ...     method="mmd",
        ...     properties=["length", "charge", "hydrophobicity"],
        ...     target_count=100,
        ...     initial_negatives=initial_negs,
        ...     extend_pool_with_initial=True,
        ...     rff_dim=512
        ... )
        >>> 
        >>> # Generate comprehensive similarity report
        >>> report = sampler.generate_similarity_report()
        >>> print(report)
    """

    def __init__(
        self,
        sampling_pool_sequences: List[str],
        pos_sequences: List[str]
    ):
        """Initialize NegSampler with candidate pool and positive sequences.

        Args:
            sampling_pool_sequences: List of candidate sequences for negative sampling.
                Properties will be computed automatically for these sequences.
            pos_sequences: List of positive peptide sequences used as reference
                for distribution matching.

        Examples:
            >>> pool_seqs = ["CANDIDATE1", "CANDIDATE2", ...]
            >>> pos_seqs = ["POSITIVE1", "POSITIVE2", ...]
            >>> sampler = NegSampler(pool_seqs, pos_seqs)
        """
        # Create initial pool dataframe and compute properties
        df = pd.DataFrame({"sequence": pd.Series(sampling_pool_sequences, dtype=str)})
        pool = compute_peptide_properties(df["sequence"].astype(str).tolist())
        self.sampling_pool = pool.drop_duplicates("sequence", keep="first").reset_index(drop=True)

        # Store positive sequences
        self.pos_sequences: List[str] = list(map(str, pos_sequences))

        # Initialize result storage (populated during sampling)
        self.neg_sequences: Optional[List[str]] = None
        self.sampled_neg_sequences: Optional[List[str]] = None
        self.initial_neg_sequences: Optional[List[str]] = None

        # Initialize components
        self.registry = SamplerRegistry()
        # Extended samplers are now included in the main registry
        logger.info(f"Available samplers: {self.registry.list_available()}")
        self.validator = DistributionValidator()

    def sample_negatives(
        self,
        method: str = "kde",
        properties: Optional[List[str]] = None,
        ratio: Optional[float] = None,
        target_count: Optional[int] = None,
        seed: int = 42,
        initial_negatives: Optional[List[str]] = None,
        extend_pool_with_initial: bool = True,
        **sampler_kwargs,
    ) -> List[str]:
        """Generate negative samples using specified sampling strategy.

        This is the main interface for negative sampling. It handles the complete
        pipeline from preprocessing to final negative sample generation.

        Args:
            method: Sampling strategy to use. Options:
                - "kde": Kernel Density Estimation importance sampling
                - "mmd": Maximum Mean Discrepancy herding  
                - "nn": Nearest neighbor matching
                - "ot": Optimal transport (Sinkhorn)
                - "moment": Moment matching via ridge regression
                - "bin": Histogram/quantile bin matching
                - "random": Uniform random sampling
            properties: List of peptide properties for distribution matching.
                Defaults to ["length", "charge"] if not specified. Available
                properties include: "length", "charge", "hydrophobicity", 
                "isoelectricpoint", "mw".
            ratio: Desired negative-to-positive ratio. Ignored if target_count
                is specified.
            target_count: Absolute number of negative samples to generate.
                Takes precedence over ratio.
            seed: Random seed for reproducible sampling.
            initial_negatives: Pre-selected negative sequences to include first.
                These are filtered against positive sequences and deduplicated.
            extend_pool_with_initial: Whether to add initial_negatives to the
                candidate pool for later evaluation and visualization.
            **sampler_kwargs: Strategy-specific parameters passed to the sampler.
                Common parameters:
                - weight_clip (kde): Tuple for clipping importance weights
                - rff_dim (mmd): Random Fourier Feature dimension
                - k_per_pos (nn): Neighbors per positive sample
                - epsilon (ot): Entropy regularization parameter
                - l2_reg (moment): Ridge regression regularization
                - n_bins (bin): Number of histogram bins

        Returns:
            List of selected negative sequence strings.

        Raises:
            ValueError: If neither ratio nor target_count is provided, or if
                the specified method is not available.

        Examples:
            Basic sampling with different strategies:
            
            >>> # KDE sampling with 1:1 ratio
            >>> negatives = sampler.sample_negatives(
            ...     method="kde", ratio=1.0, seed=42
            ... )
            >>> 
            >>> # MMD sampling with specific count
            >>> negatives = sampler.sample_negatives(
            ...     method="mmd", target_count=100, 
            ...     rff_dim=1024, seed=42
            ... )
            >>> 
            >>> # Bin matching with multiple properties
            >>> negatives = sampler.sample_negatives(
            ...     method="bin", 
            ...     properties=["length", "charge", "hydrophobicity"],
            ...     ratio=2.0, n_bins=15, seed=42
            ... )

            With initial negatives:
            
            >>> initial_negs = ["PRESET1", "PRESET2"]
            >>> negatives = sampler.sample_negatives(
            ...     method="kde", ratio=1.0,
            ...     initial_negatives=initial_negs,
            ...     extend_pool_with_initial=True,
            ...     seed=42
            ... )
        """
        # Set default properties and compute target count
        props = _default_properties(properties)
        target = self._compute_target(ratio, target_count)
        rng = np.random.default_rng(seed)

        # Handle initial negatives
        pre = self._prepare_initial(initial_negatives, extend_pool_with_initial, props)
        
        if target == 0:
            self.neg_sequences = []
            self.sampled_neg_sequences = []
            return []

        # Check if initial negatives already satisfy target
        if len(pre) >= target:
            self.neg_sequences = pre[:target]
            self.sampled_neg_sequences = []
            self.initial_neg_sequences = self.neg_sequences.copy()
            if len(pre) > target:
                logger.warning(f"Initial negatives ({len(pre)}) exceed target ({target}); truncated.")
            return self.neg_sequences

        # Build sampling context for remaining samples needed
        remaining = target - len(pre)
        context = self._build_context(props)

        # Dispatch to sampling strategy
        sampler = self.registry.create(method)
        picked = sampler.sample(context, target=remaining, rng=rng, **sampler_kwargs)

        # Merge initial and sampled negatives, fill if needed
        out = self._merge_and_fill(pre, picked, props, need=target - len(pre), rng=rng)
        
        # Store results
        self.neg_sequences = out
        self.sampled_neg_sequences = [s for s in out if s not in set(pre)]
        self.initial_neg_sequences = pre
        
        logger.info(
            f"[{method}] Final negatives={len(out)} "
            f"(initial={len(pre)}, sampled={len(self.sampled_neg_sequences)})"
        )
        return out
        
    def apply_kmer_postprocessing(
        self,
        k_list: Optional[List[int]] = None,
        js_thresholds: Optional[Dict[int, float]] = None,
        check_first: bool = True,
        properties: Optional[List[str]] = None,
        **kmer_processor_kwargs
    ) -> List[str]:
        """Apply K-mer distribution post-processing to improve negative sample quality.
        
        This method applies K-mer based post-processing to the current negative samples
        to better match the K-mer distribution of positive samples. It can optionally
        check K-mer similarity first and only apply processing if needed.
        
        Args:
            k_list: List of k-mer sizes to analyze and balance (e.g., [1, 2, 3]).
                Defaults to [1, 2] if not specified.
            js_thresholds: Thresholds for Jensen-Shannon divergence for each k-mer size.
                Defaults to {1: 0.05, 2: 0.08} if not specified.
            check_first: Whether to check K-mer similarity first and only apply
                processing if thresholds are not met. If False, always apply processing.
            properties: Properties used for rebuilding context if needed. Defaults to
                the properties used in the last sampling operation or ["length", "charge"].
            **kmer_processor_kwargs: Additional parameters passed to KmerPostProcessor.
                Common parameters include replacement strategies and selection criteria.
                
        Returns:
            List of processed negative sequence strings. The neg_sequences attribute
            is also updated with the processed results.
            
        Raises:
            ValueError: If no negative sequences are available for processing.
            ImportError: If KmerPostProcessor is not available.
            
        Examples:
            >>> # Apply K-mer post-processing with default settings
            >>> processed_negatives = sampler.apply_kmer_postprocessing()
            >>> 
            >>> # Apply with custom thresholds and always process
            >>> processed_negatives = sampler.apply_kmer_postprocessing(
            ...     k_list=[1, 2, 3],
            ...     js_thresholds={1: 0.03, 2: 0.05, 3: 0.08},
            ...     check_first=False
            ... )
            >>> 
            >>> # Apply with custom KmerPostProcessor parameters
            >>> processed_negatives = sampler.apply_kmer_postprocessing(
            ...     k_list=[1, 2],
            ...     replacement_ratio=0.3,
            ...     max_iterations=50
            ... )
        """
        if not self.neg_sequences:
            raise ValueError("No negative sequences available. Run sample_negatives first.")
            
        # Set defaults
        if k_list is None:
            k_list = [1, 2]
        if js_thresholds is None:
            js_thresholds = {1: 0.05, 2: 0.08}
        if properties is None:
            properties = DEFAULT_PROPERTIES  # ["length", "charge","hydrophobicity"]

        logger.info(f"Starting K-mer post-processing with k_list={k_list}")
        
        # Store original for comparison
        original_negatives = self.neg_sequences.copy()
        
        # Check current K-mer similarity if requested
        should_process = True
        if check_first:
            try:
                from pepbenchmark.neg_sampler.distribution_validator import check_kmer_similarity
                kmer_similar, kmer_metrics = check_kmer_similarity(
                    self.pos_sequences, self.neg_sequences, k_list, js_thresholds
                )
                
                logger.info(f"K-mer similarity check results: {kmer_metrics}")
                
                if kmer_similar:
                    logger.info("K-mer distributions already meet thresholds, skipping post-processing")
                    should_process = False
                    # Still store metrics for reporting
                    if not hasattr(self, '_kmer_metrics'):
                        self._kmer_metrics = {}
                    self._kmer_metrics.update(kmer_metrics)
                else:
                    logger.info("K-mer distributions do not meet thresholds, applying post-processing")
                    
            except Exception as e:
                logger.warning(f"K-mer similarity check failed: {e}, proceeding with post-processing")
                should_process = True
        
        if not should_process:
            return self.neg_sequences
            
        try:
            # Import KmerPostProcessor
            from pepbenchmark.neg_sampler.sampling_strategies import KmerPostProcessor
            
            # Build or rebuild context for the processor
            context = self._build_context(properties)
            
            # Initialize and apply K-mer post-processor
            kmer_processor = KmerPostProcessor()
            
            # Filter out js_thresholds as it's not needed by KmerPostProcessor
            processor_kwargs = {k: v for k, v in kmer_processor_kwargs.items() 
                              if k not in ['js_thresholds']}
            
            processed_negatives = kmer_processor.balance_kmer(
                positives=self.pos_sequences,
                base_negatives=self.neg_sequences,
                pool_df=context.pool_df,
                k_list=k_list,
                **processor_kwargs
            )
            
            # Update stored sequences
            self.neg_sequences = processed_negatives
            
            # Update sampled_neg_sequences if it exists (maintain the initial/sampled distinction)
            if self.sampled_neg_sequences is not None and self.initial_neg_sequences is not None:
                initial_set = set(self.initial_neg_sequences)
                self.sampled_neg_sequences = [s for s in processed_negatives if s not in initial_set]
                
            logger.info(f"K-mer post-processing completed successfully. "
                       f"Sequences: {len(original_negatives)} → {len(processed_negatives)}")
            
            # Store final K-mer metrics for reporting
            try:
                from pepbenchmark.neg_sampler.distribution_validator import check_kmer_similarity
                _, final_metrics = check_kmer_similarity(
                    self.pos_sequences, processed_negatives, k_list, js_thresholds
                )
                if not hasattr(self, '_kmer_metrics'):
                    self._kmer_metrics = {}
                self._kmer_metrics.update(final_metrics)
                logger.info(f"Final K-mer metrics: {final_metrics}")
            except Exception as e:
                logger.warning(f"Failed to compute final K-mer metrics: {e}")
                
            return processed_negatives
            
        except ImportError as e:
            raise ImportError(f"KmerPostProcessor not available: {e}")
        except Exception as e:
            logger.error(f"K-mer post-processing failed: {e}")
            # Restore original sequences on failure
            self.neg_sequences = original_negatives
            raise RuntimeError(f"K-mer post-processing failed: {e}")
        
    def get_distribution_diff(
        self, 
        properties: Optional[List[str]] = None,
        k_list: Optional[List[int]] = None,
        use_sampled_only: bool = False
    ) -> pd.DataFrame:
        """Get detailed distribution comparison including properties and k-mer analysis.

        Args:
            properties: Properties to compare. If None, compares default properties.
            k_list: List of k-mer sizes to analyze (e.g., [1, 2, 3]). If None, no k-mer analysis.
            use_sampled_only: If True, compare only strategy-sampled negatives
                (excluding initial negatives). If False, use all negatives.

        Returns:
            DataFrame with distribution comparison metrics including KS test
            p-values, Jensen-Shannon divergence, mean differences, etc.
            K-mer results are included with names like "1mer_js", "2mer_js".

        Raises:
            ValueError: If no positive or negative sequences are available.

        Examples:
            >>> # Compare properties and k-mers for all negatives
            >>> diff_df = sampler.get_distribution_diff(
            ...     properties=["length", "charge"], k_list=[1, 2]
            ... )
            >>> print(diff_df[["property", "ks_pvalue", "js_divergence"]])
            >>> 
            >>> # Compare only k-mers for sampled negatives only  
            >>> diff_df = sampler.get_distribution_diff(
            ...     k_list=[1, 2, 3], use_sampled_only=True
            ... )
        """
        if not self.pos_sequences:
            raise ValueError("No positive sequences available.")
            
        if use_sampled_only and self.sampled_neg_sequences is not None:
            neg_seqs = self.sampled_neg_sequences
        else:
            if not self.neg_sequences:
                raise ValueError("No negative sequences. Run sampling first.")
            neg_seqs = self.neg_sequences

        return self.validator.get_distribution_diff(
            self.pos_sequences, neg_seqs, properties, k_list
        )

    def check_similarity(
        self,
        conditions: str,
        use_sampled_only: bool = False
    ) -> Tuple[bool, Dict[str, Dict[str, Any]], List[str]]:
        """Check if negative samples satisfy similarity conditions using expression syntax.
        
        Args:
            conditions: Condition expression string like "length.ks_stat<0.2;length.p_value>0.05;1mer_js<0.05"
                Format: "property.metric operator threshold" separated by semicolons
                Supported properties: length, charge, hydrophobicity, etc.
                Supported k-mers: 1mer, 2mer, 3mer, etc. (using format "1mer_js", "2mer_ks_stat")
                Supported metrics: ks_stat, p_value, js_divergence, cohen_d, etc.
                Supported operators: <, <=, >, >=, ==, !=
            use_sampled_only: If True, analyze only strategy-sampled negatives
        
        Returns:
            Tuple of:
            - bool: True if all conditions pass
            - Dict[str, Dict[str, Any]]: Detailed results for each condition
                Format: {"length.p_value": {"value": 0.11, "condition": ">0.05", "successful": True}, ...}
            - List[str]: List of conditions that passed successfully
            
        Examples:
            >>> # Check multiple conditions
            >>> all_pass, details, passed = sampler.check_similarity("length.ks_stat<0.2;charge.p_value>0.05")
            >>> print(f"All passed: {all_pass}")
            >>> print(f"Details: {details}")
            >>> print(f"Passed conditions: {passed}")
            >>> 
            >>> # Include k-mer similarity conditions
            >>> all_pass, details, passed = sampler.check_similarity("length.js_divergence<0.1;1mer_js<0.05;2mer_ks_stat<0.3")
            >>> print(f"Details for length.js_divergence: {details['length.js_divergence']}")
            >>> print(f"Details for 1mer_js: {details['1mer_js']}")
        """
        if not self.pos_sequences:
            raise ValueError("No positive sequences available.")
            
        if use_sampled_only and self.sampled_neg_sequences is not None:
            neg_seqs = self.sampled_neg_sequences
        else:
            if not self.neg_sequences:
                raise ValueError("No negative sequences. Run sampling first.")
            neg_seqs = self.neg_sequences

        return self.validator.check_similarity_expression(
            self.pos_sequences, neg_seqs, conditions
        )

    def get_supported_similarity_conditions(
        self,
        properties: Optional[List[str]] = None,
        k_list: Optional[List[int]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate a condition expression string for check_similarity based on available properties and k-mers.
        
        This method automatically constructs a condition expression that can be used with
        check_similarity() method, based on the properties and k-mer analyses available.
        
        Args:
            properties: List of properties to include in conditions. If None, uses default properties.
            k_list: List of k-mer sizes to include (e.g., [1, 2, 3]). If None, uses [1, 2].
            thresholds: Custom thresholds for different metrics. If None, uses sensible defaults.
                Supported keys:
                - "ks_stat": KS statistic threshold (default: 0.2)
                - "ks_pvalue": KS p-value threshold (default: 0.05) 
                - "js_divergence": Jensen-Shannon divergence threshold (default: 0.1)
                - "mean_diff": Mean difference threshold (default: 3.0)
                - "kmer_js": Default JS threshold for k-mers (default: 0.05)
                - "1mer_js", "2mer_js", etc.: Specific thresholds for each k-mer size
                
        Returns:
            Condition expression string that can be used with check_similarity().
            
        Examples:
            >>> # Get default conditions for basic properties
            >>> conditions = sampler.get_supported_similarity_conditions()
            >>> print(conditions)
            'length.ks_stat<0.2;length.ks_pvalue>0.05;charge.ks_stat<0.2;charge.ks_pvalue>0.05;hydrophobicity.ks_stat<0.2;hydrophobicity.ks_pvalue>0.05;1mer_js<0.05;2mer_js<0.05'
            >>> 
            >>> # Get conditions with custom thresholds
            >>> conditions = sampler.get_supported_similarity_conditions(
            ...     properties=["length", "charge"],
            ...     k_list=[1, 2, 3],
            ...     thresholds={"ks_stat": 0.15, "ks_pvalue": 0.01, "2mer_js": 0.08}
            ... )
            >>> 
            >>> # Use the generated conditions directly
            >>> all_pass, details, passed = sampler.check_similarity(conditions)
        """
        if properties is None:
            properties = DEFAULT_PROPERTIES  # ["length", "charge", "hydrophobicity"]
        if k_list is None:
            k_list = [1, 2]
        if thresholds is None:
            thresholds = {}
            
        # Default thresholds
        default_thresholds = {
            "ks_stat": 0.2,
            "ks_pvalue": 0.05,
            "js_divergence": 0.1,
            "mean_diff": 3.0,
            "kmer_js": 0.05  # default for all k-mers
        }
        
        # Merge with user-provided thresholds
        combined_thresholds = {**default_thresholds, **thresholds}
        
        conditions = []
        
        # Add property-based conditions
        for prop in properties:
            # KS statistic condition (smaller is better)
            ks_threshold = combined_thresholds["ks_stat"]
            conditions.append(f"{prop}.ks_stat<{ks_threshold}")
            
            # KS p-value condition (larger is better)
            pvalue_threshold = combined_thresholds["ks_pvalue"]
            conditions.append(f"{prop}.ks_pvalue>{pvalue_threshold}")
            
            # Optional: JS divergence condition (smaller is better)
            # js_threshold = combined_thresholds["js_divergence"]
            # conditions.append(f"{prop}.js_divergence<{js_threshold}")
        
        # Add k-mer based conditions
        for k in k_list:
            # Check if there's a specific threshold for this k-mer size
            kmer_key = f"{k}mer_js"
            if kmer_key in combined_thresholds:
                threshold = combined_thresholds[kmer_key]
            else:
                threshold = combined_thresholds["kmer_js"]
            conditions.append(f"{k}mer_js<{threshold}")
            
        return ";".join(conditions)

    def check_similarity_auto(
        self,
        properties: Optional[List[str]] = None,
        k_list: Optional[List[int]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        use_sampled_only: bool = False
    ) -> Tuple[bool, Dict[str, Dict[str, Any]], List[str], str]:
        """Convenience method that automatically generates and checks similarity conditions.
        
        This method combines get_supported_similarity_conditions() and check_similarity()
        into a single call for ease of use.
        
        Args:
            properties: List of properties to check. If None, uses default properties.
            k_list: List of k-mer sizes to check (e.g., [1, 2, 3]). If None, uses [1, 2].
            thresholds: Custom thresholds for different metrics. If None, uses sensible defaults.
            use_sampled_only: If True, analyze only strategy-sampled negatives.
            
        Returns:
            Tuple of:
            - bool: True if all conditions pass
            - Dict[str, Dict[str, Any]]: Detailed results for each condition
            - List[str]: List of conditions that passed successfully
            - str: The condition expression that was used
            
        Examples:
            >>> # Check with default settings
            >>> all_pass, details, passed, conditions_used = sampler.check_similarity_auto()
            >>> print(f"All passed: {all_pass}")
            >>> print(f"Conditions used: {conditions_used}")
            >>> 
            >>> # Check with custom settings
            >>> all_pass, details, passed, conditions_used = sampler.check_similarity_auto(
            ...     properties=["length", "charge"],
            ...     k_list=[1, 2, 3],
            ...     thresholds={"ks_stat": 0.15, "2mer_js": 0.08}
            ... )
        """
        # Generate conditions automatically
        conditions = self.get_supported_similarity_conditions(properties, k_list, thresholds)
        
        # Check similarity using the generated conditions
        all_pass, details, passed = self.check_similarity(conditions, use_sampled_only)
        
        return all_pass, details, passed, conditions

    def visualize_distributions(
        self, 
        properties: Optional[List[str]] = None, 
        plot_type: str = "kde", 
        bins: int = 20, 
        use_sampled_only: bool = False
    ):
        """Visualize distribution comparisons between positive and negative samples.

        Args:
            properties: Properties to visualize. Defaults to ["length", "charge"].
            plot_type: Type of plot ("kde", "hist", "box").
            bins: Number of bins for histogram plots.
            use_sampled_only: Whether to visualize only strategy-sampled negatives.

        Returns:
            Matplotlib figure object or None if visualization fails.

        Examples:
            >>> # Visualize default properties with KDE plots
            >>> fig = sampler.visualize_distributions()
            >>> 
            >>> # Visualize specific properties with histograms
            >>> fig = sampler.visualize_distributions(
            ...     properties=["length", "charge", "hydrophobicity"],
            ...     plot_type="hist", bins=30
            ... )
        """
        if not self.pos_sequences:
            logger.error("No positive sequences available.")
            return None
            
        if use_sampled_only and self.sampled_neg_sequences:
            neg_seqs = self.sampled_neg_sequences
        else:
            if not self.neg_sequences:
                logger.error("No negative sequences. Run sampling first.")
                return None
            neg_seqs = self.neg_sequences
            
        if properties is None:
            properties = list(DEFAULT_PROPERTIES)
            
        return self.validator.visualize_distributions(
            self.pos_sequences, neg_seqs, properties, plot_type, bins
        )

    def generate_similarity_report(
        self,
        properties: Optional[List[str]] = None,
        pvalue_threshold: float = 0.05,
        js_threshold: float = 0.1,
        mean_diff_threshold: float = 3.0,
        ks_stat_threshold: float = 0.1,
        quantile_diff_threshold: float = 3.0,
        use_sampled_only: bool = True,
        checks: Optional[List[str]] = None,
    ) -> tuple[str, dict]:
        """Generate a comprehensive similarity assessment report.

        Args:
            properties: Properties to analyze. If None, analyzes all available.
            pvalue_threshold: Minimum p-value for Kolmogorov-Smirnov test.
            js_threshold: Maximum Jensen-Shannon divergence.
            mean_diff_threshold: Maximum z-score for mean difference.
            ks_stat_threshold: Maximum Kolmogorov-Smirnov statistic.
            quantile_diff_threshold: Maximum z-score for quantile differences.
            use_sampled_only: Whether to analyze only strategy-sampled negatives.

        Returns:
            Formatted string report with similarity assessment results.

        Examples:
            >>> # Generate comprehensive report
            >>> report,_ = sampler.generate_similarity_report()
            >>> print(report)
            >>> 
            >>> # Generate report for specific properties only
            >>> report,_ = sampler.generate_similarity_report(
            ...     properties=["length", "charge"], use_sampled_only=True
            ... )
        """
        if not self.pos_sequences:
            raise ValueError("No positive sequences available.")
            
        if use_sampled_only and self.sampled_neg_sequences is not None:
            neg_seqs = self.sampled_neg_sequences
        else:
            if not self.neg_sequences:
                raise ValueError("No negative sequences. Run sampling first.")
            neg_seqs = self.neg_sequences

        if checks is None:
            checks = [
                "ks_pvalue",
                "js_divergence",
                "ks_stat",
            ]

        # Get the base similarity report.
        report_str, report_dict = self.validator.generate_similarity_report(
            self.pos_sequences, neg_seqs, properties,
            pvalue_threshold, js_threshold, mean_diff_threshold,
            ks_stat_threshold, quantile_diff_threshold,
            checks=checks,
        )
        
        # Add k-mer distribution metrics if available.
        if hasattr(self, '_kmer_metrics') and self._kmer_metrics:
            kmer_metrics = self._kmer_metrics
            
            # Append k-mer details to the report string.
            kmer_section = "\n=== K-mer Distribution Analysis ===\n"
            for kmer_name, js_value in kmer_metrics.items():
                if isinstance(js_value, float) and not np.isnan(js_value):
                    kmer_section += f"{kmer_name.replace('_', '-').upper()}: JS divergence = {js_value:.4f}\n"
                else:
                    kmer_section += f"{kmer_name.replace('_', '-').upper()}: JS divergence = N/A\n"
            
            report_str += kmer_section
            
            # Append k-mer details to the report dictionary.
            if isinstance(report_dict, dict):
                report_dict["kmer_analysis"] = kmer_metrics
                
                # Update the summary to include k-mer checks.
                if "summary" in report_dict:
                    summary = report_dict["summary"]
                    # Check whether all k-mers pass the default thresholds.
                    kmer_thresholds = {1: 0.05, 2: 0.08}
                    kmer_pass_checks = []
                    for k in [1, 2]:
                        kmer_key = f"kmer_{k}"
                        if kmer_key in kmer_metrics:
                            js_val = kmer_metrics[kmer_key]
                            threshold = kmer_thresholds.get(k, 0.1)
                            if isinstance(js_val, float) and not np.isnan(js_val):
                                kmer_pass_checks.append(js_val <= threshold)
                            else:
                                kmer_pass_checks.append(False)
                    
                    if kmer_pass_checks:
                        summary["kmer_similar"] = all(kmer_pass_checks)
                        # Update overall similarity to include k-mer checks.
                        original_similar = summary.get("all_similar", False)
                        summary["all_similar"] = original_similar and summary["kmer_similar"]
        
        return report_str, report_dict

    # ===========================
    # Internal Helper Methods
    # ===========================

    def _compute_target(self, ratio: Optional[float], target_count: Optional[int]) -> int:
        """Compute target number of negative samples.
        
        Args:
            ratio: Desired negative-to-positive ratio.
            target_count: Absolute target count.
            
        Returns:
            Target number of negative samples.
            
        Raises:
            ValueError: If neither ratio nor target_count is provided.
        """
        if target_count is not None:
            return int(max(0, target_count))
        if ratio is None:
            raise ValueError("Either `ratio` or `target_count` must be provided.")
        return int(max(0, ratio * len(self.pos_sequences)))

    def _prepare_initial(
        self, 
        init_negs: Optional[List[str]], 
        extend_pool: bool, 
        needed_props: List[str]
    ) -> List[str]:
        """Prepare initial negative sequences by deduplication and filtering.
        
        Args:
            init_negs: Raw initial negative sequences.
            extend_pool: Whether to add to sampling pool.
            needed_props: Properties needed for pool extension.
            
        Returns:
            Deduplicated and filtered initial negatives.
        """
        if not init_negs:
            return []
            
        # Stable deduplication while filtering against positives
        seen, pre = set(), []
        pos_set = set(self.pos_sequences)
        
        for s in init_negs:
            s = str(s)
            if s not in seen and s not in pos_set:
                seen.add(s)
                pre.append(s)
                
        # Optionally extend pool with initial negatives
        if extend_pool and pre:
            self._append_to_pool(pre, needed_props)
            
        return pre

    def _append_to_pool(self, sequences: List[str], needed: List[str]) -> None:
        """Append new sequences to the sampling pool with property computation.
        
        Args:
            sequences: New sequences to add.
            needed: Required properties for validation.
        """

        new_df = compute_peptide_properties(list(map(str, sequences)))
        existing_seqs = set(self.sampling_pool["sequence"])
        new_df = new_df[~new_df["sequence"].isin(existing_seqs)]
        
        if len(new_df) == 0:
            return
            
        # Validate that needed properties are available
        for col in needed:
            if col not in new_df.columns:
                logger.warning(f"Property '{col}' not found in appended sequences.")
                
        # Merge with existing pool and deduplicate
        self.sampling_pool = (
            pd.concat([self.sampling_pool, new_df], ignore_index=True)
            .drop_duplicates("sequence", keep="first")
            .reset_index(drop=True)
        )

    def _build_context(self, properties: List[str]) -> SamplingContext:
        """Build sampling context with aligned and normalized features.
        
        Args:
            properties: List of property names to include.
            
        Returns:
            SamplingContext with processed data ready for sampling.
        """
        # Compute properties for positive sequences
        pos_raw = compute_peptide_properties(self.pos_sequences)
        pos_df = pos_raw[["sequence"] + [p for p in properties if p in pos_raw.columns]]

        # Filter pool to exclude positives and select needed properties
        pool_df = self.sampling_pool[["sequence"] + [p for p in properties if p in self.sampling_pool.columns]]
        pool_df = pool_df.drop_duplicates("sequence")
        pool_df = pool_df[~pool_df["sequence"].isin(set(pos_df["sequence"]))].reset_index(drop=True)

        # Validate that all required properties are available
        _validate_properties(pd.concat([pos_df, pool_df], axis=0).columns, properties)

        # Combine and normalize features with z-scoring
        all_df = pd.concat([pos_df[["sequence"] + properties], pool_df[["sequence"] + properties]], ignore_index=True)
        Z_all, mu, sd = _safe_zscore(all_df, properties)
        
        # Split back into positive and pool features
        n_pos = len(pos_df)
        Z_pos = Z_all.iloc[:n_pos].to_numpy()
        Z_pool = Z_all.iloc[n_pos:].to_numpy()

        return SamplingContext(
            pos_sequences=self.pos_sequences,
            pool_df=pool_df,
            pos_df=pos_df,
            Z_pos=Z_pos,
            Z_pool=Z_pool,
            properties=properties,
        )

    def _merge_and_fill(
        self,
        pre: List[str],
        picked: List[str],
        properties: List[str],
        need: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """Merge initial and sampled negatives, filling remaining slots if needed.
        
        Args:
            pre: Initial negative sequences.
            picked: Strategy-sampled sequences.
            properties: Properties used (for logging).
            need: Number of additional samples needed.
            rng: Random number generator.
            
        Returns:
            Final merged and potentially filled negative sequence list.
        """
        pos_set = set(self.pos_sequences)
        out: List[str] = []
        seen = set()
        
        # Merge while deduplicating and filtering positives
        for s in list(pre) + list(picked):
            s = str(s)
            if s in pos_set:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)

        # Top-up with random samples if needed
        if len(out) < len(pre) + need:
            delta = len(pre) + need - len(out)
            pool_left = self.sampling_pool[
                ~self.sampling_pool["sequence"].isin(set(out) | pos_set)
            ]
            
            if len(pool_left) > 0 and delta > 0:
                add = pool_left.sample(
                    n=min(delta, len(pool_left)), 
                    random_state=int(rng.integers(0, 2**31 - 1))
                )
                out.extend(add["sequence"].astype(str).tolist())

        # Ensure we don't exceed target (defensive programming)
        return out[: len(pre) + need]


class StrategySelector:
    """
    Evaluator that selects the best sampling strategy from validation results, including the experiment loop.

    Usage example
    --------
    selector = StrategySelector(verbose=True)

    # 1) Run with the default strategy set (recommended: use a condition expression)
    best, summary_rows, summary_df, strategy_results = selector.run_and_select(
        sampler,
        properties=["length", "charge"],
        ratio=1,
        seed=123,
        # Use the new condition-expression interface (recommended)
        condition_expression="length.ks_stat<0.2;charge.ks_stat<0.2;length.js_divergence<0.2;1mer_js<0.05",
        # Or use legacy report_kwargs for backward compatibility
        report_kwargs=dict(js_threshold=0.2, ks_stat_threshold=0.2, pvalue_threshold=0.05, use_sampled_only=True),
        # Common arguments passed through to sampler.sample_negatives (optional)
        sample_common_kwargs={}
    )

    # 2) Custom strategy list + condition expression
    strategies = [
        {"method": "kde", "params": {"weight_clip": (0.1, 10.0)}},
        {"method": "mmd", "params": {"rff_dim": 512}},
        {"method": "nn",  "params": {"k_per_pos": 2}},
        {"method": "ot",  "params": {"epsilon": 0.1, "max_iter": 100}},
        {"method": "moment", "params": {"l2_reg": 1e-3}},
        {"method": "bin", "params": {"n_bins": 10}},
        {"method": "random", "params": {}},
    ]
    best, summary_rows, summary_df, strategy_results = selector.run_and_select(
        sampler, 
        properties=["length", "charge"], 
        strategies=strategies,
        condition_expression="length.ks_stat<0.15;charge.p_value>0.1;1mer_js<0.1;2mer_js<0.2"
    )

    # 3) Complex conditions including k-mer analysis
    best, summary_rows, summary_df, strategy_results = selector.run_and_select(
        sampler,
        properties=["length", "charge", "hydrophobicity"],
        k_list=[1, 2, 3],  # Enable k-mer analysis.
        condition_expression="length.ks_stat<0.2;charge.ks_stat<0.2;hydrophobicity.js_divergence<0.15;1mer_js<0.05;2mer_js<0.08;3mer_js<0.1"
    )
    """

    def __init__(
        self,
        prefer: str = "condition_pass",           # Primary criterion: prefer strategies that satisfy the condition expression.
        tie_breakers: Tuple[str, str, str] = ("mean_js", "mean_ks", "mean_abs_mean_diff"),
        verbose: bool = True
    ) -> None:
        self.prefer = prefer
        self.tie_breakers = tie_breakers
        self.verbose = verbose

    # ---------- Default strategy set ----------
    @staticmethod
    def default_strategies() -> List[Dict[str, Any]]:
        return [
            {"method": "kde",   "params": {"weight_clip": (0.1, 10.0)}},
            {"method": "mmd",   "params": {"rff_dim": 512}},
            {"method": "nn",    "params": {"k_per_pos": 2}},
            {"method": "ot",    "params": {"epsilon": 0.1, "max_iter": 100}},
            {"method": "moment","params": {"l2_reg": 1e-3}},
            {"method": "bin",   "params": {"n_bins": 10}},
            {"method": "random","params": {}},
        ]

    # ---------- Utility functions ----------
    @staticmethod
    def _nanmean_safe(values: List[Any]) -> float:
        nums = []
        for v in values:
            if isinstance(v, (int, float, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
                nums.append(float(v))
        return float(np.mean(nums)) if nums else float("nan")

    @staticmethod
    def _fmt_bool(b: Optional[bool]) -> str:
        if b is None:
            return "-"
        return "YES" if b else "NO"

    @staticmethod
    def _fmt_num(x: Any, w: int = 10, p: int = 4) -> str:
        if isinstance(x, (int, float, np.floating)) and not (isinstance(x, float) and np.isnan(x)):
            return f"{float(x):.{p}f}".rjust(w)
        return "-".rjust(w)

    @staticmethod
    def _fmt_pct(x: Any, w: int = 7, p: int = 1) -> str:
        if isinstance(x, (int, float, np.floating)) and not (isinstance(x, float) and np.isnan(x)):
            return f"{100*float(x):.{p}f}%".rjust(w)
        return "-".rjust(w)

    # ---------- Run strategy loop ----------
    def run_strategies(
        self,
        sampler: Any,
        properties: List[str],
        strategies: Optional[List[Dict[str, Any]]] = None,
        *,
        ratio: int = 1,
        seed: int = 123,
        k_list: Optional[List[int]] = None,
        condition_expression: Optional[str] = None,
        sample_common_kwargs: Optional[Dict[str, Any]] = None,
        report_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the strategy loop: sample negatives -> generate a report -> collect results.

        Parameters
        ----
        sampler : Object exposing sample_negatives(...) and generate_similarity_report(...)
        properties : Property list used for sampling and validation
        strategies : Strategy config list (defaults to the built-in set)
        ratio, seed : Passed through to sample_negatives
        k_list : k values for k-mer analysis (for example [1, 2, 3])
        condition_expression : Condition expression string, e.g. "length.ks_stat<0.2;1mer_js<0.05"
        sample_common_kwargs : Extra arguments passed through to sample_negatives
        report_kwargs : Arguments passed through to generate_similarity_report (e.g. use_sampled_only=True)

        Returns
        ----
        strategy_results : dict[method] -> report_dict or error info structure
        """
        if strategies is None:
            strategies = self.default_strategies()
        if sample_common_kwargs is None:
            sample_common_kwargs = {}
        if report_kwargs is None:
            report_kwargs = {}

        strategy_results: Dict[str, Dict[str, Any]] = {}

        logger.info("=== Testing All Sampling Strategies ===")
        print("=== Testing All Sampling Strategies ===")
        for cfg in strategies:
            method = cfg["method"]
            params = cfg.get("params", {})
            logger.info(f"Testing {method.upper()} strategy with params: {params}")
            print(f"\n--- Testing {method.upper()} strategy ---\n")
            try:
                # 1) Sample
                negatives = sampler.sample_negatives(
                    method=method,
                    properties=properties,
                    ratio=ratio,
                    seed=seed,
                    **sample_common_kwargs,
                    **params
                )

                # 2) Generate the report (string + structured dict)
                report_str, report_dict = sampler.generate_similarity_report(
                    properties=properties,
                    **report_kwargs
                )

                # 3) Run extra checks if a condition expression is provided.
                condition_pass = None
                condition_details = None
                passed_conditions = None
                if condition_expression:
                    try:
                        condition_pass, condition_details, passed_conditions = sampler.check_similarity(
                            condition_expression, 
                            use_sampled_only=report_kwargs.get("use_sampled_only", True)
                        )
                        logger.info(f"{method.upper()} condition check: {'PASS' if condition_pass else 'FAIL'}")
                        if passed_conditions:
                            logger.info(f"{method.upper()} passed conditions: {passed_conditions}")
                    except Exception as e:
                        logger.warning(f"{method.upper()} condition check failed: {e}")
                        condition_pass = False
                        condition_details = {}
                        passed_conditions = []

                # Add some useful metadata.
                report_dict = dict(report_dict)
                report_dict["success"] = True
                report_dict["method"] = method
                report_dict["params"] = params
                report_dict["condition_pass"] = condition_pass
                report_dict["condition_details"] = condition_details
                report_dict["passed_conditions"] = passed_conditions
                try:
                    report_dict["n_negatives"] = len(negatives)
                except Exception:
                    report_dict["n_negatives"] = None

                strategy_results[method] = report_dict
                logger.info(f"✓ {method.upper()}: Success - generated {len(negatives)} negatives")
                
            except Exception as e:
                strategy_results[method] = {
                    "success": False,
                    "error": str(e),
                    "params": params,
                    "condition_pass": False
                }
                logger.error(f"✗ {method.upper()}: Failed with error - {e}")
                print(f"✗ {method.upper()}: Failed with error - {e}")

        return strategy_results

    # ---------- Summary ----------
    def build_summary(
        self,
        strategies: List[Dict[str, Any]],
        strategy_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """Summarize raw results into a list and DataFrame for sorting, display, and export."""
        summary_rows: List[Dict[str, Any]] = []

        for s in strategies:
            method = s.get("method")
            res = strategy_results.get(method, {})

            row: Dict[str, Any] = {
                "method": method,
                "success": False,
                "overall_similarity": None,
                "condition_pass": None,
                "n_props": 0,
                "pass_props": 0,
                "pass_rate": float("nan"),
                "mean_js": float("nan"),
                "mean_ks": float("nan"),
                "mean_abs_mean_diff": float("nan"),
                "failed_properties": [],
                "error": None,
                "params": s.get("params", {}),
            }

            # Success path: similarity_summary is present (returned by DistributionValidator.generate_similarity_report).
            if isinstance(res, dict) and res.get("success", False):
                logger.debug(f"Processing {method} results with keys: {list(res.keys())}")
                row["success"] = True
                
                # Get condition-expression check results.
                row["condition_pass"] = res.get("condition_pass")
                
                # Get overall similarity information.
                summary_info = res.get("summary", {})
                row["overall_similarity"] = bool(summary_info.get("all_similar", False))
                logger.debug(f"{method} overall_similarity: {row['overall_similarity']}")
                
                # Get detailed property-level results.
                similarity_summary = res.get("similarity_summary")
                if similarity_summary is not None:
                    # If this is a DataFrame, convert it to a list of dicts.
                    if hasattr(similarity_summary, 'to_dict'):
                        props_data = similarity_summary.to_dict('records')
                    else:
                        props_data = similarity_summary if isinstance(similarity_summary, list) else []
                else:
                    props_data = []
                
                row["n_props"] = len(props_data)
                row["pass_props"] = sum(1 for p in props_data if p.get("is_similar", False))
                row["pass_rate"] = (row["pass_props"] / row["n_props"]) if row["n_props"] > 0 else float("nan")
                logger.debug(f"{method} properties: {row['n_props']}, passed: {row['pass_props']}")
                
                # Get summary statistics from distribution_comparison.
                dist_comparison = res.get("distribution_comparison")
                if dist_comparison is not None and hasattr(dist_comparison, 'to_dict'):
                    dist_data = dist_comparison.to_dict('records')
                    row["mean_js"] = self._nanmean_safe([d.get("js_divergence") for d in dist_data])
                    row["mean_ks"] = self._nanmean_safe([d.get("ks_stat") for d in dist_data])
                    row["mean_abs_mean_diff"] = self._nanmean_safe([
                        abs(d.get("mean_diff")) if isinstance(d.get("mean_diff"), (int, float, np.floating)) else float("nan")
                        for d in dist_data
                    ])
                else:
                    row["mean_js"] = float("nan")
                    row["mean_ks"] = float("nan") 
                    row["mean_abs_mean_diff"] = float("nan")
                
                row["failed_properties"] = [p.get("property") for p in props_data if not p.get("is_similar", False)]
            else:
                # Failure path: record the error information.
                row["success"] = bool(res.get("success", False))
                row["condition_pass"] = res.get("condition_pass", False)
                row["error"] = res.get("error")

            summary_rows.append(row)

        df = pd.DataFrame(summary_rows)
        return summary_rows, df

    # ---------- Select the best ----------
    def _sort_key(self, row: Dict[str, Any]) -> Tuple:
        """Build a comparable sorting key from the current preference settings."""
        def as_num(x, default_nan_dir="worst"):
            # Map NaN to the worst sortable value.
            if isinstance(x, (int, float, np.floating)) and not (isinstance(x, float) and np.isnan(x)):
                return float(x)
            # For pass_rate (higher is better), map NaN to very small; for error-like metrics (lower is better), map NaN to very large.
            if default_nan_dir == "best":
                return -1e18  # Best possible placeholder.
            return 1e18      # Worst possible placeholder.

        # Primary criterion.
        prefer_val = row.get(self.prefer)
        # condition_pass and pass_rate are better when larger; the others are better when smaller.
        if self.prefer in ["condition_pass", "pass_rate"]:
            prefer_key = -as_num(prefer_val, default_nan_dir="best")  # Negate to achieve descending order.
        else:
            prefer_key = as_num(prefer_val)

        # Tie-breaker order: smaller is better for all of them.
        tb_vals = []
        for k in self.tie_breakers:
            tb_vals.append(as_num(row.get(k)))

        # Prefer success=True and n_props>0 to avoid empty results "winning".
        success_key = 0 if row.get("success") and row.get("n_props", 0) > 0 else 1

        return (success_key, prefer_key, *tb_vals)

    def select_best(
        self,
        summary_rows: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Pick the best strategy from the summary, or return None if none succeeded."""
        candidates = [r for r in summary_rows if r.get("success") and r.get("n_props", 0) > 0]
        if not candidates:
            return None
        return sorted(candidates, key=self._sort_key)[0]

    # ---------- Print ----------
    def print_summary(self, summary_rows: List[Dict[str, Any]], best: Optional[Dict[str, Any]] = None) -> None:
        """Print the summary in aligned text format and optionally mark the recommended strategy."""
        logger.info("=== Strategy Results Summary ===")
        print("\n=== Strategy Results Summary ===")
        print(f"Total strategies tested: {len(summary_rows)}")
        logger.info(f"Total strategies tested: {len(summary_rows)}")

        header = (
            f"{'Method':<10} {'OK':<3} {'Overall':<7} {'Condition':<9} "
            f"{'#Props':>6} {'Pass%':>7} {'MeanJS':>10} {'MeanKS':>10} {'Mean|Δμ|':>10}"
        )
        print(header)
        print("-" * len(header))

        successful_count = 0
        for r in summary_rows:
            if r.get("success"):
                successful_count += 1
                
            line = (
                f"{r['method']:<10} "
                f"{self._fmt_bool(r['success']):<3} "
                f"{self._fmt_bool(r['overall_similarity']):<7} "
                f"{self._fmt_bool(r['condition_pass']):<9} "
                f"{str(r['n_props']).rjust(6)} "
                f"{self._fmt_pct(r['pass_rate']):>7} "
                f"{self._fmt_num(r['mean_js']):>10} "
                f"{self._fmt_num(r['mean_ks']):>10} "
                f"{self._fmt_num(r['mean_abs_mean_diff']):>10}"
            )
            if best and r is best:
                line += "   <== RECOMMENDED"
            print(line)
            if not r["success"]:
                if r.get("error"):
                    print(f"    -> Error: {r['error']}")
                else:
                    print("    -> Error: Unknown failure.")
            elif r["failed_properties"]:
                print(f"    -> Failed properties: {', '.join(map(str, r['failed_properties']))}")

        logger.info(f"Strategy evaluation summary: {successful_count}/{len(summary_rows)} strategies successful")

        if best:
            print("\n=== Recommended Strategy (heuristic) ===")
            print(f"- Method: {best['method']}")
            logger.info(f"Recommended strategy: {best['method']}")
            
            if isinstance(best.get("pass_rate"), (int, float, np.floating)):
                pass_rate_str = f"{best['pass_props']}/{best['n_props']} ({best['pass_rate']*100:.1f}%)"
                print(f"- Pass rate: {pass_rate_str}")
                logger.info(f"Recommended strategy pass rate: {pass_rate_str}")
                
            if isinstance(best.get("mean_js"), (int, float, np.floating)):
                print(f"- Mean JS: {best['mean_js']:.4f}")
                logger.info(f"Recommended strategy mean JS divergence: {best['mean_js']:.4f}")
                
            if isinstance(best.get("mean_ks"), (int, float, np.floating)):
                print(f"- Mean KS: {best['mean_ks']:.4f}")
                logger.info(f"Recommended strategy mean KS statistic: {best['mean_ks']:.4f}")
                
            if isinstance(best.get("mean_abs_mean_diff"), (int, float, np.floating)):
                print(f"- Mean |Δμ|: {best['mean_abs_mean_diff']:.4f}")
                logger.info(f"Recommended strategy mean absolute mean difference: {best['mean_abs_mean_diff']:.4f}")
        else:
            print("\nNo successful strategies to recommend.")
            logger.warning("No successful strategies found for recommendation")

    # ---------- One-stop flow: run + summarize + select ----------
    def run_and_select(
        self,
        sampler: Any,
        *,
        properties: List[str],
        strategies: Optional[List[Dict[str, Any]]] = None,
        ratio: int = 1,
        seed: int = 123,
        k_list: Optional[List[int]] = None,
        condition_expression: Optional[str] = None,
        sample_common_kwargs: Optional[Dict[str, Any]] = None,
        report_kwargs: Optional[Dict[str, Any]] = None,
        print_summary: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Run the strategy loop -> build the summary -> select the best -> print optionally.

        Parameters
        ----
        sampler : NegSampler object
        properties : Property list
        strategies : Strategy configuration list
        ratio, seed : Sampling parameters
        k_list : k values for k-mer analysis
        condition_expression : Condition expression, e.g. "length.ks_stat<0.2;1mer_js<0.05"
        sample_common_kwargs : Shared sampling parameters
        report_kwargs : Report parameters
        print_summary : Whether to print the summary

        Returns
        ----
        best_row, summary_rows, summary_df, strategy_results
        """
        if strategies is None:
            strategies = self.default_strategies()

        # Log start information.
        logger.info(f"Starting strategy selection with {len(strategies)} strategies")
        if condition_expression:
            logger.info(f"Using condition expression: {condition_expression}")
        logger.info(f"Properties: {properties}, ratio: {ratio}, seed: {seed}")

        strategy_results = self.run_strategies(
            sampler,
            properties=properties,
            strategies=strategies,
            ratio=ratio,
            seed=seed,
            k_list=k_list,
            condition_expression=condition_expression,
            sample_common_kwargs=sample_common_kwargs,
            report_kwargs=report_kwargs,
        )
        
        # Log strategy execution results.
        successful_strategies = [method for method, result in strategy_results.items() 
                               if result.get("success", False)]
        failed_strategies = [method for method, result in strategy_results.items() 
                           if not result.get("success", False)]
        
        logger.info(f"Strategy execution completed: {len(successful_strategies)} successful, {len(failed_strategies)} failed")
        if successful_strategies:
            logger.info(f"Successful strategies: {', '.join(successful_strategies)}")
        if failed_strategies:
            logger.warning(f"Failed strategies: {', '.join(failed_strategies)}")
        
        summary_rows, summary_df = self.build_summary(strategies, strategy_results)
        best = self.select_best(summary_rows)
        
        # Log selection results.
        if best:
            logger.info(f"Best strategy selected: {best['method']} with pass rate {best.get('pass_rate', 'N/A')}")
            if best.get('failed_properties'):
                logger.warning(f"Best strategy failed properties: {', '.join(best['failed_properties'])}")
        else:
            logger.warning("No successful strategy found for recommendation")
        
        if print_summary:
            self.print_summary(summary_rows, best)
        return best, summary_rows, summary_df, strategy_results


# Convenience function: complete the workflow in one call (optional)
def choose_best_strategy_with_run(
    sampler: Any,
    properties: List[str],
    strategies: Optional[List[Dict[str, Any]]] = None,
    *,
    ratio: int = 1,
    seed: int = 123,
    k_list: Optional[List[int]] = None,
    condition_expression: Optional[str] = None,
    sample_common_kwargs: Optional[Dict[str, Any]] = None,
    report_kwargs: Optional[Dict[str, Any]] = None,
    prefer: str = "condition_pass",
    tie_breakers: Tuple[str, str, str] = ("mean_js", "mean_ks", "mean_abs_mean_diff"),
    verbose: bool = True,
    print_summary: bool = True
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Shortcut entry point that directly runs the strategy loop and selects the best strategy.

    Parameters
    ----
    sampler : NegSampler object
    properties : List of properties
    strategies : Strategy configuration list (optional; all default strategies are used if omitted)
    ratio, seed : Sampling parameters
    k_list : List of k values for k-mer analysis
    condition_expression : Condition expression, e.g. "length.ks_stat<0.2;1mer_js<0.05"
    sample_common_kwargs : Common sampling arguments
    report_kwargs : Report arguments (compatible with the legacy interface)
    prefer : Preferred ranking metric; "condition_pass" is recommended
    tie_breakers : Tie-breaking ranking metrics
    verbose, print_summary : Output controls

    Returns
    ----
    best_strategy, summary_rows, summary_df, strategy_results

    Example
    ----
    # Select the best strategy using a condition expression
    best, _, _, _ = choose_best_strategy_with_run(
        sampler, 
        properties=["length", "charge"],
        condition_expression="length.ks_stat<0.2;charge.ks_stat<0.2;1mer_js<0.05",
        ratio=1, 
        seed=123
    )
    """
    selector = StrategySelector(prefer=prefer, tie_breakers=tie_breakers, verbose=verbose)
    return selector.run_and_select(
        sampler,
        properties=properties,
        strategies=strategies,
        ratio=ratio,
        seed=seed,
        k_list=k_list,
        condition_expression=condition_expression,
        sample_common_kwargs=sample_common_kwargs,
        report_kwargs=report_kwargs,
        print_summary=print_summary
    )

# ===========================
# Example Usage
# ===========================

if __name__ == "__main__":
    """Example usage demonstrating different sampling strategies."""
    
    # Load example dataset
    dataset_name = "bbp"
    pos_seq = read_dataset_sequences(dataset_name)
    
    # Set up sampling pool
    pool_manager = SamplingPoolManager(
        include_datasets=INCLUSIVE_MAP[dataset_name],
        exclude_datasets=EXCLUSIVE_MAP[dataset_name],
    )
    pool_manager.remove_sequences(sequences=pos_seq)

    # Initialize NegSampler
    sampler = NegSampler(pool_manager.get_sampling_pool(), pos_seq)

    print("=== NegSampler Examples ===")
    print(f"Positive samples: {len(pos_seq)}")
    print(f"Pool size: {len(sampler.sampling_pool)}")
    properties = ["length", "charge"]


    # Example 1: KDE sampling
    print("=== Example 1: KDE Sampling ===")
    negatives = sampler.sample_negatives(
        method="kde", 
        properties=properties, 
        ratio=1.0, 
        seed=123
    )
    
    print(f"Generated {len(negatives)} negative samples")
    report_str, report_dict = sampler.generate_similarity_report(properties=properties,js_threshold=0.2,ks_stat_threshold=0.2,pvalue_threshold=0.05,use_sampled_only=True)
    print(report_str)
    print(report_dict)

    # Example 4: Sampling with initial negatives
    print("=== Example 4: KDE with Initial Negatives ===")
    initial_negs = ["GGGGGGG", "AAAAAAA", "CCCCCCC"]
    negatives = sampler.sample_negatives(
        method="kde",
        properties=["length", "charge"],
        ratio=1.0,
        initial_negatives=initial_negs,
        extend_pool_with_initial=True,
        seed=123
    )
    print(f"Generated {len(negatives)} negative samples")
    print(f"Initial negatives used: {len(sampler.initial_neg_sequences or [])}")
    print()



    # You can also omit `strategies` and use `StrategySelector.default_strategies()`.
    strategies = [
        {"method": "kde",    "params": {"weight_clip": (0.1, 10.0)}},
        {"method": "mmd",    "params": {"rff_dim": 512}},
        {"method": "nn",     "params": {"k_per_pos": 2}},
        {"method": "ot",     "params": {"epsilon": 0.1, "max_iter": 100}},
        {"method": "moment", "params": {"l2_reg": 1e-3}},
        {"method": "bin",    "params": {"n_bins": 10}},
        {"method": "random", "params": {}},
    ]

    selector = StrategySelector(prefer="pass_rate", tie_breakers=("mean_js", "mean_ks", "mean_abs_mean_diff"), verbose=True)

    # End-to-end flow: run the strategy loop -> summarize -> choose the best -> print the summary.
    best, summary_rows, summary_df, strategy_results = selector.run_and_select(
        sampler,
        properties=properties,
        strategies=strategies,              # Or omit this to use `selector.default_strategies()`.
        ratio=1,                            # Small ratio for quick testing.
        seed=123,
        sample_common_kwargs={},            # Common arguments passed to `sample_negatives` (optional).
        report_kwargs={                     # Arguments passed to `generate_similarity_report`.
            "js_threshold": 0.2,
            "ks_stat_threshold": 0.2,
            "pvalue_threshold": 0.05,
            "use_sampled_only": True,
        },
        print_summary=True,                 # Control whether to print the summary.
    )

    # Optional: rerun the best strategy or keep it for later use.
    if best:
        print("\n=== Re-running with Recommended Strategy ===")
        best_method = best["method"]
        best_params = best.get("params", {})
        negatives_best = sampler.sample_negatives(
            method=best_method,
            properties=properties,
            ratio=1,
            seed=123,
            **best_params
        )
        print(f"[{best_method}] Generated {len(negatives_best)} negative samples")
        best_report_str, best_report_dict = sampler.generate_similarity_report(
            properties=properties,
            js_threshold=0.2,
            ks_stat_threshold=0.2,
            pvalue_threshold=0.05,
            use_sampled_only=True,
        )
        print(best_report_str)
