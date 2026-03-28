"""Distribution validation module for negative sampling analysis.

This module provides functionality to validate and compare the distribution
properties between positive and negative samples to ensure the quality of
negative sampling results.

Features:
- Comprehensive distribution comparison metrics
- Statistical significance testing
- Jensen-Shannon divergence calculation
- Flexible similarity thresholds
- Detailed reporting and visualization
- High-level validator class for easy usage

The module integrates with the properties and visualization modules to provide
complete analysis and reporting capabilities.

Example:
    >>> from pepbenchmark.analyze.distribution_validator import DistributionValidator
    >>> validator = DistributionValidator()
    >>> pos_sequences = ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"]
    >>> neg_sequences = ["NEGPEP1", "NEGPEP2", "NEGPEP3"]
    >>> is_similar = validator.check_similarity(pos_sequences, neg_sequences, "length")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import math
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from pepbenchmark.analyze.fasta_level import compute_peptide_properties
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


def _require_plotting_dependencies() -> None:
    """Ensure plotting dependencies are available before visualization."""
    missing = []
    if plt is None:
        missing.append("matplotlib")
    if sns is None:
        missing.append("seaborn")
    if missing:
        raise ImportError(
            "Visualization requires optional dependencies: " + ", ".join(missing)
        )


def calculate_ks_critical_value(
    n1: int, 
    n2: int, 
    alpha: float = 0.05
) -> float:
    """Calculate the critical value for two-sample Kolmogorov-Smirnov test.
    
    Uses the asymptotic formula for large samples. For two independent samples
    of sizes n1 and n2, the critical value at significance level alpha is:
    
    D_critical = K_alpha * sqrt((n1 + n2) / (n1 * n2))
    
    where K_alpha is the critical constant depending on alpha:
    - alpha = 0.10: K_alpha ≈ 1.22
    - alpha = 0.05: K_alpha ≈ 1.36  
    - alpha = 0.025: K_alpha ≈ 1.48
    - alpha = 0.01: K_alpha ≈ 1.63
    - alpha = 0.005: K_alpha ≈ 1.73
    - alpha = 0.001: K_alpha ≈ 1.95
    
    Args:
        n1: Sample size of first group
        n2: Sample size of second group  
        alpha: Significance level (default: 0.05)
        
    Returns:
        Critical value for KS test statistic
        
    Examples:
        >>> critical_val = calculate_ks_critical_value(50, 60, alpha=0.05)
        >>> print(f"Critical value: {critical_val:.4f}")
        Critical value: 0.2587
        
    References:
        - Massey Jr, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit.
        - Stephens, M. A. (1974). EDF statistics for goodness of fit and some comparisons.
    """
    # Critical constants for different significance levels
    k_alpha_values = {
        0.10: 1.22,
        0.05: 1.36,
        0.025: 1.48,
        0.01: 1.63,
        0.005: 1.73,
        0.001: 1.95
    }
    
    # Find the closest alpha value or interpolate
    if alpha in k_alpha_values:
        k_alpha = k_alpha_values[alpha]
    else:
        # Find closest values for interpolation
        alpha_keys = sorted(k_alpha_values.keys())
        
        if alpha < alpha_keys[0]:
            # Use the smallest alpha (most conservative)
            k_alpha = k_alpha_values[alpha_keys[-1]]
            logger.warning(f"Alpha {alpha} too small, using K_alpha for {alpha_keys[-1]}")
        elif alpha > alpha_keys[-1]:
            # Use the largest alpha (least conservative) 
            k_alpha = k_alpha_values[alpha_keys[0]]
            logger.warning(f"Alpha {alpha} too large, using K_alpha for {alpha_keys[0]}")
        else:
            # Linear interpolation
            for i in range(len(alpha_keys) - 1):
                if alpha_keys[i] <= alpha <= alpha_keys[i + 1]:
                    alpha1, alpha2 = alpha_keys[i], alpha_keys[i + 1]
                    k1, k2 = k_alpha_values[alpha1], k_alpha_values[alpha2]
                    # Interpolate (note: alpha decreases as k_alpha increases)
                    weight = (alpha - alpha1) / (alpha2 - alpha1)
                    k_alpha = k1 + weight * (k2 - k1)
                    break
    
    # Calculate effective sample size factor
    n_eff_factor = (n1 + n2) / (n1 * n2)
    
    # Calculate critical value
    critical_value = k_alpha * np.sqrt(n_eff_factor)
    
    logger.debug(f"KS critical value calculation: n1={n1}, n2={n2}, alpha={alpha}, "
                f"K_alpha={k_alpha:.3f}, critical_value={critical_value:.4f}")
    
    return float(critical_value)


def interpret_ks_test_result(
    ks_stat: float,
    ks_pvalue: float, 
    ks_critical_005: float,
    ks_critical_001: float,
    property_name: str = "property"
) -> str:
    """Interpret KS test results in human-readable format.
    
    Args:
        ks_stat: KS test statistic value
        ks_pvalue: KS test p-value
        ks_critical_005: Critical value at α=0.05
        ks_critical_001: Critical value at α=0.01
        property_name: Name of the property being tested
        
    Returns:
        Human-readable interpretation string
        
    Examples:
        >>> interpretation = interpret_ks_test_result(0.15, 0.02, 0.2, 0.25)
        >>> print(interpretation)
        
    """
    # Determine significance levels
    significant_005 = ks_stat > ks_critical_005
    significant_001 = ks_stat > ks_critical_001
    
    # Build interpretation
    lines = [
        f"KS test interpretation - {property_name}:",
        f"  Statistic: {ks_stat:.4f}",
        f"  p-value: {ks_pvalue:.6f}",
        f"  Critical value (α=0.05): {ks_critical_005:.4f}",
        f"  Critical value (α=0.01): {ks_critical_001:.4f}",
        ""
    ]
    
    if significant_001:
        lines.extend([
            f"  🔴 Highly significant difference (p < 0.01)",
            f"     KS statistic {ks_stat:.4f} > critical value {ks_critical_001:.4f}",
            f"     Reject the null hypothesis at the 99% confidence level",
            f"     The two {property_name} distributions differ strongly"
        ])
    elif significant_005:
        lines.extend([
            f"  🟡 Significant difference (0.01 < p < 0.05)",
            f"     KS statistic {ks_stat:.4f} > critical value {ks_critical_005:.4f}",
            f"     Reject the null hypothesis at the 95% confidence level",
            f"     The two {property_name} distributions differ significantly"
        ])
    else:
        lines.extend([
            f"  🟢 No significant difference (p > 0.05)",
            f"     KS statistic {ks_stat:.4f} < critical value {ks_critical_005:.4f}",
            f"     Fail to reject the null hypothesis",
            f"     No significant difference is detected between the two {property_name} distributions"
        ])
    
    return "\n".join(lines)



def compare_properties_distribution(
    seqs1: List[str],
    seqs2: List[str],
    properties: Optional[List[str]] = None,
    bins: int = 30,
) -> pd.DataFrame:
    """Compare distribution of selected peptide properties between two groups of sequences.

    Computes comprehensive metrics for each property:
    - mean_diff: Difference of means (group1 - group2)
    - js_divergence: Jensen–Shannon distance between binned distributions
    - ks_stat: Kolmogorov–Smirnov statistic
    - ks_pvalue: Kolmogorov–Smirnov p-value
    - ks_critical_0.05: Critical value for KS test at α=0.05
    - ks_critical_0.01: Critical value for KS test at α=0.01
    - ks_significant_0.05: Whether KS test is significant at α=0.05
    - ks_significant_0.01: Whether KS test is significant at α=0.01
    - q25_diff, q50_diff, q75_diff: Differences in 25th, 50th, 75th percentiles

    Args:
        seqs1: List of sequences in group 1.
        seqs2: List of sequences in group 2.
        properties: List of properties to compare. If None, auto-detect numeric properties.
        bins: Number of bins used to discretize values for JS divergence.

    Returns:
        DataFrame with one row per property and the comparison metrics including
        KS critical values and significance indicators.

    Raises:
        ValueError: If any requested property is invalid or unavailable.
        
    Examples:
        >>> group1 = ["ACDEF", "GHIKL"]
        >>> group2 = ["MNPQR", "STVWY"]
        >>> comparison = compare_properties_distribution(group1, group2)
        >>> print(comparison[['property', 'ks_stat', 'ks_critical_0.05', 'ks_significant_0.05']])
        >>> # Check if distributions are significantly different
        >>> significant_props = comparison[comparison['ks_significant_0.05']]
        >>> print(f"Significantly different properties: {significant_props['property'].tolist()}")
    """
    # Import here to avoid circular imports
    from pepbenchmark.analyze.fasta_level import compute_peptide_properties
    
    # Compute property tables
    df1 = compute_peptide_properties(seqs1)
    df2 = compute_peptide_properties(seqs2)

    # Auto-detect properties if not specified
    if properties is None:
        properties = [
            col for col in df1.columns
            if col != "sequence" and pd.api.types.is_numeric_dtype(df1[col])
        ]

    # Validate requested properties
    missing_cols = [p for p in properties if p not in df1.columns or p not in df2.columns]
    if missing_cols:
        raise ValueError(f"The following properties are missing in input data: {missing_cols}")

    results: List[Dict[str, float]] = []

    for prop in properties:
        # Convert to numeric and drop NaN values
        arr1 = pd.to_numeric(df1[prop], errors="coerce").dropna().to_numpy()
        arr2 = pd.to_numeric(df2[prop], errors="coerce").dropna().to_numpy()

        if arr1.size == 0 or arr2.size == 0:
            logger.warning(f"No valid data for property '{prop}', skipping")
            continue

        # Basic statistics
        mean_diff = float(np.mean(arr1) - np.mean(arr2))
        
        # Quantile differences
        q1_25, q1_50, q1_75 = np.percentile(arr1, [25, 50, 75])
        q2_25, q2_50, q2_75 = np.percentile(arr2, [25, 50, 75])
        
        q25_diff = float(q1_25 - q2_25)
        q50_diff = float(q1_50 - q2_50)
        q75_diff = float(q1_75 - q2_75)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(arr1, arr2)

        # Jensen–Shannon divergence on discretized distributions
        try:
            # Create bins based on the combined range
            combined_data = np.concatenate([arr1, arr2])
            min_val, max_val = combined_data.min(), combined_data.max()
            
            if min_val == max_val:
                # All values are the same
                js_divergence = 0.0
            else:
                # Create bins and compute histograms
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                
                hist1, _ = np.histogram(arr1, bins=bin_edges)
                hist2, _ = np.histogram(arr2, bins=bin_edges)
                
                # Convert to probability distributions
                hist1 = hist1.astype(float)
                hist2 = hist2.astype(float)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                hist1 = hist1 + epsilon
                hist2 = hist2 + epsilon
                
                # Normalize
                hist1 = hist1 / hist1.sum()
                hist2 = hist2 / hist2.sum()
                
                # Calculate Jensen-Shannon divergence
                js_divergence = float(jensenshannon(hist1, hist2))
                
        except Exception as e:
            logger.warning(f"Failed to compute JS divergence for property '{prop}': {e}")
            js_divergence = np.nan

        results.append({
            "property": prop,
            "mean_diff": mean_diff,
            "js_divergence": js_divergence,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "q25_diff": q25_diff,
            "q50_diff": q50_diff,
            "q75_diff": q75_diff,
            "n_group1": len(arr1),
            "n_group2": len(arr2)
        })

    return pd.DataFrame(results)


# ========== Distribution Similarity Assessment ==========

def is_distribution_similar_full(
    result_df: pd.DataFrame,
    property_name: str,
    pvalue_threshold: float = 0.05,
    js_threshold: float = 0.2,
    mean_diff_threshold: float = 3.0,
    ks_stat_threshold: float = 0.1,
    quantile_diff_threshold: float = 1.0,
    checks: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, bool]]:
    """Comprehensive similarity assessment using flexible criteria.

    Evaluates distribution similarity based on multiple statistical measures.
    Default checks: ["ks_pvalue", "js_divergence", "ks_stat"].

    Args:
        result_df: Output DataFrame from compare_properties_distribution().
        property_name: Name of the property to evaluate.
        pvalue_threshold: KS test p-value threshold (similarity if p > threshold).
        js_threshold: Jensen-Shannon divergence threshold (similarity if JS < threshold).
        mean_diff_threshold: Mean difference threshold (absolute value).
        ks_stat_threshold: KS statistic threshold.
        quantile_diff_threshold: Quantile difference threshold (absolute value).
        checks: List of criteria to evaluate. If None, use default checks.

    Returns:
        Tuple of (overall_similarity, individual_results):
        - overall_similarity: Boolean indicating if all checked criteria pass
        - individual_results: Dictionary with results for all available criteria

    Raises:
        ValueError: If property_name not found in result_df or required columns missing.
        
    Examples:
        >>> comparison_df = compare_properties_distribution(group1, group2)
        >>> is_similar, details = is_distribution_similar_full(
        ...     comparison_df, "length", checks=["ks_pvalue", "js_divergence"]
        ... )
        >>> print(f"Similar: {is_similar}")
        >>> print(f"KS p-value check: {details['ks_pvalue']}")
    """
    if checks is None:
        checks = ["ks_pvalue", "js_divergence", "ks_stat"]

    if "property" not in result_df.columns:
        raise ValueError("DataFrame must contain 'property' column")

    if property_name not in result_df["property"].values:
        raise ValueError(f"Property '{property_name}' not found in results")

    row = result_df[result_df["property"] == property_name].iloc[0]
    
    logger.info(f"Checking property '{property_name}' with row data: {row.to_dict()}")
    logger.info(f"Using checks: {checks}")
    logger.info(f"Thresholds: pvalue={pvalue_threshold}, js={js_threshold}, "
               f"mean_diff={mean_diff_threshold}, ks_stat={ks_stat_threshold}, "
               f"quantile_diff={quantile_diff_threshold}")

    # Define all available criteria
    conditions = {
        "ks_pvalue": (
            lambda r: r["ks_pvalue"] > pvalue_threshold,
            lambda r: f"KS p-value {r['ks_pvalue']:.4f} <= {pvalue_threshold}"
        ),
        "js_divergence": (
            lambda r: r["js_divergence"] < js_threshold,
            lambda r: f"JS divergence {r['js_divergence']:.4f} >= {js_threshold}"
        ),
        "mean_diff": (
            lambda r: abs(r["mean_diff"]) < mean_diff_threshold,
            lambda r: f"Mean diff |{r['mean_diff']:.4f}| >= {mean_diff_threshold}"
        ),
        "ks_stat": (
            lambda r: r["ks_stat"] < ks_stat_threshold,
            lambda r: f"KS stat {r['ks_stat']:.4f} >= {ks_stat_threshold}"
        ),
        "quantiles": (
            lambda r: all(abs(r[q]) < quantile_diff_threshold 
                         for q in ["q25_diff", "q50_diff", "q75_diff"]),
            lambda r: (f"Quantile diffs q25={r['q25_diff']:.4f}, "
                       f"q50={r['q50_diff']:.4f}, "
                       f"q75={r['q75_diff']:.4f} >= {quantile_diff_threshold}")
        ),
    }

    # Evaluate all conditions
    all_results: Dict[str, bool] = {}
    for name, (check_func, error_msg_func) in conditions.items():
        try:
            all_results[name] = check_func(row)
        except Exception as e:
            logger.warning(f"Error evaluating condition '{name}': {e}")
            all_results[name] = False

    # Log failed checks that are being evaluated
    failed_checks = []
    for name in checks:
        if name not in all_results:
            logger.warning(f"Check '{name}' not available in conditions")
            continue
        if not all_results[name]:
            failed_checks.append(name)
            try:
                error_msg = conditions[name][1](row)
                logger.info(f"Failed check '{name}': {error_msg}")
            except:
                logger.info(f"Failed check '{name}': condition not met")

    # Overall result based only on specified checks
    all_ok = all(all_results.get(name, False) for name in checks)

    if all_ok:
        logger.info(f"✓ Property '{property_name}' distributions are similar")
    else:
        logger.info(f"✗ Property '{property_name}' distributions differ (failed: {failed_checks})")

    return all_ok, all_results


# ========== K-mer Distribution Validation ==========

def check_kmer_similarity(
    pos_sequences: List[str],
    neg_sequences: List[str],
    k_list: List[int] = [1, 2],
    js_thresholds: Optional[Dict[int, float]] = None
) -> Tuple[bool, Dict[str, float]]:
    """Check k-mer distribution similarity.
    
    Args:
        pos_sequences: Positive sequence list.
        neg_sequences: Negative sequence list.
        k_list: List of k-mer lengths to check.
        js_thresholds: JS divergence threshold for each k.
        
    Returns:
        Tuple of (all_similar, detailed_metrics) where all_similar indicates
        if all k-mers pass threshold checks, and detailed_metrics contains
        JS divergence values for each k.
        
    Examples:
        >>> pos_seqs = ["ACDEF", "GHIKL", "MNPQR"]
        >>> neg_seqs = ["STVWY", "ACGHI", "KLMNP"]
        >>> is_similar, metrics = check_kmer_similarity(pos_seqs, neg_seqs)
        >>> print(f"All k-mers similar: {is_similar}")
        >>> print(f"JS divergences: {metrics}")
    """
    if js_thresholds is None:
        js_thresholds = {1: 0.05, 2: 0.08}
        
    try:
        from pepbenchmark.analyze.kmer_level import KmerAnalyse
        from scipy.spatial.distance import jensenshannon
        import numpy as np
    except ImportError as e:
        logger.error(f"Required modules not available for k-mer analysis: {e}")
        return False, {}
    
    detailed_metrics = {}
    all_similar = True
    
    for k in k_list:
        try:
            # Analyze positive k-mer distribution.
            pos_analyzer = KmerAnalyse(pos_sequences, k=k)
            pos_stats = pos_analyzer.compute_stats()
            
            # Analyze negative k-mer distribution.
            neg_analyzer = KmerAnalyse(neg_sequences, k=k)
            neg_stats = neg_analyzer.compute_stats()
            
            # Get the union of all k-mers.
            all_kmers = set(pos_stats.total_occurrences.keys()) | set(neg_stats.total_occurrences.keys())
            
            if len(all_kmers) == 0:
                detailed_metrics[f"kmer_{k}"] = float('nan')
                continue
            
            # Build frequency vectors.
            pos_freqs = []
            neg_freqs = []
            
            for kmer in sorted(all_kmers):
                pos_count = pos_stats.total_occurrences.get(kmer, 0)
                neg_count = neg_stats.total_occurrences.get(kmer, 0)
                
                pos_freqs.append(pos_count)
                neg_freqs.append(neg_count)
            
            # Normalize to probability distributions.
            pos_total = sum(pos_freqs)
            neg_total = sum(neg_freqs)
            
            if pos_total == 0 or neg_total == 0:
                detailed_metrics[f"kmer_{k}"] = float('nan')
                continue
                
            pos_probs = np.array(pos_freqs) / pos_total
            neg_probs = np.array(neg_freqs) / neg_total
            
            # Compute Jensen-Shannon divergence.
            js_div = jensenshannon(pos_probs, neg_probs) ** 2  # Square to obtain JS divergence.
            detailed_metrics[f"kmer_{k}"] = float(js_div)
            
            # Check whether the threshold is satisfied.
            threshold = js_thresholds.get(k, 0.1)  # Default threshold is 0.1.
            if js_div > threshold:
                all_similar = False
                logger.debug(f"k-mer {k}: JS divergence {js_div:.4f} > threshold {threshold}")
            else:
                logger.debug(f"k-mer {k}: JS divergence {js_div:.4f} <= threshold {threshold}")
                
        except Exception as e:
            logger.warning(f"Failed to compute {k}-mer similarity: {e}")
            detailed_metrics[f"kmer_{k}"] = float('nan')
            all_similar = False
            
    return all_similar, detailed_metrics


# ========== High-Level Validator Class ==========

class DistributionValidator:
    """Validator for checking distribution similarity between positive and negative samples.

    This class provides methods to compare distributions of peptide properties
    between positive and negative samples, ensuring that the negative sampling
    process produces samples with similar distributions to the positive samples.

    The validator supports flexible similarity criteria and provides both
    detailed statistical analysis and high-level similarity assessments.

    Attributes:
        default_properties: Default list of properties to analyze if none specified.

    Examples:
        >>> validator = DistributionValidator()
        >>> pos_sequences = ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"]
        >>> neg_sequences = ["NEGPEP1", "NEGPEP2", "NEGPEP3"]
        >>>
        >>> # Property-wise comparisons
        >>> diff_df = validator.get_distribution_diff(pos_sequences, neg_sequences)
        >>>
        >>> # Check if distributions are similar for a specific property
        >>> is_similar = validator.check_similarity(
        ...     pos_sequences, neg_sequences, "length"
        ... )
        >>>
        >>> # Visualize distributions
        >>> validator.visualize_distributions(
        ...     pos_sequences, neg_sequences, ["length", "charge"]
        ... )
    """

    def __init__(self, default_properties: Optional[List[str]] = None) -> None:
        """Initialize the distribution validator.
        
        Args:
            default_properties: Default properties to analyze. If None, auto-detect.
        """
        self.default_properties = default_properties

    def get_distribution_diff(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        properties: Optional[List[str]] = None,
        k_list: Optional[List[int]] = None,
        bins: int = 30,
    ) -> pd.DataFrame:
        """Get detailed distribution comparison including properties and k-mer analysis.
        
        Args:
            pos_sequences: Positive sample sequences.
            neg_sequences: Negative sample sequences.
            properties: Properties to compare. If None, use default or auto-detect.
            k_list: List of k-mer sizes to analyze (e.g., [1, 2, 3]). If None, no k-mer analysis.
            bins: Number of bins for JS divergence calculation.
            
        Returns:
            DataFrame with comparison metrics for each property and k-mer analysis.
            Columns include: property, mean_diff, js_divergence, ks_stat, ks_pvalue, etc.
            K-mer results are prefixed with "{k}mer_" (e.g., "1mer_js", "2mer_js").
        """
        results = []
        
        # Get property-based comparisons
        if properties is not None:
            if self.default_properties is None and properties is None:
                properties = ["length", "charge"]  # fallback defaults
            elif properties is None:
                properties = self.default_properties
                
            prop_df = compare_properties_distribution(
                pos_sequences, neg_sequences, properties, bins
            )
            results.append(prop_df)
        
        # Get k-mer based comparisons
        if k_list is not None:
            kmer_results = []
            for k in k_list:
                try:
                    is_similar, kmer_metrics = check_kmer_similarity(
                        pos_sequences, neg_sequences, [k]
                    )
                    
                    # Convert k-mer metrics to property-style format
                    for kmer_key, js_value in kmer_metrics.items():
                        if kmer_key == f"kmer_{k}":
                            kmer_results.append({
                                "property": f"{k}mer_js",
                                "mean_diff": float('nan'),  # k-mer doesn't have mean_diff
                                "js_divergence": js_value,
                                "ks_stat": float('nan'),  # k-mer doesn't have ks_stat  
                                "ks_pvalue": float('nan'),  # k-mer doesn't have ks_pvalue
                                "q25_diff": float('nan'),
                                "q50_diff": float('nan'), 
                                "q75_diff": float('nan'),
                                "n_group1": len(pos_sequences),
                                "n_group2": len(neg_sequences)
                            })
                except Exception as e:
                    logger.warning(f"Failed to compute {k}-mer similarity: {e}")
                    kmer_results.append({
                        "property": f"{k}mer_js",
                        "mean_diff": float('nan'),
                        "js_divergence": float('nan'),
                        "ks_stat": float('nan'),
                        "ks_pvalue": float('nan'),
                        "q25_diff": float('nan'),
                        "q50_diff": float('nan'),
                        "q75_diff": float('nan'),
                        "n_group1": len(pos_sequences),
                        "n_group2": len(neg_sequences)
                    })
            
            if kmer_results:
                kmer_df = pd.DataFrame(kmer_results)
                results.append(kmer_df)
        
        # Combine all results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "property", "mean_diff", "js_divergence", "ks_stat", "ks_pvalue",
                "q25_diff", "q50_diff", "q75_diff", "n_group1", "n_group2"
            ])

    def check_similarity_expression(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        conditions: str,
        properties: Optional[List[str]] = None,
        k_list: Optional[List[int]] = None,
        bins: int = 30,
    ) -> Tuple[bool, Dict[str, Dict[str, Any]], List[str]]:
        """Check similarity using condition expressions.
        
        Args:
            pos_sequences: Positive sample sequences.
            neg_sequences: Negative sample sequences.
            conditions: Condition expression string with format:
                "property.metric<threshold;property.metric>threshold;..."
                
                Supported properties: length, charge, hydrophobicity, etc.
                Supported k-mer: 1mer_js, 2mer_js, etc.
                Supported metrics: 
                - js_divergence (or js): Jensen-Shannon divergence
                - ks_stat: Kolmogorov-Smirnov statistic  
                - ks_pvalue (or p_value): KS test p-value
                - mean_diff: Mean difference
                
                Examples:
                - "length.ks_stat<0.2;length.p_value>0.05;1mer_js<0.05"
                - "charge.js<0.1;hydrophobicity.ks_stat<0.15"
                - "2mer_js<0.08;length.mean_diff<2.0"
                
            properties: Properties to analyze. If None, auto-detect from conditions.
            k_list: K-mer sizes to analyze. If None, auto-detect from conditions.
            bins: Number of bins for JS divergence calculation.
            
        Returns:
            Tuple of:
            - bool: True if all conditions pass
            - Dict[str, Dict[str, Any]]: Detailed results for each condition
                Format: {"length.p_value": {"value": 0.11, "condition": ">0.05", "successful": True}, ...}
            - List[str]: List of conditions that passed successfully
            
        Examples:
            >>> all_pass, condition_details, passed_conditions = validator.check_similarity_expression(
            ...     pos_seqs, neg_seqs, 
            ...     "length.ks_stat<0.2;charge.js<0.1;1mer_js<0.05"
            ... )
            >>> print(f"All conditions passed: {all_pass}")
            >>> print(f"Passed conditions: {passed_conditions}")
            >>> print(f"Condition details: {condition_details}")
        """
        # Parse conditions
        parsed_conditions = self._parse_conditions(conditions)
        
        # Auto-detect properties and k_list from conditions if not provided
        if properties is None or k_list is None:
            auto_properties, auto_k_list = self._extract_requirements_from_conditions(parsed_conditions)
            if properties is None:
                properties = auto_properties
            if k_list is None:
                k_list = auto_k_list
        
        # Get distribution data
        result_df = self.get_distribution_diff(
            pos_sequences, neg_sequences, properties, k_list, bins
        )
        
        # Evaluate each condition
        condition_details = {}
        passed_conditions = []
        all_pass = True
        
        for condition_str, (prop_metric, operator, threshold) in parsed_conditions.items():
            try:
                import re
                # Handle k-mer format vs. regular property format
                if re.match(r'\d+mer_[a-zA-Z0-9_]+', prop_metric):
                    # K-mer format: "1mer_js" -> prop_name="1mer_js", metric_name="js_divergence"
                    prop_name = prop_metric
                    condition_key = prop_metric  # Use k-mer format as key (e.g., "1mer_js")
                    # Extract metric from k-mer property name
                    if prop_metric.endswith('_js'):
                        metric_name = 'js_divergence'
                    elif prop_metric.endswith('_ks_stat'):
                        metric_name = 'ks_stat'
                    elif prop_metric.endswith('_p_value') or prop_metric.endswith('_pvalue'):
                        metric_name = 'ks_pvalue'
                    else:
                        # Default to js_divergence for k-mer
                        metric_name = 'js_divergence'
                else:
                    # Regular property format: "length.ks_stat" -> prop_name="length", metric_name="ks_stat"
                    prop_name, metric_name = prop_metric.split('.', 1)
                    condition_key = prop_metric  # Use property.metric format as key (e.g., "length.p_value")
                
                # Handle metric name aliases
                metric_aliases = {
                    'js': 'js_divergence',
                    'p_value': 'ks_pvalue',
                    'pvalue': 'ks_pvalue'
                }
                metric_name = metric_aliases.get(metric_name, metric_name)
                
                # Find matching row in result_df
                matching_rows = result_df[result_df['property'] == prop_name]
                if len(matching_rows) == 0:
                    logger.warning(f"Property '{prop_name}' not found in results")
                    condition_details[condition_key] = {
                        'value': None,
                        'condition': f"{operator}{threshold}",
                        'successful': False,
                        'error': f"Property '{prop_name}' not found"
                    }
                    all_pass = False
                    continue
                
                row = matching_rows.iloc[0]
                
                if metric_name not in row:
                    logger.warning(f"Metric '{metric_name}' not found for property '{prop_name}'")
                    condition_details[condition_key] = {
                        'value': None,
                        'condition': f"{operator}{threshold}",
                        'successful': False,
                        'error': f"Metric '{metric_name}' not found"
                    }
                    all_pass = False
                    continue
                
                value = row[metric_name]
                
                # Handle NaN values
                if pd.isna(value):
                    condition_details[condition_key] = {
                        'value': None,
                        'condition': f"{operator}{threshold}",
                        'successful': False,
                        'error': "Value is NaN"
                    }
                    all_pass = False
                    continue
                
                # Evaluate condition
                if operator == '<':
                    passed = value < threshold
                elif operator == '<=':
                    passed = value <= threshold
                elif operator == '>':
                    passed = value > threshold
                elif operator == '>=':
                    passed = value >= threshold
                elif operator == '==':
                    passed = abs(value - threshold) < 1e-10
                elif operator == '!=':
                    passed = abs(value - threshold) >= 1e-10
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
                
                # Store result with the requested format
                condition_details[condition_key] = {
                    'value': float(value),
                    'condition': f"{operator}{threshold}",
                    'successful': passed
                }
                
                if passed:
                    passed_conditions.append(condition_key)
                else:
                    all_pass = False
                    
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition_str}': {e}")
                condition_details[condition_key] = {
                    'value': None,
                    'condition': f"{operator}{threshold}",
                    'successful': False,
                    'error': f"Error: {str(e)}"
                }
                all_pass = False
        
        return all_pass, condition_details, passed_conditions

    def _parse_conditions(self, conditions: str) -> Dict[str, Tuple[str, str, float]]:
        """Parse condition expression string into structured format.
        
        Args:
            conditions: Condition string like "length.ks_stat<0.2;charge.js>0.1"
            
        Returns:
            Dict mapping condition_string -> (property.metric, operator, threshold)
        """
        import re
        
        parsed = {}
        
        # Split by semicolon
        condition_parts = [c.strip() for c in conditions.split(';') if c.strip()]
        
        for condition in condition_parts:
            # Parse each condition with regex
            # First try standard format: property.metric operator threshold
            pattern1 = r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*([<>=!]+)\s*([0-9\.]+)'
            match1 = re.match(pattern1, condition.strip())
            
            if match1:
                prop, metric, operator, threshold_str = match1.groups()
                prop_metric = f"{prop}.{metric}"
            else:
                # Try k-mer format: Xmer_metric operator threshold (e.g., "1mer_js<0.05")
                pattern2 = r'([0-9]+mer)_([a-zA-Z0-9_]+)\s*([<>=!]+)\s*([0-9\.]+)'
                match2 = re.match(pattern2, condition.strip())
                
                if match2:
                    kmer_type, metric, operator, threshold_str = match2.groups()
                    prop_metric = f"{kmer_type}_{metric}"  # Keep as is for k-mer
                else:
                    raise ValueError(f"Invalid condition format: '{condition}'. Expected format: 'property.metric<threshold' or 'Xmer_metric<threshold'")
            
            try:
                threshold = float(threshold_str)
            except ValueError:
                raise ValueError(f"Invalid threshold value: '{threshold_str}' in condition '{condition}'")
            
            # Validate operator
            if operator not in ['<', '<=', '>', '>=', '==', '!=']:
                raise ValueError(f"Unsupported operator: '{operator}' in condition '{condition}'")
            
            parsed[condition] = (prop_metric, operator, threshold)
        
        return parsed

    def _extract_requirements_from_conditions(self, parsed_conditions: Dict[str, Tuple[str, str, float]]) -> Tuple[List[str], List[int]]:
        """Extract required properties and k-mer sizes from parsed conditions.
        
        Args:
            parsed_conditions: Output from _parse_conditions
            
        Returns:
            Tuple of (properties, k_list)
        """
        import re
        
        properties = []
        k_list = []
        
        for condition_str, (prop_metric, operator, threshold) in parsed_conditions.items():
            # Check if it's a k-mer property (format: "1mer_js", "2mer_ks_stat", etc.)
            kmer_match = re.match(r'(\d+)mer_([a-zA-Z0-9_]+)', prop_metric)
            if kmer_match:
                k_size, metric = kmer_match.groups()
                k = int(k_size)
                if k not in k_list:
                    k_list.append(k)
            else:
                # Regular property format: "property.metric"
                if '.' in prop_metric:
                    prop_name = prop_metric.split('.')[0]
                    if prop_name not in properties:
                        properties.append(prop_name)
                else:
                    logger.warning(f"Unexpected property format: '{prop_metric}'")
        
        return properties, k_list

    def check_similarity_flexible(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        criteria: Union[str, List[str], Dict[str, Dict[str, float]]],
        default_thresholds: Optional[Dict[str, float]] = None,
        default_checks: Optional[List[str]] = None,
        bins: int = 30,
    ) -> Union[bool, Tuple[bool, pd.DataFrame]]:
        """Flexible similarity checking with configurable criteria per property.

        This method provides a unified interface for similarity checking that supports:
        1. Single property checking: criteria="length"
        2. Multiple properties with shared thresholds: criteria=["length", "charge"]  
        3. Per-property custom thresholds: criteria={"length": {"js_divergence": 0.1, "ks_stat": 0.15}}

        Args:
            pos_sequences: Positive sample sequences.
            neg_sequences: Negative sample sequences.
            criteria: Flexible criteria specification:
                - str: Single property name
                - List[str]: Multiple properties with shared thresholds
                - Dict[str, Dict[str, float]]: Per-property custom thresholds
            default_thresholds: Default threshold values for all checks:
                - "pvalue_threshold": KS test p-value (default: 0.05)
                - "js_threshold": Jensen-Shannon divergence (default: 0.2)
                - "mean_diff_threshold": Mean difference (default: 3.0)
                - "ks_stat_threshold": KS statistic (default: 0.1)
                - "quantile_diff_threshold": Quantile difference (default: 1.0)
            default_checks: List of checks to perform if not specified per property.
                Defaults to ["ks_pvalue", "js_divergence", "ks_stat"].
            bins: Number of bins for JS divergence calculation.

        Returns:
            - For single property (str): bool
            - For multiple properties: Tuple[bool, pd.DataFrame] with detailed results

        Examples:
            Single property checking:
            >>> is_similar = validator.check_similarity_flexible(
            ...     pos_seqs, neg_seqs, "length"
            ... )

            Multiple properties with shared thresholds:
            >>> all_similar, details = validator.check_similarity_flexible(
            ...     pos_seqs, neg_seqs, ["length", "charge"],
            ...     default_thresholds={"js_threshold": 0.1, "ks_stat_threshold": 0.15}
            ... )

            Per-property custom thresholds:
            >>> all_similar, details = validator.check_similarity_flexible(
            ...     pos_seqs, neg_seqs, 
            ...     {
            ...         "length": {"js_divergence": 0.1, "ks_stat": 0.15},
            ...         "charge": {"js_divergence": 0.2, "ks_pvalue": 0.01},
            ...         "hydrophobicity": {"mean_diff": 2.0}
            ...     }
            ... )
        """
        # Set default thresholds
        if default_thresholds is None:
            default_thresholds = {
                "pvalue_threshold": 0.05,
                "js_threshold": 0.2,
                "mean_diff_threshold": 3.0,
                "ks_stat_threshold": 0.1,
                "quantile_diff_threshold": 1.0,
            }

        if default_checks is None:
            default_checks = ["ks_pvalue", "js_divergence", "ks_stat"]

        # Normalize criteria to dict format
        if isinstance(criteria, str):
            # Single property
            properties = [criteria]
            property_configs = {criteria: {}}
            return_single = True
        elif isinstance(criteria, list):
            # Multiple properties with shared config
            properties = criteria
            property_configs = {prop: {} for prop in properties}
            return_single = False
        elif isinstance(criteria, dict):
            # Per-property configs
            properties = list(criteria.keys())
            property_configs = criteria
            return_single = False
        else:
            raise ValueError(f"Invalid criteria type: {type(criteria)}. Expected str, list, or dict.")

        # Get distribution comparison data
        result_df = self.get_distribution_diff(
            pos_sequences, neg_sequences, properties, bins
        )

        similarity_results = []

        for prop in properties:
            if prop not in result_df["property"].values:
                logger.warning(f"Property '{prop}' not found in results")
                similarity_results.append({
                    "property": prop,
                    "is_similar": False,
                    "reason": "Property not found",
                    "checks_performed": [],
                    "check_details": {}
                })
                continue

            # Get property-specific config
            prop_config = property_configs[prop]
            
            # Build thresholds for this property (property config overrides defaults)
            prop_thresholds = default_thresholds.copy()
            
            # Map check names to threshold names
            check_to_threshold = {
                "ks_pvalue": "pvalue_threshold",
                "js_divergence": "js_threshold", 
                "mean_diff": "mean_diff_threshold",
                "ks_stat": "ks_stat_threshold",
                "quantiles": "quantile_diff_threshold"
            }
            
            # Apply property-specific overrides
            for check_name, threshold_value in prop_config.items():
                if check_name in check_to_threshold:
                    threshold_key = check_to_threshold[check_name]
                    prop_thresholds[threshold_key] = threshold_value
                elif check_name.endswith("_threshold"):
                    # Direct threshold name override
                    prop_thresholds[check_name] = threshold_value
                else:
                    # Assume it's a check name, try to map it
                    logger.warning(f"Unknown check/threshold name: {check_name}")

            # Determine which checks to perform for this property
            prop_checks = default_checks.copy()
            
            # If property config specifies threshold values, enable those checks
            config_implied_checks = [
                check for check, threshold_key in check_to_threshold.items()
                if check in prop_config or threshold_key in prop_config
            ]
            if config_implied_checks:
                # Use checks implied by the config, but keep defaults if no config
                prop_checks = list(set(prop_checks + config_implied_checks))

            try:
                is_similar, check_details = is_distribution_similar_full(
                    result_df, prop,
                    pvalue_threshold=prop_thresholds["pvalue_threshold"],
                    js_threshold=prop_thresholds["js_threshold"],
                    mean_diff_threshold=prop_thresholds["mean_diff_threshold"],
                    ks_stat_threshold=prop_thresholds["ks_stat_threshold"],
                    quantile_diff_threshold=prop_thresholds["quantile_diff_threshold"],
                    checks=prop_checks
                )

                similarity_results.append({
                    "property": prop,
                    "is_similar": is_similar,
                    "checks_performed": prop_checks,
                    "check_details": check_details,
                    "thresholds_used": {k: v for k, v in prop_thresholds.items()},
                    "reason": "Passed all checks" if is_similar else "Failed one or more checks"
                })

            except Exception as e:
                logger.error(f"Error checking property '{prop}': {e}")
                similarity_results.append({
                    "property": prop,
                    "is_similar": False,
                    "reason": f"Error: {str(e)}",
                    "checks_performed": prop_checks,
                    "check_details": {}
                })

        # Return results
        if return_single:
            # Single property: return just the boolean
            return similarity_results[0]["is_similar"]
        else:
            # Multiple properties: return (all_similar, detailed_df)
            all_similar = all(r["is_similar"] for r in similarity_results)
            
            # Create detailed results DataFrame
            summary_df = pd.DataFrame([
                {
                    "property": r["property"],
                    "is_similar": r["is_similar"],
                    "reason": r["reason"],
                    "checks_performed": ", ".join(r.get("checks_performed", [])),
                    **{f"check_{check}": r.get("check_details", {}).get(check, None) 
                       for check in ["ks_pvalue", "js_divergence", "ks_stat", "mean_diff", "quantiles"]}
                }
                for r in similarity_results
            ])
            
            return all_similar, summary_df

    def visualize_distributions(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        properties: Optional[List[str]] = None,
        plot_type: str = "kde",
        bins: int = 20
    ):
        """Visualize property distributions for positive and negative sequences.
        
        Args:
            pos_sequences: Positive sample sequences.
            neg_sequences: Negative sample sequences.
            properties: Properties to visualize. If None, use default or auto-detect.
            plot_type: Type of plot ("kde", "hist").
            bins: Number of bins for histograms.
            
        Note:
            This method uses the underlying visualize_property_distribution_compare
            function which saves the plot as "property_distribution_compare.png"
            and automatically displays it.
        """
        _require_plotting_dependencies()
        
        if properties is None:
            properties = self.default_properties
            
        visualize_property_distribution_compare(
            pos_sequences, neg_sequences, properties,
            plot_type=plot_type, bins=bins
        )

    def generate_similarity_report(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        properties: Optional[List[str]] = None,
        pvalue_threshold: float = 0.05,
        js_threshold: float = 0.2,
        mean_diff_threshold: float = 3.0,
        ks_stat_threshold: float = 0.1,
        quantile_diff_threshold: float = 1.0,
        checks: Optional[List[str]] = None,
        bins: int = 30,
    ) -> Tuple[str, Dict]:
        """Generate a comprehensive similarity report.
        
        Args:
            pos_sequences: Positive sample sequences.
            neg_sequences: Negative sample sequences.
            properties: Properties to analyze. If None, use default or auto-detect.
            pvalue_threshold: KS test p-value threshold.
            js_threshold: Jensen-Shannon divergence threshold.
            mean_diff_threshold: Mean difference threshold.
            ks_stat_threshold: KS statistic threshold.
            quantile_diff_threshold: Quantile difference threshold.
            checks: List of criteria to check. If None, use defaults.
            bins: Number of bins for JS divergence calculation.
            
        Returns:
            Tuple of (text_report, detailed_data):
            - text_report: Human-readable summary report
            - detailed_data: Dictionary with detailed analysis results
        """
        if properties is None:
            properties = self.default_properties
            
        # Get distribution comparison
        dist_df = self.get_distribution_diff(pos_sequences, neg_sequences, properties, k_list=None, bins=bins)
        
        # Check similarity for each property based on thresholds
        summary_rows = []
        all_similar = True
        
        for _, row in dist_df.iterrows():
            prop_name = row["property"]
            
            # Apply checks for this property
            is_similar = True
            failed_checks = []
            
            if "ks_pvalue" in checks and not pd.isna(row.get("ks_pvalue")):
                if row["ks_pvalue"] < pvalue_threshold:
                    is_similar = False
                    failed_checks.append(f"p-value {row['ks_pvalue']:.4f} < {pvalue_threshold}")
                    
            if "js_divergence" in checks and not pd.isna(row.get("js_divergence")):
                if row["js_divergence"] > js_threshold:
                    is_similar = False
                    failed_checks.append(f"JS divergence {row['js_divergence']:.4f} > {js_threshold}")
                    
            if "ks_stat" in checks and not pd.isna(row.get("ks_stat")):
                if row["ks_stat"] > ks_stat_threshold:
                    is_similar = False
                    failed_checks.append(f"KS stat {row['ks_stat']:.4f} > {ks_stat_threshold}")
            
            summary_rows.append({
                "property": prop_name,
                "is_similar": is_similar,
                "failed_checks": failed_checks,
                "ks_pvalue": row.get("ks_pvalue"),
                "js_divergence": row.get("js_divergence"),
                "ks_stat": row.get("ks_stat"),
                "mean_diff": row.get("mean_diff")
            })
            
            if not is_similar:
                all_similar = False
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Generate text report
        n_pos = len(pos_sequences)
        n_neg = len(neg_sequences)
        n_props = len(properties) if properties else 0
        n_similar = summary_df["is_similar"].sum()
        
        report_lines = [
            "Distribution Similarity Report",
            "=" * 40,
            f"Positive sequences: {n_pos}",
            f"Negative sequences: {n_neg}",
            f"Properties analyzed: {n_props}",
            f"Properties similar: {n_similar}/{n_props}",
            f"Overall similar: {'✓ Yes' if all_similar else '✗ No'}",
            "",
            "Property Details:",
            "-" * 20
        ]
        
        for _, row in summary_df.iterrows():
            status = "✓" if row["is_similar"] else "✗"
            if row["failed_checks"]:
                reason = f" ({'; '.join(row['failed_checks'])})"
            else:
                reason = ""
            report_lines.append(f"{status} {row['property']}{reason}")
        
        if not dist_df.empty:
            report_lines.extend([
                "",
                "Statistical Summary:",
                "-" * 20
            ])
            
            for _, row in dist_df.iterrows():
                report_lines.append(
                    f"{row['property']}: "
                    f"KS p-val={row['ks_pvalue']:.4f}, "
                    f"JS div={row['js_divergence']:.4f}, "
                    f"Mean diff={row['mean_diff']:.4f}"
                )
        
        text_report = "\n".join(report_lines)
        
        # Detailed data
        detailed_data = {
            "summary": {
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_properties": n_props,
                "n_similar": n_similar,
                "all_similar": all_similar
            },
            "distribution_comparison": dist_df,
            "similarity_summary": summary_df,
            "thresholds": {
                "pvalue_threshold": pvalue_threshold,
                "js_threshold": js_threshold,
                "mean_diff_threshold": mean_diff_threshold,
                "ks_stat_threshold": ks_stat_threshold,
                "quantile_diff_threshold": quantile_diff_threshold
            }
        }
        
        return text_report, detailed_data

    def check_kmer_similarity(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        k_list: List[int] = [1, 2],
        js_thresholds: Optional[Dict[int, float]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """Convenience wrapper for checking k-mer distribution similarity.
        
        Args:
            pos_sequences: Positive sequence list.
            neg_sequences: Negative sequence list.
            k_list: List of k-mer lengths to check.
            js_thresholds: JS divergence threshold for each k.
            
        Returns:
            Tuple of (all_similar, detailed_metrics)
            
        Examples:
            >>> validator = DistributionValidator()
            >>> pos_seqs = ["ACDEF", "GHIKL"]
            >>> neg_seqs = ["STVWY", "ACGHI"]
            >>> is_similar, metrics = validator.check_kmer_similarity(pos_seqs, neg_seqs)
        """
        return check_kmer_similarity(pos_sequences, neg_sequences, k_list, js_thresholds)

    def interpret_ks_results(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str],
        properties: Optional[List[str]] = None,
        bins: int = 30,
        verbose: bool = True
    ) -> Dict[str, str]:
        """Get interpreted KS test results with critical value analysis.
        
        Args:
            pos_sequences: Positive sample sequences
            neg_sequences: Negative sample sequences  
            properties: Properties to analyze
            bins: Number of bins for JS divergence calculation
            verbose: Whether to print results to console
            
        Returns:
            Dictionary mapping property names to interpretation strings
            
        Examples:
            >>> validator = DistributionValidator()
            >>> interpretations = validator.interpret_ks_results(pos_seqs, neg_seqs, ["length"])
            >>> print(interpretations["length"])
        """
        if properties is None:
            properties = self.default_properties or ["length", "charge"]
            
        # Get distribution comparison with critical values
        result_df = self.get_distribution_diff(
            pos_sequences, neg_sequences, properties, bins=bins
        )
        
        interpretations = {}
        
        for _, row in result_df.iterrows():
            prop = row["property"]
            
            # Skip k-mer properties (they don't have KS test results)
            if prop.endswith('_js') or prop.endswith('_ks_stat'):
                continue
                
            if pd.isna(row.get("ks_stat")):
                interpretations[prop] = f"KS test result - {prop}: insufficient data to compute"
                continue
                
            interpretation = interpret_ks_test_result(
                ks_stat=row["ks_stat"],
                ks_pvalue=row["ks_pvalue"],
                ks_critical_005=row["ks_critical_0.05"], 
                ks_critical_001=row["ks_critical_0.01"],
                property_name=prop
            )
            
            interpretations[prop] = interpretation
            
            if verbose:
                print(interpretation)
                print("-" * 60)
                print()
        
        return interpretations

    def check_similarity_with_critical_values(
        self,
        pos_sequences: List[str],
        neg_sequences: List[str], 
        properties: Optional[List[str]] = None,
        alpha_level: float = 0.05,
        bins: int = 30
    ) -> Tuple[bool, pd.DataFrame, Dict[str, str]]:
        """Check similarity using KS critical values at specified significance level.
        
        Args:
            pos_sequences: Positive sample sequences
            neg_sequences: Negative sample sequences
            properties: Properties to analyze  
            alpha_level: Significance level (0.05 or 0.01)
            bins: Number of bins for JS divergence calculation
            
        Returns:
            Tuple of:
            - bool: True if all properties pass (no significant differences)
            - pd.DataFrame: Detailed results
            - Dict[str, str]: Interpretations for each property
            
        Examples:
            >>> validator = DistributionValidator()
            >>> all_similar, results, interpretations = validator.check_similarity_with_critical_values(
            ...     pos_seqs, neg_seqs, ["length", "charge"], alpha_level=0.05
            ... )
            >>> print(f"All similar: {all_similar}")
        """
        if properties is None:
            properties = self.default_properties or ["length", "charge"]
            
        if alpha_level not in [0.05, 0.01]:
            raise ValueError("alpha_level must be 0.05 or 0.01")
            
        # Get distribution comparison with critical values
        result_df = self.get_distribution_diff(
            pos_sequences, neg_sequences, properties, bins=bins
        )
        
        # Filter to property-based results (exclude k-mer results)
        prop_results = result_df[~result_df['property'].str.contains('mer_', na=False)]
        
        # Check significance at specified level
        critical_col = f"ks_critical_{alpha_level:0.2f}".replace(".", ".")
        significant_col = f"ks_significant_{alpha_level:0.2f}".replace(".", ".")
        
        if critical_col not in prop_results.columns:
            # Fallback to 0.05 if exact column not found
            critical_col = "ks_critical_0.05"
            significant_col = "ks_significant_0.05"
        
        # All properties are similar if none show significant differences
        all_similar = not prop_results[significant_col].any()
        
        # Generate interpretations
        interpretations = {}
        for _, row in prop_results.iterrows():
            prop = row["property"]
            interpretation = interpret_ks_test_result(
                ks_stat=row["ks_stat"],
                ks_pvalue=row["ks_pvalue"],
                ks_critical_005=row["ks_critical_0.05"],
                ks_critical_001=row["ks_critical_0.01"],
                property_name=prop
            )
            interpretations[prop] = interpretation
        
        return all_similar, result_df, interpretations


# ========== Example Usage ==========

if __name__ == "__main__":
    print("=== Distribution Validator Demo ===\n")
    
    # Test sequences
    pos_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNPQRST",
        "MNPQRSTVWY",
        "ACDEFG",
        "GHIKLMNPQRSTVWY"
    ]
    
    neg_sequences = [
        "ACDEFGHIKLMNPQR",  # Similar to positive
        "MNPQRSTVWYACDEF",  # Similar length but different composition
        "GHIKLMN",          # Shorter
        "PQRSTVWY",         # Different composition
        "ACDEFGHIKL"        # Similar to positive
    ]
    
    print("1. Basic distribution comparison:")
    print("-" * 40)
    comparison = compare_properties_distribution(
        pos_sequences, neg_sequences, properties=["length", "charge"]
    )
    print(comparison[["property", "ks_pvalue", "js_divergence", "mean_diff"]])
    
    print("\n2. Similarity assessment:")
    print("-" * 40)
    validator = DistributionValidator()
    
    for prop in ["length", "charge"]:
        is_similar = validator.check_similarity(pos_sequences, neg_sequences, prop)
        print(f"  {prop}: {'✓ Similar' if is_similar else '✗ Different'}")
    
    print("\n3. Multiple properties check:")
    print("-" * 40)
    all_similar, summary = validator.check_multiple_properties(
        pos_sequences, neg_sequences, ["length", "charge", "hydrophobicity"]
    )
    print(f"All properties similar: {'✓ Yes' if all_similar else '✗ No'}")
    print(summary[["property", "is_similar"]])
    
    print("\n4. Comprehensive report:")
    print("-" * 40)
    report, data = validator.generate_similarity_report(
        pos_sequences, neg_sequences, ["length", "charge"]
    )
    print(report)
    
    print("\n✓ Distribution validator demo completed successfully!")



def visualize_property_distribution_compare(
    seqs1: list[str],
    seqs2: list[str],
    properties: Optional[list[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
    logger=None,
) -> None:
    """
    Visualize property distributions between two groups of sequences for comparison.
    Args:
        seqs1: List of peptide sequences (group 1).
        seqs2: List of peptide sequences (group 2).
        properties: List of property names to visualize. If None, auto-detect.
        plot_type: 'hist' for histogram, 'kde' for density plot.
        bins: Number of bins for histogram.
        logger: Optional logger for info output.
    """
    _require_plotting_dependencies()
    df1 = compute_peptide_properties(seqs1,properties)
    df2 = compute_peptide_properties(seqs2,properties)
    if properties is None:
        properties = [
            col
            for col in df1.columns
            if col != "sequence" and pd.api.types.is_numeric_dtype(df1[col])
        ]
    n_props = len(properties)
    n_cols = 3
    n_rows = math.ceil(n_props / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, prop in enumerate(properties):
        ax = axes[i]
        data1 = df1[prop].dropna().to_numpy()
        data2 = df2[prop].dropna().to_numpy()
        if plot_type == "hist":
            sns.histplot(
                data1,
                bins=bins,
                color="blue",
                label="Group 1",
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
            sns.histplot(
                data2,
                bins=bins,
                color="red",
                label="Group 2",
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
        elif plot_type == "kde":
            sns.kdeplot(data1, color="blue", label="Group 1", ax=ax)
            sns.kdeplot(data2, color="red", label="Group 2", ax=ax)
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. Choose 'hist' or 'kde'."
            )
        ax.set_title(f"{prop} distribution")
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig("property_distribution_compare.png")
    plt.show()




def visualize_property_distribution_single(
    seqs: list[str],
    properties: Optional[list[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
    logger=None,
) -> None:
    """
    Visualize property distributions for a single group of sequences.
    Args:
        seqs: List of peptide sequences.
        properties: List of property names to visualize. If None, auto-detect.
        plot_type: 'hist' for histogram, 'kde' for density plot.
        bins: Number of bins for histogram.
        logger: Optional logger for info output.
    """
    _require_plotting_dependencies()
    df = compute_peptide_properties(seqs, properties)
    if properties is None:
        properties = [
            col
            for col in df.columns
            if col != "sequence" and pd.api.types.is_numeric_dtype(df[col])
        ]
    n_props = len(properties)
    n_cols = 3
    n_rows = math.ceil(n_props / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, prop in enumerate(properties):
        ax = axes[i]
        data = df[prop].dropna().to_numpy()
        if plot_type == "hist":
            sns.histplot(
                data,
                bins=bins,
                color="blue",
                kde=False,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
            )
        elif plot_type == "kde":
            sns.kdeplot(data, color="blue", ax=ax)
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. Choose 'hist' or 'kde'."
            )
        ax.set_title(f"{prop} distribution")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()
