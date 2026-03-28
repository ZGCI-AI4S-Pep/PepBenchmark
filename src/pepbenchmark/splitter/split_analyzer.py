

# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split Analyzer for benchmarking and evaluating dataset split strategies.

This module provides the SplitAnalyzer class for comprehensive analysis and
benchmarking of different dataset split strategies, including:
- Class distribution analysis
- Cross-split similarity analysis  
- Data leakage detection
- Multi-strategy benchmarking with recommendations
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import warnings
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency, fisher_exact

from .split_validation import (
    analyze_split_class_distribution,
    analyze_cross_dataset_similarity,
    detect_potential_data_leakage,
    print_split_class_distribution_summary,
    print_cross_dataset_similarity_summary,
    print_data_leakage_summary
)


class SplitAnalyzer:
    """Comprehensive analyzer for dataset split strategies.
    
    This class provides tools for evaluating split quality, detecting potential
    data leakage, and benchmarking multiple split strategies.
    
    Args:
        sequences: List of sequence strings or DataFrame with sequence column.
        labels: List of binary labels (0/1) for classification tasks.
        embeddings: Optional numpy array of sequence embeddings.
        sequence_col: Column name for sequences if input is DataFrame.
    """
    
    def __init__(
        self,
        sequences: Union[List[str], pd.DataFrame],
        labels: Optional[List[int]] = None,
        embeddings: Optional[np.ndarray] = None,
        sequence_col: str = "sequence"
    ):
        # Extract sequences if DataFrame input
        if isinstance(sequences, pd.DataFrame):
            if sequence_col not in sequences.columns:
                raise ValueError(f"Column '{sequence_col}' not found in DataFrame")
            self.sequences = sequences[sequence_col].dropna().astype(str).tolist()
            
            # Auto-extract labels if available and not provided
            if labels is None and "label" in sequences.columns:
                self.labels = sequences["label"].dropna().astype(int).tolist()
            else:
                self.labels = labels
        else:
            self.sequences = list(sequences)
            self.labels = labels
            
        self.embeddings = embeddings
        self.sequence_col = sequence_col
        
        # Validate inputs
        if self.labels is not None and len(self.sequences) != len(self.labels):
            raise ValueError("Length mismatch between sequences and labels")
            
        if self.embeddings is not None and len(self.sequences) != len(self.embeddings):
            raise ValueError("Length mismatch between sequences and embeddings")

    def analyze_split_class_distribution(
        self,
        split_indices: Dict[str, List[int]],
        fold_name: str = "Single_Fold",
        verbose: bool = True
    ) -> pd.DataFrame:
        """Analyze class distribution across dataset splits.
        
        Args:
            split_indices: Dictionary with train/valid/test indices.
            fold_name: Name identifier for this fold.
            verbose: Whether to print formatted summary.
            
        Returns:
            DataFrame with class distribution statistics.
        """
        df = analyze_split_class_distribution(split_indices, self.labels, fold_name)
        
        if verbose and not df.empty:
            print_split_class_distribution_summary(split_indices, self.labels, fold_name)
            
        return df

    def analyze_cross_split_similarity(
        self,
        split_indices: Dict[str, List[int]],
        split_pairs: Optional[List[Tuple[str, str]]] = None,
        similarity_metric: str = "sliding_window",
        processes: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Analyze similarity between different splits.
        
        Args:
            split_indices: Dictionary with train/valid/test indices.
            split_pairs: List of split pairs to analyze, e.g. [("train", "valid"), ("train", "test")].
                        If None, analyzes cross-split pairs (excludes self-comparisons).
            similarity_metric: Similarity metric to use.
            processes: Number of parallel processes.
            verbose: Whether to print formatted summaries.
            
        Returns:
            DataFrame with split pairs as index and similarity metrics as columns.
        """
        # Get sequences for each split
        split_sequences = {}
        for split_name, indices in split_indices.items():
            if len(indices) > 0:  # Use len() to avoid ambiguity with arrays
                split_sequences[split_name] = [self.sequences[i] for i in indices]
        
        # Determine which pairs to analyze
        if split_pairs is None:
            # Default: analyze cross-split pairs only (exclude self-comparisons)
            available_splits = list(split_sequences.keys())
            pairs_to_analyze = []
            for i, split1 in enumerate(available_splits):
                for j, split2 in enumerate(available_splits):
                    if i < j:  # Only cross-split pairs, avoid self-comparisons and duplicates
                        pairs_to_analyze.append((split1, split2))
        else:
            # Filter out self-comparisons automatically
            pairs_to_analyze = [(s1, s2) for s1, s2 in split_pairs if s1 != s2]
        
        if verbose and pairs_to_analyze:
            print(f"🔍 Will analyze {len(pairs_to_analyze)} cross-split pairs: {pairs_to_analyze}")
        elif verbose:
            print("⚠️  No cross-split pairs to analyze")
        
        # Collect results for DataFrame
        df_data = []
        
        # Analyze specified split pairs
        for split1, split2 in pairs_to_analyze:
            if split1 in split_sequences and split2 in split_sequences:
                similarity_df = analyze_cross_dataset_similarity(
                    split_sequences[split1],
                    split_sequences[split2],
                    dataset_name1=split1,
                    dataset_name2=split2,
                    similarity_metric=similarity_metric,
                    processes=processes
                )
                
                if not similarity_df.empty:
                    # Extract key metrics from the similarity analysis
                    pair_key = f"{split1}-{split2}"
                    
                    # Select the correct row based on split order
                    # analyze_cross_dataset_similarity returns 2 rows:
                    # - Row 0: split1 vs split2 (split1 sequences analyzed against split2)
                    # - Row 1: split2 vs split1 (split2 sequences analyzed against split1)  
                    # We want the first row (split1 vs split2)
                    target_row = similarity_df.iloc[0]
                    
                    df_data.append({
                        "split_pair": pair_key,
                        "avg_max_similarity": target_row["avg_max_similarity"],
                        "max_similarity": target_row["max_similarity"], 
                        "min_similarity": target_row["min_similarity"],
                        "high_similarity_count": target_row["high_similarity_count"],
                        "high_similarity_ratio": target_row["high_similarity_ratio"],
                        "total_samples": target_row["total_samples"]
                    })
                
                if verbose and not similarity_df.empty:
                    target_row = similarity_df.iloc[0] 
                    print(f"✅ Analyzed {split1} vs {split2}: "
                          f"max_sim={target_row['max_similarity']:.4f}, "
                          f"high_sim_ratio={target_row['high_similarity_ratio']:.2%}")
            else:
                missing_splits = []
                if split1 not in split_sequences:
                    missing_splits.append(split1)
                if split2 not in split_sequences:
                    missing_splits.append(split2)
                
                if verbose:
                    print(f"Warning: Skipping pair ({split1}, {split2}) - missing splits: {missing_splits}")
        
        # Create and return DataFrame
        if df_data:
            result_df = pd.DataFrame(df_data)
            result_df.set_index("split_pair", inplace=True)
            return result_df
        else:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=[
                "avg_max_similarity", "max_similarity", "min_similarity",
                "high_similarity_count", "high_similarity_ratio", "total_samples"
            ])

    def detect_data_leakage(
        self,
        split_indices: Dict[str, List[int]],
        similarity_threshold: float = 0.9,
        similarity_metric: str = "sliding_window",
        processes: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """Detect potential data leakage between splits.
        
        Args:
            split_indices: Dictionary with train/valid/test indices.
            similarity_threshold: Threshold for considering sequences too similar.
            similarity_metric: Similarity metric to use.
            processes: Number of parallel processes.
            verbose: Whether to print formatted summaries.
            
        Returns:
            Dictionary mapping split pairs to leakage detection results.
        """
        results = {}
        
        # Get sequences for each split
        split_sequences = {}
        for split_name in ["train", "valid", "test"]:
            if split_name in split_indices:
                indices = split_indices[split_name]
                split_sequences[split_name] = [self.sequences[i] for i in indices]
        
        # Check critical pairs for leakage
        critical_pairs = [("train", "test"), ("train", "valid"), ("valid", "test")]
        
        for split1, split2 in critical_pairs:
            if split1 in split_sequences and split2 in split_sequences:
                pair_key = f"{split1}_vs_{split2}"
                
                leakage_info = detect_potential_data_leakage(
                    split_sequences[split1],
                    split_sequences[split2],
                    similarity_threshold=similarity_threshold,
                    similarity_metric=similarity_metric,
                    processes=processes
                )
                
                results[pair_key] = leakage_info
                
                if verbose:
                    print_data_leakage_summary(
                        split_sequences[split1],
                        split_sequences[split2],
                        similarity_threshold, similarity_metric, processes
                    )
        
        return results

    def evaluate_split_quality(
        self,
        split_indices: Dict[str, List[int]],
        similarity_metric: str = "sliding_window",
        similarity_threshold: float = 0.9,
        processes: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of split quality.
        
        Args:
            split_indices: Dictionary with train/valid/test indices.
            similarity_metric: Similarity metric to use.
            similarity_threshold: Threshold for leakage detection.
            processes: Number of parallel processes.
            weights: Custom weights for scoring components.
            
        Returns:
            Dictionary containing quality metrics and overall score.
        """
        # Default weights for scoring
        default_weights = {
            "class_imbalance": 1.0,
            "similarity_gap": 1.0,
            "leakage_penalty": 2.0,
            "high_similarity_penalty": 1.5
        }
        
        if weights is not None:
            default_weights.update(weights)
        
        results = {"weights": default_weights}
        
        # 1. Class distribution analysis
        if self.labels is not None:
            dist_df = self.analyze_split_class_distribution(split_indices, verbose=False)
            if not dist_df.empty:
                pos_ratios = {row["split"]: row["positive_ratio"] for _, row in dist_df.iterrows()}
                class_imbalance = max(pos_ratios.values()) - min(pos_ratios.values()) if len(pos_ratios) > 1 else 0
                results["class_imbalance"] = class_imbalance
                results["class_distribution"] = pos_ratios
            else:
                results["class_imbalance"] = 0
                results["class_distribution"] = {}
        else:
            results["class_imbalance"] = 0
            results["class_distribution"] = {}
        
        # 2. Cross-split similarity analysis
        cross_similarities_df = self.analyze_cross_split_similarity(
            split_indices, split_pairs=None, similarity_metric=similarity_metric, 
            processes=processes, verbose=False
        )
        
        similarity_metrics = {}
        if not cross_similarities_df.empty:
            for pair_key, row in cross_similarities_df.iterrows():
                similarity_metrics[pair_key.replace("-", "_vs_")] = {
                    "avg_similarity": row["avg_max_similarity"],
                    "max_similarity": row["max_similarity"],
                    "high_similarity_ratio": row["high_similarity_ratio"]
                }
        
        results["cross_split_similarities"] = similarity_metrics
        
        # Calculate similarity gap (difference between train-valid and train-test)
        train_valid_sim = similarity_metrics.get("train_vs_valid", {}).get("avg_similarity", 0)
        train_test_sim = similarity_metrics.get("train_vs_test", {}).get("avg_similarity", 0)
        similarity_gap = abs(train_valid_sim - train_test_sim)
        results["similarity_gap"] = similarity_gap
        
        # 3. Data leakage detection
        leakage_results = self.detect_data_leakage(
            split_indices, similarity_threshold, similarity_metric, processes, verbose=False
        )
        
        results["leakage_detection"] = leakage_results
        
        # Extract key leakage metrics
        train_test_leakage = leakage_results.get("train_vs_test", {}).get("leakage_ratio", 0)
        max_leakage_similarity = max(
            [info.get("max_similarity", 0) for info in leakage_results.values()]
        )
        
        # 4. Calculate overall quality score (lower is better)
        penalty_score = (
            default_weights["class_imbalance"] * results["class_imbalance"] +
            default_weights["similarity_gap"] * similarity_gap +
            default_weights["leakage_penalty"] * train_test_leakage +
            default_weights["high_similarity_penalty"] * max(0, max_leakage_similarity - similarity_threshold)
        )
        
        # Convert to quality score (higher is better)
        quality_score = max(0, 1.0 - penalty_score)
        results["quality_score"] = quality_score
        results["penalty_score"] = penalty_score
        
        return results
        
    def analyze_kmer_enrichment(
        self,
        split_indices: Dict[str, List[int]],
        k: int = 3,
        min_frequency: int = 5,
        enrichment_threshold: float = 2.0,
        statistical_test: str = "chi2",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze k-mer enrichment across different splits to detect potential bias.
        
        This method identifies k-mers that are significantly over-represented or 
        under-represented in specific splits, which could indicate data leakage
        or biased splitting.
        
        Args:
            split_indices: Dictionary with train/valid/test indices.
            k: K-mer length (default: 3).
            min_frequency: Minimum frequency for k-mer to be considered (default: 5).
            enrichment_threshold: Fold-change threshold for enrichment (default: 2.0).
            statistical_test: Statistical test to use ("chi2" or "fisher", default: "chi2").
            verbose: Whether to print formatted summary (default: True).
            
        Returns:
            Dictionary containing k-mer enrichment analysis results.
        """
        if verbose:
            print(f"🔍 Analyzing {k}-mer enrichment across splits...")
            print(f"   Parameters: min_freq={min_frequency}, enrichment_threshold={enrichment_threshold}x")
        
        # Extract sequences for each split
        split_sequences = {}
        split_sizes = {}
        for split_name, indices in split_indices.items():
            if len(indices) > 0:
                split_sequences[split_name] = [self.sequences[i] for i in indices]
                split_sizes[split_name] = len(indices)
        
        # Generate k-mers for each split
        split_kmer_counts = {}
        all_kmers = set()
        
        for split_name, sequences in split_sequences.items():
            kmer_counter = Counter()
            
            for seq in sequences:
                # Extract all k-mers from sequence
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i+k]
                    kmer_counter[kmer] += 1
                    all_kmers.add(kmer)
            
            split_kmer_counts[split_name] = kmer_counter
        
        # Filter k-mers by minimum frequency across all splits
        frequent_kmers = set()
        for kmer in all_kmers:
            total_count = sum(split_kmer_counts[split].get(kmer, 0) 
                            for split in split_kmer_counts)
            if total_count >= min_frequency:
                frequent_kmers.add(kmer)
        
        if verbose:
            print(f"   Found {len(all_kmers)} unique {k}-mers, {len(frequent_kmers)} meet frequency threshold")
        
        # Calculate enrichment statistics
        enrichment_results = []
        significant_kmers = defaultdict(list)
        
        for kmer in frequent_kmers:
            kmer_data = {"kmer": kmer}
            
            # Get counts for each split
            counts = {}
            total_count = 0
            for split_name in split_kmer_counts:
                count = split_kmer_counts[split_name].get(kmer, 0)
                counts[split_name] = count
                total_count += count
                kmer_data[f"{split_name}_count"] = count
            
            kmer_data["total_count"] = total_count
            
            # Calculate frequencies and enrichment ratios
            expected_ratios = {}
            observed_ratios = {}
            enrichment_ratios = {}
            
            for split_name in split_kmer_counts:
                # Expected ratio based on split size
                expected_ratio = split_sizes[split_name] / sum(split_sizes.values())
                expected_ratios[split_name] = expected_ratio
                
                # Observed ratio
                observed_ratio = counts[split_name] / total_count if total_count > 0 else 0
                observed_ratios[split_name] = observed_ratio
                
                # Enrichment ratio (observed/expected)
                enrichment_ratio = (observed_ratio / expected_ratio) if expected_ratio > 0 else 0
                enrichment_ratios[split_name] = enrichment_ratio
                
                kmer_data[f"{split_name}_expected_ratio"] = expected_ratio
                kmer_data[f"{split_name}_observed_ratio"] = observed_ratio
                kmer_data[f"{split_name}_enrichment_ratio"] = enrichment_ratio
            
            # Statistical significance testing
            if statistical_test == "chi2" and len(split_kmer_counts) > 1:
                try:
                    # Prepare contingency table
                    observed = [counts[split] for split in split_kmer_counts]
                    # Expected based on split sizes
                    total_other_kmers = [split_sizes[split] * 20 - counts[split] 
                                       for split in split_kmer_counts]  # Approximate
                    
                    contingency_table = [observed, total_other_kmers]
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    kmer_data["chi2_stat"] = chi2_stat
                    kmer_data["p_value"] = p_value
                    kmer_data["significant"] = p_value < 0.05
                    
                except (ValueError, ZeroDivisionError):
                    kmer_data["chi2_stat"] = 0
                    kmer_data["p_value"] = 1.0
                    kmer_data["significant"] = False
            
            # Check for enrichment
            max_enrichment = max(enrichment_ratios.values())
            min_enrichment = min(enrichment_ratios.values())
            
            kmer_data["max_enrichment"] = max_enrichment
            kmer_data["min_enrichment"] = min_enrichment
            kmer_data["enriched"] = max_enrichment >= enrichment_threshold
            kmer_data["depleted"] = min_enrichment <= (1.0 / enrichment_threshold)
            
            # Identify which split has the enrichment
            if max_enrichment >= enrichment_threshold:
                enriched_split = max(enrichment_ratios, key=enrichment_ratios.get)
                kmer_data["enriched_in"] = enriched_split
                significant_kmers[enriched_split].append(kmer)
            
            enrichment_results.append(kmer_data)
        
        # Sort results by maximum enrichment ratio
        enrichment_results.sort(key=lambda x: x["max_enrichment"], reverse=True)
        
        # Create summary statistics
        summary = {
            "total_kmers_analyzed": len(frequent_kmers),
            "enriched_kmers": sum(1 for r in enrichment_results if r["enriched"]),
            "depleted_kmers": sum(1 for r in enrichment_results if r["depleted"]),
            "significant_kmers": sum(1 for r in enrichment_results if r.get("significant", False)),
            "split_kmer_counts": {split: len(kmers) for split, kmers in significant_kmers.items()},
            "max_enrichment_ratio": max((r["max_enrichment"] for r in enrichment_results), default=0),
            "min_enrichment_ratio": min((r["min_enrichment"] for r in enrichment_results), default=1)
        }
        
        if verbose:
            print(f"\n📊 K-mer Enrichment Analysis Results:")
            print(f"   Total {k}-mers analyzed: {summary['total_kmers_analyzed']}")
            print(f"   Enriched k-mers (>{enrichment_threshold}x): {summary['enriched_kmers']}")
            print(f"   Depleted k-mers (<{1/enrichment_threshold:.2f}x): {summary['depleted_kmers']}")
            if statistical_test == "chi2":
                print(f"   Statistically significant: {summary['significant_kmers']}")
            print(f"   Max enrichment ratio: {summary['max_enrichment_ratio']:.2f}x")
            
            # Show top enriched k-mers
            top_enriched = [r for r in enrichment_results[:5] if r["enriched"]]
            if top_enriched:
                print(f"\n🔝 Top enriched {k}-mers:")
                for result in top_enriched:
                    enriched_in = result.get("enriched_in", "unknown")
                    max_enrich = result["max_enrichment"]
                    total_count = result["total_count"]
                    p_val = result.get("p_value", "N/A")
                    print(f"     {result['kmer']}: {max_enrich:.2f}x in {enriched_in} "
                          f"(count={total_count}, p={p_val:.3f})" if isinstance(p_val, float) 
                          else f"     {result['kmer']}: {max_enrich:.2f}x in {enriched_in} (count={total_count})")
        
        return {
            "summary": summary,
            "enrichment_results": enrichment_results,
            "split_significant_kmers": dict(significant_kmers),
            "parameters": {
                "k": k,
                "min_frequency": min_frequency,
                "enrichment_threshold": enrichment_threshold,
                "statistical_test": statistical_test
            }
        }

    def analyze_split_statistics(
        self,
        split_result: Dict[str, List[int]],
        total_clusters: int = 0,
        total_sequences: int = 0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze detailed statistics about the split.
        
        Args:
            split_result: Dictionary with train/valid/test indices
            total_clusters: Total number of clusters (if clustering-based split)
            total_sequences: Total number of sequences
            verbose: Whether to log detailed statistics
            
        Returns:
            Dictionary containing split statistics
        """
        total_seqs = sum(len(indices) for indices in split_result.values())
        
        # Basic split statistics
        split_stats = {}
        for split_name, indices in split_result.items():
            actual_count = len(indices)
            percentage = actual_count / total_seqs * 100 if total_seqs > 0 else 0
            split_stats[split_name] = {
                "count": actual_count,
                "percentage": percentage
            }
        
        # Data consistency checks
        consistency_checks = {
            "all_sequences_accounted": total_seqs == (total_sequences or len(self.sequences)),
            "no_overlapping_indices": self._check_no_overlaps(split_result),
            "valid_index_ranges": self._check_valid_indices(split_result, total_sequences or len(self.sequences))
        }
        
        results = {
            "split_distribution": split_stats,
            "total_sequences_in_splits": total_seqs,
            "total_sequences_expected": total_sequences or len(self.sequences),
            "total_clusters": total_clusters,
            "consistency_checks": consistency_checks,
            "data_integrity": all(consistency_checks.values())
        }
        
        if verbose:
            self._log_split_statistics_verbose(results)
        
        return results
    
    def _check_no_overlaps(self, split_result: Dict[str, List[int]]) -> bool:
        """Check if there are overlapping indices between splits."""
        all_indices = []
        for indices in split_result.values():
            all_indices.extend(indices)
        
        return len(set(all_indices)) == len(all_indices)
    
    def _check_valid_indices(self, split_result: Dict[str, List[int]], data_size: int) -> bool:
        """Check if all indices are within valid range."""
        for indices in split_result.values():
            if any(i < 0 or i >= data_size for i in indices):
                return False
        return True
    
    def _log_split_statistics_verbose(self, results: Dict[str, Any]) -> None:
        """Log detailed split statistics."""
        print("=" * 80)
        print("📊 SPLIT RESULTS - DETAILED STATISTICS")
        print("=" * 80)
        
        # Basic split statistics
        print(f"📈 Split Distribution:")
        for split_name, stats in results["split_distribution"].items():
            count = stats["count"]
            percentage = stats["percentage"]
            print(f"   • {split_name.capitalize():>10}: {count:>6} sequences ({percentage:>5.1f}%)")
        
        # Summary statistics
        print(f"")
        print(f"🎯 Summary:")
        if results["total_clusters"] > 0:
            print(f"   • Total clusters processed: {results['total_clusters']}")
        print(f"   • Total sequences processed: {results['total_sequences_expected']}")
        print(f"   • Total sequences in splits: {results['total_sequences_in_splits']}")
        
        # Data consistency checks
        checks = results["consistency_checks"]
        print(f"")
        print(f"🔍 Data Integrity Checks:")
        
        if checks["all_sequences_accounted"]:
            print(f"   ✅ All sequences accounted for")
        else:
            print(f"   ⚠️  Sequence count mismatch")
        
        if checks["no_overlapping_indices"]:
            print(f"   ✅ No overlapping indices between splits")
        else:
            print(f"   ⚠️  Overlapping indices found")
        
        if checks["valid_index_ranges"]:
            print(f"   ✅ All indices within valid range")
        else:
            print(f"   ⚠️  Index out of bounds detected")
        
        overall_status = "✅ PASSED" if results["data_integrity"] else "❌ FAILED"
        print(f"   Overall integrity: {overall_status}")
        
        print("=" * 80)
        if results["data_integrity"]:
            print("✅ SPLIT EXECUTION COMPLETED SUCCESSFULLY")
        else:
            print("⚠️  SPLIT EXECUTION COMPLETED WITH WARNINGS")
        print("=" * 80)

    def log_label_distribution_analysis(
        self,
        split_result: Dict[str, List[int]],
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Log label distribution statistics using built-in analyzer.
        
        Args:
            split_result: Dictionary with train/valid/test indices
            verbose: Whether to print detailed statistics
            
        Returns:
            DataFrame with label distribution analysis, or None if no labels
        """
        if self.labels is None:
            if verbose:
                print("⚠️  No labels provided - skipping label distribution analysis")
            return None
        
        try:
            if verbose:
                print("")
                print("=" * 80)
                print("📊 LABEL DISTRIBUTION ANALYSIS")
                print("=" * 80)
            
            # Use built-in analyze_split_class_distribution with verbose output
            df = self.analyze_split_class_distribution(
                split_indices=split_result,
                fold_name="Split_Analysis",
                verbose=verbose
            )
            
            if verbose:
                print("=" * 80)
            
            return df
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to analyze label distribution: {e}")
            return None

    def log_split_statistics_detailed(
        self,
        split_result: Dict[str, List[int]],
        total_clusters: int = 0,
        total_sequences: int = 0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Log detailed statistics about the split with comprehensive analysis.
        
        Args:
            split_result: Dictionary with train/valid/test indices
            total_clusters: Total number of clusters (if clustering-based split)
            total_sequences: Total number of sequences
            verbose: Whether to log detailed statistics
            
        Returns:
            Dictionary containing split statistics
        """
        results = self.analyze_split_statistics(
            split_result=split_result,
            total_clusters=total_clusters,
            total_sequences=total_sequences,
            verbose=verbose,
        )
        return results
