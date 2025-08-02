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

"""Dataset Statistics Analysis Module.

This module provides comprehensive statistics analysis for peptide datasets.
It extracts various statistics from combine.csv files including sample counts,
positive/negative ratios, sequence length ranges, and similarity analysis.

Example:
    >>> from pepbenchmark.analyze.dataset_stats import ClassDataAnalyzer, RegreDataAnalyzer
    >>>
    >>> # Initialize analyzer for a classification dataset
    >>> analyzer = ClassDataAnalyzer('BBP_APML')
    >>>
    >>> # Get comprehensive statistics
    >>> stats = analyzer.get_dataset_statistics()
    >>>
    >>> # Print statistics
    >>> print(f"Total samples: {stats['total_samples']}")
    >>> print(f"Positive ratio: {stats['positive_ratio']:.3f}")
    >>> print(f"Sequence length range: {stats['sequence_length_range']}")
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pepbenchmark.analyze.properties import (
    calculate_properties,
    compare_properties_distribution,
)
from pepbenchmark.analyze.similarity import (
    get_max_similarity_per_sample,
    sliding_window_aar,
)
from pepbenchmark.analyze.visualization import (
    plot_similarity_distribution,
    visualize_aas_distribution_compare,
    visualize_property_distribution_compare,
    visualize_regression_labels,
    visualize_top_kmer_frequency,
)
from pepbenchmark.raw_data import DATASET_MAP
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class BaseDataAnalyzer:
    """Base class for dataset analysis with common functionality."""

    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        similarity_threshold: float = 0.9,
        processes: Optional[int] = None,
        enable_visualization: bool = False,
        save_plots: bool = False,
        plot_save_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.similarity_threshold = similarity_threshold
        self.processes = processes
        self.enable_visualization = enable_visualization
        self.save_plots = save_plots
        self.plot_save_dir = plot_save_dir
        self.type = DATASET_MAP.get(dataset_name, {}).get("type", None)
        # Validate dataset name
        if dataset_name not in DATASET_MAP:
            available_datasets = list(DATASET_MAP.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_MAP. "
                f"Available datasets: {available_datasets}"
            )

        # Get dataset metadata
        self.metadata = DATASET_MAP[dataset_name]

        # Use base_data_dir if provided, otherwise use the path from metadata
        if not base_data_dir:
            self.dataset_path = DATASET_MAP.get(dataset_name, {}).get("path", None)
        else:
            self.dataset_path = base_data_dir

        # Validate dataset path
        if self.dataset_path is None:
            raise ValueError(f"Dataset path is None for dataset: {dataset_name}")

        self._raw_data = self.load_raw_data()

        logger.info(f"Initialized analyzer for dataset: {dataset_name}")
        logger.info(f"Dataset path: {self.dataset_path}")

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from combine.csv or processed_data.csv file.

        Returns:
            DataFrame containing sequence and label columns
        """

        if self.type == "binary_classification":
            combine_csv_path = os.path.join(self.dataset_path, "combine.csv")
        else:
            combine_csv_path = os.path.join(self.dataset_path, "processed_data.csv")

        if not os.path.exists(combine_csv_path):
            raise FileNotFoundError(
                f"combine.csv not found at {combine_csv_path}. "
                f"Please ensure the dataset path is correct."
            )

        logger.info(f"Loading raw data from {combine_csv_path}")

        # Load CSV file
        raw_data = pd.read_csv(combine_csv_path)

        # Validate required columns
        required_columns = ["sequence", "label"]
        missing_columns = [
            col for col in required_columns if col not in raw_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in combine.csv: {missing_columns}. "
                f"Available columns: {list(raw_data.columns)}"
            )

        logger.info(f"Loaded {len(raw_data)} samples")
        logger.info(f"Data shape: {raw_data.shape}")

        return raw_data

    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic dataset statistics.

        Returns:
            Dictionary containing basic statistics
        """

        # Basic counts
        total_samples = len(self._raw_data)

        # Sequence length statistics
        sequence_lengths = self._raw_data["sequence"].str.len()
        min_length = sequence_lengths.min()
        max_length = sequence_lengths.max()
        mean_length = sequence_lengths.mean()
        median_length = sequence_lengths.median()

        return {
            "total_samples": total_samples,
            "sequence_length_range": (int(min_length), int(max_length)),
            "sequence_length_mean": float(mean_length),
            "sequence_length_median": float(median_length),
            "sequence_length_std": float(sequence_lengths.std()),
        }


class ClassDataAnalyzer(BaseDataAnalyzer):
    """Classification dataset analyzer for binary classification tasks."""

    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        similarity_threshold: float = 0.9,
        processes: Optional[int] = None,
        enable_visualization: bool = False,
        save_plots: bool = False,
        plot_save_dir: Optional[str] = None,
    ):
        super().__init__(
            dataset_name,
            base_data_dir,
            similarity_threshold,
            processes,
            enable_visualization,
            save_plots,
            plot_save_dir,
        )

        dataset_type = DATASET_MAP[dataset_name]["type"]
        if not dataset_type == "binary_classification":
            raise ValueError(
                f"Dataset {dataset_name} is not a binary classification dataset. This analyzer is designed for binary classification datasets."
            )

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data and separate positive/negative sequences."""
        raw_data = super().load_raw_data()

        # Separate positive and negative sequences
        positive_data = raw_data[raw_data["label"] == 1]
        negative_data = raw_data[raw_data["label"] == 0]

        self._positive_sequences = positive_data["sequence"].tolist()
        self._negative_sequences = negative_data["sequence"].tolist()

        logger.info(
            f"Separated {len(self._positive_sequences) if self._positive_sequences else 0} positive and {len(self._negative_sequences) if self._negative_sequences else 0} negative sequences"
        )
        return raw_data

    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic dataset statistics for classification."""
        basic_stats = super().calculate_basic_statistics()

        # Add classification-specific statistics
        positive_samples = (
            len(self._positive_sequences) if self._positive_sequences else 0
        )
        negative_samples = (
            len(self._negative_sequences) if self._negative_sequences else 0
        )
        total_samples = basic_stats["total_samples"]

        positive_ratio = positive_samples / total_samples if total_samples > 0 else 0.0

        return {
            **basic_stats,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "positive_ratio": positive_ratio,
        }

    def pos_similarity_statistics(self) -> Dict[str, Any]:
        """Calculate similarity statistics within positive samples."""

        logger.info(
            f"Calculating similarity statistics for {len(self._positive_sequences)} positive samples"
        )

        # Calculate similarity statistics for positive samples only
        positive_similarity_stats = self._calculate_group_similarity(
            self._positive_sequences, "positive"
        )

        return {
            "positive_similarity": positive_similarity_stats,
        }

    def _calculate_group_similarity(
        self, sequences: List[str], group_name: str
    ) -> Dict[str, Any]:
        """Calculate similarity statistics for a group of sequences."""
        if len(sequences) < 2:
            logger.warning(
                f"Not enough sequences in {group_name} group for similarity analysis"
            )
            return {
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "similarity_std": 0.0,
                "similarity_threshold": self.similarity_threshold,
                "high_similarity_ratio": 0.0,
            }

        logger.info(
            f"Calculating similarity for {group_name} group with {len(sequences)} sequences"
        )

        # Calculate similarity for each pair
        max_similarities = get_max_similarity_per_sample(
            sequences, smi_function=sliding_window_aar, processes=self.processes
        )

        # Extract similarity values
        similarities = np.array(max_similarities)

        # Store similarities for visualization
        if group_name == "positive":
            self._positive_similarities = similarities.tolist()
        elif group_name == "negative":
            self._negative_similarities = similarities.tolist()

        # Calculate statistics
        average_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))
        min_similarity = float(np.min(similarities))
        similarity_std = float(np.std(similarities))

        # Calculate ratio of pairs above threshold
        high_similarity_count = np.sum(similarities > self.similarity_threshold)
        high_similarity_ratio = (
            high_similarity_count / len(similarities) if len(similarities) > 0 else 0.0
        )

        logger.info(f"{group_name} group similarity analysis completed")

        return {
            "average_similarity": average_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "similarity_std": similarity_std,
            "high_similarity_ratio": float(high_similarity_ratio),
            "similarity_threshold": self.similarity_threshold,
        }

    def calculate_property_statistics(self) -> Dict[str, Any]:
        """Calculate property statistics for positive and negative samples."""

        logger.info(
            f"Calculating property statistics for {len(self._positive_sequences)} positive and {len(self._negative_sequences)} negative samples"
        )

        # Calculate properties for both groups
        positive_properties = calculate_properties(self._positive_sequences)
        negative_properties = calculate_properties(self._negative_sequences)

        # Calculate property comparison metrics
        property_comparison = compare_properties_distribution(
            self._positive_sequences, self._negative_sequences
        )

        # Calculate basic statistics for each property
        property_stats = {}
        for col in positive_properties.columns:
            if col not in ["sequence", "length"] and pd.api.types.is_numeric_dtype(
                positive_properties[col]
            ):
                pos_stats = positive_properties[col].describe()
                neg_stats = negative_properties[col].describe()

                property_stats[col] = {
                    "positive": {
                        "count": int(pos_stats.get("count", 0) or 0),
                        "mean": float(pos_stats.get("mean", 0.0) or 0.0),
                        "std": float(pos_stats.get("std", 0.0) or 0.0),
                        "min": float(pos_stats.get("min", 0.0) or 0.0),
                        "25%": float(pos_stats.get("25%", 0.0) or 0.0),
                        "50%": float(pos_stats.get("50%", 0.0) or 0.0),
                        "75%": float(pos_stats.get("75%", 0.0) or 0.0),
                        "max": float(pos_stats.get("max", 0.0) or 0.0),
                    },
                    "negative": {
                        "count": int(neg_stats.get("count", 0) or 0),
                        "mean": float(neg_stats.get("mean", 0.0) or 0.0),
                        "std": float(neg_stats.get("std", 0.0) or 0.0),
                        "min": float(neg_stats.get("min", 0.0) or 0.0),
                        "25%": float(neg_stats.get("25%", 0.0) or 0.0),
                        "50%": float(neg_stats.get("50%", 0.0) or 0.0),
                        "75%": float(neg_stats.get("75%", 0.0) or 0.0),
                        "max": float(neg_stats.get("max", 0.0) or 0.0),
                    },
                }

        return {
            "property_statistics": property_stats,
            "property_comparison": property_comparison.to_dict("records"),
        }

    def generate_visualizations(self, property_stats: Dict[str, Any]) -> None:
        """Generate visualizations for property distributions and similarity."""
        if not self.enable_visualization:
            return

        # Create save paths if needed
        if self.save_plots and self.plot_save_dir:
            import os

            os.makedirs(self.plot_save_dir, exist_ok=True)

            compare_save_path = os.path.join(
                self.plot_save_dir, f"{self.dataset_name}_property_comparison.png"
            )
            aa_compare_save_path = os.path.join(
                self.plot_save_dir, f"{self.dataset_name}_aa_comparison.png"
            )
            sim_pos_save_path = os.path.join(
                self.plot_save_dir, f"{self.dataset_name}_positive_similarity.png"
            )
            kmer_pos_save_path = os.path.join(
                self.plot_save_dir, f"{self.dataset_name}_positive_7mer_frequency.png"
            )

        else:
            compare_save_path = None
            aa_compare_save_path = None
            sim_pos_save_path = None
            kmer_pos_save_path = None

        # Comparison visualizations
        if len(self._positive_sequences) > 0 and len(self._negative_sequences) > 0:
            visualize_property_distribution_compare(
                self._positive_sequences,
                self._negative_sequences,
                save_path=compare_save_path,
                label1="Positive",
                label2="Negative",
            )
            visualize_aas_distribution_compare(
                self._positive_sequences,
                self._negative_sequences,
                save_path=aa_compare_save_path,
                label1="Positive",
                label2="Negative",
            )

        # Similarity distribution visualizations
        if hasattr(self, "_positive_similarities") and self._positive_similarities:
            plot_similarity_distribution(
                self._positive_similarities,
                threshold=self.similarity_threshold,
                save_path=sim_pos_save_path,
            )

        # 5-mer frequency visualizations
        if len(self._positive_sequences) > 0:
            visualize_top_kmer_frequency(
                self._positive_sequences, k=5, top_n=20, save_path=kmer_pos_save_path
            )

        logger.info(f"Visualizations generated for dataset: {self.dataset_name}")

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics for classification."""

        logger.info(
            f"Calculating comprehensive statistics for dataset: {self.dataset_name}"
        )

        # Get basic statistics
        basic_stats = self.calculate_basic_statistics()

        # Get similarity statistics
        similarity_stats = self.pos_similarity_statistics()

        # Get property statistics
        property_stats = self.calculate_property_statistics()

        # Generate visualizations if enabled
        if self.enable_visualization:
            self.generate_visualizations(property_stats)

        # Combine all statistics
        all_stats = {
            "dataset_name": self.dataset_name,
            "similarity_threshold": self.similarity_threshold,
            **basic_stats,
            **similarity_stats,
            **property_stats,
        }

        logger.info(f"Statistics calculation completed for {self.dataset_name}")
        return all_stats

    def _flatten_statistics_for_csv(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested statistics dictionary for CSV export.

        Args:
            stats: Nested statistics dictionary

        Returns:
            Flattened dictionary suitable for CSV export
        """
        flattened = {}

        # Basic statistics
        flattened["dataset_name"] = stats["dataset_name"]
        flattened["similarity_threshold"] = stats.get(
            "similarity_threshold", self.similarity_threshold
        )
        flattened["total_samples"] = stats["total_samples"]
        flattened["positive_samples"] = stats["positive_samples"]
        flattened["negative_samples"] = stats["negative_samples"]
        flattened["positive_ratio"] = stats["positive_ratio"]
        flattened["sequence_length_min"] = stats["sequence_length_range"][0]
        flattened["sequence_length_max"] = stats["sequence_length_range"][1]
        flattened["sequence_length_mean"] = stats["sequence_length_mean"]
        flattened["sequence_length_std"] = stats["sequence_length_std"]

        # Positive similarity statistics
        pos_sim = stats.get("positive_similarity", {})
        if pos_sim and "error" not in pos_sim:
            flattened["pos_avg_similarity"] = pos_sim.get("average_similarity", 0.0)
            flattened["pos_max_similarity"] = pos_sim.get("max_similarity", 0.0)
            flattened["pos_min_similarity"] = pos_sim.get("min_similarity", 0.0)
            flattened["pos_similarity_std"] = pos_sim.get("similarity_std", 0.0)
            flattened["pos_high_similarity_ratio"] = pos_sim.get(
                "high_similarity_ratio", 0.0
            )
        else:
            flattened["pos_avg_similarity"] = 0.0
            flattened["pos_max_similarity"] = 0.0
            flattened["pos_min_similarity"] = 0.0
            flattened["pos_similarity_std"] = 0.0
            flattened["pos_high_similarity_ratio"] = 0.0
            if pos_sim and "error" in pos_sim:
                flattened["pos_similarity_error"] = pos_sim["error"]

        # Property statistics (add key properties)
        if "property_statistics" in stats:
            prop_stats = stats["property_statistics"]
            for prop_name, prop_data in prop_stats.items():
                if prop_name in [
                    "length",
                    "charge",
                    "isoelectric_point",
                    "hydrophobicity",
                ]:
                    pos_data = prop_data["positive"]
                    neg_data = prop_data["negative"]

                    flattened[f"{prop_name}_pos_mean"] = pos_data["mean"]
                    flattened[f"{prop_name}_pos_std"] = pos_data["std"]
                    flattened[f"{prop_name}_neg_mean"] = neg_data["mean"]
                    flattened[f"{prop_name}_neg_std"] = neg_data["std"]

        return flattened

    def print_statistics_summary(self, stats: Optional[Dict[str, Any]] = None) -> None:
        """Print a summary of classification dataset statistics.

        Args:
            stats: Statistics dictionary (if None, will calculate)
        """
        if stats is None:
            stats = self.get_dataset_statistics()

        print(
            f"\n=== Classification Dataset Statistics Summary: {stats['dataset_name']} ==="
        )
        print(
            f"Positive samples: {stats['positive_samples']} ({stats['positive_ratio']:.3f})"
        )
        print(f"Negative samples: {stats['negative_samples']}")
        print(f"Sequence length range: {stats['sequence_length_range']}")
        print(f"Sequence length mean: {stats['sequence_length_mean']:.2f}")

        print("\n--- Positive Sample Similarity ---")
        pos_sim = stats.get("positive_similarity", {})
        if pos_sim and "error" not in pos_sim:
            print(f"Average similarity: {pos_sim.get('average_similarity', 0.0):.4f}")
            print(f"Max similarity: {pos_sim.get('max_similarity', 0.0):.4f}")
            print(f"Min similarity: {pos_sim.get('min_similarity', 0.0):.4f}")
            similarity_threshold = stats.get(
                "similarity_threshold", self.similarity_threshold
            )
            print(
                f"High similarity ratio (≥{similarity_threshold}): {pos_sim.get('high_similarity_ratio', 0.0):.4f}"
            )
        else:
            print("Similarity analysis failed or not available")
            if pos_sim and "error" in pos_sim:
                print(f"Error: {pos_sim['error']}")


class RegressDataAnalyzer(BaseDataAnalyzer):
    """Regression dataset analyzer for regression tasks."""

    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        enable_visualization: bool = False,
        save_plots: bool = False,
        plot_save_dir: Optional[str] = None,
    ):
        super().__init__(
            dataset_name,
            base_data_dir,
            similarity_threshold=0.8,
            processes=None,
            enable_visualization=enable_visualization,
            save_plots=save_plots,
            plot_save_dir=plot_save_dir,
        )

        dataset_type = DATASET_MAP[dataset_name]["type"]
        if dataset_type == "binary_classification":
            raise ValueError(
                f"Dataset {dataset_name} is a binary classification dataset. This analyzer is designed for regression datasets."
            )

    def calculate_label_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive label statistics for regression."""

        labels = self._raw_data["label"].values

        # Basic statistics
        labels_array = np.array(labels)
        mean_val = float(np.mean(labels_array))
        median_val = float(np.median(labels_array))
        std_val = float(np.std(labels_array))
        min_val = float(np.min(labels_array))
        max_val = float(np.max(labels_array))

        # Percentiles
        q25 = float(np.percentile(labels_array, 25))
        q75 = float(np.percentile(labels_array, 75))

        # Detect outliers using IQR method
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = labels_array[
            (labels_array < lower_bound) | (labels_array > upper_bound)
        ]
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / len(labels) if len(labels) > 0 else 0.0

        return {
            "label_mean": mean_val,
            "label_median": median_val,
            "label_std": std_val,
            "label_min": min_val,
            "label_max": max_val,
            "label_q25": q25,
            "label_q75": q75,
            "label_iqr": float(iqr),
            "outlier_count": outlier_count,
            "outlier_ratio": float(outlier_ratio),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    def generate_regression_visualizations(self) -> None:
        """Generate visualizations for regression labels."""
        if not self.enable_visualization:
            return

        sequences = self._raw_data["sequence"].tolist()
        labels = self._raw_data["label"].tolist()

        # Create save paths if needed
        if self.save_plots and self.plot_save_dir:
            import os

            os.makedirs(self.plot_save_dir, exist_ok=True)

            hist_save_path = os.path.join(
                self.plot_save_dir, f"{self.dataset_name}_label_histogram.png"
            )
            violin_save_path = os.path.join(
                self.plot_save_dir, f"{self.dataset_name}_label_violin.png"
            )

        else:
            hist_save_path = None
            violin_save_path = None

        try:
            # Generate histogram plot
            visualize_regression_labels(
                sequences, labels, plot_type="hist", save_path=hist_save_path
            )

            # Generate violin plot
            visualize_regression_labels(
                sequences, labels, plot_type="violin", save_path=violin_save_path
            )

            logger.info(
                f"Regression visualizations generated for dataset: {self.dataset_name}"
            )

        except Exception as e:
            logger.error(f"Error generating regression visualizations: {e}")

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics for regression."""
        logger.info(
            f"Calculating comprehensive statistics for regression dataset: {self.dataset_name}"
        )

        # Get basic statistics
        basic_stats = self.calculate_basic_statistics()

        # Get label statistics
        label_stats = self.calculate_label_statistics()

        # Generate visualizations if enabled
        if self.enable_visualization:
            self.generate_regression_visualizations()

        # Combine all statistics
        all_stats = {"dataset_name": self.dataset_name, **basic_stats, **label_stats}

        logger.info(
            f"Regression statistics calculation completed for {self.dataset_name}"
        )
        return all_stats

    def _flatten_statistics_for_csv(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested statistics dictionary for CSV export.

        Args:
            stats: Nested statistics dictionary

        Returns:
            Flattened dictionary suitable for CSV export
        """
        flattened = {}

        # Basic statistics
        flattened["dataset_name"] = stats["dataset_name"]
        flattened["total_samples"] = stats["total_samples"]
        flattened["sequence_length_min"] = stats["sequence_length_range"][0]
        flattened["sequence_length_max"] = stats["sequence_length_range"][1]
        flattened["sequence_length_mean"] = stats["sequence_length_mean"]
        flattened["sequence_length_std"] = stats["sequence_length_std"]
        flattened["sequence_length_median"] = stats["sequence_length_median"]

        # Label statistics
        flattened["label_mean"] = stats["label_mean"]
        flattened["label_median"] = stats["label_median"]
        flattened["label_std"] = stats["label_std"]
        flattened["label_min"] = stats["label_min"]
        flattened["label_max"] = stats["label_max"]
        flattened["label_q25"] = stats["label_q25"]
        flattened["label_q75"] = stats["label_q75"]
        flattened["label_iqr"] = stats["label_iqr"]
        flattened["outlier_count"] = stats["outlier_count"]
        flattened["outlier_ratio"] = stats["outlier_ratio"]
        flattened["lower_bound"] = stats["lower_bound"]
        flattened["upper_bound"] = stats["upper_bound"]

        return flattened

    def print_statistics_summary(self, stats: Optional[Dict[str, Any]] = None) -> None:
        """Print a summary of regression dataset statistics."""
        if stats is None:
            stats = self.get_dataset_statistics()

        print(
            f"\n=== Regression Dataset Statistics Summary: {stats['dataset_name']} ==="
        )
        print(f"Total samples: {stats['total_samples']}")
        print(f"Sequence length range: {stats['sequence_length_range']}")
        print(f"Sequence length mean: {stats['sequence_length_mean']:.2f}")
        print(f"Sequence length median: {stats['sequence_length_median']:.2f}")

        print("\n--- Label Statistics ---")
        print(f"Mean: {stats['label_mean']:.4f}")
        print(f"Median: {stats['label_median']:.4f}")
        print(f"Standard deviation: {stats['label_std']:.4f}")
        print(f"Range: [{stats['label_min']:.4f}, {stats['label_max']:.4f}]")
        print(f"Q25: {stats['label_q25']:.4f}")
        print(f"Q75: {stats['label_q75']:.4f}")
        print(f"IQR: {stats['label_iqr']:.4f}")
        print(f"Outliers: {stats['outlier_count']} ({stats['outlier_ratio']:.2%})")
        print(
            f"Outlier bounds: [{stats['lower_bound']:.4f}, {stats['upper_bound']:.4f}]"
        )


def analyze_class_dataset(
    dataset_name: str,
    base_data_dir: Optional[str] = None,
    similarity_threshold: float = 0.8,
    processes: Optional[int] = None,
    print_summary: bool = True,
    enable_visualization: bool = False,
    save_plots: bool = False,
    plot_save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to analyze a classification dataset."""
    analyzer = ClassDataAnalyzer(
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        similarity_threshold=similarity_threshold,
        processes=processes,
        enable_visualization=enable_visualization,
        save_plots=save_plots,
        plot_save_dir=plot_save_dir,
    )

    stats = analyzer.get_dataset_statistics()

    if print_summary:
        analyzer.print_statistics_summary(stats)

    return stats


def analyze_regression_dataset(
    dataset_name: str,
    base_data_dir: Optional[str] = None,
    print_summary: bool = True,
    enable_visualization: bool = False,
    save_plots: bool = False,
    plot_save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to analyze a regression dataset."""
    analyzer = RegressDataAnalyzer(
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        enable_visualization=enable_visualization,
        save_plots=save_plots,
        plot_save_dir=plot_save_dir,
    )

    stats = analyzer.get_dataset_statistics()

    if print_summary:
        analyzer.print_statistics_summary(stats)

    return stats


def batch_analyze_class_datasets(
    dataset_names: List[str],
    base_data_dir: Optional[str] = None,
    similarity_threshold: float = 0.9,
    processes: Optional[int] = None,
    output_csv: str = "all_datasets_statistics.csv",
    print_summary: bool = False,
    enable_visualization: bool = False,
    save_plots: bool = False,
    plot_save_dir: Optional[str] = None,
    header_written: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch analyze multiple classification datasets and save results to CSV."""
    all_stats = []
    header_written = header_written
    successful_count = 0
    error_count = 0

    logger.info(
        f"Starting batch analysis of {len(dataset_names)} classification datasets"
    )
    logger.info(f"Results will be saved to: {output_csv}")

    for i, dataset_name in enumerate(dataset_names):
        try:
            if show_progress:
                logger.info(
                    f"Analyzing dataset: {dataset_name} ({i+1}/{len(dataset_names)})"
                )
            else:
                logger.info(f"Analyzing dataset: {dataset_name}")

            # Get dataset path
            dataset_path = DATASET_MAP.get(dataset_name, {}).get("path", None)
            if base_data_dir:
                dataset_path = base_data_dir

            # Initialize analyzer
            analyzer = ClassDataAnalyzer(
                dataset_name=dataset_name,
                base_data_dir=dataset_path,
                similarity_threshold=similarity_threshold,
                processes=processes,
                enable_visualization=enable_visualization,
                save_plots=save_plots,
                plot_save_dir=plot_save_dir,
            )

            # Get statistics
            stats = analyzer.get_dataset_statistics()

            # Flatten statistics for CSV
            flattened_stats = analyzer._flatten_statistics_for_csv(stats)
            all_stats.append(flattened_stats)

            # Write to CSV immediately
            df_row = pd.DataFrame([flattened_stats])
            if not header_written:
                df_row.to_csv(output_csv, index=False, mode="w")
                header_written = True
            else:
                df_row.to_csv(output_csv, index=False, mode="a", header=False)

            # Print summary if requested
            if print_summary:
                analyzer.print_statistics_summary(stats)

            successful_count += 1
            if show_progress:
                logger.info(
                    f"Successfully analyzed {dataset_name} and saved to CSV ({successful_count}/{i+1})"
                )
            else:
                logger.info(f"Successfully analyzed {dataset_name} and saved to CSV")

        except Exception as e:
            logger.error(f"Failed to analyze {dataset_name}: {e}")
            error_count += 1

            # Add error row
            error_stats = {"dataset_name": dataset_name, "error": str(e)}
            all_stats.append(error_stats)

            # Write error row to CSV immediately
            df_error = pd.DataFrame([error_stats])
            if not header_written:
                df_error.to_csv(output_csv, index=False, mode="w")
                header_written = True
            else:
                df_error.to_csv(output_csv, index=False, mode="a", header=False)

    # Create DataFrame for return
    df = pd.DataFrame(all_stats)

    # Final summary
    logger.info("Batch analysis completed!")
    logger.info(f"Total datasets processed: {len(df)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {error_count}")
    logger.info(f"Results saved to: {output_csv}")

    return df


def batch_analyze_regress_datasets(
    dataset_names: List[str],
    base_data_dir: Optional[str] = None,
    output_csv: str = "all_regression_datasets_statistics.csv",
    print_summary: bool = False,
    enable_visualization: bool = False,
    save_plots: bool = False,
    plot_save_dir: Optional[str] = None,
    header_written: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch analyze multiple regression datasets and save results to CSV."""
    all_stats = []
    header_written = header_written
    successful_count = 0
    error_count = 0

    logger.info(f"Starting batch analysis of {len(dataset_names)} regression datasets")
    logger.info(f"Results will be saved to: {output_csv}")

    for i, dataset_name in enumerate(dataset_names):
        try:
            if show_progress:
                logger.info(
                    f"Analyzing dataset: {dataset_name} ({i+1}/{len(dataset_names)})"
                )
            else:
                logger.info(f"Analyzing dataset: {dataset_name}")

            # Get dataset path
            dataset_path = DATASET_MAP.get(dataset_name, {}).get("path", None)
            if base_data_dir:
                dataset_path = base_data_dir

            # Initialize analyzer
            analyzer = RegressDataAnalyzer(
                dataset_name=dataset_name,
                base_data_dir=dataset_path,
                enable_visualization=enable_visualization,
                save_plots=save_plots,
                plot_save_dir=plot_save_dir,
            )

            # Get statistics
            stats = analyzer.get_dataset_statistics()

            # Flatten statistics for CSV
            flattened_stats = analyzer._flatten_statistics_for_csv(stats)
            all_stats.append(flattened_stats)

            # Write to CSV immediately
            df_row = pd.DataFrame([flattened_stats])
            if not header_written:
                df_row.to_csv(output_csv, index=False, mode="w")
                header_written = True
            else:
                df_row.to_csv(output_csv, index=False, mode="a", header=False)

            # Print summary if requested
            if print_summary:
                analyzer.print_statistics_summary(stats)

            successful_count += 1
            if show_progress:
                logger.info(
                    f"Successfully analyzed {dataset_name} and saved to CSV ({successful_count}/{i+1})"
                )
            else:
                logger.info(f"Successfully analyzed {dataset_name} and saved to CSV")

        except Exception as e:
            logger.error(f"Failed to analyze {dataset_name}: {e}")
            error_count += 1

            # Add error row
            error_stats = {"dataset_name": dataset_name, "error": str(e)}
            all_stats.append(error_stats)

            # Write error row to CSV immediately
            df_error = pd.DataFrame([error_stats])
            if not header_written:
                df_error.to_csv(output_csv, index=False, mode="w")
                header_written = True
            else:
                df_error.to_csv(output_csv, index=False, mode="a", header=False)

    # Create DataFrame for return
    df = pd.DataFrame(all_stats)

    # Final summary
    logger.info("Batch analysis completed!")
    logger.info(f"Total datasets processed: {len(df)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {error_count}")
    logger.info(f"Results saved to: {output_csv}")

    return df


if __name__ == "__main__":
    dataset_names = [
        "bbp",
        "nonfouling",
        "antibacterial",
        "antifungal",
        "antiviral",
        "antimicrobial",
        "E.coli_mic",
        "P.aeruginosa_mic",
        "S.aureus_mic",
        "antiparasitic",
        "ace_inhibitory",
        "anticancer",
        "antidiabetic",
        "antioxidant",
        "neuropeptide",
        "quorum_sensing",
        "ttca",
        "hemolytic",
        "solubility",
        "hemolytic_hc50",
        "toxicity",
        "allergen",
        "antiinflamatory",
        "antiaging",
        "anti_mammalian_cell",
        "dppiv_inhibitors",
        "cpp",
    ]

    # Batch analyze all datasets
    try:
        df = batch_analyze_regress_datasets(
            dataset_names=dataset_names,
            output_csv="all_new_datasets_statistics.csv",
            print_summary=True,
            enable_visualization=True,
            save_plots=True,
            plot_save_dir="./new_plots",
            show_progress=True,
            header_written=False,
        )
        print(
            "\nBatch analysis completed! Results saved to all_datasets_statistics.csv"
        )
        print(f"Total datasets analyzed: {len(df)}")
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        print(f"Batch analysis failed: {e}")
