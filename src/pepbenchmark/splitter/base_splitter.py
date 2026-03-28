from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import hashlib

import numpy as np
import scipy as sp

from pepbenchmark.utils.logging import get_logger
from pepbenchmark.splitter.split_analyzer import SplitAnalyzer
from pepbenchmark.cluster.utils import (
    log_cluster_analysis, 
    get_cluster_info_dict, 
    print_cluster_statistics_from_map
)

logger = get_logger(__name__)


class BaseSplitter(ABC):
    """
    Abstract base class for all data splitters.

    This class defines the pure interface that all splitter implementations must follow.

    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys
    """

    @abstractmethod
    def get_split_indices(
        self,
        data: Union[List, np.ndarray],
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        labels: Optional[List[int]] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Union[List[int], np.ndarray]]:
        """
        Generate train/validation/test split indices.

        Args:
            data: Input sequences to split (list of strings or numpy array)
            frac_train: Fraction of data for training (default: 0.8)
            frac_valid: Fraction of data for validation (default: 0.1)
            frac_test: Fraction of data for testing (default: 0.1)
            labels: Optional class labels for each sequence
            seed: Random seed for reproducibility
            **kwargs: Additional splitter-specific parameters

        Returns:
            Dictionary with 'train', 'valid', 'test' keys containing indices
            {
                "train": [1,2,...],
                "valid": [3,4,...],
                "test": [5,6,...],
            }

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def get_split_kfold_indices(
        self,
        data: Union[List, np.ndarray],
        k_folds: int = 5,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[int], np.ndarray]]]:
        """
        Generate k-fold cross-validation splits.

        Args:
            data: Input data to split
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
            **kwargs: Additional splitter-specific parameters


        Returns:
            Dictionary with keys in format "fold_X" where X is the fold index (0 to k_folds-1).
            Each fold contains train/valid/test splits where the test set is the X-th fold.
            Example:
            {
                "fold_0": {"train": [1,2,...], "valid": [3,4,...], "test": [5,6,...]},
                "fold_1": {"train": [...], "valid": [...], "test": [...]},
                ...
                "fold_k": {"train": [...], "valid": [...], "test": [...]}
            }
        """
        print("get_split_kfold_indices is not implemented yet")
        return {}

    def get_split_indices_n(
        self,
        data: Union[List, np.ndarray],
        n_splits: int = 5,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Union[List[int], int] = 42,
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[int], np.ndarray]]]:
        """
        Generate multiple random splits with different seeds.

        This method provides a default implementation that can be overridden by subclasses.
        Repeat `get_split_indices()` with different seeds n_splits times by default.

        Args:
            data: Input data to split
            n_splits: Number of splits to generate (default: 5)
            frac_train: Fraction of data for training (default: 0.8)
            frac_valid: Fraction of data for validation (default: 0.1)
            frac_test: Fraction of data for testing (default: 0.1)
            seed: Random seed or list of seeds for reproducibility
            **kwargs: Additional splitter-specific parameters
        Returns:
            Dictionary with keys in format "seed_X" where X is the split index (0 to n_splits-1).
            Each split contains train/valid/test splits with the specified fractions.
            Example:
            {
                "seed_0": {"train": [1,2,...], "valid": [3,4,...], "test": [5,6,...]},
                "seed_1": {"train": [...], "valid": [...], "test": [...]},
                ...
                "seed_n": {"train": [...], "valid": [...], "test": [...]}
            }

        """
        return self._get_split_indices_n_default(
            data, n_splits, frac_train, frac_valid, frac_test, seed, **kwargs
        )

    def _get_split_indices_n_default(
        self,
        data: Union[List, np.ndarray],
        n_splits: int,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Union[List[int], int],
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[int], np.ndarray]]]:
        """Default implementation for multiple splits."""
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError("Data must be a list or numpy array")

        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")

        # Prepare seeds
        if isinstance(seed, list):
            if len(seed) != n_splits:
                raise ValueError(
                    f"Expected {n_splits} seeds, but got {len(seed)}. "
                    f"Please provide a seed for each split."
                )
            seeds = seed
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(n_splits)]
        else:
            raise ValueError(
                "Seed must be an integer or a list of integers. "
                "If using a list, it should have the same length as n_splits."
            )

        split_results = {}
        logger.info(f"Generating {n_splits} splits")

        for i, current_seed in enumerate(seeds):
            logger.info(f"Generating split {i + 1}/{n_splits} with seed {current_seed}")

            split_indices = self.get_split_indices(
                data, frac_train, frac_valid, frac_test, seed=current_seed, **kwargs
            )

            logger.info(
                f"Split {i + 1} completed: Train={len(split_indices['train'])}, "
                f"Valid={len(split_indices['valid'])}, Test={len(split_indices['test'])}"
            )
            split_results[f"seed_{i}"] = split_indices

        logger.info(f"All {n_splits} splits completed successfully")
        return split_results


class AbstractSplitter(BaseSplitter):
    """
    Abstract splitter class that provides common functionality for concrete splitters.

    This class contains shared methods for validation, statistics, I/O operations,
    and common clustering-based splitting logic. All concrete splitters should inherit 
    from this class instead of BaseSplitter directly.

    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys

    Initializes logging and internal state tracking.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._last_split_info = None

    # Validation Methods
    def validate_fractions(
        self, frac_train: float, frac_valid: float, frac_test: float
    ) -> None:
        """
        Validate that train/valid/test fractions sum to 1.0.

        Args:
            frac_train: Training fraction
            frac_valid: Validation fraction
            frac_test: Test fraction

        Raises:
            ValueError: If fractions don't sum to approximately 1.0
        """
        total = frac_train + frac_valid + frac_test
        if not np.isclose(total, 1.0, atol=1e-10):
            raise ValueError(f"Train/valid/test fractions must sum to 1.0, got {total}")

        if any(frac < 0 for frac in [frac_train, frac_valid, frac_test]):
            raise ValueError("All fractions must be non-negative")

    def validate_split_keys(self, split_results: Dict[str, Any]) -> None:
        """
        Validate that the split dictionary contains the required keys.

        Args:
            split_results: Dictionary containing split results

        Raises:
            ValueError: If required keys are missing
        """
        required_keys = {"train", "valid", "test"}
        if not required_keys.issubset(split_results.keys()):
            raise ValueError(
                f"Split results must contain {required_keys}, but got {split_results.keys()}"
            )

    def validate_split_indices(
        self, split_results: Dict[str, Union[List[int], np.ndarray]], data_size: int
    ) -> None:
        """
        Validate that indices in splits are within the valid range.

        Args:
            split_results: Dictionary containing split results with indices
            data_size: Size of the original dataset

        Raises:
            TypeError: If indices are not of correct type
            ValueError: If indices are out of bounds
        """
        for key, indices in split_results.items():
            if not isinstance(indices, (list, np.ndarray)):
                raise TypeError(f"Indices for {key} must be a list or numpy array")
            if not all(isinstance(i, (int, np.integer)) for i in indices):
                raise TypeError(f"All indices for {key} must be integers")
            if any(i < 0 or i >= data_size for i in indices):
                raise ValueError(
                    f"Indices for {key} are out of bounds for data of size {data_size}"
                )

    def check_split_completeness(self, all_indices: np.ndarray, data_size: int) -> None:
        """
        Check if the splits cover the entire dataset.

        Args:
            all_indices: Array of all indices from all splits
            data_size: Expected size of the dataset
        """
        if len(all_indices) != data_size:
            self.logger.warning(
                f"Split is not complete. Expected {data_size} unique indices, but got {len(all_indices)}"
            )

    def check_split_overlaps(self, all_indices: np.ndarray, total_indices: int) -> None:
        """
        Check for overlapping indices between splits.

        Args:
            all_indices: Array of unique indices from all splits
            total_indices: Total number of indices across all splits
        """
        if len(all_indices) != total_indices:
            self.logger.warning(
                f"Overlapping indices found. Total indices: {total_indices}, Unique indices: {len(all_indices)}"
            )

    def validate_split_results(
        self,
        split_results: Dict[str, Union[List[int], np.ndarray]],
        data_size: int,
        check_completeness: bool = True,
        check_overlaps: bool = True,
    ) -> bool:
        """
        Validate split results for completeness and non-overlapping indices.

        Args:
            split_results: Dictionary with train/valid/test indices
            data_size: Original data size
            check_completeness: Whether to check if all data points are used
            check_overlaps: Whether to check for overlapping indices

        Returns:
            True if validation passes, False otherwise
        """
        try:
            self.validate_split_keys(split_results)
            self.validate_split_indices(split_results, data_size)

            train_indices = np.array(split_results["train"])
            valid_indices = np.array(split_results["valid"])
            test_indices = np.array(split_results["test"])

            total_indices = len(train_indices) + len(valid_indices) + len(test_indices)
            all_indices = np.unique(
                np.concatenate([train_indices, valid_indices, test_indices])
            )

            if check_completeness:
                self.check_split_completeness(all_indices, data_size)

            if check_overlaps:
                self.check_split_overlaps(all_indices, total_indices)

            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Split validation failed: {e}")
            return False

    # Statistics Methods
    def get_split_statistics(
        self, split_results: Dict[str, Union[List[int], np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the split results.

        Args:
            split_results: Dictionary with train/valid/test indices

        Returns:
            Dictionary containing split statistics:
            {
                "train_size": int,
                "valid_size": int,
                "test_size": int,
                "train_fraction": float,
                "valid_fraction": float,
                "test_fraction": float,
                "total_size": int
            }
        """
        stats = {}
        total_size = sum(len(split_results[key]) for key in ["train", "valid", "test"])

        for split_name in ["train", "valid", "test"]:
            split_size = len(split_results[split_name])
            stats[f"{split_name}_size"] = split_size
            stats[f"{split_name}_fraction"] = (
                split_size / total_size if total_size > 0 else 0
            )

        stats["total_size"] = total_size
        return stats

    # I/O Methods
    def save_split_results(
        self, split_results: Dict[str, Any], filepath: str, format: str = "json"
    ) -> None:
        """
        Save split results to file in specified format.

        Args:
            split_results: Split results to save (can be single split or multiple splits)
            filepath: Output file path
            format: Output format ('json' or 'numpy')

        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
        import json
        import os

        # Only create directory if filepath contains a directory part
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if format == "json":
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in split_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = (
                        value.tolist() if hasattr(value, "tolist") else value
                    )

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2)

        elif format == "numpy":
            np.savez(filepath, **split_results)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Split results saved to {filepath}")

    def load_split_results(self, filepath: str, format: str = "json") -> Dict[str, Any]:
        """
        Load split results from file.

        Args:
            filepath: Input file path
            format: Input format ('json' or 'numpy')

        Returns:
            Dictionary containing split results (automatically converts lists to numpy arrays)

        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        import json

        if format == "json":
            with open(filepath, "r") as f:
                results = json.load(f)
            # Convert lists back to numpy arrays
            for key, value in results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, list):
                            results[key][k] = np.array(v)
                elif isinstance(value, list):
                    results[key] = np.array(value)

        elif format == "numpy":
            results = dict(np.load(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Split results loaded from {filepath}")
        return results

    # Common Utility Methods
    def prepare_seeds(self, seed: Union[List[int], int], n_splits: int) -> List[int]:
        """
        Prepare seeds for multiple splits.

        Args:
            seed: Either a single integer or list of integers for seeding
            n_splits: Number of splits to generate

        Returns:
            List of seeds for each split

        Raises:
            ValueError: If seed list length doesn't match n_splits or invalid type
        """
        if isinstance(seed, list):
            if len(seed) != n_splits:
                raise ValueError(
                    f"Expected {n_splits} seeds, but got {len(seed)}. "
                    f"Please provide a seed for each split."
                )
            return seed
        elif isinstance(seed, int):
            return [seed + i for i in range(n_splits)]
        else:
            raise ValueError(
                "Seed must be an integer or a list of integers. "
                "If using a list, it should have the same length as n_splits."
            )

    def get_data_hash(self, data: List[str]) -> str:
        """
        Generate a hash for the data to check if it has changed.

        Args:
            data: List of sequences or data points

        Returns:
            MD5 hash of the concatenated data
        """
        data_str = "".join(str(item) for item in data)
        return hashlib.md5(data_str.encode()).hexdigest()

    # Common Clustering-based Splitting Methods
    def distribute_clusters_to_folds(
        self, cluster_items: List[Tuple[str, List[str]]], k_folds: int
    ) -> List[List[str]]:
        """
        Distribute clusters across k folds as evenly as possible.

        Uses a greedy algorithm to assign each cluster to the currently
        smallest fold to achieve balanced fold sizes.

        Args:
            cluster_items: List of (cluster_id, members) tuples
            k_folds: Number of folds to create

        Returns:
            List of folds, where each fold is a list of sequence IDs
        """
        folds = [[] for _ in range(k_folds)]
        fold_sizes = [0] * k_folds

        for _, members in cluster_items:
            smallest_fold = np.argmin(fold_sizes)
            folds[smallest_fold].extend(members)
            fold_sizes[smallest_fold] += len(members)

        # Log fold size distribution
        self.logger.info(f"Fold size distribution: {fold_sizes}")
        return folds

    def generate_kfold_results_from_folds(
        self, folds: List[List[str]], data: List[str], k_folds: int, seed: Optional[int]
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate k-fold results from distributed folds.

        For each fold, the fold itself becomes the test set, and remaining
        folds are combined and split into train/valid sets (80/20 split).

        Args:
            folds: List of folds, each containing sequence IDs
            data: Original data for creating ID-to-index mapping
            k_folds: Number of folds
            seed: Random seed for train/valid splitting

        Returns:
            Dictionary with fold results in format "fold_X"
        """
        id_to_idx = {f"seq{i}": i for i in range(len(data))}
        kfold_results = {}

        for fold_idx in range(k_folds):
            test_ids = folds[fold_idx]

            # Collect data from other folds
            remaining_ids = []
            for i in range(k_folds):
                if i != fold_idx:
                    remaining_ids.extend(folds[i])

            # Split remaining data into train and validation sets
            if seed is not None:
                rng = np.random.RandomState(seed + fold_idx)
                rng.shuffle(remaining_ids)
            else:
                np.random.shuffle(remaining_ids)

            train_size = int(len(remaining_ids) * 0.8)
            train_ids = remaining_ids[:train_size]
            valid_ids = remaining_ids[train_size:]

            kfold_results[f"fold_{fold_idx}"] = {
                "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
                "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
                "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
            }

            self.logger.info(
                f"Fold {fold_idx} completed: "
                f"Train={len(kfold_results[f'fold_{fold_idx}']['train'])}, "
                f"Valid={len(kfold_results[f'fold_{fold_idx}']['valid'])}, "
                f"Test={len(kfold_results[f'fold_{fold_idx}']['test'])}"
            )

        return kfold_results

    def generate_split_from_clusters(
        self,
        cluster_items: List[Tuple[str, List[str]]],
        data: List[str],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int],
    ) -> Dict[str, List[int]]:
        """
        Helper method to generate train/valid/test split from clustered data.

        Args:
            cluster_items: List of (cluster_id, members) tuples
            data: Original data for size calculation
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed

        Returns:
            Dictionary with train/valid/test indices
        """
        # Shuffle clusters
        cluster_items_copy = cluster_items.copy()
        if seed is not None:
            np.random.RandomState(seed).shuffle(cluster_items_copy)
        else:
            np.random.shuffle(cluster_items_copy)

        # Sort clusters by size
        cluster_items_sorted = sorted(cluster_items_copy, key=lambda x: len(x[1]))
        test_ids = []
        remaining_ids = []
        count_test = 0
        test_data_size = int(len(data) * frac_test)

        for _, members in cluster_items_sorted:
            if count_test + len(members) <= test_data_size:
                test_ids.extend(members)
                count_test += len(members)
            else:
                remaining_ids.extend(members)

        # Split remaining data into train and validation sets
        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)
        np.random.shuffle(remaining_ids)
        train_ids = remaining_ids[:train_data_size]
        valid_ids = remaining_ids[train_data_size:]

        self.logger.info(
            f"Split distribution: "
            f"Target train={train_data_size}, Actual train={len(train_ids)} | "
            f"Target valid={valid_data_size}, Actual valid={len(valid_ids)} | "
            f"Target test={test_data_size}, Actual test={len(test_ids)}"
        )

        # Create a mapping from sequence ID to index
        id_to_idx = {f"seq{i}": i for i in range(len(data))}

        return {
            "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
            "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
            "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
        }

    def print_cluster_stats(self, cluster_map: Dict[str, List[str]], data_type: str = "sequences") -> None:
        """
        Print comprehensive cluster statistics.

        Args:
            cluster_map: Dictionary mapping cluster IDs to sequence ID lists
            data_type: Type of data being clustered (sequences, molecules, etc.)
        """
        cluster_sizes = [len(members) for members in cluster_map.values()]

        # Statistical information
        total_items = sum(cluster_sizes)
        avg_cluster_size = np.mean(cluster_sizes)
        median_cluster_size = np.median(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        min_cluster_size = min(cluster_sizes)

        bins = {
            "size = 1": 0,
            "size 2–4": 0,
            "size 5–9": 0,
            "size 10–19": 0,
            "size 20+": 0,
        }

        for size in cluster_sizes:
            if size == 1:
                bins["size = 1"] += 1
            elif 2 <= size <= 4:
                bins["size 2–4"] += 1
            elif 5 <= size <= 9:
                bins["size 5–9"] += 1
            elif 10 <= size <= 19:
                bins["size 10–19"] += 1
            else:
                bins["size 20+"] += 1

        log_message = (
            f"Clustering Statistics:\n"
            f"  Total clusters: {len(cluster_map)}\n"
            f"  Total {data_type}: {total_items}\n"
            f"  Average cluster size: {avg_cluster_size:.2f}\n"
            f"  Median cluster size: {median_cluster_size:.1f}\n"
            f"  Min cluster size: {min_cluster_size}\n"
            f"  Max cluster size: {max_cluster_size}\n"
            f"  Cluster size distribution:\n"
        )

        for label, count in bins.items():
            percentage = (count / len(cluster_map)) * 100
            log_message += f"    {label}: {count} clusters ({percentage:.1f}%)\n"

        self.logger.info(log_message)

    def __str__(self) -> str:
        """String representation of the splitter."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Detailed string representation of the splitter."""
        return f"{self.__class__.__name__}()"


class AbstractClusteringSplitter(AbstractSplitter):
    """
    Abstract base class for clustering-based splitters.
    
    This class provides common functionality for splitters that rely on clustering,
    including caching mechanisms and cluster-based splitting logic.
    
    This class serves as the unified base class for all cluster-based splitting methods,
    enabling consistent handling of clustering results from different algorithms
    (CD-HIT, MMseqs2, Motif-based, etc.).
    
    Key features:
    - Unified interface for cluster-based splitting
    - Cluster integrity preservation (sequences in same cluster stay together)
    - Configurable cluster distribution strategies
    - Quality analysis and validation
    - Caching mechanisms for clustering results
    - Comprehensive statistics and I/O methods
    """

    def __init__(self, random_seed: Optional[int] = 42):
        super().__init__()
        self.random_seed = random_seed
        self.cluster_map = None
        self._cached_data_hash = None
        self._cached_params = None
        self._last_cluster_result = None
        self._last_split_result = None

    def clear_cache(self) -> None:
        """
        Clear cached clustering results.

        This forces the next clustering operation to run from scratch,
        useful when you want to ensure fresh results or free memory.
        """
        self.cluster_map = None
        self._cached_data_hash = None
        self._cached_params = None
        self._last_cluster_result = None
        self._last_split_result = None
        self.logger.info("Clustering cache cleared")

    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about the current clustering.

        Returns:
            Dictionary containing cluster statistics and parameters,
            or None if no clustering has been performed yet.
        """
        if self.cluster_map is None and self._last_cluster_result is None:
            return None

        # Use unified cluster result if available
        if self._last_cluster_result is not None:
            return get_cluster_info_dict(self._last_cluster_result)
        
        # Fallback to legacy cluster_map
        if self.cluster_map is not None:
            return get_cluster_info_dict(self.cluster_map)
        
        return None

    def _should_use_cache(self, data: List[str], **params) -> bool:
        """
        Determine if cached clustering results can be used.

        Args:
            data: Input data to check
            **params: Parameters to compare with cached parameters

        Returns:
            True if cache can be used, False otherwise
        """
        if self.cluster_map is None:
            return False

        current_data_hash = self.get_data_hash(data)
        if current_data_hash != self._cached_data_hash:
            return False

        if self._cached_params != params:
            return False

        return True

    def _update_cache(self, data: List[str], cluster_map: Dict[str, List[str]], **params) -> None:
        """
        Update the cache with new clustering results.

        Args:
            data: Input data
            cluster_map: Clustering results
            **params: Parameters used for clustering
        """
        self.cluster_map = cluster_map
        self._cached_data_hash = self.get_data_hash(data)
        self._cached_params = params.copy()

    @abstractmethod
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ):
        """
        Get clustering result for the given sequences.
        
        This method should be implemented by subclasses to call the appropriate
        clustering algorithm (CD-HIT, MMseqs2, motif-based, etc.) and return
        clustering information.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels (not used for clustering but passed through)
            **kwargs: Algorithm-specific clustering parameters
            
        Returns:
            Clustering result (format depends on implementation)
        """
        pass

    def _get_or_create_clusters(self, data: List[str], **params) -> Dict[str, List[str]]:
        """
        Get cached clusters or create new ones if parameters changed.

        Args:
            data: List of sequences/items to cluster
            **params: Clustering parameters

        Returns:
            Dictionary mapping cluster IDs to lists of sequence IDs
        """
        if self._should_use_cache(data, **params):
            self.logger.info("Using cached clustering results")
            return self.cluster_map

        # Use new unified interface
        result = self._get_clustering_result(data, **params)
        # Convert UnifiedClusterResult to legacy format
        cluster_map = {}
        for cluster_id, indices in result.cluster_assignments.items():
            cluster_map[cluster_id] = [f"seq{idx}" for idx in indices]
        self._update_cache(data, cluster_map, **params)
        return cluster_map


    def get_split_indices(
        self,
        data: Union[List[str], np.ndarray],
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        labels: Optional[List[int]] = None,
        seed: Optional[int] = None,
        cluster_distribution_strategy: str = "sort_cluster_size",
        preserve_cluster_integrity: bool = True,
        balance_labels: Optional[bool] = None,  # Deprecated: use cluster_distribution_strategy instead
        return_cluster_assignments: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """
        Generate cluster-aware split indices.
        
        Args:
            data: Input sequences (list of strings or numpy array of strings)
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set  
            frac_test: Fraction for test set
            labels: Optional class labels for each sequence (required for label balancing strategies)
            seed: Random seed for reproducibility (overrides default)
            cluster_distribution_strategy: How to distribute clusters across splits
                - "size_aware": Assign largest clusters first to balance sizes
                - "random": Randomly assign clusters to splits
                - "balanced": Shuffle clusters then assign by deficit to balance sequence counts
                - "label_balanced": Balance class labels while preserving cluster integrity
                - "label_balanced_free": Balance class labels freely (may break cluster integrity)
                - "sort_cluster_size": Sort clusters by size (largest first) and fill train->valid->test sequentially
            preserve_cluster_integrity: Keep sequences from same cluster together (ignored for label_balanced_free)
            balance_labels: [DEPRECATED] Use cluster_distribution_strategy="label_balanced" instead
            **kwargs: Clustering method specific parameters. For MotifSplitter, supports:
                
                # Basic motif extraction parameters (same names as in topk_mer.py)
                ks (int): K-mer size for motif extraction (default: 5)
                topM (int): Number of top motifs to consider (default: None)
                top_fraction (float): Alternative to topM as fraction (0-1) of total motifs
                
                # Supervised motif extraction parameters (extract_motifs_enriched)
                mode (str): Motif extraction mode - "pos", "neg", "both" (default: "pos")
                count_mode (str): Motif counting mode - "presence", "count" (default: "presence")
                test_method (str): Statistical test method - "fisher", "chi2", "ratio", "exclusive", 
                                   "logodds", "mutual_info" (default: "fisher")
                alternative (str): Test alternative - "greater", "less", "two-sided" (default: "greater")
                min_pos_count (int): Minimum positive count for motifs (default: 3)
                min_neg_count (int): Minimum negative count for motifs (default: 3)
                min_cluster_size (int): Minimum cluster size for motifs (default: 2)
                strict (bool): Use strict filtering (default: False)
                pval_threshold (float): P-value threshold for filtering (default: None)
                min_score (float): Minimum score threshold for filtering (default: None)
                min_pos (int): Minimum positive samples for filtering (default: None)
                min_neg (int): Minimum negative samples for filtering (default: None)
                filter_fn (callable): Custom filter function (default: None)
                fdr_correct (bool): Apply FDR correction (default: False)
                
                # Unsupervised motif extraction parameters (extract_motifs_frequency)
                min_count (int): Minimum count for frequency-based extraction (default: 1)
                
                # Clustering parameters
                min_support (int): Minimum support for motif merging (default: 3)
                min_jaccard (float): Minimum Jaccard similarity for motif merging (default: 0.5)
                conflict_strategy (str): Strategy for resolving conflicts - "max_support", "first", "random"
                merge_strategy (str): Strategy for motif merging - "jaccard", "overlap", "union"
                
                # Advanced clustering options
                use_generic_clustering (bool): Whether to use unified clustering interface (default: False)
                clustering_method (str): Method for unified clustering - "connected", "hierarchical", "kmeans"
                similarity_threshold (float): Threshold for similarity-based clustering (default: 0.5)
            
        Returns:
            Dictionary with train/valid/test split indices
            
        Example:
            For MotifSplitter with rich parameters:
            >>> motif_splitter = MotifSplitter()
            >>> splits = motif_splitter.get_split_indices(
            ...     data=sequences_df,
            ...     labels=labels,
            ...     # Basic motif parameters
            ...     k=6, top_m=100, top_fraction=0.1,
            ...     # Advanced motif extraction
            ...     test_method="fisher", alternative="greater", 
            ...     min_pos_count=5, fdr_correct=True,
            ...     pval_threshold=0.01, min_score=2.0,
            ...     # Clustering parameters  
            ...     min_support=5, min_jaccard=0.6,
            ...     merge_strategy="overlap",
            ...     # Split strategy
            ...     cluster_distribution_strategy="label_balanced"
            ... )
        """
        # Handle deprecated balance_labels parameter for backward compatibility
        if balance_labels is not None:
            import warnings
            warnings.warn(
                "The 'balance_labels' parameter is deprecated. Use cluster_distribution_strategy='label_balanced' or "
                "'label_balanced_free' instead.", 
                DeprecationWarning, 
                stacklevel=2
            )
            
            if balance_labels and cluster_distribution_strategy in ["size_aware", "random", "balanced", "sort_cluster_size"]:
                # Auto-convert to new strategy based on preserve_cluster_integrity
                if preserve_cluster_integrity:
                    cluster_distribution_strategy = "label_balanced"
                    self.logger.info("🔄 Auto-converted balance_labels=True to cluster_distribution_strategy='label_balanced'")
                else:
                    cluster_distribution_strategy = "label_balanced_free"
                    self.logger.info("🔄 Auto-converted balance_labels=True to cluster_distribution_strategy='label_balanced_free'")
        
        # Validate cluster_distribution_strategy
        valid_strategies = ["size_aware", "random", "balanced", "label_balanced", "label_balanced_free", "sort_cluster_size"]
        if cluster_distribution_strategy not in valid_strategies:
            raise ValueError(f"cluster_distribution_strategy must be one of {valid_strategies}, got '{cluster_distribution_strategy}'")
        
        # Check label requirements for label balancing strategies
        is_label_strategy = cluster_distribution_strategy in ["label_balanced", "label_balanced_free"]
        if is_label_strategy and labels is None:
            raise ValueError(f"cluster_distribution_strategy='{cluster_distribution_strategy}' requires labels to be provided")
        
        # Log detailed parameter information
        is_label_strategy = cluster_distribution_strategy in ["label_balanced", "label_balanced_free"]
        if is_label_strategy and labels is None:
            raise ValueError(f"cluster_distribution_strategy='{cluster_distribution_strategy}' requires labels to be provided")
        
        # Log detailed parameter information
        data_type = type(data).__name__
        data_shape = getattr(data, 'shape', f"length={len(data)}")
        
        self.logger.info("=" * 80)
        self.logger.info("🔄 CLUSTER-BASED SPLIT - DETAILED PARAMETERS")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Data Information:")
        self.logger.info(f"   • Data type: {data_type}")
        self.logger.info(f"   • Data shape/size: {data_shape}")
        
        # Check if DataFrame and log column info
        if hasattr(data, 'columns'):
            self.logger.info(f"   • DataFrame columns: {list(data.columns)}")
            if 'sequence' in data.columns or 'seq' in data.columns:
                seq_col = 'sequence' if 'sequence' in data.columns else 'seq'
                self.logger.info(f"   • Sequence column: '{seq_col}'")
        
        # Log labels information if provided
        if labels is not None:
            unique_labels = len(set(labels))
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            self.logger.info(f"   • Labels provided: {len(labels)} labels with {unique_labels} unique values")
            self.logger.info(f"   • Label distribution: {dict(sorted(label_counts.items()))}")
        else:
            self.logger.info(f"   • Labels: None (unsupervised mode)")
        
        self.logger.info(f"🎯 Split Configuration:")
        self.logger.info(f"   • Training fraction: {frac_train:.3f} ({frac_train*100:.1f}%)")
        self.logger.info(f"   • Validation fraction: {frac_valid:.3f} ({frac_valid*100:.1f}%)")
        self.logger.info(f"   • Test fraction: {frac_test:.3f} ({frac_test*100:.1f}%)")
        self.logger.info(f"   • Total fraction sum: {frac_train + frac_valid + frac_test:.6f}")
        
        self.logger.info(f"🎲 Random Seed Configuration:")
        if seed is None:
            effective_seed = self.random_seed
            self.logger.info(f"   • Provided seed: None (using default: {effective_seed})")
        else:
            effective_seed = seed
            self.logger.info(f"   • Provided seed: {seed}")
        self.logger.info(f"   • Effective seed: {effective_seed}")
        
        self.logger.info(f"🔧 Clustering Strategy:")
        self.logger.info(f"   • Distribution strategy: '{cluster_distribution_strategy}'")
        self.logger.info(f"   • Preserve cluster integrity: {preserve_cluster_integrity}")
        if cluster_distribution_strategy in ["label_balanced", "label_balanced_free"]:
            self.logger.info(f"   • Label balancing: ENABLED (strategy: {cluster_distribution_strategy})")
        else:
            self.logger.info(f"   • Label balancing: DISABLED")
        if balance_labels is not None:
            self.logger.info(f"   • [DEPRECATED] balance_labels parameter: {balance_labels}")
        
        # Log clustering-specific parameters
        if kwargs:
            self.logger.info(f"⚙️  Clustering Parameters:")
            for key, value in kwargs.items():
                if key not in ['cluster_distribution_strategy', 'preserve_cluster_integrity', 'balance_labels']:
                    self.logger.info(f"   • {key}: {value}")
        else:
            self.logger.info(f"⚙️  Clustering Parameters: Using default parameters")
        
        self.logger.info(f"🏭 Splitter Class: {self.__class__.__name__}")
        self.logger.info("=" * 80)
        
        self.logger.info(f"🚀 Starting cluster-based split execution...")
        
        # Validate fractions
        self.validate_fractions(frac_train, frac_valid, frac_test)
        
        # Set random seed
        if seed is None:
            seed = self.random_seed
        if seed is not None:
            np.random.seed(seed)
        
        # Extract sequences from input data
        sequences = self._extract_sequences(data)
        # Store sequences for later use in analysis
        self._current_sequences = sequences
        
        # Validate labels if provided
        if labels is not None and len(labels) != len(sequences):
            raise ValueError(f"Labels length ({len(labels)}) must match sequences length ({len(sequences)})")
        
        # Check if we can use cached clustering results
        clustering_params = {k: v for k, v in kwargs.items() if k not in ['cluster_distribution_strategy', 'preserve_cluster_integrity', 'balance_labels']}
        
        if self._should_use_cache(sequences, **clustering_params):
            self.logger.info("Using cached clustering results")
            cluster_result = self._last_cluster_result
        else:
            # Get clustering result
            self.logger.info(f"Running clustering with {self.__class__.__name__}")
            # Remove labels from clustering_params if it exists to avoid duplicate argument
            clustering_params_copy = clustering_params.copy()
            if 'labels' in clustering_params_copy:
                del clustering_params_copy['labels']
            cluster_result = self._get_clustering_result(sequences, labels, **clustering_params_copy)
            self._update_cache_unified(sequences, cluster_result, **clustering_params)

        # Extract cluster information from unified result
        cluster_assignments = cluster_result.cluster_assignments
        total_clusters = cluster_result.total_clusters
        total_sequences = cluster_result.total_sequences
        
        # Randomly shuffle cluster assignments to avoid label bias
        # This helps prevent issues where similar sequences (with same labels) 
        # are clustered together and then assigned to the same split
        cluster_items = list(cluster_assignments.items())
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(cluster_items)
        cluster_assignments = dict(cluster_items)
        
        self.logger.info(f"🔀 Randomly shuffled {len(cluster_assignments)} clusters to reduce label bias")
        
        # Log detailed clustering results using cluster utils
        log_cluster_analysis(cluster_result, sequences, self.logger)
        
        # Generate splits based on strategy
        self.logger.info("🔧 EXECUTING SPLIT STRATEGY")
        self.logger.info("=" * 80)
        
        # Handle label balancing strategies first
        if cluster_distribution_strategy == "label_balanced_free":
            self.logger.info(f"⚖️  Using label-balanced free split strategy:")
            self.logger.info(f"   • Strategy: '{cluster_distribution_strategy}'")
            self.logger.info(f"   • Cluster integrity: BROKEN (for optimal label balancing)")
            self.logger.info(f"   • Labels available: {len(set(labels))} unique labels")
            
            split_result = self._split_by_label_balance_free(
                cluster_assignments, labels, frac_train, frac_valid, frac_test, seed
            )
            
        elif cluster_distribution_strategy == "label_balanced":
            self.logger.info(f"⚖️  Using label-balanced cluster-preserving split strategy:")
            self.logger.info(f"   • Strategy: '{cluster_distribution_strategy}'")
            self.logger.info(f"   • Cluster integrity: PRESERVED (sequences from same cluster stay together)")
            self.logger.info(f"   • Labels available: {len(set(labels))} unique labels")
            
            split_result = self._split_by_label_balance_clusters(
                cluster_assignments, labels, frac_train, frac_valid, frac_test, seed
            )
            
        elif preserve_cluster_integrity:
            self.logger.info(f"🔒 Using cluster-preserving split strategy:")
            self.logger.info(f"   • Strategy: '{cluster_distribution_strategy}'")
            self.logger.info(f"   • Cluster integrity: PRESERVED (sequences from same cluster stay together)")
            
            split_result = self._split_by_clusters(
                cluster_assignments, total_sequences, frac_train, frac_valid, frac_test, 
                cluster_distribution_strategy, seed
            )
        else:
            self.logger.info(f"🔓 Using cluster-breaking split strategy:")
            self.logger.info(f"   • Cluster integrity: BROKEN (sequences from same cluster can be separated)")
            
            split_result = self._split_within_clusters(
                cluster_assignments, frac_train, frac_valid, frac_test, seed
            )
        
        # Handle deprecated balance_labels for backward compatibility
        if balance_labels is not None and balance_labels and cluster_distribution_strategy not in ["label_balanced", "label_balanced_free"]:
            self.logger.warning(f"⚠️  [DEPRECATED] balance_labels=True is being applied as fallback - please use cluster_distribution_strategy instead")
            self.logger.info(f"⚖️  Applying fallback label balancing:")
            self.logger.info(f"   • Labels available: {len(set(labels))} unique labels")
            self.logger.info(f"   • Balancing method: {'cluster-aware' if preserve_cluster_integrity else 'free'}")
            
            split_result = self._balance_labels_across_splits(
                split_result, labels, frac_train, frac_valid, frac_test, 
                cluster_assignments, preserve_cluster_integrity, seed
            )
        
        self._last_split_result = split_result
        
        # Log split statistics
        self._log_split_statistics(split_result, total_clusters, total_sequences)
        
        # Log label distribution if labels are provided
        if labels is not None:
            self._log_label_distribution_with_analyzer(split_result, sequences, labels)
        if return_cluster_assignments:
            return split_result, cluster_assignments
        return split_result
    
    def _extract_sequences(self, data: Union[List[str], np.ndarray]) -> List[str]:
        """Extract sequences from input data (simplified version for sequence-only input)."""
        # Handle pandas DataFrame
        if hasattr(data, 'columns'):
            if 'sequence' in data.columns:
                return data['sequence'].tolist()
            elif 'seq' in data.columns:
                return data['seq'].tolist()
            else:
                raise ValueError("DataFrame must have 'sequence' or 'seq' column")
        
        # Handle list or array
        elif isinstance(data, (list, np.ndarray)):
            if len(data) > 0 and isinstance(data[0], str):
                return list(data)
            else:
                raise ValueError("List/array must contain string sequences")
        else:
            raise ValueError("Data must be a list of strings or DataFrame with sequence column")
    
    def _extract_sequences_and_labels(self, data: Union[List[str], np.ndarray]) -> Tuple[List[str], Optional[List[int]]]:
        """Extract sequences and labels from input data."""
        # Handle pandas DataFrame
        if hasattr(data, 'columns'):
            if 'sequence' in data.columns:
                sequences = data['sequence'].tolist()
            elif 'seq' in data.columns:
                sequences = data['seq'].tolist()
            else:
                raise ValueError("DataFrame must have 'sequence' or 'seq' column")
            
            # Extract labels if available
            labels = None
            if 'label' in data.columns:
                labels = data['label'].tolist()
            elif 'target' in data.columns:
                labels = data['target'].tolist()
            
            return sequences, labels
        
        # Handle list or array
        elif isinstance(data, (list, np.ndarray)):
            if len(data) > 0 and isinstance(data[0], str):
                return list(data), None
            else:
                raise ValueError("List/array must contain string sequences")
        else:
            raise ValueError("Data must be a list of strings or DataFrame with sequence column")
    
    def _split_by_clusters(
        self,
        cluster_assignments: Dict[str, List[int]],
        total_sequences: int,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        strategy: str,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """Split by assigning entire clusters to splits."""
        if len(cluster_assignments) < 3:
            self.logger.warning(f"Only {len(cluster_assignments)} clusters available, may not achieve desired split ratios")
        
        # Get cluster items (cluster_id, sequence_indices)
        cluster_items = list(cluster_assignments.items())
        
        # Initialize splits
        splits = {"train": [], "valid": [], "test": []}
        split_sizes = {"train": 0, "valid": 0, "test": 0}
        
        # Fixed target size calculation to avoid rounding errors
        train_target = int(frac_train * total_sequences)
        valid_target = int(frac_valid * total_sequences)
        # Ensure all sequences are accounted for
        test_target = total_sequences - train_target - valid_target
        
        target_sizes = {
            "train": train_target,
            "valid": valid_target,
            "test": test_target
        }
        
        if strategy == "size_aware":
            # Sort by cluster size (largest first) for better distribution
            cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
            
            for cluster_id, seq_indices in cluster_items:
                cluster_size = len(seq_indices)
                
                # Calculate current ratios and find the split that's most behind its target
                current_ratios = {}
                target_ratios = {"train": frac_train, "valid": frac_valid, "test": frac_test}
                
                for split_name in splits.keys():
                    current_ratio = split_sizes[split_name] / total_sequences if total_sequences > 0 else 0
                    # Calculate how much this split is behind its target ratio
                    ratio_deficit = target_ratios[split_name] - current_ratio
                    current_ratios[split_name] = ratio_deficit
                
                # Choose the split with the largest ratio deficit (most behind target)
                best_split = max(current_ratios.keys(), key=lambda s: current_ratios[s])
                
                # Assign cluster to best split
                splits[best_split].extend(seq_indices)
                split_sizes[best_split] += cluster_size
                
        elif strategy == "random":
            # Completely random assignment
            if seed is not None:
                np.random.seed(seed)
            
            split_names = ["train", "valid", "test"]
            for cluster_id, seq_indices in cluster_items:
                # Randomly choose a split
                split_name = np.random.choice(split_names)
                splits[split_name].extend(seq_indices)
                split_sizes[split_name] += len(seq_indices)
                
        elif strategy == "balanced":
            # Balanced assignment - try to maintain target proportions
            if seed is not None:
                np.random.seed(seed)
            
            # Shuffle clusters for randomness
            np.random.shuffle(cluster_items)
            
            for cluster_id, seq_indices in cluster_items:
                cluster_size = len(seq_indices)
                
                # Calculate how far each split is from its target
                split_deficits = {}
                for split_name in ["train", "valid", "test"]:
                    current_ratio = split_sizes[split_name] / total_sequences if total_sequences > 0 else 0
                    if split_name == "train":
                        target_ratio = frac_train
                    elif split_name == "valid":
                        target_ratio = frac_valid
                    else:
                        target_ratio = frac_test
                    
                    split_deficits[split_name] = target_ratio - current_ratio
                
                # Assign to the split with largest deficit
                best_split = max(split_deficits.keys(), key=split_deficits.get)
                splits[best_split].extend(seq_indices)
                split_sizes[best_split] += cluster_size
        
        elif strategy == "sort_cluster_size":
            # Sort clusters by size (largest first) and fill train -> valid -> test sequentially
            cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
            
            if seed is not None:
                np.random.seed(seed)
            
            # Sequential assignment: fill train first, then valid, then test
            for cluster_id, seq_indices in cluster_items:
                cluster_size = len(seq_indices)
                
                # Try to assign to train first if it's not full
                if split_sizes["train"] + cluster_size <= train_target:
                    splits["train"].extend(seq_indices)
                    split_sizes["train"] += cluster_size
                # If train is full, try to assign to valid
                elif split_sizes["valid"] + cluster_size <= valid_target:
                    splits["valid"].extend(seq_indices)
                    split_sizes["valid"] += cluster_size
                # If both train and valid are full, assign to test
                else:
                    splits["test"].extend(seq_indices)
                    split_sizes["test"] += cluster_size
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Sort indices for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        return splits
    
    def _split_within_clusters(
        self,
        cluster_assignments: Dict[str, List[int]],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """Split sequences within clusters (break cluster integrity)."""
        splits = {"train": [], "valid": [], "test": []}
        
        if seed is not None:
            np.random.seed(seed)
        
        for cluster_id, seq_indices in cluster_assignments.items():
            # Shuffle sequences within cluster
            shuffled_indices = seq_indices.copy()
            np.random.shuffle(shuffled_indices)
            
            # Split within cluster
            n_seqs = len(shuffled_indices)
            n_train = int(n_seqs * frac_train)
            n_valid = int(n_seqs * frac_valid)
            
            splits["train"].extend(shuffled_indices[:n_train])
            splits["valid"].extend(shuffled_indices[n_train:n_train + n_valid])
            splits["test"].extend(shuffled_indices[n_train + n_valid:])
        
        # Sort indices for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        return splits
    
    def _balance_labels_across_splits(
        self,
        split_result: Dict[str, List[int]],
        labels: List[int],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        cluster_assignments: Dict[str, List[int]],
        preserve_cluster_integrity: bool,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """
        Balance class labels across splits while respecting cluster constraints.
        """
        if not preserve_cluster_integrity:
            # If cluster integrity is not preserved, we can freely rebalance
            return self._rebalance_labels_freely(split_result, labels, frac_train, frac_valid, frac_test, seed)
        else:
            # If cluster integrity must be preserved, we can only move entire clusters
            return self._rebalance_labels_by_clusters(split_result, labels, cluster_assignments, frac_train, frac_valid, frac_test, seed)
    
    def _rebalance_labels_freely(
        self,
        split_result: Dict[str, List[int]],
        labels: List[int],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """Rebalance labels freely without cluster constraints."""
        if seed is not None:
            np.random.seed(seed)
        
        # Get all sequences and their labels
        all_indices = []
        for split_indices in split_result.values():
            all_indices.extend(split_indices)
        
        # Group by labels
        from collections import defaultdict
        label_groups = defaultdict(list)
        for idx in all_indices:
            if idx < len(labels):
                label_groups[labels[idx]].append(idx)
        
        # Shuffle each label group
        for label_indices in label_groups.values():
            np.random.shuffle(label_indices)
        
        # Redistribute maintaining label proportions
        new_splits = {"train": [], "valid": [], "test": []}
        
        for label, indices in label_groups.items():
            n_indices = len(indices)
            n_train = int(n_indices * frac_train)
            n_valid = int(n_indices * frac_valid)
            
            new_splits["train"].extend(indices[:n_train])
            new_splits["valid"].extend(indices[n_train:n_train + n_valid])
            new_splits["test"].extend(indices[n_train + n_valid:])
        
        # Sort indices for consistency
        for split_name in new_splits:
            new_splits[split_name] = sorted(new_splits[split_name])
        
        return new_splits
    
    def _rebalance_labels_by_clusters(
        self,
        split_result: Dict[str, List[int]],
        labels: List[int],
        cluster_assignments: Dict[str, List[int]],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """Rebalance labels by moving entire clusters between splits."""
        if seed is not None:
            np.random.seed(seed)
        
        # For simplicity, return the original split_result
        # A more sophisticated implementation could move clusters to improve balance
        return split_result
    
    def _split_by_label_balance_free(
        self,
        cluster_assignments: Dict[str, List[int]],
        labels: List[int],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """
        Split sequences by balancing labels freely (cluster integrity not preserved).
        
        This method groups sequences by their labels and distributes each label group
        proportionally across train/valid/test splits, ignoring cluster boundaries.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get all sequence indices from clusters
        all_indices = []
        for seq_indices in cluster_assignments.values():
            all_indices.extend(seq_indices)
        
        # Group sequences by labels
        from collections import defaultdict
        label_groups = defaultdict(list)
        for idx in all_indices:
            if idx < len(labels):
                label_groups[labels[idx]].append(idx)
        
        # Log label distribution info
        total_seqs = len(all_indices)
        self.logger.info(f"📊 Label Distribution Analysis:")
        for label, indices in sorted(label_groups.items()):
            count = len(indices)
            percentage = count / total_seqs * 100 if total_seqs > 0 else 0
            self.logger.info(f"   • Label {label}: {count} sequences ({percentage:.1f}%)")
        
        # Shuffle each label group for randomness
        for label_indices in label_groups.values():
            np.random.shuffle(label_indices)
        
        # Distribute each label group proportionally across splits
        splits = {"train": [], "valid": [], "test": []}
        
        for label, indices in label_groups.items():
            n_indices = len(indices)
            n_train = int(n_indices * frac_train)
            n_valid = int(n_indices * frac_valid)
            # Ensure all sequences are assigned
            n_test = n_indices - n_train - n_valid
            
            splits["train"].extend(indices[:n_train])
            splits["valid"].extend(indices[n_train:n_train + n_valid])
            splits["test"].extend(indices[n_train + n_valid:n_train + n_valid + n_test])
        
        # Sort indices for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        # Log final split distribution by labels
        self.logger.info(f"⚖️  Final Label Balance Verification:")
        for split_name in ["train", "valid", "test"]:
            split_indices = splits[split_name]
            split_label_counts = defaultdict(int)
            for idx in split_indices:
                if idx < len(labels):
                    split_label_counts[labels[idx]] += 1
            
            split_total = len(split_indices)
            self.logger.info(f"   • {split_name.capitalize()} ({split_total} total):")
            for label in sorted(split_label_counts.keys()):
                count = split_label_counts[label]
                percentage = count / split_total * 100 if split_total > 0 else 0
                self.logger.info(f"     - Label {label}: {count} ({percentage:.1f}%)")
        
        return splits
    
    def _split_by_label_balance_clusters(
        self,
        cluster_assignments: Dict[str, List[int]],
        labels: List[int],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int]
    ) -> Dict[str, List[int]]:
        """
        Split by balancing labels while preserving cluster integrity.
        
        This method analyzes the label composition of each cluster and distributes
        clusters to splits in a way that balances label distribution while keeping
        all sequences from the same cluster together.
        """
        if seed is not None:
            np.random.seed(seed)
        
        from collections import defaultdict, Counter
        
        # Analyze label composition of each cluster
        cluster_label_info = {}
        total_sequences = 0
        
        for cluster_id, seq_indices in cluster_assignments.items():
            cluster_labels = [labels[idx] for idx in seq_indices if idx < len(labels)]
            label_counts = Counter(cluster_labels)
            cluster_label_info[cluster_id] = {
                'seq_indices': seq_indices,
                'size': len(seq_indices),
                'label_counts': label_counts,
                'dominant_label': label_counts.most_common(1)[0][0] if label_counts else 0
            }
            total_sequences += len(seq_indices)
        
        # Group clusters by their dominant label
        label_cluster_groups = defaultdict(list)
        for cluster_id, info in cluster_label_info.items():
            dominant_label = info['dominant_label']
            label_cluster_groups[dominant_label].append((cluster_id, info))
        
        # Log cluster-label analysis
        self.logger.info(f"📊 Cluster-Label Composition Analysis:")
        for label in sorted(label_cluster_groups.keys()):
            clusters = label_cluster_groups[label]
            total_seqs_for_label = sum(info['size'] for _, info in clusters)
            percentage = total_seqs_for_label / total_sequences * 100 if total_sequences > 0 else 0
            self.logger.info(f"   • Label {label}: {len(clusters)} clusters, {total_seqs_for_label} sequences ({percentage:.1f}%)")
        
        # Initialize splits
        splits = {"train": [], "valid": [], "test": []}
        split_sizes = {"train": 0, "valid": 0, "test": 0}
        split_label_counts = {"train": defaultdict(int), "valid": defaultdict(int), "test": defaultdict(int)}
        
        # Target sizes for each split
        train_target = int(frac_train * total_sequences)
        valid_target = int(frac_valid * total_sequences)
        test_target = total_sequences - train_target - valid_target
        
        # For each label group, distribute clusters to maintain proportions
        for label, cluster_list in label_cluster_groups.items():
            # Shuffle clusters within this label group
            np.random.shuffle(cluster_list)
            
            # Sort by cluster size (largest first) for better distribution
            cluster_list.sort(key=lambda x: x[1]['size'], reverse=True)
            
            for cluster_id, info in cluster_list:
                cluster_size = info['size']
                seq_indices = info['seq_indices']
                
                # Calculate current label ratios for each split
                split_scores = {}
                for split_name in ["train", "valid", "test"]:
                    # Current size ratio
                    current_ratio = split_sizes[split_name] / total_sequences if total_sequences > 0 else 0
                    if split_name == "train":
                        target_ratio = frac_train
                    elif split_name == "valid":
                        target_ratio = frac_valid
                    else:
                        target_ratio = frac_test
                    
                    size_deficit = target_ratio - current_ratio
                    
                    # Current label ratio for this specific label
                    current_label_count = split_label_counts[split_name][label]
                    total_split_size = split_sizes[split_name]
                    current_label_ratio = current_label_count / total_split_size if total_split_size > 0 else 0
                    
                    # We want balanced label distribution, so prefer splits that need this label
                    label_balance_score = 1.0 / (current_label_ratio + 0.1)  # Higher score for underrepresented labels
                    
                    # Combine size and label balance considerations
                    split_scores[split_name] = size_deficit * 2 + label_balance_score * 0.5
                
                # Choose the split with the highest score
                best_split = max(split_scores.keys(), key=split_scores.get)
                
                # Assign cluster to best split
                splits[best_split].extend(seq_indices)
                split_sizes[best_split] += cluster_size
                
                # Update label counts for this split
                for idx in seq_indices:
                    if idx < len(labels):
                        split_label_counts[best_split][labels[idx]] += 1
        
        # Sort indices for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        # Log final distribution
        self.logger.info(f"⚖️  Final Cluster-Aware Label Balance:")
        for split_name in ["train", "valid", "test"]:
            split_total = split_sizes[split_name]
            self.logger.info(f"   • {split_name.capitalize()} ({split_total} total):")
            for label in sorted(split_label_counts[split_name].keys()):
                count = split_label_counts[split_name][label]
                percentage = count / split_total * 100 if split_total > 0 else 0
                self.logger.info(f"     - Label {label}: {count} ({percentage:.1f}%)")
        
        return splits
    
    def _log_split_statistics(
        self,
        split_result: Dict[str, List[int]],
        total_clusters: int,
        total_sequences: int
    ):
        """Log detailed statistics about the split using SplitAnalyzer."""
        try:
            # Create SplitAnalyzer instance with sequences from current data
            # Note: sequences should be available from the most recent split operation
            sequences = getattr(self, '_current_sequences', [])
            if not sequences:
                # Fallback: create empty sequences list for the analyzer
                sequences = [''] * total_sequences
            
            analyzer = SplitAnalyzer(sequences=sequences)
            
            # Use SplitAnalyzer to log detailed statistics
            analyzer.log_split_statistics_detailed(
                split_result=split_result,
                total_clusters=total_clusters,
                total_sequences=total_sequences,
                verbose=True
            )
            
        except Exception as e:
            # Fallback to basic logging if SplitAnalyzer fails
            self.logger.warning(f"Failed to use SplitAnalyzer for detailed statistics: {e}")
            self._log_basic_split_statistics(split_result, total_clusters, total_sequences)
    
    def _log_basic_split_statistics(
        self,
        split_result: Dict[str, List[int]],
        total_clusters: int,
        total_sequences: int
    ):
        """Basic fallback logging for split statistics."""
        total_seqs = sum(len(indices) for indices in split_result.values())
        
        self.logger.info("=" * 80)
        self.logger.info("📊 SPLIT RESULTS - BASIC STATISTICS")
        self.logger.info("=" * 80)
        
        # Basic split statistics
        self.logger.info(f"📈 Split Distribution:")
        for split_name, indices in [("train", split_result["train"]), 
                                   ("valid", split_result["valid"]), 
                                   ("test", split_result["test"])]:
            actual_count = len(indices)
            percentage = actual_count / total_seqs * 100 if total_seqs > 0 else 0
            self.logger.info(f"   • {split_name.capitalize():>10}: {actual_count:>6} sequences ({percentage:>5.1f}%)")
        
        self.logger.info(f"")
        self.logger.info(f"🎯 Summary:")
        self.logger.info(f"   • Total clusters processed: {total_clusters}")
        self.logger.info(f"   • Total sequences processed: {total_sequences}")
        self.logger.info(f"   • Total sequences in splits: {total_seqs}")
        
        self.logger.info("=" * 80)
        self.logger.info("✅ SPLIT EXECUTION COMPLETED")
        self.logger.info("=" * 80)
    
    def _log_label_distribution_with_analyzer(
        self,
        split_result: Dict[str, List[int]],
        sequences: List[str],
        labels: List[int]
    ):
        """Log label distribution statistics using SplitAnalyzer."""
        try:
            # Create SplitAnalyzer instance with sequences and labels
            analyzer = SplitAnalyzer(sequences=sequences, labels=labels)
            
            # Use SplitAnalyzer's built-in label distribution analysis
            analyzer.log_label_distribution_analysis(
                split_result=split_result,
                verbose=True
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze label distribution: {e}")
    
    def _update_cache_unified(self, data: List[str], cluster_result, **params) -> None:
        """
        Update the cache with new clustering results (unified version).
        """
        self._last_cluster_result = cluster_result
        # Also maintain legacy cluster_map format for compatibility
        if hasattr(cluster_result, 'cluster_assignments'):
            self.cluster_map = cluster_result.cluster_assignments
        else:
            self.cluster_map = cluster_result
        self._cached_data_hash = self.get_data_hash(data)
        self._cached_params = params.copy()

    def get_split_kfold_indices(
        self,
        data: List[str],
        k_folds: int = 5,
        seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate k-fold cross-validation splits based on clustering.

        Args:
            data: List of items to split
            k_folds: Number of folds for cross-validation (default: 5)
            seed: Random seed for reproducibility (default: 42)
            **kwargs: Clustering-specific parameters

        Returns:
            Dictionary with fold results in format "fold_X"
        """
        self.logger.info(
            f"Starting clustering-aware k-fold split: data_size={len(data)}, "
            f"k_folds={k_folds}, seed={seed}"
        )

        if k_folds <= 1:
            raise ValueError(f"k_folds must be greater than 1, got {k_folds}")

        # Get clustering results
        cluster_map = self._get_or_create_clusters(data, **kwargs)
        cluster_items = list(cluster_map.items())

        # Shuffle clusters using fixed seed
        if seed is not None:
            rng = np.random.RandomState(seed)
            rng.shuffle(cluster_items)
        else:
            np.random.shuffle(cluster_items)

        # Distribute clusters as evenly as possible across k folds
        folds = self.distribute_clusters_to_folds(cluster_items, k_folds)

        # Generate k-fold split results
        kfold_results = self.generate_kfold_results_from_folds(folds, data, k_folds, seed)

        self.logger.info(
            f"All {k_folds} clustering-aware k-fold splits completed successfully"
        )
        return kfold_results

    def get_split_indices_n(
        self,
        data: Union[List[str], np.ndarray],
        n_splits: int = 5,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        labels: Optional[List[int]] = None,
        seed: Union[List[int], int] = 42,
        cluster_distribution_strategy: str = "sort_cluster_size",
        preserve_cluster_integrity: bool = True,
        balance_labels: Optional[bool] = None,  # Deprecated: use cluster_distribution_strategy instead
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate n random splits using the same clustering with support for label balancing.

        Args:
            data: Input data (list of strings or numpy array of strings)
            n_splits: Number of random splits to generate
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            labels: Optional class labels for each sequence (required for label balancing strategies)
            seed: Random seed for reproducibility (int or list of ints)
            cluster_distribution_strategy: How to distribute clusters across splits
                - "size_aware": Assign largest clusters first to balance sizes
                - "random": Randomly assign clusters to splits
                - "balanced": Shuffle clusters then assign by deficit to balance sequence counts
                - "label_balanced": Balance class labels while preserving cluster integrity
                - "label_balanced_free": Balance class labels freely (may break cluster integrity)
                - "sort_cluster_size": Sort clusters by size (largest first) and fill train->valid->test sequentially
            preserve_cluster_integrity: Keep sequences from same cluster together (ignored for label_balanced_free)
            balance_labels: [DEPRECATED] Use cluster_distribution_strategy="label_balanced" instead
            **kwargs: Clustering method specific parameters

        Returns:
            Dictionary with split results in format "seed_X"
        """
        # Handle deprecated balance_labels parameter for backward compatibility
        if balance_labels is not None:
            import warnings
            warnings.warn(
                "The 'balance_labels' parameter is deprecated. Use cluster_distribution_strategy='label_balanced' or "
                "'label_balanced_free' instead.", 
                DeprecationWarning, 
                stacklevel=2
            )
            
            if balance_labels and cluster_distribution_strategy in ["size_aware", "random", "balanced", "sort_cluster_size"]:
                # Auto-convert to new strategy based on preserve_cluster_integrity
                if preserve_cluster_integrity:
                    cluster_distribution_strategy = "label_balanced"
                    self.logger.info("🔄 Auto-converted balance_labels=True to cluster_distribution_strategy='label_balanced'")
                else:
                    cluster_distribution_strategy = "label_balanced_free"
                    self.logger.info("🔄 Auto-converted balance_labels=True to cluster_distribution_strategy='label_balanced_free'")
        
        # Validate cluster_distribution_strategy
        valid_strategies = ["size_aware", "random", "balanced", "label_balanced", "label_balanced_free", "sort_cluster_size"]
        if cluster_distribution_strategy not in valid_strategies:
            raise ValueError(f"cluster_distribution_strategy must be one of {valid_strategies}, got '{cluster_distribution_strategy}'")
        
        # Check label requirements for label balancing strategies
        is_label_strategy = cluster_distribution_strategy in ["label_balanced", "label_balanced_free"]
        if is_label_strategy and labels is None:
            raise ValueError(f"cluster_distribution_strategy='{cluster_distribution_strategy}' requires labels to be provided")
        
        # Log detailed parameter information
        data_type = type(data).__name__
        data_shape = getattr(data, 'shape', f"length={len(data)}")
        
        self.logger.info("=" * 80)
        self.logger.info("🔄 N-SPLITS CLUSTER-BASED GENERATION - DETAILED PARAMETERS")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Data Information:")
        self.logger.info(f"   • Data type: {data_type}")
        self.logger.info(f"   • Data shape/size: {data_shape}")
        
        # Check if DataFrame and log column info
        if hasattr(data, 'columns'):
            self.logger.info(f"   • DataFrame columns: {list(data.columns)}")
            if 'sequence' in data.columns or 'seq' in data.columns:
                seq_col = 'sequence' if 'sequence' in data.columns else 'seq'
                self.logger.info(f"   • Sequence column: '{seq_col}'")
        
        # Log labels information if provided
        if labels is not None:
            unique_labels = len(set(labels))
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            self.logger.info(f"   • Labels provided: {len(labels)} labels with {unique_labels} unique values")
            self.logger.info(f"   • Label distribution: {dict(sorted(label_counts.items()))}")
        else:
            self.logger.info(f"   • Labels: None (unsupervised mode)")
        
        self.logger.info(f"🎯 Split Configuration:")
        self.logger.info(f"   • Number of splits: {n_splits}")
        self.logger.info(f"   • Training fraction: {frac_train:.3f} ({frac_train*100:.1f}%)")
        self.logger.info(f"   • Validation fraction: {frac_valid:.3f} ({frac_valid*100:.1f}%)")
        self.logger.info(f"   • Test fraction: {frac_test:.3f} ({frac_test*100:.1f}%)")
        self.logger.info(f"   • Total fraction sum: {frac_train + frac_valid + frac_test:.6f}")
        
        # Handle seed information
        self.logger.info(f"🎲 Random Seed Configuration:")
        if isinstance(seed, list):
            if len(seed) != n_splits:
                raise ValueError(f"Seed list length ({len(seed)}) must match n_splits ({n_splits})")
            self.logger.info(f"   • Seed type: List of {len(seed)} seeds")
            self.logger.info(f"   • Seeds: {seed}")
        else:
            self.logger.info(f"   • Seed type: Single seed (will generate sequence)")
            self.logger.info(f"   • Base seed: {seed}")
            self.logger.info(f"   • Generated seeds: {[seed + i for i in range(n_splits)]}")
        
        self.logger.info(f"🔧 Clustering Strategy:")
        self.logger.info(f"   • Distribution strategy: '{cluster_distribution_strategy}'")
        self.logger.info(f"   • Preserve cluster integrity: {preserve_cluster_integrity}")
        if cluster_distribution_strategy in ["label_balanced", "label_balanced_free"]:
            self.logger.info(f"   • Label balancing: ENABLED (strategy: {cluster_distribution_strategy})")
        else:
            self.logger.info(f"   • Label balancing: DISABLED")
        if balance_labels is not None:
            self.logger.info(f"   • [DEPRECATED] balance_labels parameter: {balance_labels}")
        
        # Log clustering-specific parameters
        if kwargs:
            self.logger.info(f"⚙️  Clustering Parameters:")
            for key, value in kwargs.items():
                if key not in ['cluster_distribution_strategy', 'preserve_cluster_integrity', 'balance_labels']:
                    self.logger.info(f"   • {key}: {value}")
        else:
            self.logger.info(f"⚙️  Clustering Parameters: Using default parameters")
        
        self.logger.info(f"🏭 Splitter Class: {self.__class__.__name__}")
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 Starting {n_splits} cluster-based splits generation...")

        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")

        # Validate fractions
        self.validate_fractions(frac_train, frac_valid, frac_test)

        # Process seed arguments
        seeds = self.prepare_seeds(seed, n_splits)

        # Extract sequences from input data
        sequences = self._extract_sequences(data)
        
        # Validate labels if provided
        if labels is not None and len(labels) != len(sequences):
            raise ValueError(f"Labels length ({len(labels)}) must match sequences length ({len(sequences)})")

        # Get clustering results (run only once)
        clustering_params = {k: v for k, v in kwargs.items() if k not in ['cluster_distribution_strategy', 'preserve_cluster_integrity', 'balance_labels']}
        
        # Get clustering result
        self.logger.info(f"🧬 Running clustering with {self.__class__.__name__} (shared across all splits)")
        if 'labels' in clustering_params:
            del clustering_params['labels']
        cluster_result = self._get_clustering_result(sequences, labels, **clustering_params)
        
        # Extract cluster information from unified result
        cluster_assignments = cluster_result.cluster_assignments
        total_clusters = cluster_result.total_clusters
        total_sequences = cluster_result.total_sequences
        
        # Log detailed clustering results using cluster utils
        log_cluster_analysis(cluster_result, sequences, self.logger)

        split_results = {}
        for i, current_seed in enumerate(seeds):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔄 GENERATING SPLIT {i + 1}/{n_splits} (seed: {current_seed})")
            self.logger.info(f"{'='*60}")
            
            # Randomly shuffle cluster assignments for this split to avoid label bias
            cluster_items = list(cluster_assignments.items())
            np.random.seed(current_seed)
            np.random.shuffle(cluster_items)
            shuffled_cluster_assignments = dict(cluster_items)
            
            self.logger.info(f"🔀 Randomly shuffled {len(shuffled_cluster_assignments)} clusters for split {i + 1}")

            # Generate split using the same logic as get_split_indices
            if cluster_distribution_strategy == "label_balanced_free":
                self.logger.info(f"⚖️  Using label-balanced free split strategy")
                split_indices = self._split_by_label_balance_free(
                    shuffled_cluster_assignments, labels, frac_train, frac_valid, frac_test, current_seed
                )
                
            elif cluster_distribution_strategy == "label_balanced":
                self.logger.info(f"⚖️  Using label-balanced cluster-preserving split strategy")
                split_indices = self._split_by_label_balance_clusters(
                    shuffled_cluster_assignments, labels, frac_train, frac_valid, frac_test, current_seed
                )
                
            elif preserve_cluster_integrity:
                self.logger.info(f"🔒 Using cluster-preserving strategy: '{cluster_distribution_strategy}'")
                split_indices = self._split_by_clusters(
                    shuffled_cluster_assignments, total_sequences, frac_train, frac_valid, frac_test, 
                    cluster_distribution_strategy, current_seed
                )
            else:
                self.logger.info(f"🔓 Using cluster-breaking strategy")
                split_indices = self._split_within_clusters(
                    shuffled_cluster_assignments, frac_train, frac_valid, frac_test, current_seed
                )

            # Handle deprecated balance_labels for backward compatibility
            if balance_labels is not None and balance_labels and cluster_distribution_strategy not in ["label_balanced", "label_balanced_free"]:
                self.logger.warning(f"⚠️  [DEPRECATED] Applying fallback label balancing for split {i + 1}")
                split_indices = self._balance_labels_across_splits(
                    split_indices, labels, frac_train, frac_valid, frac_test, 
                    shuffled_cluster_assignments, preserve_cluster_integrity, current_seed
                )

            # Validate split results
            if not self.validate_split_results(split_indices, len(sequences)):
                self.logger.warning(f"⚠️  Split {i + 1} validation failed")

            # Log split statistics
            train_count = len(split_indices['train'])
            valid_count = len(split_indices['valid'])
            test_count = len(split_indices['test'])
            total_count = train_count + valid_count + test_count
            
            self.logger.info(f"📊 Split {i + 1} Results:")
            self.logger.info(f"   • Train: {train_count:>4} sequences ({train_count/total_count*100:>5.1f}%)")
            self.logger.info(f"   • Valid: {valid_count:>4} sequences ({valid_count/total_count*100:>5.1f}%)")
            self.logger.info(f"   • Test:  {test_count:>4} sequences ({test_count/total_count*100:>5.1f}%)")
            self.logger.info(f"   • Total: {total_count:>4} sequences")
            
            # Log label distribution for this split if labels are provided
            if labels is not None:
                self.logger.info(f"🏷️  Label Distribution for Split {i + 1}:")
                for split_name in ["train", "valid", "test"]:
                    split_indices_list = split_indices[split_name]
                    split_labels = [labels[idx] for idx in split_indices_list if idx < len(labels)]
                    if split_labels:
                        from collections import Counter
                        label_counts = Counter(split_labels)
                        total_split = len(split_labels)
                        label_info = []
                        for label in sorted(label_counts.keys()):
                            count = label_counts[label]
                            percentage = count / total_split * 100 if total_split > 0 else 0
                            label_info.append(f"Label {label}: {count} ({percentage:.1f}%)")
                        self.logger.info(f"   • {split_name.capitalize()}: {', '.join(label_info)}")
            
            self.logger.info(f"✅ Split {i + 1} completed successfully")
            split_results[f"seed_{i}"] = split_indices

        # Final summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎯 N-SPLITS GENERATION SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"✅ Successfully generated {n_splits} cluster-based splits")
        self.logger.info(f"📊 Strategy used: {cluster_distribution_strategy}")
        self.logger.info(f"🧬 Clustering algorithm: {self.__class__.__name__}")
        self.logger.info(f"📈 Total clusters: {total_clusters}")
        self.logger.info(f"📈 Total sequences: {total_sequences}")
        
        if labels is not None:
            unique_labels = len(set(labels))
            self.logger.info(f"🏷️  Label classes: {unique_labels}")
            
        # Calculate average split sizes
        avg_train = sum(len(result['train']) for result in split_results.values()) / n_splits
        avg_valid = sum(len(result['valid']) for result in split_results.values()) / n_splits
        avg_test = sum(len(result['test']) for result in split_results.values()) / n_splits
        
        self.logger.info(f"📊 Average split sizes:")
        self.logger.info(f"   • Train: {avg_train:.1f} sequences")
        self.logger.info(f"   • Valid: {avg_valid:.1f} sequences") 
        self.logger.info(f"   • Test:  {avg_test:.1f} sequences")
        self.logger.info(f"{'='*80}")
        
        return split_results

    # Validation Methods
    def validate_fractions(
        self, frac_train: float, frac_valid: float, frac_test: float
    ) -> None:
        """
        Validate that train/valid/test fractions sum to 1.0.

        Args:
            frac_train: Training fraction
            frac_valid: Validation fraction
            frac_test: Test fraction

        Raises:
            ValueError: If fractions don't sum to approximately 1.0
        """
        total = frac_train + frac_valid + frac_test
        if not np.isclose(total, 1.0, atol=1e-10):
            raise ValueError(f"Train/valid/test fractions must sum to 1.0, got {total}")

        if any(frac < 0 for frac in [frac_train, frac_valid, frac_test]):
            raise ValueError("All fractions must be non-negative")

    def validate_split_keys(self, split_results: Dict[str, Any]) -> None:
        """
        Validate that the split dictionary contains the required keys.

        Args:
            split_results: Dictionary containing split results

        Raises:
            ValueError: If required keys are missing
        """
        required_keys = {"train", "valid", "test"}
        if not required_keys.issubset(split_results.keys()):
            raise ValueError(
                f"Split results must contain {required_keys}, but got {split_results.keys()}"
            )

    def validate_split_indices(
        self, split_results: Dict[str, Union[List[int], np.ndarray]], data_size: int
    ) -> None:
        """
        Validate that indices in splits are within the valid range.

        Args:
            split_results: Dictionary containing split results with indices
            data_size: Size of the original dataset

        Raises:
            TypeError: If indices are not of correct type
            ValueError: If indices are out of bounds
        """
        for key, indices in split_results.items():
            if not isinstance(indices, (list, np.ndarray)):
                raise TypeError(f"Indices for {key} must be a list or numpy array")
            if not all(isinstance(i, (int, np.integer)) for i in indices):
                raise TypeError(f"All indices for {key} must be integers")
            if any(i < 0 or i >= data_size for i in indices):
                raise ValueError(
                    f"Indices for {key} are out of bounds for data of size {data_size}"
                )

    def check_split_completeness(self, all_indices: np.ndarray, data_size: int) -> None:
        """
        Check if the splits cover the entire dataset.

        Args:
            all_indices: Array of all indices from all splits
            data_size: Expected size of the dataset
        """
        if len(all_indices) != data_size:
            self.logger.warning(
                f"Split is not complete. Expected {data_size} unique indices, but got {len(all_indices)}"
            )

    def check_split_overlaps(self, all_indices: np.ndarray, total_indices: int) -> None:
        """
        Check for overlapping indices between splits.

        Args:
            all_indices: Array of unique indices from all splits
            total_indices: Total number of indices across all splits
        """
        if len(all_indices) != total_indices:
            self.logger.warning(
                f"Overlapping indices found. Total indices: {total_indices}, Unique indices: {len(all_indices)}"
            )

    def validate_split_results(
        self,
        split_results: Dict[str, Union[List[int], np.ndarray]],
        data_size: int,
        check_completeness: bool = True,
        check_overlaps: bool = True,
    ) -> bool:
        """
        Validate split results for completeness and non-overlapping indices.

        Args:
            split_results: Dictionary with train/valid/test indices
            data_size: Original data size
            check_completeness: Whether to check if all data points are used
            check_overlaps: Whether to check for overlapping indices

        Returns:
            True if validation passes, False otherwise
        """
        try:
            self.validate_split_keys(split_results)
            self.validate_split_indices(split_results, data_size)

            train_indices = np.array(split_results["train"])
            valid_indices = np.array(split_results["valid"])
            test_indices = np.array(split_results["test"])

            total_indices = len(train_indices) + len(valid_indices) + len(test_indices)
            all_indices = np.unique(
                np.concatenate([train_indices, valid_indices, test_indices])
            )

            if check_completeness:
                self.check_split_completeness(all_indices, data_size)

            if check_overlaps:
                self.check_split_overlaps(all_indices, total_indices)

            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Split validation failed: {e}")
            return False

    # Statistics Methods
    def get_split_statistics(
        self, split_results: Dict[str, Union[List[int], np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the split results.

        Args:
            split_results: Dictionary with train/valid/test indices

        Returns:
            Dictionary containing split statistics:
            {
                "train_size": int,
                "valid_size": int,
                "test_size": int,
                "train_fraction": float,
                "valid_fraction": float,
                "test_fraction": float,
                "total_size": int
            }
        """
        stats = {}
        total_size = sum(len(split_results[key]) for key in ["train", "valid", "test"])

        for split_name in ["train", "valid", "test"]:
            split_size = len(split_results[split_name])
            stats[f"{split_name}_size"] = split_size
            stats[f"{split_name}_fraction"] = (
                split_size / total_size if total_size > 0 else 0
            )

        stats["total_size"] = total_size
        return stats

    # I/O Methods
    def save_split_results(
        self, split_results: Dict[str, Any], filepath: str, format: str = "json"
    ) -> None:
        """
        Save split results to file in specified format.

        Args:
            split_results: Split results to save (can be single split or multiple splits)
            filepath: Output file path
            format: Output format ('json' or 'numpy')

        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
        import json
        import os

        # Only create directory if filepath contains a directory part
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if format == "json":
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in split_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = (
                        value.tolist() if hasattr(value, "tolist") else value
                    )

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2)

        elif format == "numpy":
            np.savez(filepath, **split_results)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Split results saved to {filepath}")

    def load_split_results(self, filepath: str, format: str = "json") -> Dict[str, Any]:
        """
        Load split results from file.

        Args:
            filepath: Input file path
            format: Input format ('json' or 'numpy')

        Returns:
            Dictionary containing split results (automatically converts lists to numpy arrays)

        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        import json

        if format == "json":
            with open(filepath, "r") as f:
                results = json.load(f)
            # Convert lists back to numpy arrays
            for key, value in results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, list):
                            results[key][k] = np.array(v)
                elif isinstance(value, list):
                    results[key] = np.array(value)

        elif format == "numpy":
            results = dict(np.load(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Split results loaded from {filepath}")
        return results

    # Common Utility Methods
    def prepare_seeds(self, seed: Union[List[int], int], n_splits: int) -> List[int]:
        """
        Prepare seeds for multiple splits.

        Args:
            seed: Either a single integer or list of integers for seeding
            n_splits: Number of splits to generate

        Returns:
            List of seeds for each split

        Raises:
            ValueError: If seed list length doesn't match n_splits or invalid type
        """
        if isinstance(seed, list):
            if len(seed) != n_splits:
                raise ValueError(
                    f"Expected {n_splits} seeds, but got {len(seed)}. "
                    f"Please provide a seed for each split."
                )
            return seed
        elif isinstance(seed, int):
            return [seed + i for i in range(n_splits)]
        else:
            raise ValueError(
                "Seed must be an integer or a list of integers. "
                "If using a list, it should have the same length as n_splits."
            )

    def get_data_hash(self, data: List[str]) -> str:
        """
        Generate a hash for the data to check if it has changed.

        Args:
            data: List of sequences or data points

        Returns:
            MD5 hash of the concatenated data
        """
        data_str = "".join(str(item) for item in data)
        return hashlib.md5(data_str.encode()).hexdigest()

    # Common Clustering-based Splitting Methods
    def distribute_clusters_to_folds(
        self, cluster_items: List[Tuple[str, List[str]]], k_folds: int
    ) -> List[List[str]]:
        """
        Distribute clusters across k folds as evenly as possible.

        Uses a greedy algorithm to assign each cluster to the currently
        smallest fold to achieve balanced fold sizes.

        Args:
            cluster_items: List of (cluster_id, members) tuples
            k_folds: Number of folds to create

        Returns:
            List of folds, where each fold is a list of sequence IDs
        """
        folds = [[] for _ in range(k_folds)]
        fold_sizes = [0] * k_folds

        for _, members in cluster_items:
            smallest_fold = np.argmin(fold_sizes)
            folds[smallest_fold].extend(members)
            fold_sizes[smallest_fold] += len(members)

        # Log fold size distribution
        self.logger.info(f"Fold size distribution: {fold_sizes}")
        return folds

    def generate_kfold_results_from_folds(
        self, folds: List[List[str]], data: List[str], k_folds: int, seed: Optional[int]
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate k-fold results from distributed folds.

        For each fold, the fold itself becomes the test set, and remaining
        folds are combined and split into train/valid sets (80/20 split).

        Args:
            folds: List of folds, each containing sequence IDs
            data: Original data for creating ID-to-index mapping
            k_folds: Number of folds
            seed: Random seed for train/valid splitting

        Returns:
            Dictionary with fold results in format "fold_X"
        """
        id_to_idx = {f"seq{i}": i for i in range(len(data))}
        kfold_results = {}

        for fold_idx in range(k_folds):
            test_ids = folds[fold_idx]

            # Collect data from other folds
            remaining_ids = []
            for i in range(k_folds):
                if i != fold_idx:
                    remaining_ids.extend(folds[i])

            # Split remaining data into train and validation sets
            if seed is not None:
                rng = np.random.RandomState(seed + fold_idx)
                rng.shuffle(remaining_ids)
            else:
                np.random.shuffle(remaining_ids)

            train_size = int(len(remaining_ids) * 0.8)
            train_ids = remaining_ids[:train_size]
            valid_ids = remaining_ids[train_size:]

            kfold_results[f"fold_{fold_idx}"] = {
                "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
                "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
                "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
            }

            self.logger.info(
                f"Fold {fold_idx} completed: "
                f"Train={len(kfold_results[f'fold_{fold_idx}']['train'])}, "
                f"Valid={len(kfold_results[f'fold_{fold_idx}']['valid'])}, "
                f"Test={len(kfold_results[f'fold_{fold_idx}']['test'])}"
            )

        return kfold_results

    def generate_split_from_clusters(
        self,
        cluster_items: List[Tuple[str, List[str]]],
        data: List[str],
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int],
    ) -> Dict[str, List[int]]:
        """
        Helper method to generate train/valid/test split from clustered data.

        Args:
            cluster_items: List of (cluster_id, members) tuples
            data: Original data for size calculation
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed

        Returns:
            Dictionary with train/valid/test indices
        """
        # Shuffle clusters
        cluster_items_copy = cluster_items.copy()
        if seed is not None:
            np.random.RandomState(seed).shuffle(cluster_items_copy)
        else:
            np.random.shuffle(cluster_items_copy)

        # Sort clusters by size
        cluster_items_sorted = sorted(cluster_items_copy, key=lambda x: len(x[1]))
        test_ids = []
        remaining_ids = []
        count_test = 0
        test_data_size = int(len(data) * frac_test)

        for _, members in cluster_items_sorted:
            if count_test + len(members) <= test_data_size:
                test_ids.extend(members)
                count_test += len(members)
            else:
                remaining_ids.extend(members)

        # Split remaining data into train and validation sets
        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)
        np.random.shuffle(remaining_ids)
        train_ids = remaining_ids[:train_data_size]
        valid_ids = remaining_ids[train_data_size:]

        self.logger.info(
            f"Split distribution: "
            f"Target train={train_data_size}, Actual train={len(train_ids)} | "
            f"Target valid={valid_data_size}, Actual valid={len(valid_ids)} | "
            f"Target test={test_data_size}, Actual test={len(test_ids)}"
        )

        # Create a mapping from sequence ID to index
        id_to_idx = {f"seq{i}": i for i in range(len(data))}

        return {
            "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
            "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
            "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
        }

    def print_cluster_stats(self, cluster_map: Dict[str, List[str]], data_type: str = "sequences") -> None:
        """
        Print comprehensive cluster statistics.

        Args:
            cluster_map: Dictionary mapping cluster IDs to sequence ID lists
            data_type: Type of data being clustered (sequences, molecules, etc.)
        """
        cluster_sizes = [len(members) for members in cluster_map.values()]

        # Statistical information
        total_items = sum(cluster_sizes)
        avg_cluster_size = np.mean(cluster_sizes)
        median_cluster_size = np.median(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        min_cluster_size = min(cluster_sizes)

        bins = {
            "size = 1": 0,
            "size 2–4": 0,
            "size 5–9": 0,
            "size 10–19": 0,
            "size 20+": 0,
        }

        for size in cluster_sizes:
            if size == 1:
                bins["size = 1"] += 1
            elif 2 <= size <= 4:
                bins["size 2–4"] += 1
            elif 5 <= size <= 9:
                bins["size 5–9"] += 1
            elif 10 <= size <= 19:
                bins["size 10–19"] += 1
            else:
                bins["size 20+"] += 1

        log_message = (
            f"Clustering Statistics:\n"
            f"  Total clusters: {len(cluster_map)}\n"
            f"  Total {data_type}: {total_items}\n"
            f"  Average cluster size: {avg_cluster_size:.2f}\n"
            f"  Median cluster size: {median_cluster_size:.1f}\n"
            f"  Min cluster size: {min_cluster_size}\n"
            f"  Max cluster size: {max_cluster_size}\n"
            f"  Cluster size distribution:\n"
        )

        for label, count in bins.items():
            percentage = (count / len(cluster_map)) * 100
            log_message += f"    {label}: {count} clusters ({percentage:.1f}%)\n"

        self.logger.info(log_message)

    def __str__(self) -> str:
        """String representation of the splitter."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Detailed string representation of the splitter."""
        return f"{self.__class__.__name__}()"
