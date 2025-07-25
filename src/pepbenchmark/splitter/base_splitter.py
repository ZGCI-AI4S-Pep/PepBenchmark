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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class SPLIT(Enum):
    """Enumeration for different split types."""

    RANDOM = "random"
    MMSEQS = "mmseqs"
    CDHIT = "cdhit"


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
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Union[List[int], np.ndarray]]:
        """
        Generate train/validation/test split indices.

        Args:
            data: Input data to split
            frac_train: Fraction of data for training (default: 0.8)
            frac_valid: Fraction of data for validation (default: 0.1)
            frac_test: Fraction of data for testing (default: 0.1)
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

    @abstractmethod
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
        raise NotImplementedError("This method must be implemented by subclasses.")

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

    This class contains shared methods for validation, statistics, and I/O operations.
    All concrete splitters should inherit from this class instead of BaseSplitter directly.

    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys

    Initializes logging and internal state tracking.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._last_split_info = None

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

    def __str__(self) -> str:
        """String representation of the splitter."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Detailed string representation of the splitter."""
        return f"{self.__class__.__name__}()"
