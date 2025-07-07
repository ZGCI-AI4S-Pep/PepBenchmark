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
    STRATIFIED = "stratified"
    HOMOLOGY = "homology"
    TEMPORAL = "temporal"
    CLUSTER = "cluster"


class BaseSplitter(ABC):
    """
    Abstract base class for all data splitters.

    This class defines the interface that all splitter implementations must follow.
    It provides common functionality and validation methods that can be reused
    across different splitting strategies.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._last_split_info = None

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

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def get_split_indices_n(
        self,
        data: Union[List, np.ndarray],
        n_splits: int = 5,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[int], np.ndarray]]]:
        """
        Generate multiple random splits with different seeds.

        Args:
            data: Input data to split
            n_splits: Number of random splits to generate
            frac_train: Fraction of data for training
            frac_valid: Fraction of data for validation
            frac_test: Fraction of data for testing
            seed: Base random seed for reproducibility
            **kwargs: Additional splitter-specific parameters

        Returns:
            Dictionary with split results keyed by 'seed_i'
        """
        self.logger.info(f"Generating {n_splits} random splits with base seed {seed}")
        self._validate_fractions(frac_train, frac_valid, frac_test)

        split_results = {}
        for i in range(n_splits):
            current_seed = seed + i if seed is not None else None
            self.logger.info(
                f"Generating split {i + 1}/{n_splits} with seed {current_seed}"
            )

            split_indices = self.get_split_indices(
                data, frac_train, frac_valid, frac_test, seed=current_seed, **kwargs
            )

            split_results[f"seed_{i}"] = split_indices
            self.logger.info(
                f"Split {i + 1} completed: Train={len(split_indices['train'])}, "
                f"Valid={len(split_indices['valid'])}, Test={len(split_indices['test'])}"
            )

        self.logger.info(f"All {n_splits} random splits completed successfully")
        return split_results

    def get_split_kfold_indices(
        self,
        data: Union[List, np.ndarray],
        k_folds: int = 5,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[int], np.ndarray]]]:
        """
        Generate k-fold cross-validation splits.

        This is a default implementation that can be overridden by subclasses
        for more sophisticated k-fold strategies.

        Args:
            data: Input data to split
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
            **kwargs: Additional splitter-specific parameters

        Returns:
            Dictionary with k-fold splits keyed by 'fold_i'
        """
        self.logger.info(f"Generating {k_folds}-fold cross-validation splits")

        # Generate a random permutation for all folds
        if seed is not None:
            perm = np.random.RandomState(seed).permutation(len(data))
        else:
            perm = np.random.permutation(len(data))

        # Split data into k roughly equal folds
        fold_size = len(data) // k_folds
        folds = []
        for i in range(k_folds):
            start_idx = i * fold_size
            if i == k_folds - 1:  # Last fold gets any remaining data
                end_idx = len(data)
            else:
                end_idx = (i + 1) * fold_size
            folds.append(perm[start_idx:end_idx])

        # Generate k-fold splits
        kfold_results = {}
        for fold_idx in range(k_folds):
            test_indices = folds[fold_idx]

            # Remaining folds for train/valid
            remaining_indices = []
            for i in range(k_folds):
                if i != fold_idx:
                    remaining_indices.extend(folds[i])

            # Split remaining data into train and valid (80% train, 20% valid)
            np.random.RandomState(seed + fold_idx).shuffle(remaining_indices)
            train_size = int(len(remaining_indices) * 0.8)
            train_indices = remaining_indices[:train_size]
            valid_indices = remaining_indices[train_size:]

            kfold_results[f"fold_{fold_idx}"] = {
                "train": np.array(train_indices),
                "valid": np.array(valid_indices),
                "test": np.array(test_indices),
            }

            self.logger.info(
                f"Fold {fold_idx} completed: Train={len(train_indices)}, "
                f"Valid={len(valid_indices)}, Test={len(test_indices)}"
            )

        self.logger.info(f"All {k_folds} k-fold splits completed successfully")
        return kfold_results

    def _validate_fractions(
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

    def _validate_data(self, data: Union[List, np.ndarray]) -> None:
        """
        Validate input data.

        Args:
            data: Input data to validate

        Raises:
            ValueError: If data is empty or invalid
        """
        if data is None:
            raise ValueError("Data cannot be None")

        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        if len(data) < 3:
            self.logger.warning(f"Very small dataset with only {len(data)} samples")

    def _validate_split_keys(self, split_results: Dict[str, Any]) -> None:
        """Validate that the split dictionary contains the required keys."""
        required_keys = {"train", "valid", "test"}
        if not required_keys.issubset(split_results.keys()):
            raise ValueError(
                f"Split results must contain {required_keys}, but got {split_results.keys()}"
            )

    def _validate_split_indices(
        self, split_results: Dict[str, Union[List[int], np.ndarray]], data_size: int
    ) -> None:
        """Validate that indices in splits are within the valid range."""
        for key, indices in split_results.items():
            if not isinstance(indices, (list, np.ndarray)):
                raise TypeError(f"Indices for {key} must be a list or numpy array")
            if not all(isinstance(i, (int, np.integer)) for i in indices):
                raise TypeError(f"All indices for {key} must be integers")
            if any(i < 0 or i >= data_size for i in indices):
                raise ValueError(
                    f"Indices for {key} are out of bounds for data of size {data_size}"
                )

    def _check_split_completeness(
        self, all_indices: np.ndarray, data_size: int
    ) -> None:
        """Check if the splits cover the entire dataset."""
        if len(all_indices) != data_size:
            self.logger.warning(
                f"Split is not complete. Expected {data_size} unique indices, but got {len(all_indices)}"
            )

    def _check_split_overlaps(
        self, all_indices: np.ndarray, total_indices: int
    ) -> None:
        """Check for overlapping indices between splits."""
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
            self._validate_split_keys(split_results)
            self._validate_split_indices(split_results, data_size)

            train_indices = np.array(split_results["train"])
            valid_indices = np.array(split_results["valid"])
            test_indices = np.array(split_results["test"])

            total_indices = len(train_indices) + len(valid_indices) + len(test_indices)
            all_indices = np.unique(
                np.concatenate([train_indices, valid_indices, test_indices])
            )

            if check_completeness:
                self._check_split_completeness(all_indices, data_size)

            if check_overlaps:
                self._check_split_overlaps(all_indices, total_indices)

            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f"Split validation failed: {e}")
            return False

    def get_last_split_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last performed split.

        Returns:
            Dictionary containing last split information, or None if no split has been performed
        """
        return self._last_split_info

    def _set_last_split_info(self, split_info: Dict[str, Any]) -> None:
        """
        Set internal information about the last performed split.

        This is for internal use only and should not be called directly by users.

        Args:
            split_info: Dictionary containing split information
        """
        self._last_split_info = split_info

    def get_split_statistics(
        self, split_results: Dict[str, Union[List[int], np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Get statistics about the split results.

        Args:
            split_results: Dictionary with train/valid/test indices

        Returns:
            Dictionary containing split statistics
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
        Save split results to file.

        Args:
            split_results: Split results to save
            filepath: Output file path
            format: Output format ('json' or 'numpy')
        """
        import json
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
            Dictionary containing split results
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
