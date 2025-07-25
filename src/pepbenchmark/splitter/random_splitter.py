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

from typing import Dict, List, Optional, Union

import numpy as np

from pepbenchmark.splitter.base_splitter import AbstractSplitter
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


class RandomSplitter(AbstractSplitter):
    """
    Random data splitter that randomly shuffles data before splitting.

    This splitter performs completely random splits without considering
    sequence similarity or any other relationships in the data.

    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys
    """

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
        Generate random split indices.

        This method randomly shuffles the data indices and then splits them
        according to the specified fractions.

        Args:
            data: Input data to split (List or numpy array)
            frac_train: Fraction of data for training (default: 0.8)
            frac_valid: Fraction of data for validation (default: 0.1)
            frac_test: Fraction of data for testing (default: 0.1)
            seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters (ignored for random splitting)

        Returns:
            Dictionary containing train/valid/test split indices:
            {
                "train": [indices for training set],
                "valid": [indices for validation set],
                "test": [indices for test set]
            }

        Raises:
            ValueError: If fractions don't sum to 1.0 or are invalid
        """
        logger.info(
            f"Starting random split: data_size={len(data)}, frac_train={frac_train}, "
            f"frac_valid={frac_valid}, frac_test={frac_test}, seed={seed}"
        )

        # Validate fractions using parent class method
        self.validate_fractions(frac_train, frac_valid, frac_test)

        # Generate random permutation
        if seed is not None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(data))
        else:
            perm = np.random.permutation(len(data))

        # Calculate split sizes
        train_size = int(len(data) * frac_train)
        valid_size = int(len(data) * frac_valid)

        # Create split indices
        split_result = {
            "train": perm[:train_size],
            "valid": perm[train_size : train_size + valid_size],
            "test": perm[train_size + valid_size :],
        }

        logger.info(
            f"Random split completed: Train={len(split_result['train'])}, "
            f"Valid={len(split_result['valid'])}, Test={len(split_result['test'])}"
        )

        return split_result

    def get_split_kfold_indices(
        self,
        data: Union[List, np.ndarray],
        k_folds: int = 5,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Dict[str, Union[List[int], np.ndarray]]]:
        """
        Generate k-fold cross-validation splits using random permutation.

        Args:
            data: Input data to split
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
            **kwargs: Additional parameters (ignored for random splitting)

        Returns:
            Dictionary with keys in format "fold_X" where X is the fold index (0 to k_folds-1).
            Each fold contains train/valid/test splits where the test set is the X-th fold.
        """
        logger.info(
            f"Starting k-fold split: data_size={len(data)}, k_folds={k_folds}, "
            f"seed={seed}"
        )

        if k_folds <= 1:
            raise ValueError(f"k_folds must be greater than 1, got {k_folds}")

        # Generate a single random permutation for all folds
        if seed is not None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(data))
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

            # Collect remaining indices for train/valid split
            remaining_indices = []
            for i in range(k_folds):
                if i != fold_idx:
                    remaining_indices.extend(folds[i])

            # Split remaining data into train and valid (80% train, 20% valid)
            if seed is not None:
                rng = np.random.RandomState(seed + fold_idx)
                rng.shuffle(remaining_indices)
            else:
                np.random.shuffle(remaining_indices)

            train_size = int(len(remaining_indices) * 0.8)
            train_indices = remaining_indices[:train_size]
            valid_indices = remaining_indices[train_size:]

            kfold_results[f"fold_{fold_idx}"] = {
                "train": np.array(train_indices),
                "valid": np.array(valid_indices),
                "test": np.array(test_indices),
            }

            logger.info(
                f"Fold {fold_idx} completed: Train={len(train_indices)}, "
                f"Valid={len(valid_indices)}, Test={len(test_indices)}"
            )

        logger.info(f"All {k_folds} k-fold splits completed successfully")
        return kfold_results


if __name__ == "__main__":
    from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

    # Test with a dataset
    dataset_name = "AV_APML"  # Change this to your dataset name

    dataset_manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta"]
    )
    fasta = dataset_manager.get_official_feature("fasta")

    splitter = RandomSplitter()

    # Test single split
    logger.info("=" * 50)
    logger.info("Testing single random split")
    logger.info("=" * 50)
    single_split = splitter.get_split_indices(
        fasta,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=42,
    )
    splitter.validate_split_results(single_split, len(fasta))

    # Test n random splits
    logger.info("=" * 50)
    logger.info("Testing n random splits")
    logger.info("=" * 50)
    random_splits = splitter.get_split_indices_n(
        fasta,
        n_splits=5,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=42,
    )
    # Validate first split as example
    splitter.validate_split_results(random_splits["seed_0"], len(fasta))

    # Test k-fold cross-validation splits
    logger.info("=" * 50)
    logger.info("Testing k-fold cross-validation")
    logger.info("=" * 50)
    kfold_splits = splitter.get_split_kfold_indices(
        fasta,
        k_folds=5,
        seed=42,
    )
    # Validate first fold as example
    splitter.validate_split_results(kfold_splits["fold_0"], len(fasta))

    # Test save and load split results
    logger.info("=" * 50)
    logger.info("Testing save and load split results")
    logger.info("=" * 50)

    # Test saving and loading single split results
    import os

    # Save single split results in both JSON and numpy formats
    json_path = "single_split.json"
    numpy_path = "single_split.npz"

    # Save in JSON format
    splitter.save_split_results(single_split, json_path, format="json")
    logger.info(f"Single split saved to JSON: {json_path}")

    # Save in numpy format
    splitter.save_split_results(single_split, numpy_path, format="numpy")
    logger.info(f"Single split saved to numpy: {numpy_path}")

    # Load and verify JSON format
    loaded_json = splitter.load_split_results(json_path, format="json")
    logger.info(
        f"Loaded JSON split - Train: {len(loaded_json['train'])}, Valid: {len(loaded_json['valid'])}, Test: {len(loaded_json['test'])}"
    )

    # Verify loaded data matches original
    for split_name in ["train", "valid", "test"]:
        original_indices = set(single_split[split_name])
        loaded_indices = set(loaded_json[split_name])
        if original_indices == loaded_indices:
            logger.info(f"✓ {split_name} split matches after JSON load")
        else:
            logger.error(f"✗ {split_name} split mismatch after JSON load")

    # Load and verify numpy format
    loaded_numpy = splitter.load_split_results(numpy_path, format="numpy")
    logger.info(
        f"Loaded numpy split - Train: {len(loaded_numpy['train'])}, Valid: {len(loaded_numpy['valid'])}, Test: {len(loaded_numpy['test'])}"
    )

    # Verify loaded data matches original
    for split_name in ["train", "valid", "test"]:
        original_indices = set(single_split[split_name])
        loaded_indices = set(loaded_numpy[split_name])
        if original_indices == loaded_indices:
            logger.info(f"✓ {split_name} split matches after numpy load")
        else:
            logger.error(f"✗ {split_name} split mismatch after numpy load")

    # Test saving and loading multiple splits
    multi_splits_path = "random_splits.json"
    splitter.save_split_results(random_splits, multi_splits_path, format="json")
    logger.info(f"Multiple splits saved to: {multi_splits_path}")

    loaded_multi = splitter.load_split_results(multi_splits_path, format="json")
    logger.info(f"Loaded multiple splits with keys: {list(loaded_multi.keys())}")

    # Verify structure of loaded multiple splits
    for split_key in random_splits.keys():
        if split_key in loaded_multi:
            for split_name in ["train", "valid", "test"]:
                original_count = len(random_splits[split_key][split_name])
                loaded_count = len(loaded_multi[split_key][split_name])
                if original_count == loaded_count:
                    logger.info(
                        f"✓ {split_key}.{split_name} count matches ({original_count})"
                    )
                else:
                    logger.error(
                        f"✗ {split_key}.{split_name} count mismatch: {original_count} vs {loaded_count}"
                    )
        else:
            logger.error(f"✗ Missing split key: {split_key}")

    # Test saving and loading k-fold splits
    kfold_splits_path = "kfold_splits.json"
    splitter.save_split_results(kfold_splits, kfold_splits_path, format="json")
    logger.info(f"K-fold splits saved to: {kfold_splits_path}")

    loaded_kfold = splitter.load_split_results(kfold_splits_path, format="json")
    logger.info(f"Loaded k-fold splits with keys: {list(loaded_kfold.keys())}")

    # Test get_split_statistics
    stats = splitter.get_split_statistics(single_split)
    logger.info(f"Split statistics: {stats}")

    # Verify statistics
    expected_total = len(fasta)
    if stats["total_size"] == expected_total:
        logger.info(f"✓ Total size matches: {expected_total}")
    else:
        logger.error(
            f"✗ Total size mismatch: expected {expected_total}, got {stats['total_size']}"
        )

    # Check that fractions sum to approximately 1.0
    total_fraction = (
        stats["train_fraction"] + stats["valid_fraction"] + stats["test_fraction"]
    )
    if abs(total_fraction - 1.0) < 0.001:
        logger.info(f"✓ Fractions sum to 1.0: {total_fraction}")
    else:
        logger.error(f"✗ Fractions don't sum to 1.0: {total_fraction}")

    # Clean up generated files
    for file_path in [json_path, numpy_path, multi_splits_path, kfold_splits_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")
        else:
            logger.warning(f"File not found for cleanup: {file_path}")

    logger.info("All tests completed successfully!")
