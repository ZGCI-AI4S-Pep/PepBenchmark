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

import numpy

from pepbenchmark.utils.logging import get_logger

from .base_spliter import BaseSplitter

logger = get_logger(__name__)


class RandomSplitter(BaseSplitter):
    def get_split_indices(
        self, data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42, **kwargs
    ):
        logger.info(
            f"Starting random split: data_size={len(data)}, frac_train={frac_train}, "
            f"frac_valid={frac_valid}, frac_test={frac_test}, seed={seed}, kwargs={kwargs}"
        )

        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        if seed is not None:
            perm = numpy.random.RandomState(seed).permutation(len(data))
        else:
            perm = numpy.random.permutation(len(data))

        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)

        split_result = {
            "train": perm[:train_data_size],
            "valid": perm[train_data_size : train_data_size + valid_data_size],
            "test": perm[train_data_size + valid_data_size :],
        }

        logger.info(
            f"Random split completed: Train={len(split_result['train'])}, "
            f"Valid={len(split_result['valid'])}, Test={len(split_result['test'])}"
        )

        return split_result

    def get_split_indices_n(
        self,
        data,
        n_splits=5,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=42,
        **kwargs,
    ):
        """
        Generate n random splits with different seeds.

        Args:
            data: Input sequences
            n_splits: Number of random splits to generate
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed for reproducibility

        Returns:
            Dictionary with n random splits
        """
        logger.info(
            f"Starting n random splits: data_size={len(data)}, n_splits={n_splits}, "
            f"frac_train={frac_train}, frac_valid={frac_valid}, frac_test={frac_test}, "
            f"seed={seed}, kwargs={kwargs}"
        )

        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        split_results = {}
        for i in range(n_splits):
            current_seed = seed + i if seed is not None else None
            logger.info(f"Generating split {i + 1} with seed {current_seed}")

            split_indices = self.get_split_indices(
                data, frac_train, frac_valid, frac_test, seed=current_seed, **kwargs
            )

            logger.info(
                f"Split {i + 1} completed: Train={len(split_indices['train'])}, "
                f"Valid={len(split_indices['valid'])}, Test={len(split_indices['test'])}"
            )
            split_results[f"seed_{i}"] = split_indices

        logger.info(f"All {n_splits} random splits completed successfully")
        return split_results

    def get_split_kfold_indices(
        self,
        data,
        k_folds=5,
        seed=42,
        **kwargs,
    ):
        """
        Generate k-fold cross-validation splits using random permutation.

        Args:
            data: Input sequences
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility

        Returns:
            Dictionary with k-fold splits, each containing train/valid/test indices
        """
        logger.info(
            f"Starting k-fold split: data_size={len(data)}, k_folds={k_folds}, "
            f"seed={seed}, kwargs={kwargs}"
        )

        # Generate a single random permutation for all folds
        if seed is not None:
            perm = numpy.random.RandomState(seed).permutation(len(data))
        else:
            perm = numpy.random.permutation(len(data))

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
            numpy.random.RandomState(seed + fold_idx).shuffle(remaining_indices)
            train_size = int(len(remaining_indices) * 0.8)
            train_indices = remaining_indices[:train_size]
            valid_indices = remaining_indices[train_size:]

            kfold_results[f"fold_{fold_idx}"] = {
                "train": numpy.array(train_indices),
                "valid": numpy.array(valid_indices),
                "test": numpy.array(test_indices),
            }

            logger.info(
                f"Fold {fold_idx} completed: Train={len(train_indices)}, "
                f"Valid={len(valid_indices)}, Test={len(test_indices)}"
            )

        logger.info(f"All {k_folds} k-fold splits completed successfully")
        return kfold_results

    def _validate_split_results(self, split_results, data_size):
        """
        Validate that split results are complete and non-overlapping.

        Args:
            split_results: Dictionary with train/valid/test indices
            data_size: Original data size
        """
        train_indices = set(split_results["train"])
        valid_indices = set(split_results["valid"])
        test_indices = set(split_results["test"])

        # Check for overlaps
        train_valid_overlap = train_indices.intersection(valid_indices)
        train_test_overlap = train_indices.intersection(test_indices)
        valid_test_overlap = valid_indices.intersection(test_indices)

        if train_valid_overlap:
            logger.warning(
                f"Train-Valid overlap detected: {len(train_valid_overlap)} indices"
            )
        if train_test_overlap:
            logger.warning(
                f"Train-Test overlap detected: {len(train_test_overlap)} indices"
            )
        if valid_test_overlap:
            logger.warning(
                f"Valid-Test overlap detected: {len(valid_test_overlap)} indices"
            )

        # Check completeness
        all_indices = train_indices.union(valid_indices).union(test_indices)
        expected_indices = set(range(data_size))

        missing_indices = expected_indices - all_indices
        extra_indices = all_indices - expected_indices

        if missing_indices:
            logger.warning(f"Missing indices: {len(missing_indices)}")
        if extra_indices:
            logger.warning(f"Extra indices: {len(extra_indices)}")

        # Log summary
        total_assigned = len(all_indices)
        logger.info(
            f"Split validation: {total_assigned}/{data_size} indices assigned, "
            f"overlaps: train-valid={len(train_valid_overlap)}, "
            f"train-test={len(train_test_overlap)}, valid-test={len(valid_test_overlap)}"
        )

        return (
            len(train_valid_overlap) == 0
            and len(train_test_overlap) == 0
            and len(valid_test_overlap) == 0
            and len(missing_indices) == 0
            and len(extra_indices) == 0
        )


if __name__ == "__main__":
    from pepbenchmark.single_pred.base_dataset import SingleTaskDatasetManager

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
    splitter._validate_split_results(single_split, len(fasta))

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
    splitter._validate_split_results(random_splits["seed_0"], len(fasta))

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
    splitter._validate_split_results(kfold_splits["fold_0"], len(fasta))

    # Optional: Save results to files
    # with open(f"{dataset_name}/random_single_split.json", "w") as f:
    #     json.dump({k: v.tolist() for k, v in single_split.items()}, f, indent=4)

    # with open(f"{dataset_name}/random_n_splits.json", "w") as f:
    #     json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in random_splits.items()}, f, indent=4)

    # with open(f"{dataset_name}/random_kfold_splits.json", "w") as f:
    #     json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in kfold_splits.items()}, f, indent=4)

    logger.info("All tests completed successfully!")
