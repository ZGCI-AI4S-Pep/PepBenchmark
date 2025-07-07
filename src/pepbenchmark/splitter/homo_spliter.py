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

import os
import tempfile

import numpy

from pepbenchmark.pep_utils.mmseq2 import (
    parse_cluster_tsv,
    run_mmseqs_clustering,
    save_fasta,
)
from pepbenchmark.utils.logging import get_logger

from .base_spliter import BaseSplitter

logger = get_logger()


class MMseqs2Spliter(BaseSplitter):
    def __init__(self):
        super().__init__()
        self.cluster_map = None

    def run(self, data, identity, **mmseqs_kwargs):
        logger.info(
            f"Starting MMseqs2 clustering: data_size={len(data)}, identity={identity}, "
            f"mmseqs_params={mmseqs_kwargs}"
        )
        # Use a temporary directory (auto-delete after use)
        with tempfile.TemporaryDirectory() as tmp_root:
            input_fasta = os.path.join(tmp_root, "input.fasta")
            output_dir = os.path.join(tmp_root, "output")
            tmp_dir = os.path.join(tmp_root, "tmp")

            # Save the input data to a FASTA file
            save_fasta(data, input_fasta)

            logger.info(f"Input data saved to {input_fasta}")

            tsv_path = run_mmseqs_clustering(
                input_fasta, output_dir, tmp_dir, identity, **mmseqs_kwargs
            )
            cluster_map = parse_cluster_tsv(tsv_path)
            self._print_cluster_stats(cluster_map)

            logger.info(
                f"MMseqs2 clustering completed: generated {len(cluster_map)} clusters from {len(data)} sequences"
            )
            return cluster_map

    def get_split_indices(
        self,
        data,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        identity=0.25,
        seed=42,
        **kwargs,
    ):
        # Separate MMseqs2 parameters from other kwargs
        mmseqs_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "coverage",
                "sensitivity",
                "alignment_mode",
                "seq_id_mode",
                "mask",
                "cov_mode",
                "threads",
                "max_iterations",
            ]
            or k.startswith("mmseqs_")
        }
        other_kwargs = {k: v for k, v in kwargs.items() if k not in mmseqs_params}

        logger.info(
            f"Starting single split: data_size={len(data)}, frac_train={frac_train}, "
            f"frac_valid={frac_valid}, frac_test={frac_test}, identity={identity}, seed={seed}, "
            f"mmseqs_params={mmseqs_params}, other_kwargs={other_kwargs}"
        )

        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        self.cluster_map = self.run(data, identity, **mmseqs_params)
        cluster_items = list(self.cluster_map.items())

        split_result = self._generate_split_from_clusters(
            cluster_items, data, frac_train, frac_valid, frac_test, seed
        )

        logger.info(
            f"Single split completed: Train={len(split_result['train'])}, "
            f"Valid={len(split_result['valid'])}, Test={len(split_result['test'])}"
        )
        return split_result

    def _generate_split_from_clusters(
        self, cluster_items, data, frac_train, frac_valid, frac_test, seed
    ):
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
        numpy.random.RandomState(seed).shuffle(cluster_items_copy)

        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)
        test_data_size = int(len(data) * frac_test)

        train_ids, valid_ids, test_ids = [], [], []
        count_train = count_valid = count_test = 0

        for _, members in cluster_items_copy:
            if count_train + len(members) <= train_data_size:
                train_ids.extend(members)
                count_train += len(members)
            elif count_valid + len(members) <= valid_data_size:
                valid_ids.extend(members)
                count_valid += len(members)
            else:
                test_ids.extend(members)
                count_test += len(members)

        logger.info(
            f"Finish clustering and splitting data: "
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

    def get_split_indices_n(
        self,
        data,
        n_splits=5,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        identity=0.25,
        seed=42,
        **kwargs,
    ):
        """
        Generate n random splits using the same clustering.

        Args:
            data: Input sequences
            n_splits: Number of random splits to generate
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            identity: Sequence identity threshold for clustering
            seed: Random seed for reproducibility

        Returns:
            Dictionary with n random splits
        """
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        # Separate MMseqs2 parameters from other kwargs
        mmseqs_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "coverage",
                "sensitivity",
                "alignment_mode",
                "seq_id_mode",
                "mask",
                "cov_mode",
                "threads",
                "max_iterations",
            ]
            or k.startswith("mmseqs_")
        }
        other_kwargs = {k: v for k, v in kwargs.items() if k not in mmseqs_params}

        # Run clustering only once
        logger.info(
            f"Starting n random splits: data_size={len(data)}, n_splits={n_splits}, "
            f"frac_train={frac_train}, frac_valid={frac_valid}, frac_test={frac_test}, "
            f"identity={identity}, seed={seed}, mmseqs_params={mmseqs_params}, other_kwargs={other_kwargs}"
        )
        cluster_map = self.run(data, identity, **mmseqs_params)
        cluster_items = list(cluster_map.items())

        split_results = {}
        for i in range(n_splits):
            current_seed = seed + i if seed is not None else None
            logger.info(f"Generating split {i + 1} with seed {current_seed}")
            split_indices = self._generate_split_from_clusters(
                cluster_items, data, frac_train, frac_valid, frac_test, current_seed
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
        identity=0.25,
        seed=42,
        **kwargs,
    ):
        """
        Generate k-fold cross-validation splits based on sequence clustering.

        Args:
            data: Input sequences
            k_folds: Number of folds for cross-validation
            identity: Sequence identity threshold for clustering
            seed: Random seed for reproducibility

        Returns:
            Dictionary with k-fold splits, each containing train/valid/test indices
        """
        # Separate MMseqs2 parameters from other kwargs
        mmseqs_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "coverage",
                "sensitivity",
                "alignment_mode",
                "seq_id_mode",
                "mask",
                "cov_mode",
                "threads",
                "max_iterations",
            ]
            or k.startswith("mmseqs_")
        }
        other_kwargs = {k: v for k, v in kwargs.items() if k not in mmseqs_params}

        # Run clustering only once
        logger.info(
            f"Starting k-fold split: data_size={len(data)}, k_folds={k_folds}, "
            f"identity={identity}, seed={seed}, mmseqs_params={mmseqs_params}, other_kwargs={other_kwargs}"
        )
        cluster_map = self.run(data, identity, **mmseqs_params)
        cluster_items = list(cluster_map.items())

        # Shuffle clusters with fixed seed for reproducibility
        numpy.random.RandomState(seed).shuffle(cluster_items)

        # Distribute clusters across k folds as evenly as possible
        folds = [[] for _ in range(k_folds)]
        fold_sizes = [0] * k_folds

        # Assign each cluster to the fold with the smallest current size
        for _cluster_id, members in cluster_items:
            smallest_fold = numpy.argmin(fold_sizes)
            folds[smallest_fold].extend(members)
            fold_sizes[smallest_fold] += len(members)

        # Create a mapping from sequence ID to index
        id_to_idx = {f"seq{i}": i for i in range(len(data))}

        # Generate k-fold splits
        kfold_results = {}
        for fold_idx in range(k_folds):
            test_ids = folds[fold_idx]

            # Remaining folds for train/valid
            remaining_ids = []
            for i in range(k_folds):
                if i != fold_idx:
                    remaining_ids.extend(folds[i])

            # Split remaining data into train and valid (80% train, 20% valid)
            numpy.random.RandomState(seed + fold_idx).shuffle(remaining_ids)
            train_size = int(len(remaining_ids) * 0.8)
            train_ids = remaining_ids[:train_size]
            valid_ids = remaining_ids[train_size:]

            kfold_results[f"fold_{fold_idx}"] = {
                "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
                "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
                "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
            }

            # 转换后的实际索引数量
            actual_train = len(kfold_results[f"fold_{fold_idx}"]["train"])
            actual_valid = len(kfold_results[f"fold_{fold_idx}"]["valid"])
            actual_test = len(kfold_results[f"fold_{fold_idx}"]["test"])
            logger.info(
                f"Fold {fold_idx} completed: Train={actual_train}, Valid={actual_valid}, Test={actual_test}"
            )

        logger.info(f"All {k_folds} k-fold splits completed successfully")
        return kfold_results

    def _print_cluster_stats(self, cluster_map):
        log_message = f"Total clusters: {len(cluster_map)}\n"
        cluster_sizes = [len(members) for members in cluster_map.values()]

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

        log_message += "Cluster size distribution:\n"
        for label, count in bins.items():
            log_message += f"  {label}: {count} clusters\n"
        logger.info(log_message)


if __name__ == "__main__":
    from pepbenchmark.single_pred.base_dataset import SingleTaskDatasetManager

    dataset_name = "AV_APML"  # Change this to your dataset name

    dataset_manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta"]
    )
    fasta = dataset_manager.get_official_feature("fasta")

    split = MMseqs2Spliter()

    # Test with default MMseqs2 parameters
    logger.info("=" * 50)
    logger.info("Testing with default MMseqs2 parameters")
    logger.info("=" * 50)

    # Generate 5 random splits with default parameters
    random_splits = split.get_split_indices_n(
        fasta,
        n_splits=5,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        identity=0.25,
        seed=42,
    )

    # Test with custom MMseqs2 parameters
    logger.info("=" * 50)
    logger.info("Testing with custom MMseqs2 parameters")
    logger.info("=" * 50)

    # Generate k-fold cross-validation splits with custom parameters
    kfold_splits = split.get_split_kfold_indices(
        fasta,
        k_folds=5,
        identity=0.25,
        seed=42,
        # Custom MMseqs2 parameters
        coverage=0.8,  # Higher coverage requirement
        sensitivity=7.5,  # Different sensitivity
        threads=4,  # Use 4 threads
        cov_mode=1,  # Different coverage mode
    )

    # Test with single split and additional custom parameters
    logger.info("=" * 50)
    logger.info("Testing single split with custom parameters")
    logger.info("=" * 50)

    single_split = split.get_split_indices(
        fasta,
        frac_train=0.7,
        frac_valid=0.15,
        frac_test=0.15,
        identity=0.3,
        seed=123,
        # More custom MMseqs2 parameters
        coverage=0.5,
        sensitivity=6.0,
        alignment_mode=2,
        seq_id_mode=0,
        mask=1,
    )

    # Optional: Save results to files
    # with open(f"{dataset_name}/mmseqs2_random_splits.json", "w") as f:
    #     json.dump(random_splits, f, indent=4)
    # with open(f"{dataset_name}/mmseqs2_kfold_splits.json", "w") as f:
    #     json.dump(kfold_splits, f, indent=4)
    # with open(f"{dataset_name}/mmseqs2_single_split.json", "w") as f:
    #     json.dump(single_split, f, indent=4)

    logger.info("All MMseqs2 tests completed successfully!")
