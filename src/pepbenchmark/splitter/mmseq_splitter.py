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
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np

from pepbenchmark.pep_utils.mmseq2 import (
    parse_cluster_tsv,
    run_mmseqs_clustering,
    save_fasta,
)
from pepbenchmark.utils.logging import get_logger
from pepbenchmark.splitter.base_splitter import AbstractSplitter

logger = get_logger(__name__)


class MMseqs2Splitter(AbstractSplitter):
    """
    MMseqs2-based sequence splitter for homology-aware splitting.
    
    This splitter uses MMseqs2 clustering to ensure that similar sequences
    are placed in the same split, preventing data leakage in evaluation.
    
    Result Key Naming Conventions:
    - get_split_indices_n(): Returns keys as "seed_X" (X = 0 to n_splits-1)
    - get_split_kfold_indices(): Returns keys as "fold_X" (X = 0 to k_folds-1)
    - get_split_indices(): Returns single dict with "train", "valid", "test" keys
    """
    
    def __init__(self):
        """
        Initialize the MMseqs2Splitter.
        
        Sets up caching mechanism for clustering results to avoid
        re-clustering when the same data and parameters are used.
        """
        super().__init__()
        self.cluster_map = None
        self._cached_identity = None
        self._cached_params = None
        self._cached_data_hash = None

    def get_split_indices(
        self,
        data: List[str],
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        identity: float = 0.25,
        seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """
        Generate homology-aware split indices using MMseqs2 clustering.
        
        This method performs sequence clustering using MMseqs2 to group similar sequences
        together, then distributes clusters across train/valid/test sets to prevent
        data leakage due to sequence similarity.
        
        Args:
            data: List of sequences to split
            frac_train: Fraction of data for training set (default: 0.8)
            frac_valid: Fraction of data for validation set (default: 0.1)
            frac_test: Fraction of data for test set (default: 0.1)
            identity: Sequence identity threshold for clustering (default: 0.25)
            seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional MMseqs2 parameters (coverage, sensitivity, etc.)
            
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
            f"Starting homology-aware split: data_size={len(data)}, "
            f"frac_train={frac_train}, frac_valid={frac_valid}, frac_test={frac_test}, "
            f"identity={identity}, seed={seed}"
        )

        # Use parent class validation methods
        self.validate_fractions(frac_train, frac_valid, frac_test)
        
        # Extract MMseqs2 parameters
        mmseqs_params = self._extract_mmseqs_params(kwargs)
        
        # Get clustering results
        cluster_map = self._get_or_create_clusters(data, identity, **mmseqs_params)
        cluster_items = list(cluster_map.items())

        split_result = self._generate_split_from_clusters(
            cluster_items, data, frac_train, frac_valid, frac_test, seed
        )

        logger.info(
            f"Homology-aware split completed: Train={len(split_result['train'])}, "
            f"Valid={len(split_result['valid'])}, Test={len(split_result['test'])}"
        )
        return split_result

    def get_split_kfold_indices(
        self,
        data: List[str],
        k_folds: int = 5,
        identity: float = 0.25,
        seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate k-fold cross-validation splits based on sequence clustering.
        
        This method performs MMseqs2 clustering first, then distributes clusters
        across k folds to ensure similar sequences stay within the same fold.
        For each fold, the fold itself becomes the test set, and remaining data
        is split into train/valid sets.
        
        Args:
            data: List of sequences to split
            k_folds: Number of folds for cross-validation (default: 5)
            identity: Sequence identity threshold for clustering (default: 0.25)
            seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional MMseqs2 parameters (coverage, sensitivity, etc.)
            
        Returns:
            Dictionary with keys in format "fold_X" where X is the fold index (0 to k_folds-1).
            Each fold contains train/valid/test splits where the test set is the X-th fold.
            {
                "fold_0": {"train": [...], "valid": [...], "test": [...]},
                "fold_1": {"train": [...], "valid": [...], "test": [...]},
                ...
            }
            
        Raises:
            ValueError: If k_folds <= 1
        """
        logger.info(
            f"Starting homology-aware k-fold split: data_size={len(data)}, "
            f"k_folds={k_folds}, identity={identity}, seed={seed}"
        )
        
        if k_folds <= 1:
            raise ValueError(f"k_folds must be greater than 1, got {k_folds}")
        
        # Extract MMseqs2 parameters
        mmseqs_params = self._extract_mmseqs_params(kwargs)
        
        # Get clustering results
        cluster_map = self._get_or_create_clusters(data, identity, **mmseqs_params)
        cluster_items = list(cluster_map.items())

        # Shuffle clusters using fixed seed
        if seed is not None:
            rng = np.random.RandomState(seed)
            rng.shuffle(cluster_items)
        else:
            np.random.shuffle(cluster_items)

        # Distribute clusters as evenly as possible across k folds
        folds = self._distribute_clusters_to_folds(cluster_items, k_folds)

        # Generate k-fold split results
        kfold_results = self._generate_kfold_results(folds, data, k_folds, seed)

        logger.info(f"All {k_folds} homology-aware k-fold splits completed successfully")
        return kfold_results

    def get_split_indices_n(
        self,
        data: List[str],
        n_splits: int = 5,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        identity: float = 0.25,
        seed: Union[List[int], int] = 42,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate n random splits using the same clustering.

        Args:
            data: Input sequences
            n_splits: Number of random splits to generate
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            identity: Sequence identity threshold for clustering
            seed: Random seed for reproducibility (int or list of ints)

        Returns:
            Dictionary with keys in format "seed_X" where X is the split index (0 to n_splits-1).
            Each split contains train/valid/test splits with the specified fractions.
        """
        logger.info(
            f"Starting n random splits: data_size={len(data)}, n_splits={n_splits}, "
            f"frac_train={frac_train}, frac_valid={frac_valid}, frac_test={frac_test}, "
            f"identity={identity}, seed={seed}"
        )

        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")

        # Use parent class validation methods
        self.validate_fractions(frac_train, frac_valid, frac_test)
        
        # Process seed arguments
        seeds = self._prepare_seeds(seed, n_splits)

        # Extract MMseqs2 parameters
        mmseqs_params = self._extract_mmseqs_params(kwargs)
        
        # Get clustering results (run only once)
        cluster_map = self._get_or_create_clusters(data, identity, **mmseqs_params)
        cluster_items = list(cluster_map.items())

        split_results = {}
        for i, current_seed in enumerate(seeds):
            logger.info(f"Generating split {i + 1}/{n_splits} with seed {current_seed}")
            
            split_indices = self._generate_split_from_clusters(
                cluster_items, data, frac_train, frac_valid, frac_test, current_seed
            )
            
            # Use parent class validation methods
            if not self.validate_split_results(split_indices, len(data)):
                logger.warning(f"Split {i + 1} validation failed")
            
            logger.info(
                f"Split {i + 1} completed: Train={len(split_indices['train'])}, "
                f"Valid={len(split_indices['valid'])}, Test={len(split_indices['test'])}"
            )
            split_results[f"seed_{i}"] = split_indices

        logger.info(f"All {n_splits} random splits completed successfully")
        return split_results

    def _prepare_seeds(self, seed: Union[List[int], int], n_splits: int) -> List[int]:
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

    def _extract_mmseqs_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract MMseqs2-specific parameters from kwargs.
        
        Args:
            kwargs: Dictionary of keyword arguments
            
        Returns:
            Dictionary containing only MMseqs2-related parameters
        """
        mmseqs_param_names = {
            "coverage", "sensitivity", "alignment_mode", "seq_id_mode", 
            "mask", "cov_mode", "threads", "max_iterations"
        }
        
        mmseqs_params = {}
        for key, value in kwargs.items():
            if key in mmseqs_param_names or key.startswith("mmseqs_"):
                mmseqs_params[key] = value
                
        return mmseqs_params

    def _get_data_hash(self, data: List[str]) -> str:
        """
        Generate a hash for the data to check if it has changed.
        
        Args:
            data: List of sequences
            
        Returns:
            MD5 hash of the concatenated data
        """
        import hashlib
        data_str = "".join(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_or_create_clusters(
        self, data: List[str], identity: float, **mmseqs_params: Any
    ) -> Dict[str, List[str]]:
        """
        Get cached clusters or create new ones if parameters changed.
        
        This method implements caching to avoid re-clustering when the same
        data and parameters are used multiple times.
        
        Args:
            data: List of sequences to cluster
            identity: Sequence identity threshold
            **mmseqs_params: Additional MMseqs2 parameters
            
        Returns:
            Dictionary mapping cluster IDs to lists of sequence IDs
        """
        current_data_hash = self._get_data_hash(data)
        
        # Check if we can use cached clustering results
        if (self.cluster_map is not None and 
            self._cached_identity == identity and 
            self._cached_params == mmseqs_params and
            self._cached_data_hash == current_data_hash):
            logger.info("Using cached clustering results")
            return self.cluster_map
        
        # Create new clustering
        self.cluster_map = self._run_clustering(data, identity, **mmseqs_params)
        self._cached_identity = identity
        self._cached_params = mmseqs_params.copy()
        self._cached_data_hash = current_data_hash
        
        return self.cluster_map

    def _run_clustering(
        self, data: List[str], identity: float, **mmseqs_params: Any
    ) -> Dict[str, List[str]]:
        """
        Run MMseqs2 clustering on the input sequences.
        
        Args:
            data: List of sequences to cluster
            identity: Sequence identity threshold for clustering
            **mmseqs_params: Additional MMseqs2 parameters
            
        Returns:
            Dictionary mapping cluster IDs to lists of sequence IDs
            
        Raises:
            Exception: If MMseqs2 clustering fails
        """
        logger.info(
            f"Starting MMseqs2 clustering: data_size={len(data)}, "
            f"identity={identity}, params={mmseqs_params}"
        )
        
        try:
            with tempfile.TemporaryDirectory() as tmp_root:
                input_fasta = os.path.join(tmp_root, "input.fasta")
                output_dir = os.path.join(tmp_root, "output")
                tmp_dir = os.path.join(tmp_root, "tmp")

                save_fasta(data, input_fasta)
                logger.info(f"Input data saved to {input_fasta}")

                tsv_path = run_mmseqs_clustering(
                    input_fasta, output_dir, tmp_dir, identity, **mmseqs_params
                )
                cluster_map = parse_cluster_tsv(tsv_path)
                self._print_cluster_stats(cluster_map)

            logger.info(
                f"MMseqs2 clustering completed: generated {len(cluster_map)} clusters "
                f"from {len(data)} sequences"
            )
            return cluster_map
        except Exception as e:
            logger.error(f"MMseqs2 clustering failed: {e}")
            raise

    def _distribute_clusters_to_folds(
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
        logger.info(f"Fold size distribution: {fold_sizes}")
        return folds

    def _generate_kfold_results(
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

            logger.info(
                f"Fold {fold_idx} completed: "
                f"Train={len(kfold_results[f'fold_{fold_idx}']['train'])}, "
                f"Valid={len(kfold_results[f'fold_{fold_idx}']['valid'])}, "
                f"Test={len(kfold_results[f'fold_{fold_idx}']['test'])}"
            )

        return kfold_results

    def _generate_split_from_clusters(
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

        train_data_size = int(len(data) * frac_train)
        valid_data_size = int(len(data) * frac_valid)

        train_ids, valid_ids, test_ids = [], [], []
        count_train = count_valid = 0

        for _, members in cluster_items_copy:
            if count_train + len(members) <= train_data_size:
                train_ids.extend(members)
                count_train += len(members)
            elif count_valid + len(members) <= valid_data_size:
                valid_ids.extend(members)
                count_valid += len(members)
            else:
                test_ids.extend(members)

        logger.info(
            f"Split distribution: "
            f"Target train={train_data_size}, Actual train={len(train_ids)} | "
            f"Target valid={valid_data_size}, Actual valid={len(valid_ids)} | "
            f"Target test={len(data) - train_data_size - valid_data_size}, Actual test={len(test_ids)}"
        )

        # Create a mapping from sequence ID to index
        id_to_idx = {f"seq{i}": i for i in range(len(data))}
        
        return {
            "train": [id_to_idx[x] for x in train_ids if x in id_to_idx],
            "valid": [id_to_idx[x] for x in valid_ids if x in id_to_idx],
            "test": [id_to_idx[x] for x in test_ids if x in id_to_idx],
        }

    def _print_cluster_stats(self, cluster_map: Dict[str, List[str]]) -> None:
        """
        Print comprehensive cluster statistics.
        
        Args:
            cluster_map: Dictionary mapping cluster IDs to sequence ID lists
        """
        cluster_sizes = [len(members) for members in cluster_map.values()]
        
        # Statistical information
        total_sequences = sum(cluster_sizes)
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
            f"  Total sequences: {total_sequences}\n"
            f"  Average cluster size: {avg_cluster_size:.2f}\n"
            f"  Median cluster size: {median_cluster_size:.1f}\n"
            f"  Min cluster size: {min_cluster_size}\n"
            f"  Max cluster size: {max_cluster_size}\n"
            f"  Cluster size distribution:\n"
        )
        
        for label, count in bins.items():
            percentage = (count / len(cluster_map)) * 100
            log_message += f"    {label}: {count} clusters ({percentage:.1f}%)\n"
            
        logger.info(log_message)

    def clear_cache(self) -> None:
        """
        Clear cached clustering results.
        
        This forces the next clustering operation to run from scratch,
        useful when you want to ensure fresh results or free memory.
        """
        self.cluster_map = None
        self._cached_identity = None
        self._cached_params = None
        self._cached_data_hash = None
        logger.info("Clustering cache cleared")

    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about the current clustering.
        
        Returns:
            Dictionary containing cluster statistics and parameters,
            or None if no clustering has been performed yet.
            
        Example:
            {
                "num_clusters": 150,
                "total_sequences": 1000,
                "avg_cluster_size": 6.67,
                "median_cluster_size": 4.0,
                "min_cluster_size": 1,
                "max_cluster_size": 45,
                "identity_threshold": 0.25,
                "mmseqs_params": {"coverage": 0.8, "sensitivity": 7.5}
            }
        """
        if self.cluster_map is None:
            return None
        
        cluster_sizes = [len(members) for members in self.cluster_map.values()]
        return {
            "num_clusters": len(self.cluster_map),
            "total_sequences": sum(cluster_sizes),
            "avg_cluster_size": np.mean(cluster_sizes),
            "median_cluster_size": np.median(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "identity_threshold": self._cached_identity,
            "mmseqs_params": self._cached_params.copy() if self._cached_params else {}
        }


if __name__ == "__main__":
    from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

    dataset_name = "AV_APML"  # Change this to your dataset name

    dataset_manager = SingleTaskDatasetManager(
        dataset_name=dataset_name, official_feature_names=["fasta"]
    )
    fasta = dataset_manager.get_official_feature("fasta")

    splitter = MMseqs2Splitter()

    # Test with default MMseqs2 parameters
    logger.info("=" * 50)
    logger.info("Testing with default MMseqs2 parameters")
    logger.info("=" * 50)

    # Generate 5 random splits with default parameters
    random_splits = splitter.get_split_indices_n(
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
    kfold_splits = splitter.get_split_kfold_indices(
        fasta,
        k_folds=5,
        identity=0.25,
        seed=42,
        # Custom MMseqs2 parameters
        coverage=0.8,
        sensitivity=7.5,
        threads=4,
        cov_mode=1,
    )

    # Test with single split and additional custom parameters
    logger.info("=" * 50)
    logger.info("Testing single split with custom parameters")
    logger.info("=" * 50)

    single_split = splitter.get_split_indices(
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

    # Print cluster information
    cluster_info = splitter.get_cluster_info()
    if cluster_info:
        logger.info(f"Cluster info: {cluster_info}")

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
    logger.info(f"Loaded JSON split - Train: {len(loaded_json['train'])}, Valid: {len(loaded_json['valid'])}, Test: {len(loaded_json['test'])}")
    
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
    logger.info(f"Loaded numpy split - Train: {len(loaded_numpy['train'])}, Valid: {len(loaded_numpy['valid'])}, Test: {len(loaded_numpy['test'])}")
    
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
                    logger.info(f"✓ {split_key}.{split_name} count matches ({original_count})")
                else:
                    logger.error(f"✗ {split_key}.{split_name} count mismatch: {original_count} vs {loaded_count}")
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
        logger.error(f"✗ Total size mismatch: expected {expected_total}, got {stats['total_size']}")
    
    # Check that fractions sum to approximately 1.0
    total_fraction = stats["train_fraction"] + stats["valid_fraction"] + stats["test_fraction"]
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

    logger.info("All MMseqs2 tests completed successfully!")