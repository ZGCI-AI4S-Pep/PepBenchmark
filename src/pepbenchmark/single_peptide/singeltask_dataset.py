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

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from pepbenchmark.raw_data import DATASET_MAP
from pepbenchmark.single_peptide.base_dataset import DatasetManager
from pepbenchmark.utils.logging import get_logger

logger = get_logger()

AVALIABLE_DATASET = list(DATASET_MAP.keys())
BASE_URL = "https://raw.githubusercontent.com/ZGCI-AI4S-Pep/peptide_data/main/"
# Supported official feature types.
OFFICIAL_FEATURE_TYPES = {
    "fasta",
    "esm2_150_embedding",
    "dpml_embedding",
    "smiles",
    "helm",
    "biln",
    "ecfp4",
    "ecfp6",
    "random_split",
    "mmseqs2_split",
    "cdhit_split",
    "graph",
    "label",
}

FEATURE_FILE_EXTENTION_MAP = {
    "fasta": "csv",
    "smiles": "csv",
    "helm": "csv",
    "biln": "csv",
    "ecfp4": "npz",
    "ecfp6": "npz",
    "esm2_150_embedding": "npz",
    "dpml_embedding": "npz",
    "graph": "pt",
    "random_split": "json",
    "mmseqs2_split": "json",
    "cdhit_split": "json",
    "label": "csv",
}


class SingleTaskDatasetManager(DatasetManager):
    """Single-task dataset class supporting multiple feature types and dynamic processing.

    This class provides comprehensive support for both official and user-defined datasets
    with flexible feature management and processing capabilities.

    Key features:
        - Support for official datasets and user-defined datasets
        - Multiple feature types: fasta, fasta_esm2_150, smiles, helm, biln, ecfp
        - Dynamic feature processing and preprocessing cache
        - Official data splits and custom splitting
        - Data augmentation and transformations

    Example usage:
        # Using official dataset
        dataset = SingleTaskDataset(
            dataset_name="BBP_APML",
            feature_types=["fasta", "smiles"]
        )

        # Using custom dataset
        dataset = SingleTaskDataset(
            dataset_name="user_dataset_name",
            user_dataset=True,
            feature_types=["fasta"]
        )
        dataset.add_features_by_dict({"fasta": ["AAGC", "AAGT"]})
        dataset.add_labels_by_dict({"label": [1, 0]})
    """

    def __init__(
        self,
        dataset_name: str,
        official_feature_names: Optional[List[str]] = None,
        dataset_dir: str = None,
        force_download: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            force_download=force_download,
        )

        self.official_feature_dict = {}
        self.user_feature_dict = {}

        self.split_indices = None
        self.length = None

        for feature_name in official_feature_names or self.get_metadata.get(
            "feature_types"
        ):
            self.set_official_feature(feature_name)

    def _check_official_feature_name(self, feature_name: str) -> bool:
        """Checks if the feature name is an official feature type."""
        if feature_name not in OFFICIAL_FEATURE_TYPES:
            raise ValueError(
                f"Feature type '{feature_name}' is not an official feature type. "
                f"Supported types: {OFFICIAL_FEATURE_TYPES}"
            )
        return True

    def get_official_feature(self, feature_name) -> None:
        self._check_official_feature_name(feature_name)
        if self.official_feature_dict.get(feature_name) is not None:
            logger.info(f"Feature {feature_name} already loaded, skipping download")
            return self.official_feature_dict[feature_name]

        file_extention = FEATURE_FILE_EXTENTION_MAP.get(feature_name)
        feature_path = os.path.join(
            self.dataset_dir, f"{feature_name}.{file_extention}"
        )
        full_name = f"{self.dataset_name}/{feature_name}.{file_extention}"
        print(feature_path)

        # Check if download is needed.
        need_download = False
        if not os.path.exists(feature_path):
            logger.info(
                f"Feature file ===={full_name}=== not found locally, will download"
            )
            need_download = True
        elif self.force_download:
            logger.info(
                f"Feature file ===={full_name}=== exists but force_download is True, will re-download"
            )
            need_download = True

        if need_download:
            self._download_official_feature(feature_name)

        if file_extention == "csv":
            feature = pd.read_csv(feature_path)["feature"].to_list()
        elif file_extention == "npz":
            feature = np.load(feature_path)["data"]
        elif file_extention == "pt":
            feature = torch.load(feature_path, weights_only=False)
        elif file_extention == "json":
            with open(feature_path, "r") as f:
                feature = json.load(f)
        else:
            raise ValueError(f"Unsupported feature file extension: {file_extention}")

        return feature

    def set_official_feature(self, feature_name: str) -> None:
        if feature_name in self.official_feature_dict:
            logger.info(f"Feature {feature_name} already exists in dataset")
            return
        feature = self.get_official_feature(feature_name)
        if self.length is None:
            self.length = len(feature)
        assert (
            len(feature) == self.length
        ), f"Feature {feature_name} length mismatch: expected {self.length}, got {len(feature)}"
        self.official_feature_dict[feature_name] = feature
        logger.info(f"Set official feature: {feature_name} successfully")

    def set_user_feature(
        self, feature_name: str, feature_data: Union[List[Any], np.ndarray, pd.Series]
    ) -> None:
        """Sets a user-defined feature for the dataset.

        Args:
            feature_name: Name of the user-defined feature.
            feature_data: Data for the user-defined feature.
        """
        if feature_name in self.user_feature_dict:
            logger.info(f"User feature {feature_name} already exists in dataset")
            return
        if self.length is None:
            self.length = len(feature_data)
        assert (
            len(feature_data) == self.length
        ), f"Feature {feature_name} length mismatch: expected {self.length}, got {len(feature_data)}"
        self.user_feature_dict[feature_name] = feature_data
        logger.info(f"Set user feature: {feature_name} successfully")

    def get_user_feature(self, feature_name: str):
        """Retrieves a user-defined feature from the dataset.

        Args:
            feature_name: Name of the user-defined feature.

        Returns:
            The user-defined feature data.
        """
        if feature_name not in self.user_feature_dict:
            raise ValueError(f"User feature {feature_name} not found in dataset")
        return self.user_feature_dict[feature_name]

    def remove_official_feature(self, feature_name: str) -> None:
        """Removes an official feature from the dataset.

        Args:
            feature_name: Name of the feature to remove.
        """
        if feature_name in self.official_feature_dict:
            del self.official_feature_dict[feature_name]
            logger.info(f"Removed official feature: {feature_name}")
        else:
            logger.warning(
                f"Feature {feature_name} not found in dataset, cannot remove"
            )

    def get_feature_names(self) -> List[str]:
        """Retrieves the names of all features in the dataset.

        Returns:
            A list of feature names.
        """
        # 分别添加official 和 user 前缀
        official_feature_names = [
            f"official_{name}" for name in self.official_feature_dict.keys()
        ]
        user_feature_names = [f"user_{name}" for name in self.user_feature_dict.keys()]
        return official_feature_names + user_feature_names

    def remove_user_feature(self, feature_name: str) -> None:
        """Removes a user-defined feature from the dataset.

        Args:
            feature_name: Name of the user-defined feature to remove.
        """
        if feature_name in self.user_feature_dict:
            del self.user_feature_dict[feature_name]
            logger.info(f"Removed user feature: {feature_name}")
        else:
            logger.warning(
                f"User feature {feature_name} not found in dataset, cannot remove"
            )

    def get_positive_sequences(self) -> List[str]:
        fasta_list = self.get_official_feature("fasta")
        label_list = self.get_official_feature("label")
        self.pos_seqs = [
            seq for seq, label in zip(fasta_list, label_list) if label == 1
        ]
        return self.pos_seqs

    def _download_official_feature(self, feature_name: str) -> None:
        """Downloads the data file for the specified feature type."""

        url = (
            BASE_URL
            + f"{self.dataset_name}/{feature_name}.{FEATURE_FILE_EXTENTION_MAP[feature_name]}"
        )

        file_extention = FEATURE_FILE_EXTENTION_MAP.get(feature_name)
        feature_path = os.path.join(
            self.dataset_dir, f"{feature_name}.{file_extention}"
        )
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        logger.info(
            f"Downloading ==={self.dataset_name}/{feature_name}.{file_extention}=== from {url}"
        )
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            with requests.get(url, stream=True, timeout=100, headers=headers) as r:
                r.raise_for_status()
                with open(feature_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except requests.RequestException as e:
            logger.error(f"Failed to download {feature_name}: {e}")
            raise

    def set_official_split_indices(
        self, split_type: str = "random_split", fold_seed: int = 0
    ) -> None:
        """Sets the official split indices for train, validation, and test sets.

        Args:
            split_type: Type of split to perform. Can be 'random', 'stratified', or 'custom'.
            fold_seed: Seed for random splits.
            split_ratio: Custom split ratios for train, valid, and test sets.
        """

        splits = self.get_official_feature(split_type)
        splits = splits.get(f"seed_{fold_seed}")
        self.split_indices = splits
        logger.info(
            f"Set official split ==={split_type}=== with seed ===={fold_seed}=== successfully"
        )

    def set_user_split_indices(self, split_indices):
        self.split_indices = split_indices
        logger.info("Set user-defined split indices successfully")

    def get_split_indices(self) -> Dict[str, List[int]]:
        """Retrieves the split indices for train, validation, and test sets.

        Returns:
            A dictionary with keys 'train', 'valid', and 'test' containing the respective indices.
        """
        if self.split_indices is None:
            raise ValueError(
                "Split indices are not set. Please set split indices first."
            )
        return self.split_indices

    def _select_by_indices(self, data, indices):
        """Universal index selection for any object supporting __getitem__.

        Args:
            data: Any object supporting __getitem__ (list, np.ndarray, pd.Series, torch.Tensor, etc.)
            indices: List or np.ndarray of indices.

        Returns:
            Subset of data at the given indices.
        """
        # Try direct advanced indexing (works for numpy, pandas, torch.Tensor)
        try:
            return data[indices]
        except Exception:
            pass

        # Try __getitem__ for each index (works for list, tuple, or any sequence)
        try:
            return type(data)(data[i] for i in indices)
        except Exception:
            # Fallback: return as list
            return [data[i] for i in indices]

    def get_train_val_test_features(
        self, format: str = "dict"
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Retrieves train, validation, and test features.

        Args:
            format: Format of the returned features. Can be 'dict' or 'tuple'.

        Returns:
            Features in the specified format.
        """
        if self.split_indices is None:
            raise ValueError(
                "Split indices are not set. Please set split indices first."
            )

        train_idx = self.split_indices.get("train", [])
        valid_idx = self.split_indices.get("valid", [])
        test_idx = self.split_indices.get("test", [])

        def _collect_features(feature_dict, indices, prefix):
            return {
                f"{prefix}_{k}": self._select_by_indices(v, indices)
                for k, v in feature_dict.items()
            }

        train_features = {}
        valid_features = {}
        test_features = {}

        train_features.update(
            _collect_features(self.official_feature_dict, train_idx, "official")
        )
        train_features.update(
            _collect_features(self.user_feature_dict, train_idx, "user")
        )
        valid_features.update(
            _collect_features(self.official_feature_dict, valid_idx, "official")
        )
        valid_features.update(
            _collect_features(self.user_feature_dict, valid_idx, "user")
        )
        test_features.update(
            _collect_features(self.official_feature_dict, test_idx, "official")
        )
        test_features.update(
            _collect_features(self.user_feature_dict, test_idx, "user")
        )

        if format == "dict":
            return train_features, valid_features, test_features
        elif format == "pytorch_dataset":

            class PyDataset(Dataset):
                def __init__(self, features: Dict[str, Any]):
                    self.features = features

                def __len__(self):
                    # Return the length of the first feature, which is the number of data points
                    return len(next(iter(self.features.values())))

                def __getitem__(self, idx):
                    return {k: v[idx] for k, v in self.features.items()}

            return (
                PyDataset(train_features),
                PyDataset(valid_features),
                PyDataset(test_features),
            )

        else:
            raise ValueError(f"Unsupported format: {format}")

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        if self.length is None:
            raise ValueError("Dataset length is not set. Please load features first.")
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves a single data sample by index.

        Args:
            idx: Index of the data sample to retrieve.

        Returns:
            A dictionary containing the features for the specified index.
        """
        if self.length is None:
            raise ValueError("Dataset length is not set. Please load features first.")
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {self.length}"
            )

        sample = {}
        for feature_name, feature_data in self.official_feature_dict.items():
            sample[f"official_{feature_name}"] = feature_data[idx]
        for feature_name, feature_data in self.user_feature_dict.items():
            sample[f"user_{feature_name}"] = feature_data[idx]

        return sample

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Retrieves metadata for the dataset.

        Returns:
            A dictionary containing metadata such as dataset name, path, and feature types.
        """
        return DATASET_MAP.get(self.dataset_name, {})
