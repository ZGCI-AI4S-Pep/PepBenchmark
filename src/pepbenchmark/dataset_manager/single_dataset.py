"""Dataset manager for official single-peptide benchmark datasets."""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset
from pepbenchmark.metadata import DATASET_DIR, DATASET_MAP
from pepbenchmark.utils.logging import get_logger

logger = get_logger()

BASE_URL = "https://huggingface.co/datasets/jiahuizhang/PepBenchData/resolve/main/"
DOWNLOAD_ROOT_CANDIDATES = ("", "PepBenchData-50/")




class SinglePeptideDatasetManager:
    """Load official single-peptide datasets and feature artifacts.

    The manager keeps a lightweight in-memory registry of loaded features and
    optional split indices. It can read local artifacts directly or download
    missing files from the hosted benchmark repository.

    Args:
        dataset_name: Official dataset identifier defined in
            :mod:`pepbenchmark.metadata`.
        official_feature_names: Optional list of official feature names to load
            during initialization.
        dataset_dir: Root directory that stores official dataset folders.
        force_download: Whether to re-download official files even when a local
            copy already exists.
    """
    OFFICIAL_FEATURE_MAP: Dict[str, str] = {
        "fasta": "fasta.csv",
        "smiles": "smiles.csv",
        "helm": "helm.csv",
        "biln": "biln.csv",
        "ecfp4": "ecfp4.npz",
        "ecfp6": "ecfp6.npz",
        "esm2_150_embedding": "esm2_150_embedding.npz",
        "dplm_150_embedding": "dplm_150_embedding.npz",
        "graph": "graph.pt",
        "label": "label.csv",
    }

    OFFICIAL_SPLIT_MAP: Dict[str, str] = {
        "random_split": "random_split.json",
        "mmseqs2_split": "mmseqs2_split.json",
        "cdhit_split": "cdhit_split.json",
        "kmer_split": "kmer_split.json",
        "ecfp_split": "ecfp_split.json",
        "hybrid_split": "hybrid_split.json",
    }


    def __init__(
        self,
        dataset_name: str,
        official_feature_names: Optional[List[str]] = None,
        dataset_dir: str = DATASET_DIR,
        force_download: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(dataset_dir, dataset_name)
        self._check_official_dataset_name()
        self.dataset_metadata = DATASET_MAP.get(self.dataset_name)
        if self.dataset_metadata is None:
            raise ValueError(f"Dataset metadata not found for: {self.dataset_name}")

        self.force_download = force_download
        self.features = {}
        self.length = None
        self.split_indices = None

        for feature_name in official_feature_names or []:
            self.set_official_feature(feature_name)
    def _check_official_dataset_name(self) -> bool:
        """Validate that the dataset name exists in the official registry."""
        if self.dataset_name not in DATASET_MAP:
            raise ValueError(
                f"Unknown dataset name: {self.dataset_name}. "
                f"Available datasets: {list(DATASET_MAP.keys())}"
            )
        return True

    def _check_official_feature_name(self, feature_name: str) -> bool:
        """Validate that a requested feature belongs to the official feature set."""
        if feature_name not in self.OFFICIAL_FEATURE_MAP:
            raise ValueError(
                f"Feature type '{feature_name}' is not an official feature type. "
                f"Supported types: {self.OFFICIAL_FEATURE_MAP}"
            )
        return True

    def set_official_feature(self, feature_name: str) -> None:
        """Load an official feature file into the manager.

        Args:
            feature_name: Name of the official feature to load, such as
                ``"fasta"``, ``"label"``, or ``"ecfp4"``.
        """
        self._check_official_feature_name(feature_name)
        if feature_name in self.features:
            logger.info(f"Feature {feature_name} already exists")
            return
        feature_path = os.path.join( self.dataset_path, self.OFFICIAL_FEATURE_MAP[feature_name] )
        # Check if download is needed.
        need_download = False
        if not os.path.exists(feature_path):
            logger.info( f"Feature file ===={feature_path}=== not found locally, will download" )
            need_download = True
        elif self.force_download:
            logger.info( f"Feature file ===={feature_path}=== exists but force_download is True, will re-download" )
            need_download = True
        if need_download:
            self._download_official_file(feature_name)
        file_extention = self.OFFICIAL_FEATURE_MAP[feature_name].split(".")[-1]
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


        if self.length is None:
            self.length = len(feature)

        assert len(feature) == self.length, (
            f"Feature {feature_name} length mismatch: "
            f"expected {self.length}, got {len(feature)}"
        )

        self.features[feature_name] = feature
        logger.info(f"Loaded official feature: {feature_name}")

    def set_user_feature(
        self, feature_name: str, feature_data: Union[List[Any], np.ndarray, pd.Series]
    ) -> None:
        """Register a user-provided feature array.

        Args:
            feature_name: Feature key to store.
            feature_data: Feature values aligned with the dataset order.
        """
        if feature_name in self.features:
            raise ValueError(f"Feature {feature_name} already exists")

        if self.length is None:
            self.length = len(feature_data)

        assert len(feature_data) == self.length, (
            f"Feature {feature_name} length mismatch: "
            f"expected {self.length}, got {len(feature_data)}"
        )

        self.features[feature_name] = feature_data
        logger.info(f"Added user feature: {feature_name}")

    def get_feature(self, feature_name: str):
        """Return a previously loaded feature by name.

        Args:
            feature_name: Feature key stored in the manager.

        Returns:
            The loaded feature payload.
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not found")
        return self.features[feature_name]

    def get_all_features(self) -> Dict[str, Any]:
        """Return the internal feature dictionary."""
        return self.features

    def get_all_feature_names(self) -> List[str]:
        """Return the names of all currently loaded features."""
        return list(self.features.keys())

    def remove_feature(self, feature_name: str):
        """Remove a feature from the in-memory registry.

        Args:
            feature_name: Name of the feature to delete.
        """
        if feature_name in self.features:
            del self.features[feature_name]
            logger.info(f"Removed feature: {feature_name}")

    def _download_official_file(self, name: str) -> None:
        """Download a feature or split artifact from the hosted dataset store.

        Args:
            name: Official feature or split name defined in this module.
        """
        if name in self.OFFICIAL_FEATURE_MAP:
            path = self.OFFICIAL_FEATURE_MAP[name]
        elif name in self.OFFICIAL_SPLIT_MAP:
            path = self.OFFICIAL_SPLIT_MAP[name]
        else:
            raise ValueError(f"Name '{name}' is not an official feature or split type.")

        file_path = os.path.join(self.dataset_path, path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}

        attempted_urls: List[str] = []
        for root in DOWNLOAD_ROOT_CANDIDATES:
            url = f"{BASE_URL}{root}{self.dataset_name}/{path}"
            attempted_urls.append(url)
            logger.info(f"Downloading from {url} and store in {file_path}")
            try:
                with requests.get(url, stream=True, timeout=100, headers=headers) as r:
                    if r.status_code == 404:
                        logger.warning(f"File not found at {url}, try next candidate")
                        continue
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        delete=False,
                        dir=os.path.dirname(file_path),
                        prefix=".download_",
                    ) as f:
                        temp_path = f.name
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    os.replace(temp_path, file_path)
                    return
            except requests.RequestException as e:
                logger.warning(f"Download attempt failed for {url}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error while downloading {url}: {e}")
                try:
                    if "temp_path" in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass

        attempted = "\n - ".join(attempted_urls)
        raise requests.HTTPError(
            "Failed to download official file. Tried URLs:\n"
            f" - {attempted}\n"
            f"Local target path: {file_path}"
        )

    def set_official_split_indices(
        self, split_type: str = "random_split", fold_seed: int = 0
    ) -> Dict[str, List[int]]:
        """Load one official split definition into the manager.

        Args:
            split_type: Official split artifact name such as
                ``"random_split"`` or ``"mmseqs2_split"``.
            fold_seed: Split seed identifier stored in the JSON file.
        """
        if split_type not in self.OFFICIAL_SPLIT_MAP:
            raise ValueError(f"Split type '{split_type}' is not an official split type.")
        
        split_path = os.path.join(self.dataset_path, self.OFFICIAL_SPLIT_MAP[split_type] )
        # Check if download is needed.
        need_download = False
        if not os.path.exists(split_path):
            logger.info( f"Split file ===={split_path}=== not found locally, will download" )
            need_download = True
        elif self.force_download:
            logger.info( f"Split file ===={split_path}=== exists but force_download is True, will re-download" )
            need_download = True
        if need_download:
            self._download_official_file(split_type)
        with open(split_path, "r") as f:
            all_splits = json.load(f)
        splits = all_splits.get(f"seed_{fold_seed}")
        if splits is None:
            available_seeds = sorted(
                k for k in all_splits.keys() if isinstance(k, str) and k.startswith("seed_")
            )
            raise ValueError(
                f"Seed {fold_seed} not found in split '{split_type}'. "
                f"Available seeds: {available_seeds}"
            )
        self.split_indices = splits
        logger.info(
            f"Set official split ==={split_type}=== with seed ===={fold_seed}=== successfully"
        )
        return splits

    def set_user_split_indices(self, split_indices):
        """Store user-defined split indices.

        Args:
            split_indices: Mapping with ``train``, ``valid``, and ``test`` keys.
        """
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
        """Select a subset from common container types.

        Args:
            data: Container that supports item access.
            indices: Integer indices to extract.

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
        """Return train/validation/test subsets for all loaded features.

        Args:
            format: Output format. Use ``"dict"`` for raw feature dictionaries
                or ``"pytorch_dataset"`` for lightweight dataset wrappers.

        Returns:
            Either three feature dictionaries or three PyTorch-style datasets.
        """
        if self.split_indices is None:
            raise ValueError(
                "Split indices are not set. Please set split indices first."
            )

        train_idx = self.split_indices.get("train", [])
        valid_idx = self.split_indices.get("valid", [])
        test_idx = self.split_indices.get("test", [])

        def _collect_features(indices):
            return {
                k: self._select_by_indices(v, indices)
                for k, v in self.features.items()
            }

        train_features = _collect_features(train_idx)
        valid_features = _collect_features(valid_idx)
        test_features = _collect_features(test_idx)

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
        """Return the number of samples represented by the loaded features."""
        if self.length is None:
            raise ValueError("Dataset length is not set. Please load features first.")
        return self.length

    def __getitem__(self, idx: int):
        """Return all loaded feature values for a single sample index.

        Args:
            idx: Sample index in the current dataset ordering.

        Returns:
            A dictionary that maps feature names to sample-level values.
        """

        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds")

        return {
            k: v[idx]
            for k, v in self.features.items()
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return metadata associated with the current dataset.

        Returns:
            Dataset metadata stored in :mod:`pepbenchmark.metadata`.
        """
        return DATASET_MAP.get(self.dataset_name, {})
