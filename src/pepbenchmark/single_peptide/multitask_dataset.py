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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from torch.utils.data import Dataset

from pepbenchmark.pep_utils.neg_sample import MultiTaskNegSampler
from pepbenchmark.utils.logging import get_logger
from src.pepbenchmark.single_peptide.base_dataset import DatasetManager

logger = get_logger()

LABEL_TYPE = ["Antibacterial", "Antiviral"]

BASE_URL = "https://raw.githubusercontent.com/ZGCI-AI4S-Pep/peptide_data/main/"


class MultiTaskDatasetManager(DatasetManager):
    def __init__(
        self,
        dataset_name: str = "multitask_peptidepedia",
        labels: Optional[
            List[str]
        ] = None,  # activity names, et.al. ["Antibacterial", "Antiviral"]
        dataset_dir: str = None,
        force_download: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            force_download=force_download,
        )

        if not labels:
            raise ValueError(
                f"The labels parameter must be defined as a list of strings."
                f" Please choose one or more labels from {LABEL_TYPE}."
            )

        self._check_label_name(labels)
        self.labels = labels
        self.user_feature_dict = {}
        self.split_indices = None
        self.length = None
        self.official_feature_dict = {}
        self.user_feature_dict = {}
        self.set_data()

    def _check_label_name(self, label_name: list) -> bool:
        not_in_official = [label for label in label_name if label not in LABEL_TYPE]
        if not_in_official:
            raise ValueError(
                f"Label names {not_in_official} not found in the official label list."
                f" Please choose one or more labels from {LABEL_TYPE}."
            )
        return True

    def _download_fasta_file(self, label_name: str):
        url = f"{BASE_URL}{self.dataset_name}/{label_name}.fasta"
        fasta_path = os.path.join(self.dataset_dir, f"{label_name}.fasta")
        os.makedirs(os.path.dirname(fasta_path), exist_ok=True)

        logger.info(f"Downloading {label_name}.fasta from {url}")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(fasta_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.RequestException as e:
            logger.error(f"Download failed for {label_name}.fasta: {e}")
            raise

    def set_data(self):
        sequence_to_multilabel = defaultdict(lambda: [0] * len(self.labels))

        for i, label_name in enumerate(self.labels):
            fasta_path = os.path.join(self.dataset_dir, f"{label_name}.fasta")

            need_download = False
            if not os.path.exists(fasta_path):
                logger.info(f"{fasta_path} not found locally.")
                need_download = True
            elif self.force_download:
                logger.info(f"{fasta_path} exists.")
                need_download = True

            if need_download:
                self._download_fasta_file(label_name)

            for record in SeqIO.parse(fasta_path, "fasta"):
                seq = str(record.seq)
                sequence_to_multilabel[seq][i] = 1

        sequences = list(sequence_to_multilabel.keys())
        multilabels = list(sequence_to_multilabel.values())
        self.pos_seq = sequences
        self.length = len(sequences)
        self.official_feature_dict["fasta"] = sequences

        for i, label_name in enumerate(self.labels):
            self.official_feature_dict[label_name] = [label[i] for label in multilabels]

        logger.info(f"MultitaskDataset for {self.labels} loaded.")

    def get_fasta(self):
        return self.official_feature_dict["fasta"]

    def set_user_feature(
        self, feature_name: str, feature_data: Union[List[Any], np.ndarray, pd.Series]
    ) -> None:
        """Sets a user-defined feature for the dataset.

        Args:
            feature_name: Name of the user-defined feature.
            feature_data: Data for the user-defined feature.
        """

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

    def negative_sampling(self, ratio: int, **kwargs):
        seed = kwargs.get("seed", 42)
        if not self.labels:
            raise ValueError("labels not found.")

        sampler = MultiTaskNegSampler(self.labels)
        neg_seqs = sampler(self.pos_seq, ratio, seed)
        self.official_feature_dict["fasta"] = self.pos_seq + neg_seqs
        for key, values in self.official_feature_dict.items():
            if key == "fasta":
                continue
            self.official_feature_dict[key] = values + [0] * len(neg_seqs)

        self.length = len(self.official_feature_dict["fasta"])

        if self.user_feature_dict:
            for key in list(self.user_feature_dict.keys()):
                self.remove_user_feature(key)
            logger.info(
                f"User features {list(self.user_feature_dict.keys())} need to be recomputed after negative sampling."
            )

    def get_data(self):
        combined_dict = {**self.official_feature_dict, **self.user_feature_dict}
        df = pd.DataFrame(combined_dict)
        return df

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

        def _collect_features(feature_dict, indices):
            return {
                f"{k}": self._select_by_indices(v, indices)
                for k, v in feature_dict.items()
            }

        train_features = {}
        valid_features = {}
        test_features = {}

        train_features.update(_collect_features(self.official_feature_dict, train_idx))
        train_features.update(_collect_features(self.user_feature_dict, train_idx))
        valid_features.update(_collect_features(self.official_feature_dict, valid_idx))
        valid_features.update(_collect_features(self.user_feature_dict, valid_idx))
        test_features.update(_collect_features(self.official_feature_dict, test_idx))
        test_features.update(_collect_features(self.user_feature_dict, test_idx))

        if format == "dict":
            return train_features, valid_features, test_features
        elif format == "pytorch_dataset":

            class PyDataset(Dataset):
                def __init__(self, features: Dict[str, Any]):
                    self.features = features

                def __len__(self):
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
