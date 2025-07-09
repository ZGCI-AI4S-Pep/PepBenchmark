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

"""Peptide Dataset Preprocessing Module.

This module provides comprehensive preprocessing functionality for peptide datasets.
It handles loading raw data from combine.csv files and generating various types of
features based on OFFICIAL_FEATURE_TYPES.

Key features:
    - Load dataset from DATASET_MAP and read combine.csv
    - Generate multiple feature types: fasta, smiles, helm, biln, ecfp, embeddings
    - Support batch processing and caching
    - Generate train/validation/test splits
    - Export processed features in multiple formats

Example:
    >>> from pepbenchmark.preprocess import DatasetPreprocessor
    >>>
    >>> # Initialize preprocessor for a specific dataset
    >>> preprocessor = DatasetPreprocessor('BBP_APML')
    >>>
    >>> # Load raw data from combine.csv
    >>> preprocessor.load_raw_data()
    >>>
    >>> # Generate all available features
    >>> preprocessor.generate_all_features()
    >>>
    >>> # Save processed features
    >>> preprocessor.save_features()
    >>>
    >>> # Or generate specific features
    >>> preprocessor.generate_features(['fasta', 'smiles', 'ecfp4'])
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from pepbenchmark.metadata import DATASET_MAP
from pepbenchmark.pep_utils.convert import (
    Fasta2Biln,
    Fasta2Embedding,
    Fasta2Helm,
    Fasta2Smiles,
    Smiles2FP,
    Smiles2Graph,
)
from pepbenchmark.single_peptide.singeltask_dataset import (
    FEATURE_FILE_EXTENTION_MAP,
    OFFICIAL_FEATURE_TYPES,
)
from pepbenchmark.splitter.homo_spliter import MMseqs2Spliter
from pepbenchmark.splitter.random_spliter import RandomSplitter
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class DatasetPreprocessor:
    """Comprehensive dataset preprocessing class for peptide datasets.

    This class handles loading raw data from combine.csv files and generating
    various feature types based on OFFICIAL_FEATURE_TYPES.

    Args:
        dataset_name: Name of the dataset from DATASET_MAP
        base_data_dir: Base directory for data storage (optional)
        force_regenerate: Whether to force regeneration of existing features

    """

    def __init__(
        self,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        force_regenerate: bool = False,
    ):
        self.dataset_name = dataset_name
        self.force_regenerate = force_regenerate

        # Validate dataset name
        if dataset_name not in DATASET_MAP:
            available_datasets = list(DATASET_MAP.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found in DATASET_MAP. "
                f"Available datasets: {available_datasets}"
            )

        # Get dataset metadata
        self.metadata = DATASET_MAP[dataset_name]

        # Use base_data_dir if provided, otherwise use the path from metadata
        if base_data_dir is not None:
            self.dataset_path = DATASET_MAP.get(dataset_name, {}).get("path", None)
        else:
            self.dataset_path = base_data_dir

        # Initialize converters (lazy loading)
        self._converters = {}
        self._raw_data = None
        self._features = {}

        logger.info(f"Initialized preprocessor for dataset: {dataset_name}")
        logger.info(f"Dataset path: {self.dataset_path}")

    def _get_converter(self, feature_type: str):
        """Get or create converter for specific feature type."""
        if feature_type not in self._converters:
            if feature_type == "smiles":
                self._converters[feature_type] = Fasta2Smiles()
            elif feature_type == "helm":
                self._converters[feature_type] = Fasta2Helm()
            elif feature_type == "biln":
                self._converters[feature_type] = Fasta2Biln()
            elif feature_type == "ecfp4":
                self._converters["smiles"] = Fasta2Smiles()
                self._converters[feature_type] = Smiles2FP(
                    fp_type="Morgan", radius=2, nBits=2048
                )
            elif feature_type == "ecfp6":
                self._converters["smiles"] = Fasta2Smiles()
                self._converters[feature_type] = Smiles2FP(
                    fp_type="Morgan", radius=3, nBits=2048
                )
            elif feature_type == "graph":
                self._converters["smiles"] = Fasta2Smiles()
                self._converters[feature_type] = Smiles2Graph()
            elif feature_type == "esm2_150_embedding":
                self._converters[feature_type] = Fasta2Embedding(
                    model="facebook/esm2_t30_150M_UR50D"
                )
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

        return self._converters[feature_type]

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from combine.csv file.

        Returns:
            DataFrame containing sequence and label columns
        """
        combine_csv_path = os.path.join(self.dataset_path, "combine.csv")

        if not os.path.exists(combine_csv_path):
            raise FileNotFoundError(
                f"combine.csv not found at {combine_csv_path}. "
                f"Please ensure the dataset path is correct."
            )

        logger.info(f"Loading raw data from {combine_csv_path}")

        # Load CSV file
        self._raw_data = pd.read_csv(combine_csv_path)

        # Validate required columns
        required_columns = ["sequence", "label"]
        missing_columns = [
            col for col in required_columns if col not in self._raw_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in combine.csv: {missing_columns}. "
                f"Available columns: {list(self._raw_data.columns)}"
            )

        logger.info(f"Loaded {len(self._raw_data)} samples")
        logger.info(f"Data shape: {self._raw_data.shape}")
        logger.info(f"Label distribution:\n{self._raw_data['label'].value_counts()}")

        return self._raw_data

    def generate_feature(self, feature_type: str) -> Union[List, np.ndarray]:
        """Generate a specific feature type.

        Args:
            feature_type: Type of feature to generate

        Returns:
            Generated features
        """
        if self._raw_data is None:
            raise ValueError("Raw data not loaded. Please call load_raw_data() first.")

        if feature_type not in OFFICIAL_FEATURE_TYPES:
            raise ValueError(
                f"Feature type '{feature_type}' not supported. "
                f"Supported types: {OFFICIAL_FEATURE_TYPES}"
            )

        logger.info(f"Generating feature: {feature_type}")

        # Handle special cases
        if feature_type == "fasta":
            return self._raw_data["sequence"].tolist()

        elif feature_type == "label":
            return self._raw_data["label"].tolist()

        elif feature_type in ["random_split", "mmseqs2_split"]:
            return self._generate_split(feature_type)

        else:
            converter = self._get_converter(feature_type)
            sequences = self._raw_data["sequence"].tolist()

            if feature_type in ["ecfp4", "ecfp6", "graph"]:
                smiles_converter = self._get_converter("smiles")
                logger.info("Converting FASTA to SMILES...")
                smiles_list = smiles_converter(sequences)

                logger.info(f"Converting SMILES to {feature_type.upper()}...")
                features = converter(smiles_list)

                # ECFP需要转换为numpy数组
                if feature_type.startswith("ecfp"):
                    return np.array(features)
                else:
                    return features

            else:
                features = converter(sequences)
                return features

    def _generate_split(self, split_type: str) -> Dict[str, Any]:
        """Generate train/validation/test splits.

        Args:
            split_type: Type of split ('random_split' or 'mmseqs2_split')

        Returns:
            Dictionary containing split indices for multiple seeds
        """
        if self._raw_data is None:
            raise ValueError("Raw data not loaded. Please call load_raw_data() first.")

        sequences = self._raw_data["sequence"].tolist()

        split_results = {}

        # 生成多个种子的划分
        for seed in [0, 1, 2, 3, 4]:
            logger.info(f"Generating {split_type} with seed {seed}")

            if split_type == "random_split":
                splitter = RandomSplitter()
                split_result = splitter.get_split_indices(
                    data=sequences,
                    frac_train=0.8,
                    frac_valid=0.1,
                    frac_test=0.1,
                    seed=seed,
                )
                train_idx = split_result["train"]
                valid_idx = split_result["valid"]
                test_idx = split_result["test"]

            elif split_type == "mmseqs2_split":
                splitter = MMseqs2Spliter()
                split_result = splitter.get_split_indices(
                    data=sequences,
                    frac_train=0.8,
                    frac_valid=0.1,
                    frac_test=0.1,
                    seed=seed,
                )
                train_idx = split_result["train"]
                valid_idx = split_result["valid"]
                test_idx = split_result["test"]

            split_results[f"seed_{seed}"] = {
                "train": train_idx.tolist()
                if hasattr(train_idx, "tolist")
                else list(train_idx),
                "valid": valid_idx.tolist()
                if hasattr(valid_idx, "tolist")
                else list(valid_idx),
                "test": test_idx.tolist()
                if hasattr(test_idx, "tolist")
                else list(test_idx),
            }

        return split_results

    def generate_features(self, feature_types: List[str]) -> Dict[str, Any]:
        """Generate multiple feature types.

        Args:
            feature_types: List of feature types to generate

        Returns:
            Dictionary containing all generated features
        """
        results = {}

        for feature_type in feature_types:
            try:
                features = self.generate_feature(feature_type)
                results[feature_type] = features
                self._features[feature_type] = features
                logger.info(f"Successfully generated feature: {feature_type}")
            except Exception as e:
                logger.error(f"Failed to generate feature {feature_type}: {e}")
                results[feature_type] = None

        return results

    def generate_all_features(self) -> Dict[str, Any]:
        """Generate all supported feature types.

        Returns:
            Dictionary containing all generated features
        """
        return self.generate_features(list(OFFICIAL_FEATURE_TYPES))

    def save_feature(self, feature_type: str, features: Any) -> None:
        """Save a single feature to file.

        Args:
            feature_type: Type of feature
            features: Feature data
        """
        if features is None:
            logger.warning(f"No features to save for {feature_type}")
            return

        output_dir = self.dataset_path
        os.makedirs(output_dir, exist_ok=True)
        extension = FEATURE_FILE_EXTENTION_MAP.get(feature_type, "csv")
        output_path = os.path.join(output_dir, f"{feature_type}.{extension}")
        logger.info(f"Saving {feature_type} features to {output_path}")

        try:
            if extension == "csv":
                self._save_csv(features, output_path)
            elif extension == "npz":
                self._save_npz(features, output_path)
            elif extension == "pt":
                self._save_pt(features, output_path, feature_type)
            elif extension == "json":
                self._save_json(features, output_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
            logger.info(f"Successfully saved {feature_type} features")
        except Exception as e:
            logger.error(f"Failed to save {feature_type} features: {e}")
            raise

    def _save_csv(self, features, output_path):
        if isinstance(features, (list, np.ndarray)):
            df = pd.DataFrame({"feature": features})
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported data type for CSV: {type(features)}")

    def _save_npz(self, features, output_path):
        if isinstance(features, list):
            features = np.array(features)
        np.savez_compressed(output_path, data=features)

    def _save_pt(self, features, output_path, feature_type):
        if feature_type == "graph":
            torch.save(features, output_path)
        else:
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            torch.save(features, output_path)

    def _save_json(self, features, output_path):
        with open(output_path, "w") as f:
            json.dump(features, f, indent=2)

    def save_features(self, feature_types: Optional[List[str]] = None) -> None:
        """Save generated features to files.

        Args:
            feature_types: List of feature types to save (None for all)
        """
        if feature_types is None:
            feature_types = list(self._features.keys())

        for feature_type in feature_types:
            if feature_type in self._features:
                self.save_feature(feature_type, self._features[feature_type])

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about loaded features.

        Returns:
            Dictionary containing feature information
        """
        info = {
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path),
            "raw_data_shape": self._raw_data.shape
            if self._raw_data is not None
            else None,
            "loaded_features": list(self._features.keys()),
            "feature_shapes": {},
        }

        for feature_type, features in self._features.items():
            if features is not None:
                if isinstance(features, (list, tuple)):
                    info["feature_shapes"][feature_type] = len(features)
                elif isinstance(features, np.ndarray):
                    info["feature_shapes"][feature_type] = features.shape
                elif isinstance(features, dict):
                    info["feature_shapes"][feature_type] = "dict"
                else:
                    info["feature_shapes"][feature_type] = type(features).__name__

        return info


def preprocess_dataset(
    dataset_name: str,
    feature_types: Optional[List[str]] = None,
    base_data_dir: Optional[str] = None,
    force_regenerate: bool = False,
    save_results: bool = True,
) -> DatasetPreprocessor:
    """Convenience function to preprocess a dataset.

    Args:
        dataset_name: Name of the dataset from DATASET_MAP
        feature_types: List of feature types to generate (None for all)
        base_data_dir: Base directory for data storage (optional)
        force_regenerate: Whether to force regeneration of existing features
        save_results: Whether to save generated features to files

    Returns:
        DatasetPreprocessor instance with generated features
    """
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        force_regenerate=force_regenerate,
    )

    # Load raw data
    preprocessor.load_raw_data()

    # Generate features
    if feature_types is None:
        preprocessor.generate_all_features()
    else:
        preprocessor.generate_features(feature_types)

    # Save results
    if save_results:
        preprocessor.save_features()

    # Print summary
    info = preprocessor.get_feature_info()
    logger.info(f"Preprocessing completed for {dataset_name}")
    logger.info(f"Generated features: {info['loaded_features']}")

    return preprocessor


if __name__ == "__main__":
    # Example usage

    # Example 1: Process a single dataset with all features
    dataset_name = "AV_APML"
    base_data_dir = DATASET_MAP.get(dataset_name, {}).get("path", None)
    try:
        preprocessor = preprocess_dataset(
            dataset_name=dataset_name,
            base_data_dir=base_data_dir,
            force_regenerate=False,
        )
        print("Preprocessing completed successfully!")
        print(f"Feature info: {preprocessor.get_feature_info()}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")

    # # Example 2: Manual step-by-step processing with custom base directory
    # try:
    #     # Initialize preprocessor with custom base directory
    #     preprocessor = DatasetPreprocessor(
    #         dataset_name=dataset_name,
    #         base_data_dir=base_data_dir
    #     )
    #     # Load raw data
    #     raw_data = preprocessor.load_raw_data()
    #     print(f"Loaded {len(raw_data)} samples from custom path")

    #     # Generate specific features
    #     features = preprocessor.generate_features(["fasta", "esm2_150_embedding"])

    #     # Save features
    #     preprocessor.save_features()

    #     print("Manual preprocessing with custom path completed!")

    # except Exception as e:
    #     logger.error(f"Manual preprocessing failed: {e}")
