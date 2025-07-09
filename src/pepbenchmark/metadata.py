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

"""
Peptide Dataset Metadata and Registry.
This module contains metadata definitions and configuration for all peptide
datasets included in the PepBenchmark suite. It serves as a central registry
for dataset information, including dataset keys, categories, data paths, and
validation utilities.
The metadata system enables:
- Consistent dataset naming and organization
- Automated dataset discovery and loading
- Validation of dataset configurations
- Easy extension with new datasets

Dataset Categories:
- Natural Binary: Binary classification tasks with natural amino acids
- Synthetic Binary: Binary classification with synthetic/modified peptides
- Natural Regression: Regression tasks with natural amino acids
- Synthetic Regression: Regression with synthetic/modified peptides

Each dataset is identified by a unique key and contains information about:
- Data source and preprocessing details
- Task type (classification/regression)
- Peptide type (natural/synthetic)
- Performance baselines and evaluation metrics

Example:
    >>> from pepbenchmark.metadata import natural_binary_keys, get_all_datasets
    >>>
    >>> # List available natural binary classification datasets
    >>> print(natural_binary_keys)
    ['BBP_APML', 'cCPP_Pepland', 'Solubility', ...]
    >>>
    >>> # Get all dataset metadata
    >>> all_datasets = get_all_datasets()
    >>> print(f"Total datasets: {len(all_datasets)}")
"""

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for processed peptide datasets
# DEFAULT_DATA_DIR = os.path.expanduser("~/.pepbenchmark_cache/")
DEFAULT_DATA_DIR = os.path.expanduser("~/assist/git_review/PepBenchmark/data_share/")
DATA_DIR = os.environ.get("PEPBENCHMARK_DATA_DIR", DEFAULT_DATA_DIR)

"""
- path: 数据集默认存储的路径;
- type: 任务类型（分类 回归）
- num_class: 类别数量（仅对分类任务适用）；对于回归任务是1；
- size: 数据集大小（样本数量）
- size_range: 氨基酸数量的范围
- max_len: 最大氨基酸数量
- nature: 三个值分别是True（自然氨基酸）、False（非自然氨基酸）和"mixed"（同时包含天然和非天然氨基酸）
- description: 数据集描述
- group: 数据集所属组
- format: 数据集文件格式
"""

DATASET_MAP = {
    "BBP_APML": {
        "path": os.path.join(DATA_DIR, "ADME/BBP_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 238,
        "size_range": "5–53",
        "max_len": 53,
        "nature": True,
        "description": "Blood–brain barrier penetrating peptides (APML dataset)",
        "group": "ADME",
        "format": "FASTA",
    },
    "cCPP_Pepland": {
        "path": os.path.join(DATA_DIR, "ADME/cCPP_Pepland"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 2324,
        "size_range": "2–50",
        "max_len": 50,
        "nature": True,
        "description": "Cell penetrating peptides (Pepland dataset)",
        "group": "ADME",
        "format": "FASTA",
    },
    "ncCPP_CycPeptMPDB-PAMA": {
        "path": os.path.join(DATA_DIR, "ADME/ncCPP_CycPeptMPDB-PAMA"),
        "type": "regression",
        "num_class": 1,
        "size": 7239,
        "size_range": "2–15",
        "max_len": 15,
        "nature": False,
        "description": "Non-natural cell penetrating peptides (CycPeptMPDB-PAMA dataset)",
        "group": "ADME",
        "format": "HELM",
    },
    "Nonfouling": {
        "path": os.path.join(DATA_DIR, "ADME/Nonfouling"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 17178,
        "size_range": "5–198",
        "max_len": 198,
        "nature": True,
        "description": "Nonfouling peptides",
        "group": "ADME",
        "format": "FASTA",
    },
    "Solubility": {
        "path": os.path.join(DATA_DIR, "ADME/Solubility"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 18453,
        "size_range": "19–198",
        "max_len": 198,
        "nature": True,
        "description": "Soluble vs insoluble peptides",
        "group": "ADME",
        "format": "FASTA",
    },
    "AF_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AF_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 1616,
        "size_range": "5–100",
        "max_len": 100,
        "nature": True,
        "description": "Antifungal peptides (APML dataset)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "all-AMP": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AMP-MIC/all-AMP"),
        "type": "regression",
        "num_class": 1,
        "size": 6760,
        "size_range": "1–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC regression, all species)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "E.coli": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AMP-MIC/E.coli"),
        "type": "regression",
        "num_class": 1,
        "size": 5102,
        "size_range": "1–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC regression, E. coli)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "P.aeruginosa": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AMP-MIC/P.aeruginosa"),
        "type": "regression",
        "num_class": 1,
        "size": 2852,
        "size_range": "1–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC regression, P. aeruginosa)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "S.aureus": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AMP-MIC/S.aureus"),
        "type": "regression",
        "num_class": 1,
        "size": 4582,
        "size_range": "1–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC regression, S. aureus)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "AMP_PepDiffusion": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AMP_PepDiffusion"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 44614,
        "size_range": "2–50",
        "max_len": 50,
        "nature": True,
        "description": "Antimicrobial peptides (PepDiffusion dataset)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "AP_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AP_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 497,
        "size_range": "5–141",
        "max_len": 141,
        "nature": True,
        "description": "Antiparasite peptides (APML dataset)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "AV_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/AV_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 4796,
        "size_range": "5–100",
        "max_len": 100,
        "nature": True,
        "description": "Antiviral peptides (APML dataset)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "cAB_APML2": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/cAB_APML2"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 9774,
        "size_range": "5–50",
        "max_len": 50,
        "nature": True,
        "description": "Antibacterial peptides (canonical, APML2 dataset)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "ncAB_APML2": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/ncAB_APML2"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 1726,
        "size_range": "1–49",
        "max_len": 49,
        "nature": False,
        "description": "Antibacterial peptides (non-canonical, APML2 dataset)",
        "group": "Therapeutic-AMP",
        "format": "BILN",
    },
    "ncAV_APML2": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/ncAV_APML2"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 381,
        "size_range": "3–56",
        "max_len": 56,
        "nature": False,
        "description": "Antiviral peptides (non-canonical, APML2 dataset)",
        "group": "Therapeutic-AMP",
        "format": "BILN",
    },
    "ACE_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/ACE_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 1982,
        "size_range": "5–84",
        "max_len": 84,
        "nature": True,
        "description": "ACE-inhibitory peptides (APML dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "ACP_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/ACP_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 1408,
        "size_range": "2–50",
        "max_len": 50,
        "nature": True,
        "description": "Anticancer peptides (APML dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "Aox_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/Aox_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 822,
        "size_range": "2–28",
        "max_len": 28,
        "nature": True,
        "description": "Antioxidative peptides (APML dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "DLAD_BioDADPep": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/DLAD_BioDADPep"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 834,
        "size_range": "1–20",
        "max_len": 20,
        "nature": True,
        "description": "Antidiabetic peptides (BioDADPep dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "DPPIV_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/DPPIV_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 1232,
        "size_range": "2–33",
        "max_len": 33,
        "nature": True,
        "description": "Dipeptidyl peptidase inhibitor peptides (APML dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "Neuro_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/Neuro_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 4778,
        "size_range": "5–99",
        "max_len": 99,
        "nature": True,
        "description": "Neuropeptides (APML dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "QS_APML": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/QS_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 436,
        "size_range": "5–49",
        "max_len": 49,
        "nature": True,
        "description": "Quorum sensing peptides (APML dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "TTCA_TCAHybrid": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/TTCA_TCAHybrid"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 955,
        "size_range": "8–37",
        "max_len": 37,
        "nature": True,
        "description": "Tumor T cell antigens (TCAHybrid dataset)",
        "group": "Therapeutic-Other",
        "format": "FASTA",
    },
    "HemoPI2": {
        "path": os.path.join(DATA_DIR, "Tox/HemoPI2"),
        "type": "regression",
        "num_class": 1,
        "size": 1926,
        "size_range": "6–39",
        "max_len": 39,
        "nature": True,
        "description": "Hemolytic peptides with regression labels (HemoPI2 dataset)",
        "group": "Tox",
        "format": "FASTA",
    },
    "Hemo_PeptideBERT": {
        "path": os.path.join(DATA_DIR, "Tox/Hemo_PeptideBERT"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 5611,
        "size_range": "1–190",
        "max_len": 190,
        "nature": True,
        "description": "Hemolytic peptides (PeptideBERT dataset)",
        "group": "Tox",
        "format": "FASTA",
    },
    "Tox_APML": {
        "path": os.path.join(DATA_DIR, "Tox/Tox_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 3087,
        "size_range": "2–50",
        "max_len": 50,
        "nature": True,
        "description": "Toxic peptides (APML dataset)",
        "group": "Tox",
        "format": "FASTA",
    },
    "multitask_peptidepedia": {
        "path": os.path.join(DATA_DIR, "multitask_peptidepedia"),
        "type": "binary_classification",
        "num_class": 1,
        "nature": True,
        "description": "Activity peptides from PeptidePedia",
        "format": "FASTA",
    },
}


def _generate_dataset_keys():
    """
    从DATASET_MAP中动态生成各种数据集分类列表
    """
    natural_binary = []
    natural_multiclass = []
    natural_regression = []
    non_natural_binary = []
    non_natural_multiclass = []
    non_natural_regression = []

    for dataset_name, info in DATASET_MAP.items():
        is_natural = info["nature"]
        task_type = info["type"]

        if is_natural:
            if task_type == "binary_classification":
                natural_binary.append(dataset_name)
            elif task_type == "multiclass_classification":
                natural_multiclass.append(dataset_name)
            elif task_type == "regression":
                natural_regression.append(dataset_name)
        else:  # non-natural
            if task_type == "binary_classification":
                non_natural_binary.append(dataset_name)
            elif task_type == "multiclass_classification":
                non_natural_multiclass.append(dataset_name)
            elif task_type == "regression":
                non_natural_regression.append(dataset_name)

    return {
        "natural_binary": natural_binary,
        "natural_multiclass": natural_multiclass,
        "natural_regression": natural_regression,
        "non_natural_binary": non_natural_binary,
        "non_natural_multiclass": non_natural_multiclass,
        "non_natural_regression": non_natural_regression,
    }


_dataset_keys = _generate_dataset_keys()
natural_binary_keys = _dataset_keys["natural_binary"]
natural_multiclass_keys = _dataset_keys["natural_multiclass"]
natural_regression_keys = _dataset_keys["natural_regression"]
non_natural_binary_keys = _dataset_keys["non_natural_binary"]
non_natural_multiclass_keys = _dataset_keys["non_natural_multiclass"]
non_natural_regression_keys = _dataset_keys["non_natural_regression"]


def get_all_datasets():
    """
    Get metadata for all available datasets.

    Returns:
        dict: Complete DATASET_MAP containing all dataset metadata

    Examples:
        >>> datasets = get_all_datasets()
        >>> print(f"Total datasets: {len(datasets)}")
        >>> print("Available datasets:", list(datasets.keys()))
    """
    return DATASET_MAP.copy()


def get_datasets_by_category():
    """
    Get datasets organized by category and task type.

    Returns:
        dict: Nested dictionary with structure:
            {
                'natural': {
                    'binary': [...],
                    'multiclass': [...],
                    'regression': [...]
                },
                'synthetic': {
                    'binary': [...],
                    'multiclass': [...],
                    'regression': [...]
                }
            }

    Examples:
        >>> categories = get_datasets_by_category()
        >>> print("Natural binary datasets:", categories['natural']['binary'])
        >>> print("Synthetic regression datasets:", categories['synthetic']['regression'])
    """
    dataset_keys = _generate_dataset_keys()
    return {
        True: {
            "binary": dataset_keys["natural_binary"].copy(),
            "multiclass": dataset_keys["natural_multiclass"].copy(),
            "regression": dataset_keys["natural_regression"].copy(),
        },
        "synthetic": {
            "binary": dataset_keys["non_natural_binary"].copy(),
            "multiclass": dataset_keys["non_natural_multiclass"].copy(),
            "regression": dataset_keys["non_natural_regression"].copy(),
        },
    }


def get_dataset_info(dataset_name):
    """
    Get detailed information about a specific dataset.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        dict: Dataset metadata including path, type, size, etc.

    Raises:
        ValueError: If dataset_name is not found

    Examples:
        >>> info = get_dataset_info("BBP_APML")
        >>> print(f"Dataset type: {info['type']}")
        >>> print(f"Dataset size: {info['size']}")
        >>> print(f"Description: {info['description']}")
    """
    if dataset_name not in DATASET_MAP:
        available = list(DATASET_MAP.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. " f"Available datasets: {available}"
        )
    return DATASET_MAP[dataset_name].copy()


def validate_dataset_files(dataset_name):
    """
    Validate that all expected files exist for a dataset.

    Args:
        dataset_name (str): Name of the dataset to validate

    Returns:
        dict: Validation results with missing files and status

    Examples:
        >>> results = validate_dataset_files("BBP_APML")
        >>> if results['valid']:
        ...     print("All files present")
        >>> else:
        ...     print("Missing files:", results['missing'])
    """
    if dataset_name not in DATASET_MAP:
        return {"valid": False, "error": f"Dataset {dataset_name} not found"}

    base_path = DATASET_MAP[dataset_name]["path"]
    missing_files = []

    # Check combine.csv
    combine_path = os.path.join(base_path, "combine.csv")
    if not os.path.exists(combine_path):
        missing_files.append("combine.csv")

    # Check common split directories
    for split_type in ["Random_split", "Homology_based_split"]:
        split_dir = os.path.join(base_path, split_type)
        if os.path.exists(split_dir):
            # Check for random1 subdirectory (most common)
            random_dir = os.path.join(split_dir, "random1")
            if os.path.exists(random_dir):
                for subset in ["train.csv", "test.csv", "valid.csv"]:
                    subset_path = os.path.join(random_dir, subset)
                    if not os.path.exists(subset_path):
                        missing_files.append(f"{split_type}/random1/{subset}")

    return {
        "valid": len(missing_files) == 0,
        "missing": missing_files,
        "base_path": base_path,
    }


if __name__ == "__main__":
    print("Available datasets:")
    for category, tasks in DATASET_MAP.items():
        print(f"  {category}:")
        for task, datasets in tasks.items():
            print(f"    {task}: {datasets}")
    print()
