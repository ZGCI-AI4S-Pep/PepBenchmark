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

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for processed peptide datasets
# DEFAULT_DATA_DIR = os.path.expanduser("~/.pepbenchmark_cache/")

# Six versions of official datasets
DEFAULT_OFFICIAL_NATURE_DATASET_V1 = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_nature_dataset_v1"
)
DEFAULT_OFFICIAL_NATURE_DATASET_50_V1 = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_nature_dataset_50_v1"
)
DEFAULT_OFFICIAL_NONNATURE_DATASET_50_V1 = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_nonnature_dataset_50_v1"
)
DEFAULT_OFFICIAL_NONNATURE_DATASET_V1 = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_nonnature_dataset_v1"
)
DEFAULT_OFFICIAL_MIX_DATASET_V1 = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_mix_dataset_v1"
)
DEFAULT_OFFICIAL_MIX_DATASET_50_V1 = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_mix_dataset_50_v1"
)

# Official dataset mapping
OFFICIAL_DATASET_MAP = {
    "nature": DEFAULT_OFFICIAL_NATURE_DATASET_V1,
    "nature_50": DEFAULT_OFFICIAL_NATURE_DATASET_50_V1,
    "nonnature_50": DEFAULT_OFFICIAL_NONNATURE_DATASET_50_V1,
    "nonnature": DEFAULT_OFFICIAL_NONNATURE_DATASET_V1,
    "mix": DEFAULT_OFFICIAL_MIX_DATASET_V1,
    "mix_50": DEFAULT_OFFICIAL_MIX_DATASET_50_V1,
}

# Default to the first official dataset
DEFAULT_OFFICIAL_DATASET = "nature"
DATA_DIR = OFFICIAL_DATASET_MAP.get(
    DEFAULT_OFFICIAL_DATASET, DEFAULT_OFFICIAL_NATURE_DATASET_V1
)


def set_official_data_version(dataset_name: str) -> str:
    """
    Set the official dataset to use for data paths.

    Args:
        dataset_name (str): Name of the official dataset to use

    Returns:
        str: The path of the selected dataset

    Raises:
        ValueError: If the dataset name is not found in OFFICIAL_DATASET_MAP
    """
    global DATA_DIR

    if dataset_name not in OFFICIAL_DATASET_MAP:
        available = list(OFFICIAL_DATASET_MAP.keys())
        raise ValueError(
            f"Unknown official dataset name: {dataset_name}. "
            f"Available official datasets: {available}"
        )

    DATA_DIR = OFFICIAL_DATASET_MAP[dataset_name]
    logger.info(f"Set official dataset to: {dataset_name} -> {DATA_DIR}")
    return DATA_DIR


def get_available_official_data_versions() -> list:
    """
    Get list of available official dataset names.

    Returns:
        list: List of available official dataset names
    """
    return list(OFFICIAL_DATASET_MAP.keys())


def get_current_official_data_version() -> str:
    """
    Get the current official dataset name.

    Returns:
        str: Current official dataset name
    """
    for name, path in OFFICIAL_DATASET_MAP.items():
        if path == DATA_DIR:
            return name
    return DEFAULT_OFFICIAL_DATASET


DATASET_MAP = {
    "bbp": {
        "path": os.path.join(DATA_DIR, "ADME/bbp"),
        "property_name": "Blood brain barrier penetrating",
        "type": "binary_classification",
    },
    "cpp": {
        "path": os.path.join(DATA_DIR, "ADME/cpp"),
        "property_name": "Cell penetrating",
        "type": "binary_classification",
    },
    "nonfouling": {
        "path": os.path.join(DATA_DIR, "ADME/nonfouling"),
        "type": "binary_classification",
    },
    "solubility": {
        "path": os.path.join(DATA_DIR, "ADME/solubility"),
        "type": "binary_classification",
    },
    "antibacterial": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antibacterial"),
        "property_name": "Antibacterial",
        "type": "binary_classification",
    },
    "antifungal": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antifungal"),
        "property_name": "Anti fungal",
        "type": "binary_classification",
    },
    "antiviral": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antiviral"),
        "property_name": "Antiviral",
        "type": "binary_classification",
    },
    "antimicrobial": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antimicrobial"),
        "property_name": "Antimicrobial",
        "type": "binary_classification",
    },
    "E.coli_mic": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antimicrobial_mic/E.coli_mic"),
        "type": "regression",
        "size": 5465,
        "size_range": "2–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC, E. coli)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "P.aeruginosa_mic": {
        "path": os.path.join(
            DATA_DIR, "Theraputic-AMP/antimicrobial_mic/P.aeruginosa_mic"
        ),
        "type": "regression",
        "size": 2523,
        "size_range": "2–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC, P. aeruginosa)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "S.aureus_mic": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antimicrobial_mic/S.aureus_mic"),
        "type": "regression",
        "size": 5070,
        "size_range": "2–190",
        "max_len": 190,
        "nature": True,
        "description": "Antimicrobial peptides (MIC, S. aureus)",
        "group": "Therapeutic-AMP",
        "format": "FASTA",
    },
    "antiparasitic": {
        "path": os.path.join(DATA_DIR, "Theraputic-AMP/antiparasitic"),
        "property_name": "Antiparasitic",
        "type": "binary_classification",
    },
    "ace_inhibitory": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/ace_inhibitory"),
        "property_name": "Angiotensin-converting enzyme (ace) inhibitors",
        "type": "binary_classification",
    },
    "anticancer": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/anticancer"),
        "property_name": "Anticancer",
        "type": "binary_classification",
    },
    "antidiabetic": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/antidiabetic"),
        "property_name": "Anti diabetic",
        "type": "binary_classification",
    },
    "antioxidant": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/antioxidant"),
        "property_name": "Anti oxidative",
        "type": "binary_classification",
    },
    "neuropeptide": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/neuropeptide"),
        "property_name": "Neuropeptide",
        "type": "binary_classification",
    },
    "quorum_sensing": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/quorum_sensing"),
        "property_name": "Quorum sensing",
        "type": "binary_classification",
    },
    "ttca": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/ttca"),
        "type": "binary_classification",
    },
    "hemolytic": {
        "path": os.path.join(DATA_DIR, "Tox/hemolytic"),
        "property_name": "Hemolytic",
        "type": "binary_classification",
    },
    "hemolytic_hc50": {
        "path": os.path.join(DATA_DIR, "Tox/hemolytic_hc50"),
        "property_name": "Hemolytic",
        "type": "regression",
    },
    "toxicity": {
        "path": os.path.join(DATA_DIR, "Tox/toxicity"),
        "property_name": ["Cytotoxic", "Neurotoxin"],
        "type": "binary_classification",
    },
    "allergen": {
        "path": os.path.join(DATA_DIR, "Tox/allergen"),
        "property_name": "Allergen",
        "type": "binary_classification",
    },
    "antiinflamatory": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/antiinflamatory"),
        "property_name": "Anti inflamatory",
        "type": "binary_classification",
    },
    "antiaging": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/antiaging"),
        "property_name": "Anti aging",
        "type": "binary_classification",
    },
    "anti_mammalian_cell": {
        "path": os.path.join(DATA_DIR, "Tox/anti_mammalian_cell"),
        "property_name": "Anti mammalian cell",
        "type": "binary_classification",
    },
    "dppiv_inhibitors": {
        "path": os.path.join(DATA_DIR, "Theraputic-Other/dppiv_inhibitors"),
        "type": "binary_classification",
    },
    "PpI": {
        "path": os.path.join(DATA_DIR, "PepPI/PpI"),
        "property_name": "Peptide-Protein Interaction",
        "type": "binary_classification",
        "size": 9500,
        "size_range": "5–25",
        "max_len": 25,
        "nature": True,
        "description": "Peptide-protein interaction prediction (PepPI dataset)",
        "group": "Interaction",
        "format": "CSV",
    },
    "PpI_ba": {
        "path": os.path.join(DATA_DIR, "PepPI/PpI_ba"),
        "property_name": "Peptide-Protein Interaction",
        "type": "regression",
        "size": 1806,
        "size_range": "51–1287",
        "max_len": 1287,
        "nature": True,
        "description": "Peptide-protein interaction prediction with binding affinity (PepPI dataset)",
        "group": "Interaction",
        "format": "CSV",
    },
}
