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

DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../data_share/peptide_dataset/processed_2025.6.12v/",
    )
)
print(DATA_DIR)


natural_binary_keys = [
    "BBP_APML",
    "cCPP_Pepland",
    "Nonfouling",
    "Solubility",
    "AF_APML",
    "AMP_PepDiffusion",
    "AP_APML",
    "AV_APML",
    "cAB_APML2",
    "ACE_APML",
    "ACP_APML",
    "Aox_APML",
    "DLAD_BioDADPep",
    "DPPIV_APML",
    "Neuro_APML",
    "QS_APML",
    "TTCA_TCAHybrid",
    "Hemo_PeptideBERT",
    "Tox_APML",
]

natural_multiclass_keys = ["cMultitask_Peptidepedia"]


natural_regression_keys = ["all-AMP", "E.coli", "P.aeruginosa", "S.aureus", "HemoPI2"]

non_natural_binary_keys = ["ncAB_APML2", "ncAV_APML2"]

non_natural_multiclass_keys = []

non_natural_regression_keys = ["ncCPP_CycPeptMPDB-PAMA"]


DATASET_MAP = {
    "BBP_APML": {
        "path": os.path.join(DATA_DIR, "ADME/BBP_APML"),
        "type": "binary_classification",
        "num_class": 2,
        "size": 238,
        "size_range": "5–53",
        "max_len": 53,
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "non-natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "non-natural",
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
        "nature": "non-natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
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
        "nature": "natural",
        "description": "Toxic peptides (APML dataset)",
        "group": "Tox",
        "format": "FASTA",
    },
}


def get_dataset_path(dataset_name, split=None, fold_seed=1, type="train"):
    """
    Retrieve the path for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.
        split (str, optional): The type of split to use ('Random_split', 'Homology_based_split'). If None, returns the path to 'combine.csv'.
        fold_seed (int, optional): The seed for the random split. Ignored if `split` is None.
        type (str, optional): The specific subset of the split to use ('train', 'test', or 'valid'). Ignored if `split` is None.

    Returns:
        str: The path to the dataset if it exists, otherwise None.
    """
    print(dataset_name, "*" * 200)
    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. Please choose from {list(DATASET_MAP.keys())}."
        )

    base_dir = DATASET_MAP.get(dataset_name)["path"]

    if split is None:
        path = os.path.join(base_dir, "combine.csv")
    else:
        split_path = os.path.join(base_dir, split, "random" + str(fold_seed))
        if type not in ["train", "test", "valid"]:
            raise ValueError("Type must be one of 'train', 'test', or 'valid'.")
        elif type == "train":
            path = os.path.join(split_path, "train.csv")
        elif type == "test":
            path = os.path.join(split_path, "test.csv")
        elif type == "valid":
            path = os.path.join(split_path, "valid.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    return path
