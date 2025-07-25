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
# DEFAULT_DATA_DIR = os.path.expanduser("~/.pepbenchmark_cache/")S
DEFAULT_POS_DATA_DIR = os.path.expanduser(
    "/home/dataset-assist-0/rouyi/rouyi/Projects/PepBenchmark/data_share/peptide_dataset/official_dataset_50_v1"
)
POS_DATA_DIR = os.environ.get("PEPBENCHMARK_DATA_DIR", DEFAULT_POS_DATA_DIR)


DATASET_MAP = {
    "bbp": {
        "path": os.path.join(POS_DATA_DIR, "ADME/bbp"),
        "property_name": "Blood brain barrier penetrating",
        "type": "binary_classification",
    },
    "cpp": {
        "path": os.path.join(POS_DATA_DIR, "ADME/cpp"),
        "property_name": "Cell penetrating",
        "type": "binary_classification",
    },
    "nonfouling": {
        "path": os.path.join(POS_DATA_DIR, "ADME/nonfouling"),
        "type": "binary_classification",
    },
    "solubility": {
        "path": os.path.join(POS_DATA_DIR, "ADME/solubility"),
        "type": "binary_classification",
    },
    "antibacterial": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-AMP/antibacterial"),
        "property_name": "Antibacterial",
        "type": "binary_classification",
    },
    "antifungal": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-AMP/antifungal"),
        "property_name": "Anti fungal",
        "type": "binary_classification",
    },
    "antiviral": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-AMP/antiviral"),
        "property_name": "Antiviral",
        "type": "binary_classification",
    },
    "antimicrobial": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-AMP/antimicrobial"),
        "property_name": "Antimicrobial",
        "type": "binary_classification",
    },
    "antiparasitic": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-AMP/antiparasitic"),
        "property_name": "Antiparasitic",
        "type": "binary_classification",
    },
    "ace_inhibitory": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/ace_inhibitory"),
        "property_name": "Angiotensin-converting enzyme (ace) inhibitors",
        "type": "binary_classification",
    },
    "anticancer": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/anticancer"),
        "property_name": "Anticancer",
        "type": "binary_classification",
    },
    "antidiabetic": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/antidiabetic"),
        "property_name": "Anti diabetic",
        "type": "binary_classification",
    },
    "antioxidant": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/antioxidant"),
        "property_name": "Anti oxidative",
        "type": "binary_classification",
    },
    "neuropeptide": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/neuropeptide"),
        "property_name": "Neuropeptide",
        "type": "binary_classification",
    },
    "quorum_sensing": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/quorum_sensing"),
        "property_name": "Quorum sensing",
        "type": "binary_classification",
    },
    "ttca": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/ttca"),
        "type": "binary_classification",
    },
    "hemolytic": {
        "path": os.path.join(POS_DATA_DIR, "Tox/hemolytic"),
        "property_name": "Hemolytic",
        "type": "binary_classification",
    },
    "toxicity": {
        "path": os.path.join(POS_DATA_DIR, "Tox/toxicity"),
        "property_name": ["Cytotoxic", "Neurotoxin"],
        "type": "binary_classification",
    },
    "allergen": {
        "path": os.path.join(POS_DATA_DIR, "Tox/allergen"),
        "property_name": "Allergen",
        "type": "binary_classification",
    },
    "antiinflamatory": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/antiinflamatory"),
        "property_name": "Anti inflamatory",
        "type": "binary_classification",
    },
    "antiaging": {
        "path": os.path.join(POS_DATA_DIR, "Theraputic-Other/antiaging"),
        "property_name": "Anti aging",
        "type": "binary_classification",
    },
    "anti_mammalian_cell": {
        "path": os.path.join(POS_DATA_DIR, "Tox/anti_mammalian_cell"),
        "property_name": "Anti mammalian cell",
        "type": "binary_classification",
    },
}
