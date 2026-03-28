import json
import os
import pandas as pd
from collections import defaultdict

from pepbenchmark.metadata import BASE_DIR

NEG_POOL_DIR = BASE_DIR / "PepBenchData_raw"

NEG_POOL_MAP = {
    # ======================== ADME ========================
    "bbp": {
        "pos_path": os.path.join(NEG_POOL_DIR, "ADME/nature/bbp/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "ADME/nature/bbp/neg.csv"),
        "group": "ADME",
        "type": "classification",
        "is_nature": True
    },
    "cpp": {
        "pos_path": os.path.join(NEG_POOL_DIR, "ADME/nature/cpp/pos.csv"),
        "neg_path": None,
        "group": "ADME",
        "type": "classification",
        "is_nature": True
    },
    "nc-cpp": {
        "pos_path": os.path.join(NEG_POOL_DIR, "ADME/non-nature/nc-cpp/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "ADME/non-nature/nc-cpp/neg.csv"),
        "group": "ADME",
        "type": "classification",
        "is_nature": False
    },
    "nc-cpp_pampa": {
        "raw_path": os.path.join(NEG_POOL_DIR, "ADME/non-nature/nc-cpp_pampa/raw.csv"),
        "group": "ADME",
        "type": "regression",
        "is_nature": False
    },

    "antibacterial": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antibacterial/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": True
    },


    "antifungal": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antifungal/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": True
    },


    "antimicrobial": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antimicrobial/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": True
    },


    "antimicrobial_E_coli_mic": {
        "raw_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antimicrobial_E_coli_mic/raw.csv"),
        "group": "AMP",
        "type": "regression",
        "is_nature": True
    },
    "antimicrobial_P_aeruginosa_mic": {
        "raw_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antimicrobial_P_aeruginosa_mic/raw.csv"),
        "group": "AMP",
        "type": "regression",
        "is_nature": True
    },


    "antimicrobial_S_aureus_mic": {
        "raw_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antimicrobial_S_aureus_mic/raw.csv"),
        "group": "AMP",
        "type": "regression",
        "is_nature": True
    },
    "antiparasitic": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antiparasitic/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": True
    },
    "antiviral": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/nature/antiviral/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": True
    },
    "nc-antibacterial": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/non-nature/nc-antibacterial/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": False
    },
    "nc-antifungal": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/non-nature/nc-antifungal/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": False
    },
    "nc-antimicrobial": {
        "pos_path": os.path.join(NEG_POOL_DIR, "AMP/non-nature/nc-antimicrobial/pos.csv"),
        "neg_path": None,
        "group": "AMP",
        "type": "classification",
        "is_nature": False
    },

    # ======================== Metabolic ========================
    "ace_inhibitory": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Metabolic/nature/ace_inhibitory/pos.csv"),
        "neg_path": None,
        "group": "Metabolic",
        "type": "classification",
        "is_nature": True
    },
    "ace_inhibitory_ic50": {
        "raw_path": os.path.join(NEG_POOL_DIR, "Metabolic/nature/ace_inhibitory_ic50/raw.csv"),
        "group": "Metabolic",
        "type": "regression",
        "is_nature": True
    },
    "antidiabetic": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Metabolic/nature/antidiabetic/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "Metabolic/nature/antidiabetic/neg.csv"),
        "group": "Metabolic",
        "type": "classification",
        "is_nature": True
    },
    "dppiv_inhibitors": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Metabolic/nature/dppiv_inhibitors/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "Metabolic/nature/dppiv_inhibitors/neg.csv"),
        "group": "Metabolic",
        "type": "classification",
        "is_nature": True
    },

    # ======================== Oncology ========================
    "anticancer": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Oncology/nature/anticancer/pos.csv"),
        "neg_path": None,
        "group": "Oncology",
        "type": "classification",
        "is_nature": True
    },
    "ttca": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Oncology/nature/ttca/pos.csv"),
        "neg_path": None,
        "group": "Oncology",
        "type": "classification",
        "is_nature": True
    },

    # ======================== Others ========================
    "antiaging": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Others/nature/antiaging/pos.csv"),
        "neg_path": None,
        "group": "Others",
        "type": "classification",
        "is_nature": True
    },

    "antiinflamatory": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Others/nature/antiinflamatory/pos.csv"),
        "neg_path": None,
        "group": "Others",
        "type": "classification",
        "is_nature": True
    },

    "antioxidant": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Others/nature/antioxidant/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "Others/nature/antioxidant/neg.csv"),
        "group": "Others",
        "type": "classification",
        "is_nature": True
    },
    "neuropeptide": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Others/nature/neuropeptide/pos.csv"),
        "neg_path": None,
        "group": "Others",
        "type": "classification",
        "is_nature": True
    },
    "quorum_sensing": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Others/nature/quorum_sensing/pos.csv"),
        "neg_path": None,
        "group": "Others",
        "type": "classification",
        "is_nature": True
    },

    # ======================== PepPI ========================
    "PpI": {
        "pos_path": os.path.join(NEG_POOL_DIR, "PepPI/nature/PpI/pos.csv"),
        "neg_path": None,
        "group": "PepPI",
        "type": "classification",
        "is_nature": True
    },
    "PpI_ba": {
        "raw_path": os.path.join(NEG_POOL_DIR, "PepPI/nature/PpI_ba/raw.csv"),
        "group": "PepPI",
        "type": "regression",
        "is_nature": True
    },
    "PpI_ba_X": {
        "raw_path": os.path.join(NEG_POOL_DIR, "PepPI/nature/PpI_ba_X/raw.csv"),
        "group": "PepPI",
        "type": "regression",
        "is_nature": True
    },
    "PpI_X": {
        "pos_path": os.path.join(NEG_POOL_DIR, "PepPI/nature/PpI_X/raw.csv"),
        "neg_path": None,
        "group": "PepPI",
        "type": "regression",
        "is_nature": True
    },
    "nc-PpI_ba": {
        "raw_path": os.path.join(NEG_POOL_DIR, "PepPI/non-nature/nc-PpI_ba/raw.csv"),
        "group": "PepPI",
        "type": "regression",
        "is_nature": False
    },

    # ======================== PhysChem ========================
    "nonfouling": {
        "pos_path": os.path.join(NEG_POOL_DIR, "PhysChem/nature/nonfouling/pos.csv"),
        "neg_path": None,
        "group": "PhysChem",
        "type": "classification",
        "is_nature": True
    },
    "solubility": {
        "pos_path": os.path.join(NEG_POOL_DIR, "PhysChem/nature/solubility/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "PhysChem/nature/solubility/neg.csv"),
        "group": "PhysChem",
        "type": "classification",
        "is_nature": True
    },

    # ======================== Tox ========================
    "allergen": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Tox/nature/allergen/pos.csv"),
        "neg_path": None,
        "group": "Tox",
        "type": "classification",
        "is_nature": True
    },
    "hemolytic": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Tox/nature/hemolytic/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "Tox/nature/hemolytic/neg.csv"),
        "group": "Tox",
        "type": "classification",
        "is_nature": True
    },
    "hemolytic_hc50": {
        "raw_path": os.path.join(NEG_POOL_DIR, "Tox/nature/hemolytic_hc50/raw.csv"),
        "group": "Tox",
        "type": "regression",
        "is_nature": True
    },
    "hemolytic_strict": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Tox/nature/hemolytic_strict/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "Tox/nature/hemolytic_strict/neg.csv"),
        "group": "Tox",
        "type": "classification",
        "is_nature": True
    },
    "neurotoxin": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Tox/nature/neurotoxin/pos.csv"),
        "neg_path": None,
        "group": "Tox",
        "type": "classification",
        "is_nature": True
    },
    "toxicity": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Tox/nature/toxicity/pos.csv"),
        "neg_path": None,
        "group": "Tox",
        "type": "classification",
        "is_nature": True
    },
    "nc-hemolytic": {
        "pos_path": os.path.join(NEG_POOL_DIR, "Tox/non-nature/nc-hemolytic/pos.csv"),
        "neg_path": os.path.join(NEG_POOL_DIR, "Tox/non-nature/nc-hemolytic/neg.csv"),
        "group": "Tox",
        "type": "classification",
        "is_nature": False
    }
}


OTHER_PATH = os.path.join(NEG_POOL_DIR, "others.csv"),
ALL_POOLING_DATASET = list(NEG_POOL_MAP.keys())
ALL_POOLING_DATASET.remove("nc-PpI_ba")
NATURE_DATASET = [dataset for dataset in ALL_POOLING_DATASET if NEG_POOL_MAP[dataset]["is_nature"]]
NATURE_CLS_DATASET = [dataset for dataset in NATURE_DATASET if NEG_POOL_MAP[dataset]["type"] == "classification"]


NATURE_DATASET_WITHOUT_NEG = [dataset for dataset in NATURE_CLS_DATASET if NEG_POOL_MAP[dataset]["neg_path"] is None]




NATURE_CLS_DATASET_SINGLE = [dataset for dataset in NATURE_DATASET if NEG_POOL_MAP[dataset]["type"] == "classification" and  NEG_POOL_MAP[dataset]["group"] != "PepPI"]
NATURE_REG_DATASET_SINGLE = [dataset for dataset in NATURE_DATASET if NEG_POOL_MAP[dataset]["type"] == "regression" and  NEG_POOL_MAP[dataset]["group"] != "PepPI"]



NATURE_CLS_DATASET_PPI = [dataset for dataset in NATURE_DATASET if NEG_POOL_MAP[dataset]["type"] == "classification" and  NEG_POOL_MAP[dataset]["group"] == "PepPI"]
NON_NATURE_DATASET = [dataset for dataset in ALL_POOLING_DATASET if not NEG_POOL_MAP[dataset]["is_nature"]]



def read_dataset_sequences(dataset_name,flag="all",as_dataframe=False):
    """
    Read sequences from a dataset based on its configuration.
    
    Args:
        dataset_name: Name of the dataset
        flag: "all" to read all sequences, "pos" to read only positive sequences,"neg" to read only negative sequences (for classification datasets)

    Returns:
        List of sequences
    """
    dataset_info = NEG_POOL_MAP.get(dataset_name, {})

    
    # Determine which column to read based on dataset properties
    if dataset_info["group"] == "PepPI":
        seq_col = "pep_seq" if dataset_info["is_nature"] else "pep_smi"
    else:
        seq_col = "sequence" if dataset_info["is_nature"] else "HELM"
    sequences = []
    if dataset_info["type"] == "classification":
        # Read positive samples
        if "pos_path" in dataset_info and dataset_info["pos_path"] and (flag == "all" or flag == "pos"):
            pos_df = pd.read_csv(dataset_info["pos_path"])
            sequences.extend(pos_df[seq_col].tolist())
        
        # Read negative samples if they exist
        if "neg_path" in dataset_info and dataset_info["neg_path"] and (flag == "all" or flag == "neg"):
            neg_df = pd.read_csv(dataset_info["neg_path"])
            sequences.extend(neg_df[seq_col].tolist())
    
    elif dataset_info["type"] == "regression":
        # Read raw data
        if "raw_path" in dataset_info and dataset_info["raw_path"]:
            raw_df = pd.read_csv(dataset_info["raw_path"])
            sequences.extend(raw_df[seq_col].tolist())
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_info['type']}")
    if len(sequences) == 0:
        return None
    if as_dataframe:
        return pd.DataFrame({"sequence": sequences})
    return sequences



def filter_X(sequences):
    """
    Filter out sequences containing 'X'.
    Args:
        sequences: List of sequences to filter
    Returns:
        Filtered list of sequences
    """
    return [seq for seq in sequences if "X" not in str(seq)]


# Generate hierarchical statistics
def get_hierarchical_stats():
    """Generate hierarchical statistics for datasets."""
    stats = defaultdict(int)
    
    for dataset_name, dataset_info in NEG_POOL_MAP.items():
        group = dataset_info["group"]
        is_nature = "nature" if dataset_info["is_nature"] else "non-nature"
        task_type = dataset_info["type"]
        
        # Count at different levels
        stats[group] += 1
        stats[f"{group}.{is_nature}"] += 1
        stats[f"{group}.{is_nature}.{task_type}"] += 1
    
    return dict(stats)

# Generate dataset quantity statistics
def get_dataset_quantity_stats(min_length=None, max_length=None):
    """Generate quantity statistics for each dataset."""
    quantity_stats = {}
    
    # Initialize aggregate statistics
    all_nature_stats = {"pos": 0, "neg": 0, "total": 0, "pos_filter": 0, "neg_filter": 0, "total_filter": 0, "reg": 0, "reg_filter": 0}
    all_non_nature_stats = {"pos": 0, "neg": 0, "total": 0, "pos_filter": 0, "neg_filter": 0, "total_filter": 0, "reg": 0, "reg_filter": 0}
    
    all_nature_sequences = []
    all_non_nature_sequences = []
    
    for dataset_name, dataset_info in NEG_POOL_MAP.items():
        if dataset_name == "nc-PpI_ba":
            print("Skipping nc-PpI_ba dataset (smiles)")
            continue
        stats = {"pos": 0, "neg": 0, "total": 0, "pos_filter": 0, "neg_filter": 0, "total_filter": 0, "reg": 0, "reg_filter": 0}
        
        if dataset_info["type"] == "classification":
            pos_sequences = []
            neg_sequences = []
            
            # Count positive samples
            if "pos_path" in dataset_info and dataset_info["pos_path"]:
                pos_df = pd.read_csv(dataset_info["pos_path"])
                # Determine sequence column
                if dataset_info["group"] == "PpI":
                    seq_col = "pep_seq" if dataset_info["is_nature"] else "pep_helm"
                else:
                    seq_col = "sequence" if dataset_info["is_nature"] else "HELM"
                
                pos_sequences = pos_df[seq_col].tolist()
                
                # Apply length filter for natural datasets
                if dataset_info["is_nature"] and (min_length is not None or max_length is not None):
                    filtered_pos = []
                    for seq in pos_sequences:
                        seq_len = len(str(seq))
                        if (min_length is None or seq_len >= min_length) and \
                            (max_length is None or seq_len <= max_length):
                            filtered_pos.append(seq)
                    pos_sequences = filtered_pos
                
                stats["pos"] = len(pos_sequences)
                stats["pos_filter"] = len(set(pos_sequences))
            
            # Count negative samples
            if "neg_path" in dataset_info and dataset_info["neg_path"]:
                neg_df = pd.read_csv(dataset_info["neg_path"])
                # Determine sequence column
                if dataset_info["group"] == "PpI":
                    seq_col = "pep_seq" if dataset_info["is_nature"] else "pep_helm"
                else:
                    seq_col = "sequence" if dataset_info["is_nature"] else "HELM"
                
                neg_sequences = neg_df[seq_col].tolist()
                
                # Apply length filter for natural datasets
                if dataset_info["is_nature"] and (min_length is not None or max_length is not None):
                    filtered_neg = []
                    for seq in neg_sequences:
                        seq_len = len(str(seq))
                        if (min_length is None or seq_len >= min_length) and \
                            (max_length is None or seq_len <= max_length):
                            filtered_neg.append(seq)
                    neg_sequences = filtered_neg
                
                stats["neg"] = len(neg_sequences)
                stats["neg_filter"] = len(set(neg_sequences))
            
            all_sequences = pos_sequences + neg_sequences
            stats["total"] = len(all_sequences)
            stats["total_filter"] = len(set(all_sequences))
            for seq in all_sequences:
                if "X" in str(seq):
                    # print("X", dataset_name, seq)
                    all_sequences.remove(seq)
                if "U" in str(seq):
                    print("U", dataset_name, seq)
                    all_sequences.remove(seq)
                # if "O" in str(seq):
                #     print("O", dataset_name, seq)
                #     # all_sequences.remove(seq)
            
            # Add to aggregate statistics
            if dataset_info["is_nature"]:
                all_nature_stats["pos"] += stats["pos"]
                all_nature_stats["neg"] += stats["neg"]
                all_nature_stats["total"] += stats["total"]
                all_nature_sequences.extend(all_sequences)
            else:
                all_non_nature_stats["pos"] += stats["pos"]
                all_non_nature_stats["neg"] += stats["neg"]
                all_non_nature_stats["total"] += stats["total"]
                all_non_nature_sequences.extend(all_sequences)
        
        elif dataset_info["type"] == "regression":
            # Count total samples in raw data
            if "raw_path" in dataset_info and dataset_info["raw_path"]:
                raw_df = pd.read_csv(dataset_info["raw_path"])
                # Determine sequence column
                if dataset_info["group"] == "PpI":
                    seq_col = "pep_seq" if dataset_info["is_nature"] else "pep_helm"
                else:
                    seq_col = "sequence" if dataset_info["is_nature"] else "HELM"
                
                all_sequences = raw_df[seq_col].tolist()
                
                # Apply length filter for natural datasets
                if dataset_info["is_nature"] and (min_length is not None or max_length is not None):
                    filtered_sequences = []
                    for seq in all_sequences:
                        seq_len = len(str(seq))
                        if (min_length is None or seq_len >= min_length) and \
                            (max_length is None or seq_len <= max_length):
                            filtered_sequences.append(seq)
                    all_sequences = filtered_sequences
                
                stats["reg"] = len(all_sequences)
                stats["reg_filter"] = len(set(all_sequences))
                stats["total"] = len(all_sequences)
                stats["total_filter"] = len(set(all_sequences))
                
                # Add to aggregate statistics
                if dataset_info["is_nature"]:
                    all_nature_stats["reg"] += stats["reg"]
                    all_nature_stats["total"] += stats["total"]
                    all_nature_sequences.extend(all_sequences)
                else:
                    all_non_nature_stats["reg"] += stats["reg"]
                    all_non_nature_stats["total"] += stats["total"]
                    all_non_nature_sequences.extend(all_sequences)
        
        quantity_stats[dataset_name] = stats
    
    # Calculate filtered totals for aggregates
    all_nature_stats["pos_filter"] = all_nature_stats["pos_filter"]  # Already calculated above
    all_nature_stats["neg_filter"] = all_nature_stats["neg_filter"]  # Already calculated above
    all_nature_stats["reg_filter"] = all_nature_stats["reg_filter"]  # Already calculated above
    all_nature_stats["total_filter"] = len(set(all_nature_sequences))
    
    all_non_nature_stats["pos_filter"] = all_non_nature_stats["pos_filter"]  # Already calculated above
    all_non_nature_stats["neg_filter"] = all_non_nature_stats["neg_filter"]  # Already calculated above
    all_non_nature_stats["reg_filter"] = all_non_nature_stats["reg_filter"]  # Already calculated above
    all_non_nature_stats["total_filter"] = len(set(all_non_nature_sequences))
    
    # Add aggregate statistics to results
    quantity_stats["all_nature"] = all_nature_stats
    quantity_stats["all_non_nature"] = all_non_nature_stats
    
    return quantity_stats




EXCLUSIVE_MAP = {
  "bbp": [
    "antibacterial",
    "antifungal",
    "antimicrobial",
    "bbp",
    "cpp",
    "neuropeptide"
  ],
  "cpp": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "antibacterial": [
    "antibacterial",
    "anticancer",
    "antidiabetic",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antioxidant",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "dppiv_inhibitors",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict",
    "neurotoxin",
    "toxicity"
  ],
  "antifungal": [
    "antibacterial",
    "anticancer",
    "antidiabetic",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antioxidant",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "dppiv_inhibitors",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict",
    "neurotoxin",
    "toxicity"
  ],
  "antimicrobial": [
    "ace_inhibitory",
    "ace_inhibitory_ic50",
    "antibacterial",
    "anticancer",
    "antidiabetic",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antioxidant",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "dppiv_inhibitors",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict",
    "neuropeptide",
    "neurotoxin",
    "toxicity"
  ],
  "antimicrobial_E_coli_mic": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "antimicrobial_P_aeruginosa_mic": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "antimicrobial_S_aureus_mic": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "antiparasitic": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict",
    "neurotoxin",
    "toxicity"
  ],
  "antiviral": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "ace_inhibitory": [
    "ace_inhibitory_ic50",
    "ace_inhibitory",
    "antidiabetic",
    "antimicrobial",
    "antioxidant",
    "dppiv_inhibitors",
    "neuropeptide"
  ],
  "antidiabetic": [
    "ace_inhibitory",
    "ace_inhibitory_ic50",
    "antibacterial",
    "antidiabetic",
    "antifungal",
    "antiinflamatory",
    "antimicrobial",
    "dppiv_inhibitors"
  ],
  "dppiv_inhibitors": [
    "ace_inhibitory",
    "ace_inhibitory_ic50",
    "antibacterial",
    "anticancer",
    "antidiabetic",
    "antifungal",
    "antimicrobial",
    "antioxidant",
    "dppiv_inhibitors",
    "neuropeptide"
  ],
  "anticancer": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antioxidant",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "dppiv_inhibitors",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict",
    "neurotoxin",
    "toxicity"
  ],
  "ttca": [
    "ttca"
  ],
  "antiaging": [
    "antiaging"
  ],
  "antiinflamatory": [
    "antidiabetic",
    "antiinflamatory"
  ],
  "antioxidant": [
    "ace_inhibitory",
    "ace_inhibitory_ic50",
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antioxidant",
    "dppiv_inhibitors"
  ],
  "neuropeptide": [
    "ace_inhibitory",
    "ace_inhibitory_ic50",
    "antimicrobial",
    "bbp",
    "dppiv_inhibitors",
    "neuropeptide"
  ],
  "quorum_sensing": [
    "quorum_sensing"
  ],
  "PpI": [
    "PpI",
    "PpI_X",
    "PpI_ba",
    "PpI_ba_X"
  ],
  "PpI_ba": [
    "PpI",
    "PpI_X",
    "PpI_ba",
    "PpI_ba_X"
  ],
  "PpI_ba_X": [
    "PpI",
    "PpI_X",
    "PpI_ba",
    "PpI_ba_X"
  ],
  "PpI_X": [
    "PpI",
    "PpI_X",
    "PpI_ba",
    "PpI_ba_X"
  ],
  "nonfouling": [
    "nonfouling"
  ],
  "solubility": [
    "solubility"
  ],
  "allergen": [
    "allergen"
  ],
  "hemolytic": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "hemolytic_hc50": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "hemolytic_strict": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antimicrobial_E_coli_mic",
    "antimicrobial_P_aeruginosa_mic",
    "antimicrobial_S_aureus_mic",
    "antiparasitic",
    "antiviral",
    "bbp",
    "cpp",
    "hemolytic",
    "hemolytic_hc50",
    "hemolytic_strict"
  ],
  "neurotoxin": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antiparasitic",
    "neurotoxin",
    "toxicity"
  ],
  "toxicity": [
    "antibacterial",
    "anticancer",
    "antifungal",
    "antimicrobial",
    "antiparasitic",
    "neurotoxin",
    "toxicity"
  ]
}


EXCLUSIVE_MAP = {
    key: list(
        (set(value) | set(NON_NATURE_DATASET) | {"PpI_X","PpI_ba_X","nonfouling"}) 
        if NEG_POOL_MAP[key]["is_nature"] 
        else (set(value) | set(NATURE_DATASET))
    )
    for key, value in EXCLUSIVE_MAP.items()
}

# cpp and solubility are physicochemical property datasets, so negative samples are generated from experimental test results and are not included in negative sample selection.
# antimicrobial2, hemolytic1, and hemolytic2 have positive and negative sample labels determined by specific experimental thresholds, so they are not included in negative sample selection.
# Regression tasks and PPI tasks are not included in negative sample selection.





INCLUSIVE_MAP = {
    key: list(set(ALL_POOLING_DATASET) - set(exclusive_values))
    for key, exclusive_values in EXCLUSIVE_MAP.items()
}

# Add exceptions.
# INCLUSIVE_MAP["nonfouling"] = ["hemolytic", "anti mammalian cell"]
# Sampled from hemolytic and cytotoxic peptides, as such peptides are more likely to engage in nonspecific interactions - such as binding to membranes and causing membrane disruption.



if __name__ == "__main__":
    print(INCLUSIVE_MAP)
    for dataset_name in NATURE_CLS_DATASET:
        print(f"Dataset: {dataset_name}")
        sequences = read_dataset_sequences(dataset_name, flag="all")
        print(f"{dataset_name}: {len(sequences)}")
    # print(EXCLUSIVE_MAP)
    # print(NATURE_DATASET)
    # print(INCLUSIVE_MAP.get("nonfouling"))
    # print(INCLUSIVE_MAP.get("toxicity"))

    # hierarchical_stats = get_hierarchical_stats()
    # # dataset_quantity_stats = get_dataset_quantity_stats(max_length=50)
    # dataset_quantity_stats = get_dataset_quantity_stats()

    # for key,value in dataset_quantity_stats.items():
    #     print(f"{key}: {value}")