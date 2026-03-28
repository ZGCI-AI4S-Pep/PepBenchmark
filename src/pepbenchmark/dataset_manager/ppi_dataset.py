"""Dataset manager for official peptide-protein interaction benchmarks."""

import json
import os
from tokenize import Single
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import scipy as sp
import torch
from torch.utils.data import Dataset

from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager
from pepbenchmark.metadata import DATASET_DIR, DATASET_MAP
from pepbenchmark.utils.logging import get_logger

logger = get_logger()

BASE_URL = "https://huggingface.co/datasets/jiahuizhang/PepBenchData/resolve/main/"



class PPIDatasetManager(SinglePeptideDatasetManager):
        
    OFFICIAL_FEATURE_MAP: Dict[str, str] = {
        "pep_fasta": "pep_fasta.csv",
        "prot_fasta": "prot_fasta.csv",
        "pep_smiles": "pep_smiles.csv",
        "pep_helm": "pep_helm.csv",
        "pep_biln": "pep_biln.csv",
        "pep_ecfp4": "pep_ecfp4.npz",
        "pep_ecfp6": "pep_ecfp6.npz",
        "pep_graph": "pep_graph.pt",
        "prot_ecfp4": "prot_ecfp4.npz",
        "prot_ecfp6": "prot_ecfp6.npz",
        "prot_graph": "prot_graph.pt",
        "pep_esm2_150_embedding": "pep_esm2_150_embedding.npz",
        "prot_esm2_150_embedding": "prot_esm2_150_embedding.npz",
        "pep_dpml_embedding": "pep_dpml_embedding.npz",
        "prot_dpml_embedding": "prot_dpml_embedding.npz",
        "interaction_graph": "interaction_graph.pt",
        "label": "label.csv",
    }

    OFFICIAL_SPLIT_MAP: Dict[str, str] = {
        "protein_random_cold_split": "protein_random_cold_split.json",
        "protein_mmseqs_cold_split": "protein_mmseqs_cold_split.json",
        "protein_kmer_cold_split": "protein_kmer_cold_split.json",
        "protein_hybrid_cold_split": "protein_hybrid_cold_split.json",
        "peptide_random_cold_split": "peptide_random_cold_split.json",
        "peptide_mmseqs_cold_split": "peptide_mmseqs_cold_split.json",
        "peptide_kmer_cold_split": "peptide_kmer_cold_split.json",
        "peptide_hybrid_cold_split": "peptide_hybrid_cold_split.json",
    }

    


