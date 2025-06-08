import random
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AtomPairs import Pairs, Torsions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import argparse
import os
from pepbenchmark.metadata import DATASET_MAP

# External libraries for xgboost and lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np
from rdkit.Chem import rdFingerprintGenerator as rfp
class FP_Converter():
 

    def __init__(self, type: str, nbits: int, radius: int):
        self.nbits = nbits
        self.radius = radius
        self.generator = self._load_generator(type)




    def __call__(self, mol):

        if mol is None:
            print("Warning: Invalid molecule, returning zero fingerprint.")
            fp = np.zeros((1, self.nbits))
        else:
            fp = self.generator.GetCountFingerprintAsNumPy(mol)
        return fp
    def _load_generator(self, fp_type: str):
        

        if 'ecfp' in fp_type:
            return AllChem.GetMorganGenerator(radius=self.radius,
                                            includeChirality=True,
                                            fpSize=self.nbits,
                                            countSimulation='count' in fp_type)

        elif 'fcfp' in fp_type:
            invgen = AllChem.GetMorganFeatureAtomInvGen()
            return AllChem.GetMorganGenerator(radius=self.radius,
                                            fpSize=self.nbits,
                                            includeChirality=True,
                                            useBondTypes=True,
                                            atomInvariantsGenerator=invgen,
                                            includeRingMembership=True,
                                            countSimulation=True)

        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
# ---------- Data Loading and Featurization ----------

def load_data(path):
    df = pd.read_csv(path)
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    return sequences, labels


def seq_to_mol(sequence):
    """Convert peptide sequence to RDKit Mol object"""
    return Chem.MolFromSequence(sequence)


def seq_to_smiles(sequence):
    """Convert peptide sequence to SMILES representation"""
    mol = seq_to_mol(sequence)
    return Chem.MolToSmiles(mol) if mol else ""


def featurize(sequences,converter):
    """Convert list of sequences to fingerprint features"""
    features = []
    for seq in sequences:
        mol = seq_to_mol(seq)
        fp = converter(mol)
        features.append(fp)
    return features

# ---------- Main Script ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)