import random
from typing import Union, List

import pandas as pd
import numpy as np
from rdkit import Chem
from.convert import fasta2smiles, _seq_to_mol, helm2smiles, biln2smiles
from rdkit.Chem import AllChem, MACCSkeys,Descriptors


class PeptideFeaturizer():

    """
    Arguments:
        input_format (str, optional): the peptide sequence format ('fasta', 'helm', 'biln', 'smiles')
        feature_types (str, optional): the type of features ('esm2_embedding', 'onehot', 'ecfp', 'graph', 'modred_descriptor])
        is_nature (bool, optional): whether the peptide is natural or synthetic

    Returns:
        pd.DataFrame/dict[str, pd.DataFrame]

    Raises:
        AttributeError: format not supported
    """


    def __init__(self,
                 input_format: str,
                 feature_type: str,
                 is_natural: bool = True, **kwargs):
        self.input_format = input_format
        self.feature_type = feature_type


        if is_natural == True:
            assert feature_type in ['esm2_embedding', 'onehot', 'pep_descriptors', 'ecfp', 'fcfp', 'graph', 'rdkit_des'],\
                "Feature type not supported. Please choose from ['esm2_embedding', 'onehot', 'pep_descriptors', 'ecfp', 'fcfp','graph', 'rdkit_des']"
        else:
            assert input_format in ['helm', 'biln', 'smiles'],\
                "Input format not supported. Please choose from ['helm', 'biln','smiles']"
            assert feature_type in ['ecfp', 'fcfp', 'graph', 'rdkit_des'],\
                "Feature type not supported. Please choose from ['ecfp', 'fcfp', 'graph', 'rdkit_des']"


    def __call__(self, sequences: Union[str, List[str]]) -> np.ndarray:

        if isinstance(sequences, str):
            sequences = [sequences]

        if self.input_format == 'fasta':
            return featurizer_fasta(sequences, self.feature_type)
        else:
            if self.input_format == 'helm':
                smi_seqs = [helm2smiles(seq) for seq in sequences]
            elif self.input_format == 'biln':
                smi_seqs = [biln2smiles(seq) for seq in sequences]
            elif self.input_format =='smiles':
                smi_seqs = sequences

            return featurizer_smi(smi_seqs, self.feature_type)



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


class DescriptorCalculator():
    """
    计算给定肽分子的常见物化描述符。
    输入为 RDKit 分子对象，输出为 descriptor 向量。
    """

    def __init__(self, type: str = 'rdkit_des'):

        self.descriptor_type = type.lower()
        self.descriptor_funcs = self._load_calculator(self.descriptor_type)

    def _load_calculator(self, type: str):
        if type == 'rdkit_des':
            return Descriptors._descList
        else:
            raise ValueError(f"Unknown descriptor type: {type}")

    def __call__(self, mol):
        if mol is None:
            print("Warning: Invalid molecule, returning zero vector.")

        else:

            features = []
            for name, func in self.descriptor_funcs:
                try:
                    val = func(mol)
                except Exception:
                    val = np.nan
                features.append(val)

            column_names = [name for name, _ in self.descriptor_funcs]
            return pd.DataFrame([features], columns = column_names)


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a peptide sequence"""
    STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
    one_hot = np.zeros((len(sequence), len(STANDARD_AAS)))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            one_hot[i, AA_TO_INDEX[aa]] = 1
    return one_hot

class pep_descriptor():
    ""




def FP_featurize(input_format, sequences,converter):
    """Convert list of sequences to fingerprint features"""
    features = []
    for seq in sequences:
        if input_format == 'fasta':
            mol = _seq_to_mol(seq)
        if input_format == 'smiles':
            mol = Chem.MolFromSmiles(seq)

        fp = converter(mol)
        features.append(fp)

    return features


def DP_featurize(input_format, sequences,converter):
    """Convert list of sequences to descriptor features"""
    features = []
    for seq in sequences:
        if input_format == 'fasta':
            mol = _seq_to_mol(seq)
        if input_format == 'smiles':
            mol = Chem.MolFromSmiles(seq)

        feature_row = converter(mol)
        features.append(feature_row)

    return pd.concat(features, axis=0).reset_index(drop=True)


def featurizer_fasta(fastas: str, feature_type: str) -> Union[np.ndarray, List]:


    assert feature_type in ['esm2_embedding', 'onehot', 'pep_descriptors', 'ecfp', 'fcfp', 'graph', 'rdkit_des'], \
        "Feature type not supported. Please choose from ['esm2_embedding', 'onehot', 'pep_descriptors', 'ecfp', 'fcfp', 'graph', 'rdkit_des']"

    if feature_type in ['graph']:
        smi_seqs = [fasta2smiles(seq) for seq in fastas]
        featurizer_smi(smi_seqs, feature_type)

    elif feature_type in ['ecfp', 'fcfp']:
        converter = FP_Converter(type=feature_type, nbits=1024, radius=2)
        return FP_featurize('fasta', fastas, converter)

    elif feature_type == 'esm2_embedding':
        ""

    elif feature_type == 'onehot':
        return pd.DataFrame([{'onehot': one_hot_encode(seq)} for seq in fastas])

    elif feature_type == 'rdkit_des':
        converter = DescriptorCalculator(type=feature_type)
        return DP_featurize('fasta', fastas, converter)

    elif feature_type == 'pep_descriptors':
        ""



def featurizer_smi(smiles:  str, feature_type: str):
    assert feature_type in ['ecfp', 'fcfp', 'graph', 'rdkit_des'],\
        "Feature type not supported. Please choose from ['ecfp', 'fcfp', 'graph', 'rdkit_des']"
    if feature_type in ['ecfp', 'fcfp']:
        converter = FP_Converter(type=feature_type, nbits=1024, radius=2)
        return FP_featurize('smiles', smiles, converter)
    elif feature_type == 'rdkit_des':
        converter = DescriptorCalculator(type=feature_type)
        return DP_featurize('smiles', smiles, converter)

    elif feature_type == 'graph':
        ""

