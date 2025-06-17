from typing import Union, List

import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.error")
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.Chem import MACCSkeys
except:
    raise ImportError(
        "Please install rdkit by 'conda install -c conda-forge rdkit'! ")



class Peptide():
    """
    Arguments:
        sequence (str, optional): the peptide sequence.
        raw_format (str, optional): the peptide sequence format ('fasta', 'helm', 'biln', 'smiles'])

    Returns:
        pd.DataFrame/dict[str, pd.DataFrame]

    Raises:
        AttributeError: format not supported
    """

    def __init__(self, sequence: str, format: str, all_nature: bool = True):

        assert format in ['fasta', 'helm', 'biln','smiles'],\
            "Format not supported. Please choose from ['fasta', 'helm', 'biln','smiles']"

        self.sequence = sequence
        self.format = format
        self.all_nature = all_nature

    def to(self, output_format):

        assert output_format in ['helm', 'biln','smiles'],\
            "Format not supported. Please choose from ['helm', 'biln', 'smiles']"

        if output_format == self.format:
            print("The input and output formats are the same!")
            new_sequence = self.sequence

        if output_format =='smiles':
            if self.format == 'fasta':
                new_sequence = fasta2smiles(self.sequence)
            elif self.format == 'helm':
                new_sequence = helm2smiles(self.sequence)
            elif self.format == 'biln':
                new_sequence = biln2smiles(self.sequence)

        elif output_format == 'helm':
            if self.format == 'biln':
                seq_smi = biln2smiles(self.sequence)
                new_sequence = smiles2helm(seq_smi)
            elif self.format =='smiles':
                new_sequence = smiles2helm(self.sequence)

        elif output_format == 'biln':
            if self.format == 'helm':
                seq_smi = helm2smiles(self.sequence)
                new_sequence = smiles2biln(seq_smi)
            elif self.format =='smiles':
                new_sequence = smiles2biln(self.sequence)

        return new_sequence



# ---------- Conversion  ----------

def _seq_to_mol(fasta):
    """Convert peptide sequence to RDKit Mol object"""
    if Chem.MolFromSequence(fasta) is None:
        print(fasta)
    return Chem.MolFromSequence(fasta)

def fasta2smiles(fasta: str):
    """Convert peptide sequence to SMILES representation"""
    mol = _seq_to_mol(fasta)
    return Chem.MolToSmiles(mol) if mol else ""

def biln2smiles(biln: str):
    ''
def helm2smiles(helm: str):
    ''
def smiles2biln(smiles: str):
    ''
def smiles2helm(smiles: str):
    ''










