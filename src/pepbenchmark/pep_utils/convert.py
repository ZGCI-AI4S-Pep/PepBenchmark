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


try:
    from rdkit import Chem, rdBase

    rdBase.DisableLog("rdApp.error")
except ImportError as err:
    raise ImportError(
        "Please install rdkit by 'conda install -c conda-forge rdkit'! "
    ) from err

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Peptide:
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
        assert format in [
            "fasta",
            "helm",
            "biln",
            "smiles",
        ], "Format not supported. Please choose from ['fasta', 'helm', 'biln','smiles']"

        self.sequence = sequence
        self.format = format
        self.all_nature = all_nature

    def to(self, output_format):
        assert output_format in [
            "helm",
            "biln",
            "smiles",
        ], "Format not supported. Please choose from ['helm', 'biln', 'smiles']"

        if output_format == self.format:
            logger.info("The input and output formats are the same!")
            return self.sequence

        if output_format == "smiles":
            return self._to_smiles()
        elif output_format == "helm":
            return self._to_helm()
        elif output_format == "biln":
            return self._to_biln()

    def _to_smiles(self):
        if self.format == "fasta":
            return fasta2smiles(self.sequence)
        elif self.format == "helm":
            return helm2smiles(self.sequence)
        elif self.format == "biln":
            return biln2smiles(self.sequence)

    def _to_helm(self):
        if self.format == "biln":
            return smiles2helm(biln2smiles(self.sequence))
        elif self.format == "smiles":
            return smiles2helm(self.sequence)

    def _to_biln(self):
        if self.format == "helm":
            return smiles2biln(helm2smiles(self.sequence))
        elif self.format == "smiles":
            return smiles2biln(self.sequence)


# ---------- Conversion  ----------


def _seq_to_mol(fasta):
    """Convert peptide sequence to RDKit Mol object"""
    return Chem.MolFromSequence(fasta)


def fasta2smiles(fasta: str):
    """Convert peptide sequence to SMILES representation"""
    mol = _seq_to_mol(fasta)
    return Chem.MolToSmiles(mol) if mol else ""


def biln2smiles(biln: str):
    """"""


def helm2smiles(helm: str):
    """"""


def smiles2biln(smiles: str):
    """"""


def smiles2helm(smiles: str):
    """"""
