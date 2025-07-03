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
Peptide Sequence Conversion and Manipulation Module.

This module provides the Peptide class for handling peptide sequences in various
formats and performing conversions between different representations. It supports
multiple peptide sequence formats including FASTA, HELM, BiLN, and SMILES.

The module enables:
- Peptide sequence format validation and conversion
- Structure-based representations (SMILES, HELM)
- Bioinformatics standard formats (FASTA)
- Custom peptide notation systems (BiLN)

Classes:
    Peptide: Main class for peptide sequence handling and conversion

Dependencies:
    - rdkit: For chemical structure handling and SMILES processing
    - logging: For operation logging and debugging

Example:
    >>> # Create peptide from FASTA sequence
    >>> peptide = Peptide("ALAGGGPCR", format="fasta")
    >>>
    >>> # Convert to SMILES representation
    >>> smiles = peptide.to("smiles")
    >>>
    >>> # Convert to HELM notation
    >>> helm = peptide.to("helm")
"""

# Configure logging for this module
import numpy as np
import torch
from pepbenchmark.external.pep.builder import MolBuilder
from pepbenchmark.external.pep.library import MonomerLibrary
from pepbenchmark.external.pep.parsers.biln_parser import BilnParser, BilnSerializer
from pepbenchmark.external.pep.parsers.fasta_parser import FastaParser, FastaSerializer
from pepbenchmark.external.pep.parsers.helm_parser import HelmParser, HelmSerializer
from pepbenchmark.utils.logging import get_logger
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from transformers import AutoModel, AutoTokenizer
from ogb.utils import smiles2graph
from torch_geometric.data import Data

logger = get_logger()

lib = MonomerLibrary.from_sdf_file(
    "test_library",
    "E:/pycharm/PepBenchmark/src/pepbenchmark/external/pep/resources/monomers.sdf",
)


class FormatTransform:
    """
    Base class for peptide format transformations.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__ method.")


class Fasta2Smiles(FormatTransform):
    """
    Transform a protein sequence in FASTA format into a SMILES string representing the peptide.
    """

    def __call__(self, fasta: str) -> str:
        # Parse the FASTA: remove headers and join sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(seq_lines)

        if not sequence:
            raise ValueError("No sequence found in FASTA input.")

        # Use RDKit to build a peptide from the sequence
        # Chem.MolFromSequence handles standard amino acids
        peptide = Chem.MolFromSequence(sequence)
        if peptide is None:
            raise ValueError(f"Failed to generate molecule from sequence: {sequence}")

        # Convert the molecule to SMILES
        smiles = Chem.MolToSmiles(peptide)
        return smiles


class Smiles2FP(FormatTransform):
    """
    Transform a SMILES string into a single molecular fingerprint as a numpy array.

    Parameters in __init__:
        fp_type (str): fingerprint type to compute. Options:
            - 'Morgan'
            - 'RDKit'
            - 'MACCS'
            - 'TopologicalTorsion'
            - 'AtomPair'
        kwargs: hyperparameters for the chosen fingerprint:
            - Morgan: radius (int), nBits (int)
            - RDKit: fpSize (int)
            - TopologicalTorsion: nBits (int)
            - AtomPair: nBits (int)
    """

    def __init__(self, fp_type: str = "Morgan", **kwargs):
        self.available_fps = [
            "Morgan",
            "RDKit",
            "MACCS",
            "TopologicalTorsion",
            "AtomPair",
        ]
        if fp_type not in self.available_fps:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        self.fp_type = fp_type
        # Default hyperparameters
        self.params = {
            "Morgan": {"radius": 2, "nBits": 2048},
            "RDKit": {"fpSize": 2048},
            "MACCS": {},
            "TopologicalTorsion": {"nBits": 2048},
            "AtomPair": {"nBits": 2048},
        }
        # Override defaults with any provided kwargs
        self.params[self.fp_type].update(kwargs)

    def __call__(self, smiles: str) -> np.ndarray:
        # Parse SMILES into RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        fp_type = self.fp_type
        p = self.params[fp_type]
        # Compute the specified fingerprint
        if fp_type == "Morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=p["radius"], nBits=p["nBits"]
            )
        elif fp_type == "RDKit":
            fp = Chem.RDKFingerprint(mol, fpSize=p["fpSize"])
        elif fp_type == "MACCS":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == "TopologicalTorsion":
            fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=p["nBits"]
            )
        elif fp_type == "AtomPair":
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=p["nBits"])
        else:
            # Should never happen
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

        # Convert bit vector to numpy array of ints
        bit_str = fp.ToBitString()
        arr = np.fromiter(bit_str, dtype=int)
        return arr


# Example usage:
# transformer_fp = Smiles2FP(fp_type='Morgan', radius=3, nBits=1024)
# smiles_str = 'CC(=O)Oc1ccccc1C(=O)O'
# fingerprint = transformer_fp(smiles_str)
# print(fingerprint)  # List of ints for each bit


class Fasta2Embedding(FormatTransform):
    """
    Convert FASTA sequence to molecular embedding using a pretrained model.
    If `model` is a string, initialize with Transformers; if it is a PyTorch model instance,
    it can be any `torch.nn.Module`. In that case, the model should provide a `tokenizer` attribute
    or have `config.name_or_path` for inference.

    Args:
        model: HuggingFace model identifier (str) or a PyTorch `nn.Module` instance.
        device: Optional device string (e.g., 'cuda', 'cpu'). Defaults to GPU if available else CPU.
    """

    def __init__(self, model, device: str = None):
        # Select device: use provided or default to GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model based on input type
        if isinstance(model, str):
            # Load from HuggingFace Hub
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
            self.model = AutoModel.from_pretrained(model)
        elif isinstance(model, torch.nn.Module):
            # Use provided PyTorch model instance
            self.model = model
            # Try to get tokenizer from model attribute
            if hasattr(model, "tokenizer"):
                self.tokenizer = model.tokenizer
            else:
                # Attempt to infer from config name
                model_id = getattr(getattr(model, "config", None), "name_or_path", None)
                if model_id:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id, use_fast=False
                    )
                else:
                    raise ValueError(
                        "Cannot infer tokenizer for provided PyTorch model. "
                        "Please attach a `tokenizer` attribute or supply a model name string."
                    )
        else:
            raise ValueError(
                "`model` must be a string model identifier or a PyTorch nn.Module instance."
            )

        # Move model to chosen device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, fasta: str):
        """
        Convert a FASTA-formatted string to an embedding using the pretrained model.

        Args:
            fasta: FASTA string (header lines start with '>') containing one sequence.

        Returns:
            numpy.ndarray: Embedding vector of shape (hidden_dim,) obtained by mean-pooling the last hidden states.
        """
        # Parse FASTA: remove header lines and concatenate sequence lines
        lines = fasta.strip().splitlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

        # Tokenize sequence and move inputs to device
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        # Forward pass without gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract last hidden states and apply mean pooling over sequence dimension
        hidden_states = (
            outputs.last_hidden_state
        )  # tensor of shape (1, seq_len, hidden_dim)
        embedding = hidden_states.mean(dim=1).squeeze(
            0
        )  # tensor of shape (hidden_dim,)

        # Return as NumPy array on CPU
        return embedding.cpu().numpy()


class Fasta2Helm(FormatTransform):
    def __call__(self, fasta: str) -> str:
        # Parse the FASTA: remove headers and join sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(seq_lines)

        if not sequence:
            raise ValueError("No sequence found in FASTA input.")

        parsed_data = FastaParser(lib).parse(sequence)
        serializer = HelmSerializer(lib)
        return serializer.serialize(parsed_data)


class Fasta2Biln(FormatTransform):
    def __call__(self, fasta: str) -> str:
        # Parse the FASTA: remove headers and join sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(seq_lines)

        if not sequence:
            raise ValueError("No sequence found in FASTA input.")

        parsed_data = FastaParser(lib).parse(sequence)
        serializer = BilnSerializer(lib)
        return serializer.serialize(parsed_data)


class Smiles2Fasta(FormatTransform):
    def __call__(self, smiles: str) -> str:
        logger.warning("SMILES to FASTA conversion not yet implemented")
        return ""


class Smiles2Helm(FormatTransform):
    def __call__(self, smiles: str) -> str:
        logger.warning("SMILES to HELM conversion not yet implemented")
        return ""


class Smiles2Biln(FormatTransform):
    def __call__(self, smiles: str) -> str:
        logger.warning("SMILES to BiLN conversion not yet implemented")
        return ""


class Helm2Fasta(FormatTransform):
    def __call__(self, helm: str) -> str:
        # Parse HELM notation into a structured format
        parsed_data = HelmParser(lib).parse(helm)
        # Serialize back to FASTA format
        serializer = FastaSerializer(lib)
        return serializer.serialize(parsed_data)


class Helm2Smiles(FormatTransform):
    def __call__(self, helm: str) -> str:
        # Parse HELM notation into a structured format
        parsed_data = HelmParser(lib).parse(helm)
        mol = MolBuilder(parsed_data).build()
        if mol is None:
            raise ValueError(f"Failed to build molecule from HELM: {helm}")
        return Chem.MolToSmiles(mol)


class Helm2Biln(FormatTransform):
    def __call__(self, helm: str) -> str:
        # Parse HELM notation into a structured format
        parsed_data = HelmParser(lib).parse(helm)
        # Serialize to BiLN format
        serializer = BilnSerializer(lib)
        return serializer.serialize(parsed_data)


class Biln2Fasta(FormatTransform):
    def __call__(self, biln: str) -> str:
        # Parse BiLN notation into a structured format
        parsed_data = BilnParser(lib).parse(biln)
        # Serialize back to FASTA format
        serializer = FastaSerializer(lib)
        return serializer.serialize(parsed_data)


class Biln2Smiles(FormatTransform):
    def __call__(self, biln: str) -> str:
        # Parse BiLN notation into a structured format
        parsed_data = BilnParser(lib).parse(biln)
        # Convert to SMILES using MolBuilder
        mol = MolBuilder(parsed_data).build()
        if mol is None:
            raise ValueError(f"Failed to build molecule from BiLN: {biln}")
        return Chem.MolToSmiles(mol)


class Biln2Helm(FormatTransform):
    def __call__(self, biln: str) -> str:
        # Parse BiLN notation into a structured format
        parsed_data = BilnParser(lib).parse(biln)
        # Serialize to HELM format
        serializer = HelmSerializer(lib)
        return serializer.serialize(parsed_data)


class Mol2Fingerprint(FormatTransform):
    """
    Convert a SMILES or RDKit Mol object to a molecular fingerprint (e.g., Morgan).
    """

    def __call__(self, mol_or_smiles, radius=2, nbits=2048):
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles
        if mol is None:
            logger.warning("Invalid molecule for fingerprint generation.")
            return None
        from rdkit.Chem import AllChem

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        # Return as numpy array for ML usage
        import numpy as np

        arr = np.zeros((nbits,), dtype=int)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(fp, arr)
        return arr


class Mol2Embedding(FormatTransform):
    """
    Use a pretrained model to get molecular embedding from SMILES or Mol.
    Placeholder for actual model integration.
    """

    def __init__(self, model=None):
        self.model = model  # Placeholder for actual model

    def __call__(self, mol_or_smiles):
        # Placeholder: return dummy embedding or raise NotImplementedError
        logger.warning(
            "Pretrained model embedding not implemented. Please integrate your model."
        )
        return None


class Sequence2Embedding(FormatTransform):
    """
    Convert FASTA sequence to molecular embedding using a pretrained model.
    If `model` is a string, initialize with Transformers; if it is a PyTorch model instance,
    it can be any `torch.nn.Module`. In that case, the model should provide a `tokenizer` attribute
    or have `config.name_or_path` for inference.

    Args:
        model: HuggingFace model identifier (str)
        device: Optional device string (e.g., 'cuda', 'cpu'). Defaults to GPU if available else CPU.
    """

    def __init__(self, model, device: str = None):
        # Select device: use provided or default to GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model based on input type
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        self.model = AutoModel.from_pretrained(model)

        # Move model to chosen device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, fasta: str):
        """
        Convert a FASTA-formatted string to an embedding using the pretrained model.

        Args:
            fasta: FASTA string (header lines start with '>') containing one sequence.

        Returns:
            numpy.ndarray: Embedding vector of shape (hidden_dim,) obtained by mean-pooling the last hidden states.
        """
        # Parse FASTA: remove header lines and concatenate sequence lines
        lines = fasta.strip().splitlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

        # Tokenize sequence and move inputs to device
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        # Forward pass without gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract last hidden states and apply mean pooling over sequence dimension
        hidden_states = (
            outputs.last_hidden_state
        )  # tensor of shape (1, seq_len, hidden_dim)
        embedding = hidden_states.mean(dim=1).squeeze(
            0
        )  # tensor of shape (hidden_dim,)

        # Return as NumPy array on CPU
        return embedding.cpu().numpy()
    

class Smiles2Graph(FormatTransform):
    def __call__(self, smiles: str, label: torch.Tensor = None) -> dict:
        """
        Convert a SMILES string to a graph representation.

        Args:
            smiles: SMILES string of the molecule.
            label: Optional label tensor for the graph.

        Returns:
            dict: Graph representation with nodes and edges.
        """
        # Convert SMILES to graph format by ogb
        graph_data = smiles2graph(smiles)

        # Create a PyTorch Geometric Data object
        graph = Data(
            x=torch.from_numpy(graph_data['node_feat']),
            edge_index=torch.from_numpy(graph_data['edge_index']),
            edge_attr=torch.from_numpy(graph_data['edge_feat'])
        )

        # If a label is provided, assign it to the graph
        if label is not None:
            graph.y = label
        
        # Return the graph object on CPU
        return graph


AVALIABLE_TRANSFORM = {
    ("fasta", "smiles"): Fasta2Smiles,
    ("fasta", "helm"): Fasta2Helm,
    ("fasta", "biln"): Fasta2Biln,
    ("fasta", "embedding"): Fasta2Embedding,
    ("smiles", "fasta"): Smiles2Fasta,
    ("smiles", "helm"): Smiles2Helm,
    ("smiles", "biln"): Smiles2Biln,
    ("helm", "fasta"): Helm2Fasta,
    ("helm", "smiles"): Helm2Smiles,
    ("helm", "biln"): Helm2Biln,
    ("biln", "fasta"): Biln2Fasta,
    ("biln", "smiles"): Biln2Smiles,
    ("biln", "helm"): Biln2Helm,
    ("mol", "fingerprint"): Mol2Fingerprint,
    ("mol", "embedding"): Mol2Embedding,
    ("smiles", "fingerprint"): Smiles2FP,
    ("smiles", "embedding"): Mol2Embedding,
    ("smiles", "graph"): Smiles2Graph,
}

if __name__ == "__main__":
    fasta2smiles = Fasta2Smiles()
    fasta2helm = Fasta2Helm()
    fasta2biln = Fasta2Biln()
    helm2fasta = Helm2Fasta()
    helm2smiles = Helm2Smiles()
    helm2biln = Helm2Biln()
    biln2fasta = Biln2Fasta()
    biln2smiles = Biln2Smiles()
    biln2helm = Biln2Helm()

    fasta = "ALAGGGPCR"

    smile = fasta2smiles(fasta)
    helm = fasta2helm(fasta)
    biln = fasta2biln(fasta)
    logger.info(f"FASTA: {fasta}")
    logger.info(f"SMILES: {smile}")
    logger.info(f"HELM: {helm}")
    logger.info(f"BiLN: {biln}")
    logger.info(f"FASTA from HELM: {helm2fasta(helm)}")
    logger.info(f"SMILES from HELM: {helm2smiles(helm)}")
    logger.info(f"BiLN from HELM: {helm2biln(helm)}")
    logger.info(f"FASTA from BiLN: {biln2fasta(biln)}")
    logger.info(f"SMILES from BiLN: {biln2smiles(biln)}")
    logger.info(f"HELM from BiLN: {biln2helm(biln)}")
