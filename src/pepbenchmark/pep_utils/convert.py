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

"""Peptide Sequence Conversion and Manipulation Module.

This module provides comprehensive functionality for handling peptide sequences in various
formats and performing conversions between different representations. It supports
multiple peptide sequence formats including FASTA, HELM, BiLN, and SMILES, along with
advanced molecular descriptors and embeddings.

Example:
    Basic format conversion:
        >>> fasta2smiles = Fasta2Smiles()
        >>> smiles = fasta2smiles("ALAGGGPCR")
        >>> print(smiles)

    Molecular fingerprint generation:
        >>> fp_generator = Smiles2FP(fp_type='Morgan', radius=3, nBits=2048)
        >>> fingerprint = fp_generator(smiles)
        >>> print(f"Fingerprint length: {len(fingerprint)}")

    Neural embedding:
        >>> embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
        >>> embedding = embedder("ALAGGGPCR")
        >>> print(f"Embedding shape: {embedding.shape}")
"""

# Configure logging for this module
import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetTopologicalTorsionGenerator,
)
from rdkit.DataStructs import ConvertToNumpyArray
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from pepbenchmark.external.pep.builder import MolBuilder
from pepbenchmark.external.pep.library import MonomerLibrary
from pepbenchmark.external.pep.parsers.biln_parser import BilnParser, BilnSerializer
from pepbenchmark.external.pep.parsers.fasta_parser import FastaParser, FastaSerializer
from pepbenchmark.external.pep.parsers.helm_parser import HelmParser, HelmSerializer
from pepbenchmark.utils.logging import get_logger

logger = get_logger()


class FormatTransform:
    """Base class for peptide format transformations.

    This abstract class defines the interface for all peptide format conversion
    operations. All transformation classes should inherit from this base class
    and implement the __call__ method.

    Methods:
        __call__: Abstract method that performs the actual transformation
    """

    def __call__(self, *args, **kwargs):
        """Perform the format transformation.

        Args:
            *args: Variable length argument list specific to each transformation
            **kwargs: Arbitrary keyword arguments specific to each transformation

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def batch_convert(self, inputs: list) -> list:
        """
        Perform batch conversion of multiple inputs.
        Args:
            inputs (list): List of inputs to convert. Each input should be
                compatible with the __call__ method of the subclass.
        Returns:
            list: List of converted outputs corresponding to each input.
        """
        return [self.__call__(input) for input in inputs]


class Fasta2Smiles(FormatTransform):
    """Transform a sequence in FASTA format into a SMILES string.

    This class converts amino acid sequences from FASTA format into their
    corresponding SMILES (Simplified Molecular Input Line Entry System)
    representation using RDKit's molecular building capabilities.

    The conversion handles standard amino acids and builds a complete
    peptide molecule with proper bond connectivity.

    Example:
        >>> converter = Fasta2Smiles()
        >>> smiles = converter("ALAGGGPCR")
        >>> print(smiles)  # Returns SMILES string representation
    """

    def __call__(self, fasta: str) -> str:
        """Convert FASTA sequence to SMILES representation.

        Args:
            fasta (str): FASTA-formatted string containing peptide sequence.
                Can include header lines (starting with '>') which will be ignored.

        Returns:
            str: SMILES string representation of the peptide molecule.

        Raises:
            ValueError: If no sequence is found in the input or if RDKit
                fails to generate a molecule from the sequence.

        Example:
            >>> converter = Fasta2Smiles()
            >>> smiles = converter(">peptide1\\nALAGGGPCR")
            >>> print(type(smiles))  # <class 'str'>
        """
        # Parse the FASTA: remove headers and join sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(seq_lines)

        if not sequence:
            raise ValueError("No sequence found in FASTA input.")

        # Use RDKit to build a peptide from the sequence
        peptide = Chem.MolFromSequence(sequence)
        if peptide is None:
            raise ValueError(f"Failed to generate molecule from sequence: {sequence}")

        # Convert the molecule to SMILES
        smiles = Chem.MolToSmiles(peptide)
        return smiles


class Fasta2Embedding(FormatTransform):
    """
    Convert FASTA sequence to molecular embedding using pretrained models.

    This class generates dense vector representations of sequences
    using pretrained transformer models from HuggingFace.

    The embedding is computed by mean-pooling the last hidden states from
    the transformer model, providing a fixed-size representation regardless
    of sequence length.

    Attributes:
        device (str): Device for model computation ('cuda' or 'cpu')
        tokenizer: HuggingFace tokenizer for sequence preprocessing
        model: Pretrained transformer model for embedding generation
        pooling (str): Pooling strategy ('mean', 'max', 'cls')
    Args:
        model (str or torch.nn.Module): Either a HuggingFace model identifier
            string or a PyTorch model instance. For model instances, must
            have a 'tokenizer' attribute or 'config.name_or_path' for
            tokenizer inference.
        device (str, optional): Device for computation. Defaults to GPU
            if available, otherwise CPU.
        pooling (str, optional): Pooling strategy: 'mean', 'max', or 'cls'.
            Defaults to 'mean'.
            - 'mean': Mean-pooling over sequence tokens
            - 'max': Max-pooling over sequence tokens
            - 'cls': Use the CLS token embedding

    Raises:
        ValueError: If model type is unsupported or tokenizer cannot be
            inferred for PyTorch model instances.
    Example:
        >>> embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
        >>> embedding = embedder("ALAGGGPCR")
        >>> print(embedding.shape)  # (640,) for ESM-2 150M model
    """

    def __init__(self, model, device: str = None, pooling: str = "mean"):
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "max", "cls"}:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

        # Load model and tokenizer
        if isinstance(model, str):
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                model, use_fast=False
            )
            self.model: PreTrainedModel = AutoModel.from_pretrained(model)
        elif isinstance(model, torch.nn.Module):
            self.model = model
            if hasattr(model, "tokenizer") and isinstance(
                model.tokenizer, PreTrainedTokenizer
            ):
                self.tokenizer = model.tokenizer
            else:
                model_id = getattr(getattr(model, "config", None), "name_or_path", None)
                if model_id:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id, use_fast=False
                    )
                else:
                    raise ValueError(
                        "Cannot infer tokenizer for provided model. "
                        "Attach a `tokenizer` attribute or supply a model name string."
                    )
        else:
            raise ValueError(
                "`model` must be a HuggingFace model identifier string or a torch.nn.Module instance."
            )

        # Prepare model
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, fasta: str) -> np.ndarray:
        """Generate embedding vector from a single FASTA sequence."""
        # Parse FASTA
        lines = fasta.strip().splitlines()
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))
        if not seq:
            raise ValueError("No sequence found in FASTA input.")

        # Tokenize and infer
        inputs = self.tokenizer(seq, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)

        hidden = out.last_hidden_state  # (1, L, D)
        # Pooling
        if self.pooling == "mean":
            emb = hidden.mean(dim=1).squeeze(0)
        elif self.pooling == "max":
            emb = hidden.max(dim=1).values.squeeze(0)
        else:  # 'cls'
            emb = hidden[:, 0, :].squeeze(0)

        return emb.cpu().numpy()


class Fasta2Helm(FormatTransform):
    """Convert FASTA sequence to HELM (Hierarchical Editing Language for Macromolecules) notation.

    HELM is a standard notation for representing complex biological molecules
    including peptides, nucleic acids, and small molecules. This converter
    transforms amino acid sequences into proper HELM syntax.

    Example:
        >>> converter = Fasta2Helm()
        >>> helm = converter("ALAGGGPCR")
        >>> print(helm)  # HELM notation string
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.fasta_parser = FastaParser(self.lib)
        self.helm_serializer = HelmSerializer(self.lib)

    def __call__(self, fasta: str) -> str:
        """Convert FASTA sequence to HELM notation.

        Args:
            fasta (str): FASTA-formatted string containing peptide sequence.

        Returns:
            str: HELM notation representation of the peptide.

        Raises:
            ValueError: If no sequence is found in the input.
        """
        # Parse the FASTA: remove headers and join sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(seq_lines)

        if not sequence:
            raise ValueError("No sequence found in FASTA input.")

        parsed_data = self.fasta_parser.parse(sequence)
        return self.helm_serializer.serialize(parsed_data)


class Fasta2Biln(FormatTransform):
    """Convert FASTA sequence to BiLN (Biological Linear Notation) format.

    BiLN is a linear notation system for representing biological macromolecules.
    This converter transforms standard amino acid sequences into BiLN format
    for specialized applications.

    Example:
        >>> converter = Fasta2Biln()
        >>> biln = converter("ALAGGGPCR")
        >>> print(biln)  # BiLN notation string
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        # Initialize the FastaParser and BilnSerializer with the monomer library
        self.fasta_parser = FastaParser(self.lib)
        self.biln_serializer = BilnSerializer(self.lib)

    def __call__(self, fasta: str) -> str:
        """Convert FASTA sequence to BiLN notation.

        Args:
            fasta (str): FASTA-formatted string containing peptide sequence.

        Returns:
            str: BiLN notation representation of the peptide.

        Raises:
            ValueError: If no sequence is found in the input.
        """
        # Parse the FASTA: remove headers and join sequence lines
        lines = fasta.strip().splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(seq_lines)

        if not sequence:
            raise ValueError("No sequence found in FASTA input.")

        parsed_data = self.fasta_parser.parse(sequence)
        return self.biln_serializer.serialize(parsed_data)


class Smiles2Fasta(FormatTransform):
    """Convert SMILES notation to FASTA sequence format.

    Note:
        This conversion is not yet implemented and currently returns an empty string.
        SMILES to sequence conversion requires complex molecular analysis and
        sequence inference algorithms.
    """

    def __call__(self, smiles: str) -> str:
        """Convert SMILES to FASTA sequence (not implemented).

        Args:
            smiles (str): SMILES notation string.

        Returns:
            str: Empty string (conversion not implemented).
        """
        logger.warning("SMILES to FASTA conversion not yet implemented")
        return ""


class Smiles2Helm(FormatTransform):
    """Convert SMILES notation to HELM format.

    Note:
        This conversion is not yet implemented and currently returns an empty string.
    """

    def __call__(self, smiles: str) -> str:
        """Convert SMILES to HELM notation (not implemented).

        Args:
            smiles (str): SMILES notation string.

        Returns:
            str: Empty string (conversion not implemented).
        """
        logger.warning("SMILES to HELM conversion not yet implemented")
        return ""


class Smiles2Biln(FormatTransform):
    """Convert SMILES notation to BiLN format.

    Note:
        This conversion is not yet implemented and currently returns an empty string.
    """

    def __call__(self, smiles: str) -> str:
        """Convert SMILES to BiLN notation (not implemented).

        Args:
            smiles (str): SMILES notation string.

        Returns:
            str: Empty string (conversion not implemented).
        """
        logger.warning("SMILES to BiLN conversion not yet implemented")
        return ""


class Smiles2FP(FormatTransform):
    """Transform SMILES string into molecular fingerprints.

    This class generates various types of molecular fingerprints from SMILES
    representations using RDKit. Supported fingerprint types include Morgan,
    RDKit topological, MACCS keys, Topological Torsion, and Atom Pair fingerprints.
    Args:
        fp_type (str): Type of fingerprint to generate. Must be one of
            available_fps. Defaults to "Morgan".
        **kwargs: Hyperparameters specific to the chosen fingerprint type:
            - Morgan: radius (int), nBits (int)
            - RDKit: fpSize (int)
            - MACCS: no parameters
            - TopologicalTorsion: nBits (int)
            - AtomPair: nBits (int)

    Raises:
        ValueError: If fp_type is not in available_fps.

    Example:
        >>> # Default Morgan fingerprint
        >>> fp_gen = Smiles2FP()
        >>>
        >>> # Custom Morgan parameters
        >>> fp_gen = Smiles2FP(fp_type='Morgan', radius=4, nBits=4096)
        >>>
        >>> # RDKit fingerprint
        >>> fp_gen = Smiles2FP(fp_type='RDKit', fpSize=1024)
    Attributes:
        available_fps (list): List of supported fingerprint types
        fp_type (str): Selected fingerprint type
        params (dict): Hyperparameters for each fingerprint type

    Example:
        >>> # Morgan fingerprint with custom parameters
        >>> fp_gen = Smiles2FP(fp_type='Morgan', radius=3, nBits=1024)
        >>> fingerprint = fp_gen('CCO')  # ethanol
        >>> print(f"Fingerprint length: {len(fingerprint)}")  # 1024

        >>> # MACCS keys (fixed length)
        >>> maccs_gen = Smiles2FP(fp_type='MACCS')
        >>> maccs_fp = maccs_gen('CCO')
        >>> print(f"MACCS length: {len(maccs_fp)}")  # 167
    """

    available_fps = [
        "Morgan",
        "RDKit",
        "MACCS",
        "TopologicalTorsion",
        "AtomPair",
    ]

    def __init__(self, fp_type: str = "Morgan", **kwargs):
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
        """Generate molecular fingerprint from SMILES string.

        Args:
            smiles (str): Valid SMILES notation string representing a molecule.

        Returns:
            numpy.ndarray: Binary fingerprint as integer array. Length depends
                on fingerprint type and parameters:
                - Morgan/RDKit/TopologicalTorsion/AtomPair: configurable via nBits/fpSize
                - MACCS: fixed length of 167 bits

        Raises:
            ValueError: If SMILES string is invalid or fingerprint type is unsupported.

        Example:
            >>> fp_gen = Smiles2FP(fp_type='Morgan', radius=2, nBits=2048)
            >>> fingerprint = fp_gen('CC(=O)OC1=CC=CC=C1C(=O)O')  # aspirin
            >>> print(f"Non-zero bits: {np.sum(fingerprint)}")
            >>> print(f"Total bits: {len(fingerprint)}")  # 2048
        """
        # Parse SMILES into RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        fp_type = self.fp_type
        p = self.params[fp_type]

        # Compute the specified fingerprint using new generators
        if fp_type == "Morgan":
            generator = GetMorganGenerator(radius=p["radius"], fpSize=p["nBits"])
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((p["nBits"],), dtype=int)
            ConvertToNumpyArray(fp, arr)
            return arr
        elif fp_type == "RDKit":
            fp = Chem.RDKFingerprint(mol, fpSize=p["fpSize"])
        elif fp_type == "MACCS":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == "TopologicalTorsion":
            generator = GetTopologicalTorsionGenerator(fpSize=p["nBits"])
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((p["nBits"],), dtype=int)
            ConvertToNumpyArray(fp, arr)
            return arr
        elif fp_type == "AtomPair":
            generator = GetAtomPairGenerator(fpSize=p["nBits"])
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((p["nBits"],), dtype=int)
            ConvertToNumpyArray(fp, arr)
            return arr
        else:
            # Should never happen
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

        # Convert bit vector to numpy array of ints (for RDKit and MACCS)
        bit_str = fp.ToBitString()
        arr = np.fromiter(bit_str, dtype=int)
        return arr


class Helm2Fasta(FormatTransform):
    """Convert HELM notation to FASTA sequence format.

    This converter parses HELM (Hierarchical Editing Language for Macromolecules)
    notation and converts it back to standard FASTA amino acid sequence format.

    Example:
        >>> converter = Helm2Fasta()
        >>> fasta = converter("PEPTIDE1{A.L.A.G.G.G.P.C.R}$$$$")
        >>> print(fasta)  # "ALAGGGPCR"
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.helm_parser = HelmParser(self.lib)
        self.fasta_serializer = FastaSerializer(self.lib)

    def __call__(self, helm: str) -> str:
        """Convert HELM notation to FASTA sequence.

        Args:
            helm (str): Valid HELM notation string.

        Returns:
            str: FASTA sequence representation without header.

        Raises:
            ValueError: If HELM string cannot be parsed or converted.
        """
        # Parse HELM notation into a structured format
        parsed_data = self.helm_parser.parse(helm)
        # Serialize back to FASTA format
        return self.fasta_serializer.serialize(parsed_data)


class Helm2Smiles(FormatTransform):
    """Convert HELM notation to SMILES representation.

    This converter parses HELM notation, builds the corresponding molecular
    structure, and generates the SMILES representation.

    Example:
        >>> converter = Helm2Smiles()
        >>> smiles = converter("PEPTIDE1{A.L.A.G.G.G.P.C.R}$$$$")
        >>> print(smiles)  # SMILES string
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.helm_parser = HelmParser(self.lib)

    def __call__(self, helm: str) -> str:
        """Convert HELM notation to SMILES string.

        Args:
            helm (str): Valid HELM notation string.

        Returns:
            str: SMILES representation of the molecule.

        Raises:
            ValueError: If HELM cannot be parsed or molecule cannot be built.
        """
        # Parse HELM notation into a structured format
        parsed_data = self.helm_parser.parse(helm)
        mol = MolBuilder(parsed_data).build()
        if mol is None:
            raise ValueError(f"Failed to build molecule from HELM: {helm}")
        return Chem.MolToSmiles(mol)


class Helm2Biln(FormatTransform):
    """Convert HELM notation to BiLN format.

    This converter transforms HELM (Hierarchical Editing Language) notation
    into BiLN (Biological Linear Notation) format.

    Example:
        >>> converter = Helm2Biln()
        >>> biln = converter("PEPTIDE1{A.L.A.G.G.G.P.C.R}$$$$")
        >>> print(biln)  # BiLN representation
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.helm_parser = HelmParser(self.lib)
        self.biln_serializer = BilnSerializer(self.lib)

    def __call__(self, helm: str) -> str:
        """Convert HELM notation to BiLN format.

        Args:
            helm (str): Valid HELM notation string.

        Returns:
            str: BiLN representation of the molecule.

        Raises:
            ValueError: If HELM string cannot be parsed.
        """
        # Parse HELM notation into a structured format
        parsed_data = self.helm_parser.parse(helm)
        # Serialize to BiLN format
        return self.biln_serializer.serialize(parsed_data)


class Biln2Fasta(FormatTransform):
    """Convert BiLN notation to FASTA sequence format.

    This converter parses BiLN (Biological Linear Notation) and converts
    it to standard FASTA amino acid sequence format.

    Example:
        >>> converter = Biln2Fasta()
        >>> fasta = converter("biln_notation_here")
        >>> print(fasta)  # "ALAGGGPCR"
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.biln_parser = BilnParser(self.lib)
        self.fasta_serializer = FastaSerializer(self.lib)

    def __call__(self, biln: str) -> str:
        """Convert BiLN notation to FASTA sequence.

        Args:
            biln (str): Valid BiLN notation string.

        Returns:
            str: FASTA sequence representation without header.

        Raises:
            ValueError: If BiLN string cannot be parsed.
        """
        # Parse BiLN notation into a structured format
        parsed_data = self.biln_parser.parse(biln)
        # Serialize back to FASTA format
        return self.fasta_serializer.serialize(parsed_data)


class Biln2Smiles(FormatTransform):
    """Convert BiLN notation to SMILES representation.

    This converter parses BiLN notation, builds the molecular structure,
    and generates the corresponding SMILES string.

    Example:
        >>> converter = Biln2Smiles()
        >>> smiles = converter("biln_notation_here")
        >>> print(smiles)  # SMILES string
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.biln_parser = BilnParser(self.lib)

    def __call__(self, biln: str) -> str:
        """Convert BiLN notation to SMILES string.

        Args:
            biln (str): Valid BiLN notation string.

        Returns:
            str: SMILES representation of the molecule.

        Raises:
            ValueError: If BiLN cannot be parsed or molecule cannot be built.
        """
        # Parse BiLN notation into a structured format
        parsed_data = BilnParser(self.lib).parse(biln)
        # Convert to SMILES using MolBuilder
        mol = MolBuilder(parsed_data).build()
        if mol is None:
            raise ValueError(f"Failed to build molecule from BiLN: {biln}")
        return Chem.MolToSmiles(mol)


class Biln2Helm(FormatTransform):
    """Convert BiLN notation to HELM format.

    This converter transforms BiLN (Biological Linear Notation) into
    HELM (Hierarchical Editing Language for Macromolecules) format.

    Example:
        >>> converter = Biln2Helm()
        >>> helm = converter("biln_notation_here")
        >>> print(helm)  # HELM notation
    """

    def __init__(self):
        self.lib = MonomerLibrary.from_sdf_file(
            "test_library",
            os.path.join(
                (os.path.dirname(__file__)),
                "..",
                "external",
                "pep",
                "resources",
                "monomers.sdf",
            ),
        )
        self.biln_parser = BilnParser(self.lib)
        self.helm_serializer = HelmSerializer(self.lib)

    def __call__(self, biln: str) -> str:
        """Convert BiLN notation to HELM format.

        Args:
            biln (str): Valid BiLN notation string.

        Returns:
            str: HELM representation of the molecule.

        Raises:
            ValueError: If BiLN string cannot be parsed.
        """
        # Parse BiLN notation into a structured format
        parsed_data = self.biln_parser.parse(biln)
        # Serialize to HELM format
        return self.helm_serializer.serialize(parsed_data)


class SMILES2Graph(FormatTransform):
    """Convert SMILES string into graph representation.

    This class is intended to convert SMILES notation into molecular graph
    representations suitable for graph neural networks and other graph-based
    machine learning applications.

    Note:
        This conversion is currently not implemented and serves as a placeholder
        for future development.

    Example:
        >>> converter = SMILES2Graph()
        >>> graph = converter("CCO")  # Returns None (not implemented)
    """

    def __call__(self, smiles: str):
        """Convert SMILES to graph representation (not implemented).

        Args:
            smiles (str): SMILES notation string.

        Returns:
            None: Conversion not yet implemented.
        """
        logger.warning("SMILES to Graph conversion not yet implemented")
        return None


AVAILABLE_TRANSFORM = {
    ("fasta", "smiles"): Fasta2Smiles,
    ("fasta", "helm"): Fasta2Helm,
    ("fasta", "biln"): Fasta2Biln,
    ("fasta", "embedding"): Fasta2Embedding,
    ("smiles", "graph"): SMILES2Graph,
    ("smiles", "fasta"): Smiles2Fasta,
    ("smiles", "helm"): Smiles2Helm,
    ("smiles", "biln"): Smiles2Biln,
    ("smiles", "fingerprint"): Smiles2FP,
    ("helm", "fasta"): Helm2Fasta,
    ("helm", "smiles"): Helm2Smiles,
    ("helm", "biln"): Helm2Biln,
    ("biln", "fasta"): Biln2Fasta,
    ("biln", "smiles"): Biln2Smiles,
    ("biln", "helm"): Biln2Helm,
}

if __name__ == "__main__":
    fasta = "ALAGGGPCR"
    print(f"Original FASTA: {fasta}")
    print("=" * 80)

    # Test FASTA conversions
    print("🧬 FASTA Conversion Tests:")
    print("-" * 40)

    # FASTA to SMILES
    try:
        fasta2smiles = Fasta2Smiles()
        smiles = fasta2smiles(fasta)
        print(f"FASTA → SMILES: {smiles}")
    except Exception as e:
        print(f"FASTA → SMILES conversion failed: {e}")

    # FASTA to HELM
    try:
        fasta2helm = Fasta2Helm()
        helm = fasta2helm(fasta)
        print(f"FASTA → HELM: {helm}")
    except Exception as e:
        print(f"FASTA → HELM conversion failed: {e}")

    # FASTA to BiLN
    try:
        fasta2biln = Fasta2Biln()
        biln = fasta2biln(fasta)
        print(f"FASTA → BiLN: {biln}")
    except Exception as e:
        print(f"FASTA → BiLN conversion failed: {e}")

    print("\n" + "=" * 80)

    # If SMILES conversion succeeded, test SMILES-related conversions
    try:
        smiles = fasta2smiles(fasta)
        print("⚗️ SMILES Conversion Tests:")
        print("-" * 40)

        # SMILES to fingerprints
        print("SMILES → Fingerprints:")
        for fp_type in Smiles2FP.available_fps:
            try:
                convert = Smiles2FP(fp_type=fp_type, radius=3, nBits=2048)
                fp = convert(smiles)
                print(f"{fp_type:15} → {len(fp)} bits, non-zero bits: {np.sum(fp)}")
            except Exception as e:
                print(f"  {fp_type:15} → conversion failed: {e}")

        # SMILES to other formats (not yet implemented)
        smiles2fasta = Smiles2Fasta()
        smiles2helm = Smiles2Helm()
        smiles2biln = Smiles2Biln()
        print(f"SMILES → FASTA: {smiles2fasta(smiles) or '(not implemented)'}")
        print(f"SMILES → HELM: {smiles2helm(smiles) or '(not implemented)'}")
        print(f"SMILES → BiLN: {smiles2biln(smiles) or '(not implemented)'}")

    except Exception as e:
        print(f"SMILES conversion tests skipped: {e}")

    print("\n" + "=" * 80)

    # If HELM conversion succeeded, test HELM-related conversions
    try:
        helm = fasta2helm(fasta)
        print("HELM Conversion Tests:")
        print("-" * 40)

        # HELM to other formats
        helm2fasta = Helm2Fasta()
        helm2smiles = Helm2Smiles()
        helm2biln = Helm2Biln()

        try:
            helm_to_fasta = helm2fasta(helm)
            print(f"HELM → FASTA: {helm_to_fasta}")
        except Exception as e:
            print(f"HELM → FASTA conversion failed: {e}")

        try:
            helm_to_smiles = helm2smiles(helm)
            print(f"HELM → SMILES: {helm_to_smiles}")
        except Exception as e:
            print(f"HELM → SMILES conversion failed: {e}")

        try:
            helm_to_biln = helm2biln(helm)
            print(f"HELM → BiLN: {helm_to_biln}")
        except Exception as e:
            print(f"HELM → BiLN conversion failed: {e}")

    except Exception as e:
        print(f"HELM conversion tests skipped: {e}")

    print("\n" + "=" * 80)

    # If BiLN conversion succeeded, test BiLN-related conversions
    try:
        biln = fasta2biln(fasta)
        print("🧪 BiLN Conversion Tests:")
        print("-" * 40)

        # BiLN to other formats
        biln2fasta = Biln2Fasta()
        biln2smiles = Biln2Smiles()
        biln2helm = Biln2Helm()

        try:
            biln_to_fasta = biln2fasta(biln)
            print(f"BiLN → FASTA: {biln_to_fasta}")
        except Exception as e:
            print(f"BiLN → FASTA conversion failed: {e}")

        try:
            biln_to_smiles = biln2smiles(biln)
            print(f"BiLN → SMILES: {biln_to_smiles}")
        except Exception as e:
            print(f"BiLN → SMILES conversion failed: {e}")

        try:
            biln_to_helm = biln2helm(biln)
            print(f"BiLN → HELM: {biln_to_helm}")
        except Exception as e:
            print(f"BiLN → HELM conversion failed: {e}")

    except Exception as e:
        print(f"BiLN conversion tests skipped: {e}")

    print("\n" + "=" * 80)

    # Test embedding generator (requires a model—shown here as example only)
    try:
        print("🧪 Embedding Conversion Tests:")
        embedding_generator = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
        embedding = embedding_generator(fasta)
        print(f"FASTA → Embedding: {embedding[:10]}... (length: {len(embedding)})")

    except Exception as e:
        print(f"Embedding conversion tests skipped: {e}")

    try:
        smiles2graph = SMILES2Graph()
        graph = smiles2graph("CCO")  # Ethanol
        print(f"SMILES → Graph: {graph}")  # Should return None (not implemented)
    except Exception as e:
        print(f"SMILES → Graph conversion failed: {e}")

    print("\nAll tests completed!")
