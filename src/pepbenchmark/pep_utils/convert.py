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

All converters support both single input and batch processing:

Example:
    Single input processing:
        >>> fasta2smiles = Fasta2Smiles()
        >>> smiles = fasta2smiles("ALAGGGPCR")
        >>> print(smiles)

    Batch processing:
        >>> fasta_list = ["ALAGGGPCR", "PEPTIDE"]
        >>> smiles_list = fasta2smiles(fasta_list)
        >>> print(smiles_list)  # Returns list of SMILES strings

    Molecular fingerprint generation:
        >>> fp_generator = Smiles2FP(fp_type='Morgan', radius=3, nBits=2048)
        >>> # Single fingerprint
        >>> fingerprint = fp_generator(smiles)
        >>> print(f"Fingerprint length: {len(fingerprint)}")
        >>> # Batch fingerprints
        >>> fingerprints = fp_generator(["CCO", "CC(=O)O"])
        >>> print(f"Number of fingerprints: {len(fingerprints)}")

    Neural embedding:
        >>> embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
        >>> # Single embedding
        >>> embedding = embedder("ALAGGGPCR")
        >>> print(f"Embedding shape: {embedding.shape}")
        >>> # Batch embeddings
        >>> embeddings = embedder(["ALAGGGPCR", "PEPTIDE"])
        >>> print(f"Number of embeddings: {len(embeddings)}")
"""

# Configure logging for this module``
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from ogb.utils import smiles2graph
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetTopologicalTorsionGenerator,
)
from rdkit.DataStructs import ConvertToNumpyArray
from torch_geometric.data import Data
from tqdm import tqdm
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

    Attributes:
        desc (str): Description for progress bar (set by subclasses)

    Methods:
        __call__: Abstract method that performs the actual transformation
        _process_single: Process a single input item
        _process_batch: Handle batch processing
    """

    def __init__(self):
        self.desc = "Processing batch"

    def __call__(
        self, inputs: Union[Any, List[Any]], **kwargs: Any
    ) -> Union[Any, List[Any]]:
        """Perform the format transformation on single input or batch.

        Args:
            inputs: Single input item or list of input items
            **kwargs: Arbitrary keyword arguments specific to each transformation

        Returns:
            Single output or list of outputs matching input format

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        return self._process_batch(inputs, **kwargs)

    def _process_batch(self, inputs: List[Any], **kwargs: Any) -> List[Any]:
        """Handle batch processing.

        Args:
            inputs: Single input or list of inputs
            **kwargs: Arguments to pass to _process_single

        Returns:
            Single output or list of outputs
        """
        if isinstance(inputs, (list, tuple)):
            # Batch processing with progress bar using class-specific description
            results = []
            for item in tqdm(inputs, desc=self.desc):
                results.append(self._process_single(item, **kwargs))
            return results
        else:
            # Single item processing
            return self._process_single(inputs, **kwargs)

    def _process_single(self, input_item: Any, **kwargs: Any) -> Any:
        """Process a single input item.

        Args:
            input_item: Single input item to process
            **kwargs: Additional arguments for processing

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _process_single method.")


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

        >>> # Batch processing
        >>> smiles_list = converter(["ALAGGGPCR", "PEPTIDE"])
        >>> print(smiles_list)  # Returns list of SMILES strings
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting FASTA to SMILES"

    def _process_single(self, fasta: str) -> str:
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
            >>> smiles = converter._process_single(">peptide1\\nALAGGGPCR")
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

        >>> # Batch processing
        >>> embeddings = embedder(["ALAGGGPCR", "PEPTIDE"])
        >>> print(len(embeddings))  # 2
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        device: Optional[str] = None,
        pooling: str = "mean",
    ):
        super().__init__()
        self.desc = "Generating embeddings"

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
        else:
            raise ValueError(
                "`model` must be a HuggingFace model identifier string or a torch.nn.Module instance."
            )

        # Prepare model
        self.model.to(self.device)
        self.model.eval()

    def _process_single(self, fasta: str) -> np.ndarray:
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
            emb = hidden.max(dim=1).squeeze(0)
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

        >>> # Batch processing
        >>> helm_list = converter(["ALAGGGPCR", "PEPTIDE"])
        >>> print(helm_list)  # List of HELM notation strings
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting FASTA to HELM"

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

    def _process_single(self, fasta: str) -> str:
        """Convert FASTA sequence to HELM notation.

        Args:
            fasta (str): FASTA-formatted string containing peptide sequence.
                Can include header lines (starting with '>') which will be ignored.

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

        >>> # Batch processing
        >>> biln_list = converter(["ALAGGGPCR", "PEPTIDE"])
        >>> print(biln_list)  # List of BiLN notation strings
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting FASTA to BiLN"

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

    def _process_single(self, fasta: str) -> str:
        """Convert FASTA sequence to BiLN representation.

        Args:
            fasta (str): FASTA-formatted string containing peptide sequence.
                Can include header lines (starting with '>') which will be ignored.

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

    def __init__(self):
        super().__init__()
        self.desc = "Converting SMILES to FASTA"

    def _process_single(self, smiles: str) -> str:
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

    def __init__(self):
        super().__init__()
        self.desc = "Converting SMILES to HELM"

    def _process_single(self, smiles: str) -> str:
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

    def __init__(self):
        super().__init__()
        self.desc = "Converting SMILES to BiLN"

    def _process_single(self, smiles: str) -> str:
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

        >>> # Batch processing
        >>> fp_list = fp_gen(['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O'])
        >>> print(len(fp_list))  # 2
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
        super().__init__()
        self.desc = f"Generating {fp_type} fingerprints"

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

    def _process_single(self, smiles: str) -> np.ndarray:
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
            >>> fingerprint = fp_gen._process_single('CC(=O)OC1=CC=CC=C1C(=O)O')  # aspirin
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

        >>> # Batch processing
        >>> fasta_list = converter(["PEPTIDE1{A.L.A.G.G.G.P.C.R}$$$$", "HELM2"])
        >>> print(fasta_list)  # List of FASTA sequences
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting HELM to FASTA"

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

    def _process_single(self, helm: str) -> str:
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

        >>> # Batch processing
        >>> smiles_list = converter(["PEPTIDE1{A.L.A.G.G.G.P.C.R}$$$$", "HELM2"])
        >>> print(smiles_list)  # List of SMILES strings
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting HELM to SMILES"

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

    def _process_single(self, helm: str) -> str:
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

        >>> # Batch processing
        >>> biln_list = converter(["PEPTIDE1{A.L.A.G.G.G.P.C.R}$$$$", "HELM2"])
        >>> print(biln_list)  # List of BiLN representations
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting HELM to BiLN"

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

    def _process_single(self, helm: str) -> str:
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

        >>> # Batch processing
        >>> fasta_list = converter(["biln1", "biln2"])
        >>> print(fasta_list)  # List of FASTA sequences
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting BiLN to FASTA"

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

    def _process_single(self, biln: str) -> str:
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

        >>> # Batch processing
        >>> smiles_list = converter(["biln1", "biln2"])
        >>> print(smiles_list)  # List of SMILES strings
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting BiLN to SMILES"

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

    def _process_single(self, biln: str) -> str:
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

        >>> # Batch processing
        >>> helm_list = converter(["biln1", "biln2"])
        >>> print(helm_list)  # List of HELM notations
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting BiLN to HELM"

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

    def _process_single(self, biln: str) -> str:
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


class Smiles2Graph(FormatTransform):
    """Convert SMILES notation to graph representation.

    This converter transforms a SMILES (Simplified Molecular-Input Line-Entry System)
    string into a pyg graph format suitable for machine learning tasks.

    Example:
        >>> converter = Smiles2Graph()
        >>> graph = converter("CCO")
        >>> print(graph)  # PyTorch Geometric Data object

        >>> # With label
        >>> labeled_graph = converter("CCO", label=torch.tensor([1]))
        >>> print(labeled_graph.y)  # tensor([1])

        >>> # Batch processing
        >>> graph_list = converter(["CCO", "CCC"])
        >>> print(graph_list)  # List of PyTorch Geometric Data objects
    """

    def __init__(self):
        super().__init__()
        self.desc = "Converting SMILES to graph representation"

    def _process_single(
        self, smiles: str, label: Optional[torch.Tensor] = None
    ) -> Data:
        """Convert SMILES string to graph representation (PyTorch Geometric Data object).

        Args:
            smiles (str): SMILES string of the molecule.
            label (torch.Tensor, optional): Label tensor for the graph.

        Returns:
            Data: Graph representation with nodes and edges (PyTorch Geometric Data object).
        """
        # Convert SMILES to graph format by ogb
        graph_data = smiles2graph(smiles)

        # Create a PyTorch Geometric Data object
        graph = Data(
            x=torch.from_numpy(graph_data["node_feat"]),
            edge_index=torch.from_numpy(graph_data["edge_index"]),
            edge_attr=torch.from_numpy(graph_data["edge_feat"]),
        )

        # If a label is provided, assign it to the graph
        if label is not None:
            graph.y = label

        return graph

    def __call__(
        self, inputs: Union[str, List[str]], label: Optional[torch.Tensor] = None
    ) -> Union[Data, List[Data]]:
        """Call method to handle both single and batch inputs.

        Args:
            inputs (str or list of str): Single SMILES string or a list of SMILES strings.
            label (torch.Tensor, optional): Label tensor for a single graph.

        Returns:
            dict or list of dict: Single or list of graph representations.
        """
        if isinstance(inputs, (list, tuple)):
            return [self._process_single(smiles) for smiles in inputs]
        else:
            return self._process_single(inputs, label=label)


if __name__ == "__main__":
    fasta = "ALAGGGPCR"
    fasta_list = ["ALAGGGPCR", "PEPTIDE"]
    print(f"Original FASTA: {fasta}")
    print(f"FASTA List: {fasta_list}")
    print("=" * 80)

    # Test FASTA conversions - both single and batch
    print("🧬 FASTA Conversion Tests:")
    print("-" * 40)

    # FASTA to SMILES
    fasta2smiles = Fasta2Smiles()
    smiles = fasta2smiles(fasta)
    smiles_list = fasta2smiles(fasta_list)
    print(f"FASTA → SMILES (single): {smiles}")
    print(f"FASTA → SMILES (batch): {smiles_list}")

    # FASTA to HELM
    fasta2helm = Fasta2Helm()
    helm = fasta2helm(fasta)
    helm_list = fasta2helm(fasta_list)
    print(f"FASTA → HELM (single): {helm}")
    print(f"FASTA → HELM (batch): {helm_list}")

    # FASTA to BiLN
    fasta2biln = Fasta2Biln()
    biln = fasta2biln(fasta)
    biln_list = fasta2biln(fasta_list)
    print(f"FASTA → BiLN (single): {biln}")
    print(f"FASTA → BiLN (batch): {biln_list}")

    print("\n" + "=" * 80)

    # SMILES Conversion Tests
    smiles = fasta2smiles(fasta)
    smiles_list = ["CCO", "CC(=O)O"]
    print("⚗️ SMILES Conversion Tests:")
    print("-" * 40)

    # SMILES to fingerprints
    print("SMILES → Fingerprints:")
    for fp_type in Smiles2FP.available_fps:
        convert = Smiles2FP(fp_type=fp_type, radius=3, nBits=2048)
        fp = convert(smiles)
        fp_list = convert(smiles_list)
        print(f"{fp_type:15} → Single: {len(fp)} bits, non-zero: {np.sum(fp)}")
        print(
            f"{fp_type:15} → Batch: {len(fp_list)} items, first non-zero: {np.sum(fp_list[0])}"
        )

    # SMILES to other formats
    smiles2fasta = Smiles2Fasta()
    smiles2helm = Smiles2Helm()
    smiles2biln = Smiles2Biln()
    print(f"SMILES → FASTA (single): {smiles2fasta(smiles) or '(not implemented)'}")
    print(f"SMILES → FASTA (batch): {smiles2fasta(smiles_list) or '(not implemented)'}")
    print(f"SMILES → HELM (single): {smiles2helm(smiles) or '(not implemented)'}")
    print(f"SMILES → BiLN (single): {smiles2biln(smiles) or '(not implemented)'}")

    print("\n" + "=" * 80)

    # HELM Conversion Tests
    helm = fasta2helm(fasta)
    helm_list = fasta2helm(fasta_list)
    print("🧪 HELM Conversion Tests:")
    print("-" * 40)

    helm2fasta = Helm2Fasta()
    helm2smiles = Helm2Smiles()
    helm2biln = Helm2Biln()

    helm_to_fasta = helm2fasta(helm)
    helm_to_fasta_list = helm2fasta(helm_list)
    print(f"HELM → FASTA (single): {helm_to_fasta}")
    print(f"HELM → FASTA (batch): {helm_to_fasta_list}")

    helm_to_smiles = helm2smiles(helm)
    print(f"HELM → SMILES (single): {helm_to_smiles}")

    helm_to_biln = helm2biln(helm)
    print(f"HELM → BiLN (single): {helm_to_biln}")

    print("\n" + "=" * 80)

    # BiLN Conversion Tests
    biln = fasta2biln(fasta)
    biln_list = fasta2biln(fasta_list)
    print("🧪 BiLN Conversion Tests:")
    print("-" * 40)

    biln2fasta = Biln2Fasta()
    biln2smiles = Biln2Smiles()
    biln2helm = Biln2Helm()

    biln_to_fasta = biln2fasta(biln)
    biln_to_fasta_list = biln2fasta(biln_list)
    print(f"BiLN → FASTA (single): {biln_to_fasta}")
    print(f"BiLN → FASTA (batch): {biln_to_fasta_list}")

    biln_to_smiles = biln2smiles(biln)
    print(f"BiLN → SMILES (single): {biln_to_smiles}")

    biln_to_helm = biln2helm(biln)
    print(f"BiLN → HELM (single): {biln_to_helm}")

    print("\n" + "=" * 80)

    # Embedding Conversion Tests
    print("🧪 Embedding Conversion Tests:")
    embedding_generator = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
    embedding = embedding_generator(fasta)
    embedding_list = embedding_generator(fasta_list)
    print(f"FASTA → Embedding (single): {embedding[:10]}... (length: {len(embedding)})")
    print(
        f"FASTA → Embedding (batch): {len(embedding_list)} items, first shape: {embedding_list[0].shape}"
    )

    smi2graph = Smiles2Graph()
    graph = smi2graph("CCO")
    graph_list = smi2graph(["CCO", "CC(=O)O"])
    print(f"SMILES → Graph (single): {graph}")
    print(f"SMILES → Graph (batch): {graph_list}")

    print(
        "\nAll tests completed! All converters now support both single input and batch processing."
    )
