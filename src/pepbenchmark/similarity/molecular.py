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
Molecular similarity computation module.

This module provides specialized similarity computation functionality for molecular data,
including SMILES-based fingerprint similarity calculations.
"""

from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from pepbenchmark.pep_utils.convert import FormatTransform, Smiles2FP
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level fingerprint similarity helpers (previously in fp.py)
# ---------------------------------------------------------------------------

def _to_explicit_bv(fp) -> ExplicitBitVect:
    """Ensure fingerprint is RDKit ExplicitBitVect."""
    if fp is None:
        return None
    if hasattr(fp, "ToBitString"):
        return fp
    bitstring = "".join("1" if int(x) else "0" for x in fp)
    return DataStructs.CreateFromBitString(bitstring)


def _sim_pair(f1: ExplicitBitVect, f2: ExplicitBitVect, metric: str) -> float:
    if metric == "tanimoto":
        return DataStructs.TanimotoSimilarity(f1, f2)
    elif metric == "dice":
        return DataStructs.DiceSimilarity(f1, f2)
    elif metric == "cosine":
        return DataStructs.CosineSimilarity(f1, f2)
    elif metric == "sokal":
        return DataStructs.SokalSimilarity(f1, f2)
    raise ValueError(f"Unsupported similarity metric: {metric}")


def fingerprint_similarity(
    mols1: Optional[List[str]] = None,
    mols2: Optional[List[str]] = None,
    fps1: Optional[List] = None,
    fps2: Optional[List] = None,
    fp_gen: Optional[FormatTransform] = None,
    sim_metric: str = "tanimoto",
) -> np.ndarray:
    """Compute fingerprint similarity matrix.

    Args:
        mols1: List of molecules (SMILES). Required if fps1 not provided.
        mols2: List of molecules (SMILES). If None, self-comparison on mols1.
        fps1: Precomputed fingerprints for mols1 (ExplicitBitVect or ndarray).
        fps2: Precomputed fingerprints for mols2.
        fp_gen: Fingerprint generator (FormatTransform). Required if fps not provided.
        sim_metric: "tanimoto", "dice", "cosine", "sokal".

    Returns:
        np.ndarray of shape (n1, n2).
    """
    if fps1 is None:
        if fp_gen is None or mols1 is None:
            raise ValueError("Must provide fps1 or (mols1 + fp_gen)")
        fps1 = [fp_gen(m) for m in mols1]

    if mols2 is None and fps2 is None:
        mols2, fps2 = mols1, fps1

    if fps2 is None:
        if fp_gen is None or mols2 is None:
            raise ValueError("Must provide fps2 or (mols2 + fp_gen)")
        fps2 = [fp_gen(m) for m in mols2]

    fps1 = [_to_explicit_bv(fp) for fp in fps1]
    fps2 = [_to_explicit_bv(fp) for fp in fps2]
    metric = sim_metric.lower()

    n1, n2 = len(fps1), len(fps2)
    mat = np.zeros((n1, n2), dtype=np.float32)
    for i, f1 in enumerate(fps1):
        for j, f2 in enumerate(fps2):
            mat[i, j] = _sim_pair(f1, f2, metric)
    return mat


class FingerprintType(str, Enum):
    """Supported molecular fingerprint types."""
    MORGAN = "Morgan"
    RDKIT = "RDKit"
    MACCS = "MACCS"
    TOPOLOGICAL_TORSION = "TopologicalTorsion"
    ATOM_PAIR = "AtomPair"


class MolecularSimilarityMethod(str, Enum):
    """Supported molecular similarity methods."""
    TANIMOTO = "tanimoto"
    COSINE = "cosine"
    DICE = "dice"
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"


def compute_molecular_fingerprints(
    smiles: List[str],
    fp_type: Union[str, FingerprintType] = FingerprintType.MORGAN,
    radius: int = 2,
    nBits: int = 2048,
    processes: int = 1,
    **kwargs: Any
) -> np.ndarray:
    """
    Compute molecular fingerprints from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        fp_type: Fingerprint type
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprint
        processes: Number of parallel processes
        **kwargs: Additional parameters for fingerprint computation
        
    Returns:
        Numpy array of molecular fingerprints, shape (n_molecules, nBits)
        
    Raises:
        ImportError: If molecular fingerprint dependencies are not available
        ValueError: If invalid parameters are provided
    """
    if len(smiles) == 0:
        return np.array([]).reshape(0, nBits)

    # Ensure fp_type is string
    if isinstance(fp_type, FingerprintType):
        fp_type = fp_type.value

    converter = Smiles2FP(
        fp_type=fp_type,
        radius=radius,
        nBits=nBits,
        processes=processes,
        **kwargs
    )

    fps = converter(smiles)

    # Ensure numpy array format
    if isinstance(fps, list):
        if len(fps) > 0 and hasattr(fps[0], 'toarray'):
            # Convert sparse matrix list to dense array
            fps = np.array([fp.toarray().flatten() for fp in fps])
        else:
            fps = np.array(fps)

    logger.info(f"Computed {fp_type} fingerprints for {len(smiles)} molecules (shape: {fps.shape})")
    return fps


def compute_molecular_similarity_matrix(
    smiles1: List[str],
    smiles2: Optional[List[str]] = None,
    fp_type: Union[str, FingerprintType] = FingerprintType.MORGAN,
    similarity_method: Union[str, MolecularSimilarityMethod] = MolecularSimilarityMethod.TANIMOTO,
    radius: int = 2,
    nBits: int = 2048,
    processes: int = 1,
    mode: str = "full",
    **kwargs: Any
) -> Tuple[np.ndarray, dict]:
    """
    Compute molecular similarity matrix from SMILES strings.
    
    Args:
        smiles1: First set of SMILES strings
        smiles2: Optional second set of SMILES strings (if None, compute self-similarity)
        fp_type: Fingerprint type
        similarity_method: Similarity computation method
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprint
        processes: Number of parallel processes
        mode: Matrix computation mode ("full" or "blockwise")
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (similarity_matrix, metadata_dict)
        
    Raises:
        ImportError: If required dependencies are not available
        ValueError: If invalid parameters are provided
    """
    if len(smiles1) == 0:
        return np.array([]).reshape(0, 0), {}
    
    if isinstance(fp_type, FingerprintType):
        fp_type = fp_type.value
    if isinstance(similarity_method, MolecularSimilarityMethod):
        similarity_method = similarity_method.value

    fps1 = compute_molecular_fingerprints(
        smiles1,
        fp_type=fp_type,
        radius=radius,
        nBits=nBits,
        processes=processes,
        **kwargs,
    )

    if smiles2 is not None and len(smiles2) > 0:
        fps2 = compute_molecular_fingerprints(
            smiles2,
            fp_type=fp_type,
            radius=radius,
            nBits=nBits,
            processes=processes,
            **kwargs,
        )
    else:
        smiles2 = smiles1
        fps2 = fps1

    metric_alias = {
        MolecularSimilarityMethod.TANIMOTO.value: "tanimoto",
        MolecularSimilarityMethod.JACCARD.value: "tanimoto",
        MolecularSimilarityMethod.COSINE.value: "cosine",
        MolecularSimilarityMethod.DICE.value: "dice",
        MolecularSimilarityMethod.EUCLIDEAN.value: "cosine",
    }
    backend_metric = metric_alias.get(similarity_method, "tanimoto")

    sim_matrix = fingerprint_similarity(
        mols1=smiles1,
        mols2=smiles2,
        fps1=fps1,
        fps2=fps2,
        sim_metric=backend_metric,
    )
    metadata = {
        "fingerprint_type": fp_type,
        "radius": radius,
        "nBits": nBits,
        "similarity_method": similarity_method,
        "backend_metric": backend_metric,
        "mode": mode,
        "n_smiles1": len(smiles1),
        "n_smiles2": len(smiles2),
    }

    logger.info(
        "Computed molecular similarity matrix (%s) with shape %s",
        similarity_method,
        sim_matrix.shape,
    )
    return sim_matrix, metadata


def compute_tanimoto_similarity(
    smiles1: List[str],
    smiles2: Optional[List[str]] = None,
    fp_type: Union[str, FingerprintType] = FingerprintType.MORGAN,
    radius: int = 2,
    nBits: int = 2048,
    processes: int = 1,
    **kwargs: Any
) -> np.ndarray:
    """
    Compute Tanimoto similarity matrix for molecular fingerprints.
    
    This is a convenience function specifically for Tanimoto similarity,
    which is the most commonly used similarity measure for molecular fingerprints.
    
    Args:
        smiles1: First set of SMILES strings
        smiles2: Optional second set of SMILES strings
        fp_type: Fingerprint type
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprint
        processes: Number of parallel processes
        **kwargs: Additional parameters
        
    Returns:
        Tanimoto similarity matrix
    """
    sim_matrix, _ = compute_molecular_similarity_matrix(
        smiles1=smiles1,
        smiles2=smiles2,
        fp_type=fp_type,
        similarity_method=MolecularSimilarityMethod.TANIMOTO,
        radius=radius,
        nBits=nBits,
        processes=processes,
        **kwargs
    )
    return sim_matrix


def find_similar_molecules(
    query_smiles: str,
    database_smiles: List[str],
    similarity_threshold: float = 0.8,
    fp_type: Union[str, FingerprintType] = FingerprintType.MORGAN,
    similarity_method: Union[str, MolecularSimilarityMethod] = MolecularSimilarityMethod.TANIMOTO,
    radius: int = 2,
    nBits: int = 2048,
    processes: int = 1,
    top_k: Optional[int] = None,
    **kwargs: Any
) -> List[Tuple[int, str, float]]:
    """
    Find similar molecules to a query molecule in a database.
    
    Args:
        query_smiles: Query SMILES string
        database_smiles: Database of SMILES strings to search
        similarity_threshold: Minimum similarity threshold
        fp_type: Fingerprint type
        similarity_method: Similarity computation method
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprint
        processes: Number of parallel processes
        top_k: Optional limit on number of results to return
        **kwargs: Additional parameters
        
    Returns:
        List of tuples (index, smiles, similarity_score) sorted by similarity (descending)
    """
    if len(database_smiles) == 0:
        return []
    
    # Compute similarity matrix
    sim_matrix, _ = compute_molecular_similarity_matrix(
        smiles1=[query_smiles],
        smiles2=database_smiles,
        fp_type=fp_type,
        similarity_method=similarity_method,
        radius=radius,
        nBits=nBits,
        processes=processes,
        **kwargs
    )
    
    # Extract similarities for the query molecule
    similarities = sim_matrix[0, :]
    
    # Find molecules above threshold
    similar_indices = np.where(similarities >= similarity_threshold)[0]
    
    # Create result list with (index, smiles, similarity)
    results = [
        (int(idx), database_smiles[idx], float(similarities[idx]))
        for idx in similar_indices
    ]
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Limit results if requested
    if top_k is not None and top_k > 0:
        results = results[:top_k]
    
    logger.info(f"Found {len(results)} similar molecules (threshold: {similarity_threshold})")
    return results


# Convenience functions for different fingerprint types
def compute_morgan_similarity(
    smiles1: List[str],
    smiles2: Optional[List[str]] = None,
    radius: int = 2,
    nBits: int = 2048,
    **kwargs: Any
) -> np.ndarray:
    """Compute Morgan fingerprint Tanimoto similarity."""
    return compute_tanimoto_similarity(
        smiles1, smiles2, fp_type=FingerprintType.MORGAN, radius=radius, nBits=nBits, **kwargs
    )


def compute_rdkit_similarity(
    smiles1: List[str],
    smiles2: Optional[List[str]] = None,
    nBits: int = 2048,
    **kwargs: Any
) -> np.ndarray:
    """Compute RDKit fingerprint Tanimoto similarity."""
    return compute_tanimoto_similarity(
        smiles1, smiles2, fp_type=FingerprintType.RDKIT, nBits=nBits, **kwargs
    )


def compute_maccs_similarity(
    smiles1: List[str],
    smiles2: Optional[List[str]] = None,
    **kwargs: Any
) -> np.ndarray:
    """Compute MACCS fingerprint Tanimoto similarity."""
    return compute_tanimoto_similarity(
        smiles1, smiles2, fp_type=FingerprintType.MACCS, nBits=167, **kwargs
    )


if __name__ == "__main__":
    # Test code for development and validation
    test_smiles = [
        "CCO",  # Ethanol
        "CCCO",  # Propanol
        "CC(C)O",  # Isopropanol
        "c1ccccc1",  # Benzene
        "c1ccc2ccccc2c1"  # Naphthalene
    ]
    
    print("Testing molecular similarity computation")
    
    try:
        # Test fingerprint computation
        fps = compute_molecular_fingerprints(test_smiles, fp_type="Morgan")
        print(f"Computed fingerprints shape: {fps.shape}")
        
        # Test similarity matrix computation
        sim_matrix, metadata = compute_molecular_similarity_matrix(test_smiles)
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        print(f"Metadata: {metadata}")
        
        # Test molecule search
        query = "CCO"
        similar = find_similar_molecules(query, test_smiles, similarity_threshold=0.5)
        print(f"Similar molecules to {query}: {similar}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("Testing completed.")
