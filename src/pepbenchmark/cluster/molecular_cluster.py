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
Molecular clustering module for SMILES-based clustering.

This module provides specialized clustering functionality for molecular data,
including ECFP/Morgan fingerprint-based clustering with various algorithms.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from pepbenchmark.cluster.interfaces import AbstractClusterer, ClusterConfig, UnifiedClusterResult
from pepbenchmark.cluster.smilarity_cluster import SimilarityClusterer, SimilarityClusterConfig
from pepbenchmark.pep_utils.convert import Smiles2FP
from pepbenchmark.similarity.similarity import compute_similarity_matrix
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MolecularClusterConfig(ClusterConfig):
    """Configuration for molecular clustering algorithms."""
    fp_type: str = "Morgan"
    radius: int = 2
    nBits: int = 2048
    similarity_threshold: float = 0.8
    clustering_method: str = "connected"
    processes: int = 1
    
    # Similarity-based clustering parameters
    coverage_threshold: Optional[float] = None
    n_clusters: Optional[int] = None
    k: int = 50  # for ANN graph
    linkage_method: str = "single"  # for hierarchical clustering


class MolecularClusterer(AbstractClusterer):
    """
    Molecular clusterer specialized for SMILES data using molecular fingerprints.
    
    This clusterer:
    1. Converts SMILES to molecular fingerprints (ECFP/Morgan)
    2. Computes similarity matrices based on fingerprints
    3. Applies various clustering algorithms (connected components, hierarchical, etc.)
    
    Supports multiple fingerprint types and clustering methods for molecular similarity.
    """
    
    def __init__(self, config: MolecularClusterConfig):
        super().__init__(config)
        self.config: MolecularClusterConfig = config
        
        # Cache for fingerprints and similarity matrices
        self._cached_fingerprints: Optional[np.ndarray] = None
        self._cached_smiles: Optional[List[str]] = None
        
        logger.info(f"MolecularClusterer initialized: fp_type={config.fp_type}, "
                   f"radius={config.radius}, nBits={config.nBits}, "
                   f"method={config.clustering_method}, threshold={config.similarity_threshold}")
    
    def cluster_sequences(
        self,
        sequences: List[str],
        similarity_matrix: Optional[np.ndarray] = None,
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Cluster SMILES sequences using molecular fingerprints.
        
        Args:
            sequences: List of SMILES strings to cluster
            similarity_matrix: Optional precomputed similarity matrix
            labels: Optional labels (not used in molecular clustering)
            **kwargs: Additional clustering parameters
        
        Returns:
            UnifiedClusterResult with cluster assignments
        """
        if len(sequences) == 0:
            return self._create_empty_result()
        
        logger.info(f"Clustering {len(sequences)} SMILES using molecular fingerprints")
        
        # Compute fingerprints if needed
        if similarity_matrix is None:
            fingerprints = self._compute_molecular_fingerprints(sequences)
            similarity_matrix = self._compute_similarity_matrix(fingerprints)
        
        # Use SimilarityClusterer for the actual clustering
        similarity_config = SimilarityClusterConfig(
            similarity_threshold=self.config.similarity_threshold,
            coverage_threshold=self.config.coverage_threshold,
            method=self.config.clustering_method,
            n_clusters=self.config.n_clusters,
            k=self.config.k,
            linkage_method=self.config.linkage_method
        )
        
        similarity_clusterer = SimilarityClusterer(similarity_config)
        
        # Convert sequences to string indices for SimilarityClusterer
        sequence_ids = [str(i) for i in range(len(sequences))]
        
        result = similarity_clusterer.cluster_sequences(
            sequences=sequence_ids,
            similarity_matrix=similarity_matrix,
            **kwargs
        )
        
        # Update result metadata with molecular-specific information
        result.algorithm = f"MolecularCluster-{self.config.clustering_method}"
        result.parameters.update({
            "fp_type": self.config.fp_type,
            "radius": self.config.radius,
            "nBits": self.config.nBits,
            "processes": self.config.processes
        })
        
        if hasattr(self, '_cached_fingerprints') and self._cached_fingerprints is not None:
            result.metadata = result.metadata or {}
            result.metadata.update({
                "fingerprint_shape": self._cached_fingerprints.shape,
                "fingerprint_type": self.config.fp_type
            })
        
        logger.info(f"Molecular clustering completed: {result.total_clusters} clusters from {result.total_sequences} molecules")
        return result
    
    def _compute_molecular_fingerprints(self, smiles: List[str]) -> np.ndarray:
        """
        Compute molecular fingerprints for SMILES strings.
        
        Args:
            smiles: List of SMILES strings
            
        Returns:
            Numpy array of molecular fingerprints
        """
        # Check cache first
        if (self._cached_fingerprints is not None and 
            self._cached_smiles is not None and 
            self._cached_smiles == smiles):
            logger.info("Using cached molecular fingerprints")
            return self._cached_fingerprints

        converter = Smiles2FP(
            fp_type=self.config.fp_type,
            radius=self.config.radius,
            nBits=self.config.nBits,
            processes=self.config.processes
        )

        fps = converter(smiles)

        # Ensure numpy array format
        if isinstance(fps, list):
            if len(fps) > 0 and hasattr(fps[0], 'toarray'):
                # Convert sparse matrix list to dense array
                fps = np.array([fp.toarray().flatten() for fp in fps])
            else:
                fps = np.array(fps)

        # Cache results
        self._cached_fingerprints = fps
        self._cached_smiles = smiles.copy()

        logger.info(f"Computed {self.config.fp_type} fingerprints for {len(smiles)} molecules (shape: {fps.shape})")
        return fps
    
    def _compute_similarity_matrix(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix from molecular fingerprints.
        
        Args:
            fingerprints: Molecular fingerprint array
            
        Returns:
            Similarity matrix
        """
        logger.info("Computing molecular fingerprint similarity matrix")

        sim_matrix = compute_similarity_matrix(
            fingerprints, fingerprints,
            input_type="embedding",
            method="cosine",  # Cosine similarity for binary fingerprints (equivalent to Tanimoto)
            mode="full"
        )

        logger.info(f"Computed similarity matrix with shape {sim_matrix.shape}")
        return sim_matrix
    
    def _create_empty_result(self) -> UnifiedClusterResult:
        """Create empty result for edge cases."""
        return UnifiedClusterResult(
            cluster_assignments={},
            total_clusters=0,
            total_sequences=0,
            algorithm=f"MolecularCluster-{self.config.clustering_method}",
            parameters=self.config.to_dict()
        )
    
    def get_cached_fingerprints(self) -> Optional[np.ndarray]:
        """Get cached molecular fingerprints."""
        return self._cached_fingerprints
    
    def clear_cache(self) -> None:
        """Clear cached fingerprints and SMILES."""
        self._cached_fingerprints = None
        self._cached_smiles = None
        logger.info("Cleared molecular clustering cache")


def create_molecular_clusterer(
    fp_type: str = "Morgan",
    radius: int = 2,
    nBits: int = 2048,
    similarity_threshold: float = 0.8,
    clustering_method: str = "connected",
    processes: int = 1,
    **kwargs
) -> MolecularClusterer:
    """
    Factory function to create MolecularClusterer.
    
    Args:
        fp_type: Fingerprint type ("Morgan", "RDKit", "MACCS", etc.)
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprint
        similarity_threshold: Similarity threshold for clustering
        clustering_method: Clustering algorithm
        processes: Number of parallel processes
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured MolecularClusterer instance
    """
    config = MolecularClusterConfig(
        fp_type=fp_type,
        radius=radius,
        nBits=nBits,
        similarity_threshold=similarity_threshold,
        clustering_method=clustering_method,
        processes=processes,
        **kwargs
    )
    return MolecularClusterer(config)


if __name__ == "__main__":
    # Test code for development and validation
    test_smiles = [
        "CCO",  # Ethanol
        "CCO",  # Ethanol (duplicate)
        "CCCO",  # Propanol
        "CC(C)O",  # Isopropanol
        "c1ccccc1",  # Benzene
        "c1ccc2ccccc2c1"  # Naphthalene
    ]
    
    print("Testing molecular clustering with sample SMILES")
    
    try:
        # Test molecular clustering
        clusterer = create_molecular_clusterer(
            fp_type="Morgan",
            radius=2,
            similarity_threshold=0.8,
            clustering_method="connected"
        )
        
        result = clusterer.cluster_sequences(sequences=test_smiles)
        print(f"Clusters found: {result.total_clusters}")
        print(f"Cluster assignments: {result.cluster_assignments}")
        print(f"Algorithm: {result.algorithm}")
        print(f"Parameters: {result.parameters}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("Testing completed.")
