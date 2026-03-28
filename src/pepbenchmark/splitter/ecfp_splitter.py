"""
ECFP-based molecular fingerprint splitter for homology-aware splitting.

This module provides ECFPSplitter, which uses molecular fingerprints (ECFP/Morgan)
to cluster SMILES strings and create similarity-aware data splits. The clustering
logic has been moved to the cluster module for better interface isolation.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from pepbenchmark.cluster.interfaces import UnifiedClusterResult
from pepbenchmark.cluster.molecular_cluster import create_molecular_clusterer
from pepbenchmark.similarity.molecular import (
    compute_molecular_fingerprints,
    compute_molecular_similarity_matrix,
    find_similar_molecules,
)
from pepbenchmark.utils.logging import get_logger

# Import the unified base class
from pepbenchmark.splitter.base_splitter import AbstractClusteringSplitter
_BASE_CLASS = AbstractClusteringSplitter

logger = get_logger(__name__)


class ECFPSplitter(_BASE_CLASS):
    """
    ECFP-based molecular fingerprint splitter for homology-aware splitting.

    This splitter uses molecular clustering to ensure that similar molecules are placed in the
    same split, preventing data leakage in molecular property prediction tasks.
    
    The clustering logic is delegated to the MolecularClusterer in the cluster module,
    while this class focuses on the splitting functionality and parameter management.
    """

    def __init__(
        self,
        fp_type: str = "Morgan",
        radius: int = 3,
        nBits: int = 1024,
        similarity_threshold: float = 0.8,
        clustering_method: str = "connected",
        processes: int = 1,
        random_seed: Optional[int] = 42,
        verbose: bool = False
    ):
        """
        Initialize ECFPSplitter with molecular fingerprint parameters.
        
        Args:
            fp_type: Fingerprint type ("Morgan", "RDKit", "MACCS", "TopologicalTorsion", "AtomPair")
            radius: Radius for Morgan fingerprints (default: 2)
            nBits: Number of bits for fingerprint (default: 2048)
            similarity_threshold: Similarity threshold for clustering (default: 0.8)
            clustering_method: Clustering algorithm ("connected", "hierarchical", "threshold", "ann_graph")
            processes: Number of parallel processes for fingerprint calculation (default: 1)
            random_seed: Random seed for reproducibility (default: 42)
            verbose: Enable verbose logging (default: False)
        """
        # Initialize parent class
        super().__init__(random_seed=random_seed)
        
        # Store molecular fingerprint parameters
        self.fp_type = fp_type
        self.radius = radius
        self.nBits = nBits
        self.similarity_threshold = similarity_threshold
        self.clustering_method = clustering_method
        self.processes = processes
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Cache for clustering results
        self._last_cluster_result: Optional[UnifiedClusterResult] = None
        
        logger.info(f"ECFPSplitter initialized: fp_type={fp_type}, radius={radius}, "
                   f"nBits={nBits}, similarity_threshold={similarity_threshold}, "
                   f"clustering_method={clustering_method}")
        
    def _get_clustering_result(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Get clustering result using molecular fingerprint clustering.
        
        This method delegates the actual clustering to the MolecularClusterer
        in the cluster module, providing better separation of concerns.
        
        Args:
            sequences: List of SMILES strings to cluster
            labels: Optional labels (not used for clustering but passed through)
            **kwargs: Additional molecular clustering parameters
            
        Returns:
            UnifiedClusterResult containing clustering information
        """
        logger.info(f"Performing molecular fingerprint clustering on {len(sequences)} SMILES")

        # Update parameters if provided in kwargs
        clustering_params = {
            'fp_type': kwargs.get('fp_type', self.fp_type),
            'radius': kwargs.get('radius', self.radius),
            'nBits': kwargs.get('nBits', self.nBits),
            'similarity_threshold': kwargs.get('similarity_threshold', self.similarity_threshold),
            'clustering_method': kwargs.get('clustering_method', self.clustering_method),
            'processes': kwargs.get('processes', self.processes)
        }

        # Create molecular clusterer
        clusterer = create_molecular_clusterer(**clustering_params)

        # Perform clustering
        cluster_result = clusterer.cluster_sequences(
            sequences=sequences,
            labels=labels,
            **kwargs
        )

        # Cache result
        self._last_cluster_result = cluster_result

        logger.info(f"ECFP clustering completed: {cluster_result.total_clusters} clusters")
        return cluster_result
    
    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the last clustering result.
        
        Returns:
            Dictionary with clustering statistics, or None if not computed yet
        """
        if self._last_cluster_result is None:
            return None
        
        return {
            "total_clusters": self._last_cluster_result.total_clusters,
            "total_sequences": self._last_cluster_result.total_sequences,
            "algorithm": self._last_cluster_result.algorithm,
            "parameters": self._last_cluster_result.parameters,
            "cluster_sizes": {
                cluster_id: len(indices) 
                for cluster_id, indices in self._last_cluster_result.cluster_assignments.items()
            }
        }
    
    def compute_similarity_matrix(self, smiles: List[str]) -> np.ndarray:
        """
        Compute full similarity matrix for SMILES data.
        
        This method delegates to the molecular similarity module for better
        separation of concerns.
        
        Args:
            smiles: List of SMILES strings
            
        Returns:
            N×N similarity matrix
        """
        sim_matrix, _ = compute_molecular_similarity_matrix(
            smiles1=smiles,
            smiles2=None,  # Self-similarity
            fp_type=self.fp_type,
            similarity_method="tanimoto",
            radius=self.radius,
            nBits=self.nBits,
            processes=self.processes,
            mode="full"
        )
        return sim_matrix
    
    def find_similar_molecules(
        self,
        query_smiles: str,
        database_smiles: List[str],
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Find similar molecules to a query molecule.
        
        Args:
            query_smiles: Query SMILES string
            database_smiles: Database of SMILES strings to search
            similarity_threshold: Minimum similarity threshold (uses instance default if None)
            top_k: Optional limit on number of results to return
            
        Returns:
            List of tuples (index, smiles, similarity_score) sorted by similarity
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        return find_similar_molecules(
            query_smiles=query_smiles,
            database_smiles=database_smiles,
            similarity_threshold=similarity_threshold,
            fp_type=self.fp_type,
            similarity_method="tanimoto",
            radius=self.radius,
            nBits=self.nBits,
            processes=self.processes,
            top_k=top_k
        )
    
    def get_molecular_fingerprints(self, smiles: List[str]) -> np.ndarray:
        """
        Compute molecular fingerprints for SMILES strings.
        
        Args:
            smiles: List of SMILES strings
            
        Returns:
            Numpy array of molecular fingerprints
        """
        return compute_molecular_fingerprints(
            smiles=smiles,
            fp_type=self.fp_type,
            radius=self.radius,
            nBits=self.nBits,
            processes=self.processes
        )


# Convenience factory function
def create_ecfp_splitter(
    fp_type: str = "Morgan",
    radius: int = 2,
    nBits: int = 2048,
    similarity_threshold: float = 0.8,
    clustering_method: str = "connected",
    processes: int = 1,
    random_seed: Optional[int] = 42,
    **kwargs
) -> ECFPSplitter:
    """
    Factory function to create ECFPSplitter.
    
    Args:
        fp_type: Fingerprint type
        radius: Radius for Morgan fingerprints
        nBits: Number of bits for fingerprint
        similarity_threshold: Similarity threshold for clustering
        clustering_method: Clustering algorithm
        processes: Number of parallel processes
        random_seed: Random seed for reproducibility
        **kwargs: Additional parameters
        
    Returns:
        Configured ECFPSplitter instance
    """
    return ECFPSplitter(
        fp_type=fp_type,
        radius=radius,
        nBits=nBits,
        similarity_threshold=similarity_threshold,
        clustering_method=clustering_method,
        processes=processes,
        random_seed=random_seed,
        **kwargs
    )


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
    
    print("Testing ECFPSplitter with sample SMILES")
    
    try:
        # Test ECFP splitter
        splitter = create_ecfp_splitter(
            fp_type="Morgan",
            radius=2,
            similarity_threshold=0.8,
            clustering_method="connected"
        )
        
        # Test split generation
        splits = splitter.get_split_indices(
            data=test_smiles,
            frac_train=0.6,
            frac_valid=0.2,
            frac_test=0.2
        )
        
        print(f"Generated splits:")
        for split_name, indices in splits.items():
            print(f"  {split_name}: {indices}")
        
        # Test cluster info
        cluster_info = splitter.get_cluster_info()
        if cluster_info:
            print(f"Cluster info: {cluster_info}")
        
        # Test similarity search
        query = "CCO"
        similar = splitter.find_similar_molecules(query, test_smiles, similarity_threshold=0.5)
        print(f"Similar molecules to {query}: {similar}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("Testing completed.")