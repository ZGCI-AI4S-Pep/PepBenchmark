from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pepbenchmark.cluster.interfaces import AbstractClusterer, ClusterConfig, UnifiedClusterResult
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimilarityClusterConfig(ClusterConfig):
    """Configuration for similarity-based clustering algorithms."""
    similarity_threshold: float = 0.9
    coverage_threshold: Optional[float] = None
    method: str = "connected"  # connected, hierarchical, graph_connected, ann_graph
    n_clusters: Optional[int] = None  # for k-means like methods
    k: int = 50  # for ANN graph
    linkage_method: str = "single"  # for hierarchical clustering


class SimilarityClusterer(AbstractClusterer):
    """
    Clusterer that works with precomputed similarity matrices.
    
    Supports multiple similarity-based clustering algorithms:
    - connected: Union-Find based threshold clustering  
    - hierarchical: Hierarchical clustering with dendrogram cutting
    - graph_connected: Connected components on similarity graph
    - ann_graph: Approximate nearest neighbor graph clustering
    """
    
    def __init__(self, config: SimilarityClusterConfig):
        super().__init__(config)
        self.config: SimilarityClusterConfig = config
    
    def cluster_sequences(
        self, 
        sequences: List[str],
        similarity_matrix: Optional[np.ndarray] = None,
        labels: Optional[List[int]] = None,
        **kwargs: Any
    ) -> UnifiedClusterResult:
        """
        Cluster sequences using similarity matrix.
        
        Args:
            sequences: List of sequences to cluster
            similarity_matrix: Precomputed N×N similarity matrix 
            labels: Optional labels (not used in similarity-based clustering)
            **kwargs: Additional parameters
        
        Returns:
            UnifiedClusterResult with cluster assignments
        """
        if similarity_matrix is None:
            raise ValueError("SimilarityClusterer requires a precomputed similarity_matrix")
        
        n = len(sequences)
        if n == 0:
            return self._create_empty_result()
        
        if similarity_matrix.shape != (n, n):
            raise ValueError(f"Similarity matrix shape {similarity_matrix.shape} doesn't match sequences count {n}")
        
        # Select clustering method
        method = kwargs.get('method', self.config.method)
        
        if method == "connected":
            cluster_assignments = self._cluster_by_threshold(
                similarity_matrix,
                self.config.similarity_threshold,
                self.config.coverage_threshold
            )
        elif method == "hierarchical":
            cluster_assignments = self._cluster_by_hierarchical(
                similarity_matrix,
                self.config.similarity_threshold,
                kwargs.get('linkage_method', self.config.linkage_method)
            )
        elif method == "graph_connected":
            cluster_assignments = self._cluster_by_connected_components(
                similarity_matrix,
                self.config.similarity_threshold
            )
        elif method == "ann_graph":
            cluster_assignments = self._cluster_by_ann_graph(
                similarity_matrix,
                self.config.similarity_threshold,
                kwargs.get('k', self.config.k)
            )
        else:
            raise ValueError(f"Unsupported similarity clustering method: {method}")
        
        # Create result
        result = UnifiedClusterResult(
            cluster_assignments=cluster_assignments,
            total_clusters=len(cluster_assignments),
            total_sequences=n,
            algorithm=f"SimilarityCluster-{method}",
            parameters=self.config.to_dict()
        )
        
        self._last_result = result
        return result
    
    def _create_empty_result(self) -> UnifiedClusterResult:
        """Create empty result for edge cases."""
        return UnifiedClusterResult(
            cluster_assignments={},
            total_clusters=0,
            total_sequences=0,
            algorithm=f"SimilarityCluster-{self.config.method}",
            parameters=self.config.to_dict()
        )
    
    def _cluster_by_threshold(
        self, 
        similarity_matrix: np.ndarray,
        threshold: float,
        coverage_threshold: Optional[float] = None
    ) -> Dict[str, List[int]]:
        """Union-Find based threshold clustering."""
        n = similarity_matrix.shape[0]
        
        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
        
        # Merge based on threshold
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    union(i, j)
        
        # Build clusters
        clusters_dict = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters_dict[root].append(i)
        
        # Convert to result format
        cluster_assignments = {}
        for cluster_id, indices in enumerate(clusters_dict.values()):
            cluster_assignments[str(cluster_id)] = indices
        
        return cluster_assignments
    
    def _cluster_by_hierarchical(
        self,
        similarity_matrix: np.ndarray,
        threshold: float,
        linkage_method: str = "single"
    ) -> Dict[str, List[int]]:
        """Hierarchical clustering using similarity matrix."""
        n = similarity_matrix.shape[0]
        
        if n <= 1:
            return {"0": list(range(n))}
        
        # Convert to distance matrix
        dist_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        
        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method=linkage_method)
        cluster_labels = fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")
        
        # Group by cluster labels
        clusters_dict = defaultdict(list)
        for i, cid in enumerate(cluster_labels):
            clusters_dict[cid].append(i)
        
        # Convert to result format
        cluster_assignments = {}
        for cluster_id, (_, indices) in enumerate(clusters_dict.items()):
            cluster_assignments[str(cluster_id)] = indices
        
        return cluster_assignments
    
    def _cluster_by_connected_components(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> Dict[str, List[int]]:
        """Connected components clustering."""
        n = similarity_matrix.shape[0]
        
        # Create adjacency matrix
        adj_matrix = (similarity_matrix >= threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
        
        # Find connected components
        graph = csr_matrix(adj_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        # Group by cluster labels
        clusters_dict = defaultdict(list)
        for i, cid in enumerate(labels):
            clusters_dict[cid].append(i)
        
        # Convert to result format
        cluster_assignments = {}
        for cluster_id, indices in enumerate(clusters_dict.values()):
            cluster_assignments[str(cluster_id)] = indices
        
        return cluster_assignments
    
    def _cluster_by_ann_graph(
        self,
        similarity_matrix: np.ndarray,
        threshold: float,
        k: int = 50
    ) -> Dict[str, List[int]]:
        """Approximate nearest neighbor graph clustering."""
        n = similarity_matrix.shape[0]
        
        if n == 0:
            return {}
        
        # Build k-NN graph
        rows, cols = [], []
        for i in range(n):
            similarities = similarity_matrix[i, :]
            # Get top-k neighbors (excluding self)
            top_indices = np.argsort(similarities)[-(k+1):-1]
            top_sims = similarities[top_indices]
            
            for j, sim in zip(top_indices, top_sims):
                if sim >= threshold:
                    rows.append(i)
                    cols.append(j)
        
        if len(rows) == 0:
            # No edges found, each item is its own cluster
            cluster_assignments = {}
            for i in range(n):
                cluster_assignments[str(i)] = [i]
            return cluster_assignments
        
        # Create sparse graph and find connected components
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        # Group by cluster labels
        clusters_dict = defaultdict(list)
        for i, cid in enumerate(labels):
            clusters_dict[cid].append(i)
        
        # Convert to result format
        cluster_assignments = {}
        for cluster_id, indices in enumerate(clusters_dict.values()):
            cluster_assignments[str(cluster_id)] = indices
        
        return cluster_assignments


def create_similarity_clusterer(
    method: str = "connected",
    similarity_threshold: float = 0.9,
    coverage_threshold: Optional[float] = None,
    **kwargs
) -> SimilarityClusterer:
    """
    Factory function to create SimilarityClusterer.
    
    Args:
        method: Clustering method ("connected", "hierarchical", "graph_connected", "ann_graph")
        similarity_threshold: Similarity threshold for clustering
        coverage_threshold: Coverage threshold (for sequence-specific methods)
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured SimilarityClusterer instance
    """
    config = SimilarityClusterConfig(
        method=method,
        similarity_threshold=similarity_threshold,
        coverage_threshold=coverage_threshold,
        **kwargs
    )
    return SimilarityClusterer(config)


if __name__ == "__main__":
    # Test code for development and validation
    test_sequences = [
        "PEPTIDE",
        "PEPTIDE", 
        "DIFFERENT",
        "ANOTHER"
    ]
    print("Test clustering with sample sequences")
    
    # Example: Create a dummy similarity matrix
    import numpy as np
    n = len(test_sequences)
    similarity_matrix = np.random.random((n, n))
    np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1.0
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric
    
    print("Testing similarity-based clustering...")
    
    try:
        # Test similarity-based clustering
        clusterer = create_similarity_clusterer(
            method="connected",
            similarity_threshold=0.8
        )
        result = clusterer.cluster_sequences(
            sequences=test_sequences,
            similarity_matrix=similarity_matrix
        )
        print(f"Clusters found: {result.total_clusters}")
        print(f"Cluster assignments: {result.cluster_assignments}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Testing completed.")

