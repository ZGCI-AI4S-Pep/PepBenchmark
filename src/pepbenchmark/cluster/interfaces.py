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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import numpy as np


@dataclass
class UnifiedClusterResult:
    """
    Unified representation of clustering results.
    
    This class provides a common interface for clustering results
    from different algorithms (CD-HIT, MMseqs2, similarity-based, etc.)
    """
    cluster_assignments: Dict[str, List[int]]
    total_clusters: int
    total_sequences: int
    algorithm: str
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def _ordered_cluster_ids(self) -> List[str]:
        """Return cluster IDs in a stable human-friendly order for compatibility helpers."""
        def sort_key(cid: str) -> Tuple[int, Union[int, str]]:
            cid_str = str(cid)
            if cid_str.isdigit():
                return (0, int(cid_str))

            trailing_num = re.search(r"(\d+)$", cid_str)
            if trailing_num:
                return (1, int(trailing_num.group(1)))

            return (2, cid_str)

        return sorted(self.cluster_assignments.keys(), key=sort_key)

    def _resolve_cluster_id(self, cluster_id: Union[int, str]) -> Optional[str]:
        """Resolve legacy int/string cluster identifiers to actual cluster keys."""
        if cluster_id in self.cluster_assignments:
            return cluster_id

        cid_as_str = str(cluster_id)
        if cid_as_str in self.cluster_assignments:
            return cid_as_str

        cid_with_prefix = f"cluster_{cid_as_str}"
        if cid_with_prefix in self.cluster_assignments:
            return cid_with_prefix

        if isinstance(cluster_id, int):
            ordered = self._ordered_cluster_ids()
            if 0 <= cluster_id < len(ordered):
                return ordered[cluster_id]

        return None

    def cluster_count(self) -> int:
        """Backward-compatible alias for total cluster count."""
        return self.total_clusters

    @property
    def representatives(self) -> List[int]:
        """Backward-compatible ordered representative sequence indices."""
        rep_lookup: Dict[str, int] = {}
        if self.metadata is not None:
            rep_lookup = self.metadata.get("cluster_representatives", {}) or {}

        rep_list: List[int] = []
        for cid in self._ordered_cluster_ids():
            if cid in rep_lookup:
                rep_list.append(rep_lookup[cid])
            else:
                member_indices = self.cluster_assignments.get(cid, [])
                rep_list.append(member_indices[0] if member_indices else -1)

        return rep_list

    def get_cluster(self, cluster_id: Union[int, str]) -> List[int]:
        """Backward-compatible cluster lookup by index or cluster key."""
        resolved_cluster_id = self._resolve_cluster_id(cluster_id)
        if resolved_cluster_id is None:
            return []
        return list(self.cluster_assignments.get(resolved_cluster_id, []))

    def summary_stats(self) -> Dict[str, Any]:
        """Backward-compatible summary statistics alias."""
        stats = dict(self.get_statistics())
        cluster_sizes = list(self.get_cluster_sizes().values())

        if cluster_sizes:
            stats.setdefault("cluster_count", self.total_clusters)
            stats.setdefault("total_sequences", self.total_sequences)
            stats.setdefault("std_cluster_size", float(np.std(cluster_sizes)))

        return stats

    def cluster_distribution(self) -> Dict[str, int]:
        """Backward-compatible alias returning cluster size mapping."""
        return self.get_cluster_sizes()
    
    def get_sequence_to_cluster_map(self) -> Dict[int, str]:
        """
        Get a mapping from sequence index to cluster ID.
        
        Returns:
            Dictionary mapping sequence indices to cluster IDs
        """
        seq_cluster_map: Dict[int, str] = {}
        for cid, member_indices in self.cluster_assignments.items():
            for seq_idx in member_indices:
                seq_cluster_map[seq_idx] = cid
        return seq_cluster_map
    
    def get_cluster_sizes(self) -> Dict[str, int]:
        """
        Get the size of each cluster.
        
        Returns:
            Dictionary mapping cluster IDs to their sizes
        """
        return {cluster_id: len(seq_indices) for cluster_id, seq_indices in self.cluster_assignments.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic clustering statistics.
        
        Returns:
            Dictionary with clustering statistics
        """
        cluster_sizes = list(self.get_cluster_sizes().values())
        if not cluster_sizes:
            return {"empty": True}
        
        return {
            "total_clusters": self.total_clusters,
            "total_sequences": self.total_sequences,
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "mean_cluster_size": np.mean(cluster_sizes),
            "median_cluster_size": np.median(cluster_sizes),
            "compression_ratio": self.total_sequences / self.total_clusters if self.total_clusters > 0 else 1.0
        }
    
    def get_cluster_distribution(self) -> Dict[str, Any]:
        """
        Get detailed cluster size distribution statistics.
        
        Returns:
            Dictionary with distribution statistics
        """
        cluster_sizes = list(self.get_cluster_sizes().values())
        if not cluster_sizes:
            return {"empty": True}
            
        # Calculate percentiles
        percentiles = np.percentile(cluster_sizes, [25, 50, 75, 90, 95, 99])
        
        return {
            "size_distribution": {
                "min": int(np.min(cluster_sizes)),
                "max": int(np.max(cluster_sizes)),
                "mean": float(np.mean(cluster_sizes)),
                "median": float(np.median(cluster_sizes))
            },
            "percentiles": {
                "25th": percentiles[0],
                "50th": percentiles[1], 
                "75th": percentiles[2],
                "90th": percentiles[3],
                "95th": percentiles[4],
                "99th": percentiles[5]
            },
            "std_cluster_size": np.std(cluster_sizes),
            "largest_cluster_pct": max(cluster_sizes) / self.total_sequences * 100,
            "compression_ratio": self.total_sequences / self.total_clusters if self.total_clusters > 0 else 1.0
        }
    
    def get_largest_clusters(self, top_k: int = 10) -> List[Tuple[str, int, float]]:
        """
        Get the largest clusters by size.
        
        Args:
            top_k: Number of top clusters to return
            
        Returns:
            List of tuples (cluster_id, size, percentage_of_total)
        """
        cluster_sizes = self.get_cluster_sizes()
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        top_clusters: List[Tuple[str, int, float]] = []
        for cid, size in sorted_clusters[:top_k]:
            pct = (size / self.total_sequences * 100) if self.total_sequences > 0 else 0
            top_clusters.append((cid, size, pct))
        
        return top_clusters
    
    def analyze_cluster_balance(self) -> Dict[str, Any]:
        """
        Analyze how balanced the clustering is.
        
        Returns:
            Dictionary with balance metrics
        """
        cluster_sizes = list(self.get_cluster_sizes().values())
        if not cluster_sizes:
            return {"empty": True}
        
        # Calculate entropy (higher = more balanced)
        total = sum(cluster_sizes)
        probabilities = [size / total for size in cluster_sizes]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(len(cluster_sizes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Gini coefficient (0 = perfectly balanced, 1 = perfectly unbalanced)
        sorted_sizes = sorted(cluster_sizes)
        n = len(sorted_sizes)
        cumsum = np.cumsum(sorted_sizes)
        gini = (2 * sum((i + 1) * size for i, size in enumerate(sorted_sizes))) / (n * sum(sorted_sizes)) - (n + 1) / n
        
        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,  # 0-1, higher is more balanced
            "gini_coefficient": gini,  # 0-1, lower is more balanced
            "balance_score": normalized_entropy,  # Overall balance score
            "is_balanced": normalized_entropy > 0.8,  # Threshold for "balanced"
            "largest_cluster_dominance": max(cluster_sizes) / total * 100  # % of sequences in largest cluster
        }
    
    def get_summary_table(self) -> str:
        """
        Generate a formatted summary table of clustering results.
        
        Returns:
            Formatted string table
        """
        stats = self.get_statistics()
        balance = self.analyze_cluster_balance()
        largest = self.get_largest_clusters(5)
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"Clustering Summary ({self.algorithm})")
        lines.append("=" * 60)
        lines.append(f"Total Sequences: {self.total_sequences:,}")
        lines.append(f"Total Clusters: {self.total_clusters:,}")
        lines.append(f"Compression Ratio: {stats.get('compression_ratio', 0):.2f}")
        lines.append("")
        
        lines.append("Cluster Size Statistics:")
        lines.append(f"  Min: {stats.get('min_cluster_size', 0):,}")
        lines.append(f"  Max: {stats.get('max_cluster_size', 0):,}")
        lines.append(f"  Mean: {stats.get('mean_cluster_size', 0):.1f}")
        lines.append(f"  Median: {stats.get('median_cluster_size', 0):.1f}")
        lines.append("")
        
        if not balance.get("empty", False):
            lines.append("Balance Metrics:")
            lines.append(f"  Normalized Entropy: {balance['normalized_entropy']:.3f}")
            lines.append(f"  Gini Coefficient: {balance['gini_coefficient']:.3f}")
            lines.append(f"  Is Balanced: {'Yes' if balance['is_balanced'] else 'No'}")
            lines.append("")
        
        lines.append("Top 5 Largest Clusters:")
        for i, (cluster_id, size, pct) in enumerate(largest, 1):
            lines.append(f"  {i}. {cluster_id}: {size:,} sequences ({pct:.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def export_cluster_map(self) -> Dict[int, str]:
        """
        Export sequence-to-cluster mapping.
        
        Returns:
            Dictionary mapping sequence indices to cluster IDs
        """
        return self.get_sequence_to_cluster_map()
    
    def filter_clusters(self, 
                       min_size: Optional[int] = None,
                       max_size: Optional[int] = None,
                       cluster_ids: Optional[List[str]] = None) -> 'UnifiedClusterResult':
        """
        Filter clusters based on size or specific cluster IDs.
        
        Args:
            min_size: Minimum cluster size (inclusive)
            max_size: Maximum cluster size (inclusive)
            cluster_ids: Specific cluster IDs to keep
            
        Returns:
            New UnifiedClusterResult with filtered clusters
        """
        filtered_clusters = {}
        
        for cluster_id, sequence_indices in self.cluster_assignments.items():
            cluster_size = len(sequence_indices)
            
            # Apply cluster ID filter
            if cluster_ids is not None and cluster_id not in cluster_ids:
                continue
                
            # Apply size filters
            if min_size is not None and cluster_size < min_size:
                continue
                
            if max_size is not None and cluster_size > max_size:
                continue
                
            filtered_clusters[cluster_id] = sequence_indices
        
        # Calculate new totals
        new_total_clusters = len(filtered_clusters)
        new_total_sequences = sum(len(indices) for indices in filtered_clusters.values())
        
        return UnifiedClusterResult(
            cluster_assignments=filtered_clusters,
            total_clusters=new_total_clusters,
            total_sequences=new_total_sequences,
            algorithm=f"filtered_{self.algorithm}",
            parameters={**self.parameters, "filter_applied": True},
            metadata={
                "original_total_clusters": self.total_clusters,
                "original_total_sequences": self.total_sequences,
                "filter_params": {
                    "min_size": min_size,
                    "max_size": max_size,
                    "cluster_ids": cluster_ids
                }
            }
        )
    
    def get_cluster_sequences(self, cluster_id: Optional[str] = None) -> Union[Dict[str, List[int]], List[int]]:
        """
        Get the sequence labels (indices) for each cluster.
        
        Args:
            cluster_id: If specified, return only the sequences for that cluster;
                if None, return sequences for all clusters.
            
        Returns:
            If `cluster_id` is None: Dict[cluster_id, List[sequence_indices]].
            If `cluster_id` is specified: List[sequence_indices] or an empty list
                if the cluster does not exist.
        """
        if cluster_id is not None:
            # Return sequences for the requested cluster.
            if cluster_id in self.cluster_assignments:
                return self.cluster_assignments[cluster_id]
            else:
                return []
        else:
            # Return sequences for all clusters.
            return dict(self.cluster_assignments)
    
    def validate_clustering(self, expected_total_sequences: Optional[int] = None) -> Dict[str, Any]:
        """
        Check whether clusters cover all samples and whether clusters overlap.
        
        Args:
            expected_total_sequences: Expected total number of sequences.
                If None, use `self.total_sequences`.
            
        Returns:
            Validation result dictionary containing:
            - is_valid: bool, whether the clustering is valid overall
            - coverage_complete: bool, whether coverage is complete
            - has_duplicates: bool, whether duplicates exist
            - missing_sequences: List[int], missing sequence indices
            - duplicate_sequences: List[int], duplicated sequence indices
            - coverage_rate: float, coverage rate (0-1)
            - duplicate_rate: float, duplicate rate (0-1)
            - validation_summary: str, summary of validation results
        """
        expected_total = expected_total_sequences or self.total_sequences
        
        # Collect all sequences assigned to clusters.
        seen_seq_indices: set = set()
        duplicate_sequences = []
        
        for _cid, member_list in self.cluster_assignments.items():
            for seq_idx in member_list:
                if seq_idx in seen_seq_indices:
                    duplicate_sequences.append(seq_idx)
                seen_seq_indices.add(seq_idx)
        
        # Find missing sequences, assuming sequence indices start at 0.
        full_index_set = set(range(expected_total))
        missing_sequences = list(full_index_set - seen_seq_indices)
        
        # Compute summary metrics.
        coverage_rate = len(seen_seq_indices) / expected_total if expected_total > 0 else 0
        duplicate_rate = len(duplicate_sequences) / expected_total if expected_total > 0 else 0
        
        has_duplicates = len(duplicate_sequences) > 0
        coverage_complete = len(missing_sequences) == 0
        is_valid = coverage_complete and not has_duplicates
        
        # Build the validation summary.
        summary_parts = []
        summary_parts.append(f"Total sequences: {expected_total}")
        summary_parts.append(f"Clustered sequences: {len(seen_seq_indices)}")
        summary_parts.append(f"Coverage rate: {coverage_rate:.2%}")
        
        if has_duplicates:
            summary_parts.append(f"⚠️ Found {len(duplicate_sequences)} duplicate sequences")
        
        if not coverage_complete:
            summary_parts.append(f"⚠️ Missing {len(missing_sequences)} sequences")
            
        if is_valid:
            summary_parts.append("✅ Clustering is valid: fully covered with no duplicates")
        else:
            summary_parts.append("❌ Clustering is invalid")
        
        validation_summary = "\n".join(summary_parts)
        
        return {
            "is_valid": is_valid,
            "coverage_complete": coverage_complete,
            "has_duplicates": has_duplicates,
            "missing_sequences": sorted(missing_sequences),
            "duplicate_sequences": sorted(duplicate_sequences),
            "coverage_rate": coverage_rate,
            "duplicate_rate": duplicate_rate,
            "validation_summary": validation_summary,
            "statistics": {
                "total_expected": expected_total,
                "total_clustered": len(seen_seq_indices),
                "total_clusters": self.total_clusters,
                "missing_count": len(missing_sequences),
                "duplicate_count": len(duplicate_sequences)
            }
        }

    def get_cluster_representatives(self) -> Dict[str, int]:
        """
        Get representative sequence for each cluster (first sequence in each cluster).
        
        Returns:
            Dictionary mapping cluster IDs to representative sequence indices
        """
        representatives = {}
        for cluster_id, seq_indices in self.cluster_assignments.items():
            if seq_indices:  # Non-empty cluster
                representatives[cluster_id] = seq_indices[0]
        return representatives
    
    def calculate_intra_cluster_similarity(self, similarity_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate intra-cluster similarity statistics if similarity matrix is provided.
        
        Args:
            similarity_matrix: Pairwise similarity matrix between sequences
            
        Returns:
            Dictionary with similarity statistics per cluster
        """
        if similarity_matrix is None:
            return {"error": "Similarity matrix not provided"}
        
        cluster_similarities = {}
        
        for cluster_id, seq_indices in self.cluster_assignments.items():
            if len(seq_indices) < 2:
                cluster_similarities[cluster_id] = {
                    "mean_similarity": 1.0,
                    "min_similarity": 1.0,
                    "max_similarity": 1.0,
                    "std_similarity": 0.0,
                    "size": len(seq_indices)
                }
                continue
            
            # Extract similarities within this cluster
            similarities = []
            for i in range(len(seq_indices)):
                for j in range(i + 1, len(seq_indices)):
                    idx1, idx2 = seq_indices[i], seq_indices[j]
                    similarities.append(similarity_matrix[idx1, idx2])
            
            cluster_similarities[cluster_id] = {
                "mean_similarity": np.mean(similarities),
                "min_similarity": np.min(similarities),
                "max_similarity": np.max(similarities),
                "std_similarity": np.std(similarities),
                "size": len(seq_indices)
            }
        
        return cluster_similarities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cluster_assignments": self.cluster_assignments,
            "total_clusters": self.total_clusters,
            "total_sequences": self.total_sequences,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedClusterResult':
        """Create from dictionary representation."""
        return cls(
            cluster_assignments=data["cluster_assignments"],
            total_clusters=data["total_clusters"],
            total_sequences=data["total_sequences"],
            algorithm=data["algorithm"],
            parameters=data["parameters"],
            metadata=data.get("metadata")
        )


@dataclass
class ClusterConfig:
    """
    Base configuration class for clustering algorithms.
    
    This provides a common interface for clustering configuration
    that can be extended by specific algorithms.
    """
    # Common parameters
    random_seed: Optional[int] = 42
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)


class AbstractClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.
    
    This provides a unified interface for different clustering methods
    including CD-HIT, MMseqs2, similarity-based clustering, etc.
    """
    
    def __init__(self, config: ClusterConfig):
        """
        Initialize the clusterer with configuration.
        
        Args:
            config: Configuration object for the clustering algorithm
        """
        self.config = config
        self._last_result: Optional[UnifiedClusterResult] = None
    
    @abstractmethod
    def cluster_sequences(self, sequences: List[str], **kwargs: Any) -> UnifiedClusterResult:
        """
        Perform clustering on the input sequences.
        
        Args:
            sequences: List of sequences to cluster
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Unified clustering result
        """
        pass
    
    def get_last_result(self) -> Optional[UnifiedClusterResult]:
        """
        Get the result of the last clustering operation.
        
        Returns:
            Last clustering result or None if no clustering has been performed
        """
        return self._last_result
    
    def get_algorithm_name(self) -> str:
        """
        Get the name of the clustering algorithm.
        
        Returns:
            Algorithm name
        """
        return self.__class__.__name__
    
    def get_config(self) -> ClusterConfig:
        """
        Get the current configuration.
        
        Returns:
            Current configuration object
        """
        return self.config
    
    def update_config(self, **kwargs: Any) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
