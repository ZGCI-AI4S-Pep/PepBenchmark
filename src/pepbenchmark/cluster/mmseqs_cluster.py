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
MMseqs2 clustering implementation with unified interface.

This module provides a clean, unified implementation of MMseqs2 clustering
that integrates with the clustering factory and uses common utilities.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np

from pepbenchmark.utils.logging import get_logger
from pepbenchmark.cluster.interfaces import AbstractClusterer, ClusterConfig, UnifiedClusterResult
from pepbenchmark.cluster.utils import (
    save_fasta, 
    dict_to_cli_args, 
    run_command, 
    validate_parameters,
    print_cluster_statistics,
    cluster_map_to_labels
)

logger = get_logger(__name__)


# =============================================================================
# MMseqs2 File Format Parsers
# =============================================================================

def parse_mmseqs_tsv_file(tsv_path: str) -> Dict[str, List[str]]:
    """
    Parse MMseqs2 TSV file to extract cluster information.
    
    Args:
        tsv_path: Path to the TSV file
        
    Returns:
        Dictionary mapping representative sequence ID to list of member sequence IDs
    """
    cluster_dict = {}
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep_id, member_id = parts[0], parts[1]
                cluster_dict.setdefault(rep_id, []).append(member_id)
    return cluster_dict


def get_representative_ids_from_mmseqs_tsv(tsv_path: str) -> List[str]:
    """
    Extract representative sequence IDs from MMseqs2 TSV file.
    
    Args:
        tsv_path: Path to the TSV file
        
    Returns:
        List of representative sequence IDs
    """
    cluster_dict = parse_mmseqs_tsv_file(tsv_path)
    return list(cluster_dict.keys())


@dataclass  
class MMseqs2Config(ClusterConfig):
    """Configuration for MMseqs2 clustering with native parameters."""
    # Basic thresholds
    min_seq_id: float = 0.8  # --min-seq-id minimum sequence identity
    c: float = 0.8  # -c coverage threshold
    
    # Clustering parameters
    cov_mode: int = 2  # --cov-mode coverage mode (0: query, 1: target, 2: shorter)
    alignment_mode: int = 3  # --alignment-mode (0: auto, 1: score, 2: coverage, 3: score+coverage)
    seq_id_mode: int = 2  # --seq-id-mode sequence identity mode (0: alignment, 1: shorter, 2: longer)
    
    # Sensitivity and performance
    s: float = 8.0  # -s sensitivity (1.0-20.0)
    kmer_per_seq: int = 50  # --kmer-per-seq k-mer per sequence
    cluster_mode: int = 2  # --cluster-mode (0: greedy set cover, 1: connected component, 2: greedy incremental)
    
    # Performance parameters
    threads: Optional[int] = None  # --threads number of threads
    
    # Additional parameters
    min_cluster_size: int = 1  # Post-processing: minimum cluster size
    
    def __post_init__(self):
        """Post-initialization to handle parameter aliases."""
        # Handle identity parameter alias for backward compatibility
        if hasattr(self, '_temp_identity'):
            self.min_seq_id = self._temp_identity
            delattr(self, '_temp_identity')
    
    def get_mmseqs2_params(self) -> Dict[str, Any]:
        """Get MMseqs2-specific parameters as a dictionary for CLI conversion."""
        params = {
            'min-seq-id': self.min_seq_id,
            'c': self.c,
            'cov-mode': self.cov_mode,
            'alignment-mode': self.alignment_mode,
            'seq-id-mode': self.seq_id_mode,
            's': self.s,
            'kmer-per-seq': self.kmer_per_seq,
            'cluster-mode': self.cluster_mode,
        }
        
        # Only add threads parameter if specified
        if self.threads is not None:
            params['threads'] = self.threads
            
        return params
    
    @property 
    def similarity_threshold(self) -> float:
        """Backward compatibility property."""
        return self.min_seq_id
        
    @similarity_threshold.setter
    def similarity_threshold(self, value: float):
        """Backward compatibility setter.""" 
        self.min_seq_id = value


# MMseqs2ClusterResult class removed - now using UnifiedClusterResult directly


def run_mmseqs_clustering(
    input_fasta: str,
    output_dir: str,
    tmp_dir: str,
    identity: float,
    **mmseqs_kwargs: Any,
) -> str:
    """
    Run MMseqs2 clustering on the input FASTA file.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory for output files
        tmp_dir: Temporary directory for MMseqs2
        identity: Minimum sequence identity threshold (--min-seq-id)
        **mmseqs_kwargs: MMseqs2 parameters (auto-converted to CLI flags)

    Returns:
        str: Path to the generated TSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    db = os.path.join(output_dir, "db")
    result = os.path.join(output_dir, "result")
    tsv_path = os.path.join(output_dir, "cluster_map.tsv")

    # Step 1: createdb
    cmd_createdb = ["mmseqs", "createdb", input_fasta, db]
    run_command(cmd_createdb)

    # Step 2: cluster
    cmd_cluster = ["mmseqs", "cluster", db, result, tmp_dir]
    
    # Convert parameters to command line arguments with proper prefix handling
    for key, value in mmseqs_kwargs.items():
        if key == 'c':
            # Coverage parameter uses single dash
            cmd_cluster.extend(["-c", str(value)])
        elif key == 's':
            # Sensitivity parameter uses single dash
            cmd_cluster.extend(["-s", str(value)])
        else:
            # Most other parameters use double dash
            cmd_cluster.extend([f"--{key}", str(value)])

    run_command(cmd_cluster)

    # Step 3: createtsv
    cmd_tsv = ["mmseqs", "createtsv", db, db, result, tsv_path]
    run_command(cmd_tsv)

    return tsv_path


def get_representative_ids_mmseqs(tsv_path: str) -> List[str]:
    """
    Extract representative sequence IDs from MMseqs2 clustering result TSV.

    Args:
        tsv_path: Path to the cluster_map.tsv file.

    Returns:
        List of representative sequence IDs.
    """
    return get_representative_ids_from_mmseqs_tsv(tsv_path)


def validate_mmseqs_params(params: Dict[str, Any]) -> bool:
    """
    Validate MMseqs2 parameters and provide warnings for potentially problematic values.

    Args:
        params: Dictionary of MMseqs2 parameters
        
    Returns:
        True if all parameters are valid
    """
    param_validations = {
        "min-seq-id": ((0.0, 1.0), "Sequence identity should be between 0.0 and 1.0, got {}"),
        "c": ((0.0, 1.0), "Coverage should be between 0.0 and 1.0, got {}"),
        "s": ((1.0, 20.0), "Sensitivity should typically be between 1.0 and 20.0, got {}"),
        "alignment-mode": ([0, 1, 2, 3], "Alignment mode should be 0, 1, 2, or 3, got {}"),
        "seq-id-mode": ([0, 1, 2], "Sequence identity mode should be 0, 1, or 2, got {}"),
        "cov-mode": ([0, 1, 2], "Coverage mode should be 0, 1, or 2, got {}"),
        "cluster-mode": ([0, 1, 2], "Cluster mode should be 0, 1, or 2, got {}"),
        "kmer-per-seq": ((1, 1000), "K-mer per sequence should be between 1 and 1000, got {}"),
    }

    warnings = validate_parameters(params, param_validations)

    if "threads" in params and params["threads"] <= 0:
        warnings.append(f"Thread count should be positive, got {params['threads']}")

    return len(warnings) == 0


class MMseqs2Clusterer(AbstractClusterer):
    """
    MMseqs2 clustering adapter that implements the unified clustering interface.
    """
    
    def __init__(self, config: Optional[MMseqs2Config] = None):
        if config is None:
            config = MMseqs2Config()
        super().__init__(config)
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        params = self.config.get_mmseqs2_params()
        if not validate_mmseqs_params(params):
            logger.warning("Some MMseqs2 parameters may be invalid")

    def cluster_sequences(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> UnifiedClusterResult:
        """
        Perform MMseqs2 clustering on input sequences.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels (ignored for MMseqs2)
            **kwargs: Additional parameters
            
        Returns:
            UnifiedClusterResult containing clustering results
        """
        if not sequences:
            return UnifiedClusterResult(
                clusters={},
                total_clusters=0,
                total_sequences=0,
                algorithm="mmseqs2",
                parameters=self.config.to_dict()
            )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sequences to temporary FASTA file
            input_fasta = os.path.join(temp_dir, "input.fasta")
            save_fasta(sequences, input_fasta)
            
            # Create temporary directory for MMseqs2
            tmp_subdir = os.path.join(temp_dir, "tmp")
            
            # Run MMseqs2 clustering
            tsv_path = run_mmseqs_clustering(
                input_fasta=input_fasta,
                output_dir=temp_dir,
                tmp_dir=tmp_subdir,
                identity=self.config.min_seq_id,
                **self.config.get_mmseqs2_params()
            )
            
            # Parse clustering results
            cluster_map = parse_mmseqs_tsv_file(tsv_path)
            
            # Convert to unified format
            unified_result = self._convert_to_unified_result(cluster_map, sequences)
            self._last_result = unified_result
            
            if self.config.verbose:
                print_cluster_statistics(
                    unified_result.cluster_assignments,
                    algorithm_name="MMseqs2",
                    data_type="sequences"
                )
            
            return unified_result
    
    def _convert_to_unified_result(
        self, 
        cluster_map: Dict[str, List[str]], 
        sequences: List[str]
    ) -> UnifiedClusterResult:
        """
        Convert MMseqs2 cluster map to unified result format.
        
        Args:
            cluster_map: Dict mapping representative sequence ID to member sequence IDs
            sequences: Original list of sequences
            
        Returns:
            UnifiedClusterResult
        """
        # Create sequence ID to index mapping
        seq_to_idx = {f"seq{i}": i for i in range(len(sequences))}
        
        # Convert to cluster assignments (cluster_id -> list of sequence indices)
        cluster_assignments = {}
        cluster_representatives = {}
        cluster_metadata = {}
        
        cluster_id = 0
        for rep_seq_id, member_seq_ids in cluster_map.items():
            cluster_key = f"cluster_{cluster_id}"
            
            # Get indices of member sequences
            member_indices = []
            rep_idx = None
            
            for seq_id in member_seq_ids:
                if seq_id in seq_to_idx:
                    idx = seq_to_idx[seq_id]
                    member_indices.append(idx)
                    if seq_id == rep_seq_id:
                        rep_idx = idx
            
            if member_indices:
                cluster_assignments[cluster_key] = member_indices
                cluster_representatives[cluster_key] = rep_idx if rep_idx is not None else member_indices[0]
                cluster_metadata[cluster_key] = {
                    'size': len(member_indices),
                    'representative_sequence': sequences[cluster_representatives[cluster_key]],
                    'representative_id': rep_seq_id
                }
                cluster_id += 1
        
        return UnifiedClusterResult(
            cluster_assignments=cluster_assignments,
            total_clusters=len(cluster_assignments),
            total_sequences=len(sequences),
            algorithm="mmseqs2",
            parameters=self.config.to_dict(),
            metadata={
                'cluster_representatives': cluster_representatives,
                'cluster_metadata': cluster_metadata
            }
        )
    
    def cluster_sequences_simple(
        self,
        sequences: List[str],
        **kwargs
    ) -> Dict[str, List[int]]:
        """
        Simple clustering interface returning cluster assignments.
        
        Args:
            sequences: List of sequences to cluster
            **kwargs: Additional parameters
            
        Returns:
            Dict mapping cluster IDs to lists of sequence indices
        """
        result = self.cluster_sequences(sequences, **kwargs)
        return result.cluster_assignments


def create_mmseqs2_clusterer(
    identity: float = None,  # Accept identity parameter
    min_seq_id: float = None,  # Accept native parameter
    c: float = 0.8,
    cov_mode: int = 2,
    alignment_mode: int = 3,
    seq_id_mode: int = 2,
    s: float = 8.0,
    kmer_per_seq: int = 50,
    cluster_mode: int = 2,
    min_cluster_size: int = 1,
    **kwargs
) -> MMseqs2Clusterer:
    """
    Create an MMseqs2 clusterer with specified parameters.
    
    Args:
        identity: Sequence identity threshold (0.0-1.0) - alias for min_seq_id
        min_seq_id: Sequence identity threshold (0.0-1.0) - native parameter
        c: Coverage threshold (0.0-1.0)
        cov_mode: Coverage mode (0: query, 1: target, 2: shorter)
        alignment_mode: Alignment mode (0: auto, 1: score, 2: coverage, 3: score+coverage)
        seq_id_mode: Seq identity mode (0: alignment, 1: shorter, 2: longer)
        s: Sensitivity (1.0-20.0)
        kmer_per_seq: K-mer per sequence
        cluster_mode: Clustering mode (0: greedy set cover, 1: connected component, 2: greedy incremental)
        min_cluster_size: Minimum cluster size
        **kwargs: Additional parameters
        
    Returns:
        Configured MMseqs2Clusterer instance
    """
    # Handle parameter aliases
    if identity is not None and min_seq_id is not None:
        raise ValueError("Cannot specify both 'identity' and 'min_seq_id' parameters")
    
    # Use identity parameter if provided, otherwise use min_seq_id or default
    actual_min_seq_id = identity if identity is not None else (min_seq_id if min_seq_id is not None else 0.8)
    
    config = MMseqs2Config(
        min_seq_id=actual_min_seq_id,
        c=c,
        cov_mode=cov_mode,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        s=s,
        kmer_per_seq=kmer_per_seq,
        cluster_mode=cluster_mode,
        min_cluster_size=min_cluster_size,
        **kwargs
    )
    return MMseqs2Clusterer(config)


