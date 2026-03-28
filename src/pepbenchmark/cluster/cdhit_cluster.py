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
CD-HIT clustering implementation with unified interface.

This module provides a clean, unified implementation of CD-HIT clustering
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
)

logger = get_logger(__name__)


# =============================================================================
# CD-HIT File Format Parsers
# =============================================================================

def parse_cdhit_clstr_file(clstr_path: str) -> Dict[str, List[str]]:
    """
    Parse CD-HIT .clstr file to extract cluster information.
    
    Args:
        clstr_path: Path to the .clstr file
        
    Returns:
        Dictionary mapping representative sequence ID to list of member sequence IDs
    """
    cluster_dict = {}
    current_cluster = []
    current_representative = None
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                # Process previous cluster
                if current_cluster and current_representative:
                    cluster_dict[current_representative] = current_cluster
                # Start new cluster
                current_cluster = []
                current_representative = None
            elif line and not line.startswith(">"):
                # Parse sequence line
                parts = line.split()
                if len(parts) >= 2:
                    seq_id = parts[2].rstrip("...").lstrip(">")
                    current_cluster.append(seq_id)
                    # Check if this is the representative (contains "*")
                    if line.endswith("*"):
                        current_representative = seq_id
    
    # Process the last cluster
    if current_cluster and current_representative:
        cluster_dict[current_representative] = current_cluster
    
    return cluster_dict


def get_representative_ids_from_cdhit_clstr(clstr_path: str) -> List[str]:
    """
    Extract representative sequence IDs from CD-HIT .clstr file.
    
    Args:
        clstr_path: Path to the .clstr file
        
    Returns:
        List of representative sequence IDs
    """
    cluster_dict = parse_cdhit_clstr_file(clstr_path)
    return list(cluster_dict.keys())


@dataclass  
class CDHitConfig(ClusterConfig):
    """Configuration for CD-HIT clustering with native parameters."""
    # Basic thresholds
    c: float = 0.9  # -c sequence identity threshold
    aL: float = 0.7  # -aL alignment coverage for longer sequence
    aS: float = 0.0  # -aS alignment coverage for shorter sequence
    
    # Advanced CD-HIT parameters
    G: int = 0  # -G global/local alignment (0: global, 1: local)
    t: int = 0  # -t tolerance for redundancy (0: exact, 1-5: allow differences)
    n: int = 2  # -n word length (2-5, auto-set based on -c if not specified)
    d: int = 0  # -d length of description in .clstr file (0: unlimited)
    l: int = 2  # -l minimum length of thrown away sequences
    g: int = 1  # -g accurate mode (0: fast, 1: accurate)
    
    # Performance parameters
    T: Optional[int] = None  # -T number of threads
    M: int = 0  # -M memory limit in MB (0: unlimited)
    
    # Additional parameters
    min_cluster_size: int = 1  # Post-processing: minimum cluster size
    
    def get_cdhit_params(self) -> Dict[str, Any]:
        """Get CD-HIT-specific parameters as a dictionary for CLI conversion."""
        params = {
            'c': self.c,
            'aL': self.aL,
            'aS': self.aS,
            'G': self.G,
            't': self.t,
            'n': self.n,
            'd': self.d,
            'l': self.l,
            'g': self.g,
            'M': self.M,
        }
        
        # Only add threads parameter if specified
        if self.T is not None:
            params['T'] = self.T
            
        return params
    
    @property 
    def similarity_threshold(self) -> float:
        """Backward compatibility property."""
        return self.c
        
    @similarity_threshold.setter
    def similarity_threshold(self, value: float):
        """Backward compatibility setter.""" 
        self.c = value



def run_cdhit_clustering(
    input_fasta: str,
    output_dir: str,
    identity: float,
    **cdhit_kwargs: Any,
) -> str:
    """
    Run CD-HIT clustering on the input FASTA file.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory for output files
        identity: Sequence identity threshold (-c)
        **cdhit_kwargs: CD-HIT parameters (auto-converted to CLI flags)

    Returns:
        str: Path to the generated clustered FASTA file
    """
    os.makedirs(output_dir, exist_ok=True)

    result_path = os.path.join(output_dir, "clustered_sequences")

    # Build CD-HIT command
    cmd_cluster = [
        "cd-hit",
        "-i", input_fasta,
        "-o", result_path,
        "-c", str(identity)
    ]
    cmd_cluster.extend(dict_to_cli_args(cdhit_kwargs))

    # Run the command using unified utility
    result = run_command(cmd_cluster)
    return result_path


def get_representative_ids_cdhit(clstr_path: str) -> List[str]:
    """
    Extract representative sequence IDs from CD-HIT clustering result .clstr file.

    Args:
        clstr_path: Path to the .clstr file.

    Returns:
        List of representative sequence IDs.
    """
    return get_representative_ids_from_cdhit_clstr(clstr_path)


def validate_cdhit_params(params: Dict[str, Any]) -> bool:
    """
    Validate CD-HIT parameters and provide warnings for potentially problematic values.

    Args:
        params: Dictionary of CD-HIT parameters
        
    Returns:
        True if all parameters are valid
    """
    param_validations = {
        "c": ((0.4, 1.0), "Sequence identity should be between 0.4 and 1.0, got {}"),
        "aL": ((0.0, 1.0), "Alignment coverage for longer sequence should be between 0.0 and 1.0, got {}"),
        "aS": ((0.0, 1.0), "Alignment coverage for shorter sequence should be between 0.0 and 1.0, got {}"),
        "G": ([0, 1], "Global alignment flag should be 0 or 1, got {}"),
        "t": ((0, 5), "Tolerance should be between 0 and 5, got {}"),
        "n": ([2, 3, 4, 5], "Word length should be 2, 3, 4, or 5, got {}"),
        "d": ((0, 256), "Description length should be between 0 and 256, got {}"),
        "l": ((1, 1000), "Minimum length should be between 1 and 1000, got {}"),
        "g": ([0, 1], "Accurate mode flag should be 0 or 1, got {}"),
        "M": ((0, 100000), "Memory limit should be between 0 and 100000 MB, got {}"),
    }

    warnings = validate_parameters(params, param_validations)

    if "T" in params and params["T"] <= 0:
        warnings.append(f"Thread count should be positive, got {params['T']}")

    return len(warnings) == 0


class CDHitClusterer(AbstractClusterer):
    """
    CD-HIT clustering adapter that implements the unified clustering interface.
    """
    
    def __init__(self, config: Optional[CDHitConfig] = None):
        if config is None:
            config = CDHitConfig()
        super().__init__(config)
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        params = self.config.get_cdhit_params()
        if not validate_cdhit_params(params):
            logger.warning("Some CD-HIT parameters may be invalid")

    def cluster_sequences(
        self, 
        sequences: List[str], 
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> UnifiedClusterResult:
        """
        Perform CD-HIT clustering on input sequences.
        
        Args:
            sequences: List of sequences to cluster
            labels: Optional labels (ignored for CD-HIT)
            **kwargs: Additional parameters
            
        Returns:
            UnifiedClusterResult containing clustering results
        """
        if not sequences:
            return UnifiedClusterResult(
                clusters={},
                total_clusters=0,
                total_sequences=0,
                algorithm="cdhit",
                parameters=self.config.to_dict()
            )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sequences to temporary FASTA file
            input_fasta = os.path.join(temp_dir, "input.fasta")
            save_fasta(sequences, input_fasta)
            
            # Run CD-HIT clustering
            result_path = run_cdhit_clustering(
                input_fasta=input_fasta,
                output_dir=temp_dir,
                identity=self.config.c,
                **self.config.get_cdhit_params()
            )
            
            # Parse clustering results
            clstr_path = result_path + ".clstr"
            cluster_map = parse_cdhit_clstr_file(clstr_path)
            
            # Convert to unified format
            unified_result = self._convert_to_unified_result(cluster_map, sequences)
            self._last_result = unified_result
            
            if self.config.verbose:
                print_cluster_statistics(
                    unified_result.cluster_assignments,
                    algorithm_name="CD-HIT",
                    data_type="sequences"
                )
            
            return unified_result
    
    def _convert_to_unified_result(
        self, 
        cluster_map: Dict[str, List[str]], 
        sequences: List[str]
    ) -> UnifiedClusterResult:
        """
        Convert CD-HIT cluster map to unified result format.
        
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
            algorithm="cdhit",
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


def create_cdhit_clusterer(
    c: float = 0.9,
    aL: float = 0.7,
    aS: float = 0.0,
    G: int = 0,
    t: int = 0,
    n: int = 2,
    d: int = 0,
    l: int = 2,
    g: int = 1,
    min_cluster_size: int = 1,
    **kwargs
) -> CDHitClusterer:
    """
    Create a CD-HIT clusterer with specified parameters.
    
    Args:
        c: Sequence identity threshold (0.4-1.0)
        aL: Alignment coverage for longer sequence (0.0-1.0)
        aS: Alignment coverage for shorter sequence (0.0-1.0)
        G: Global alignment flag (0: global, 1: local)
        t: Tolerance for redundancy (0-5)
        n: Word length (2-5)
        d: Length of description in .clstr file (0: unlimited)
        l: Minimum length of sequences
        g: Accurate mode (0: fast, 1: accurate)
        min_cluster_size: Minimum cluster size
        **kwargs: Additional parameters
        
    Returns:
        Configured CDHitClusterer instance
    """
    config = CDHitConfig(
        c=c,
        aL=aL,
        aS=aS,
        G=G,
        t=t,
        n=n,
        d=d,
        l=l,
        g=g,
        min_cluster_size=min_cluster_size,
        **kwargs
    )
    return CDHitClusterer(config)


