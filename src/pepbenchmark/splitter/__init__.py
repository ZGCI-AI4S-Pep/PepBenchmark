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
PepBenchmark Splitter Module

This module provides comprehensive data splitting functionality for peptide datasets,
including various splitting strategies and quality analysis tools.

Available Splitters:
    - RandomSplitter: Random data splitting
    - CDHitSplitter: CD-HIT based homology-aware splitting  
    - MMseqs2Splitter: MMseqs2 based homology-aware splitting
    - ECFPSplitter: ECFP fingerprint based similarity-aware splitting
    - ColdSplitter: Entity-based cold splitting

Analysis Tools:
    - SplitAnalyzer: Unified split quality analysis and comparison
    - Split validation functions for class distribution and similarity analysis
"""

# Core splitter classes
from .base_splitter import AbstractSplitter, AbstractClusteringSplitter, BaseSplitter
from .cdhit_splitter import CDHitSplitter
from .cold_splitter import ColdSplitter
from .ecfp_splitter import ECFPSplitter
from .hybrid_splitter import HybridSplitter
from .kmer_splitter import KmerSplitter
from .mmseq_splitter import MMseqs2Splitter
from .random_splitter import RandomSplitter
from .unified_result_splitter import UnifiedResultSplitter

# Analysis and validation tools
from .split_analyzer import SplitAnalyzer
from .split_validation import (
    analyze_split_class_distribution,
    print_split_class_distribution_summary,
    analyze_cross_dataset_similarity,
    print_cross_dataset_similarity_summary,
    detect_potential_data_leakage,
    print_data_leakage_summary,
)

# Available splitter classes for easy access (canonical names only)
AVAILABLE_SPLITTERS = {
    "random": RandomSplitter,
    "mmseqs": MMseqs2Splitter,
    "cold": ColdSplitter,
    "cdhit": CDHitSplitter,
    "ecfp": ECFPSplitter,
    "kmer": KmerSplitter,
    "hybrid": HybridSplitter,
}

SPLITTER_ALIASES = {
    "mmseqs2": "mmseqs",
    "hydra": "hybrid",
}

if KmerSplitter is not None:
    AVAILABLE_SPLITTERS["kmer"] = KmerSplitter

if HybridSplitter is not None:
    AVAILABLE_SPLITTERS["hybrid"] = HybridSplitter
    SPLITTER_ALIASES["hydra"] = "hybrid"

# Base public API
__all__ = [
    "BaseSplitter",
    "AbstractSplitter",
    "AbstractClusteringSplitter",
    "RandomSplitter",
    "MMseqs2Splitter",
    "ColdSplitter",
    "CDHitSplitter",
    "ECFPSplitter",
    "KmerSplitter",
    "HybridSplitter",
    "UnifiedResultSplitter",
    "SplitAnalyzer",
    "AVAILABLE_SPLITTERS",
    "SPLITTER_ALIASES",
    "create_splitter",
    "get_splitter",
    "list_available_splitters",
    "analyze_split_class_distribution",
    "print_split_class_distribution_summary",
    "analyze_cross_dataset_similarity",
    "print_cross_dataset_similarity_summary",
    "detect_potential_data_leakage",
    "print_data_leakage_summary",
]


def get_splitter(splitter_name: str, **kwargs):
    """
    Factory function to get a splitter instance by name.
    
    Args:
        splitter_name: Name of the splitter ("random", "cdhit", "mmseqs", "ecfp", "cold", "kmer", "hybrid")
        **kwargs: Additional arguments to pass to the splitter constructor
        
    Returns:
        Splitter instance
        
    Raises:
        ValueError: If splitter name is not recognized
        
    Example:
        >>> splitter = get_splitter("random")
        >>> split_indices = splitter.get_split_indices(sequences)
    """
    splitter_name = splitter_name.lower()
    splitter_name = SPLITTER_ALIASES.get(splitter_name, splitter_name)
    
    if splitter_name not in AVAILABLE_SPLITTERS:
        available = ", ".join(AVAILABLE_SPLITTERS.keys())
        raise ValueError(f"Unknown splitter '{splitter_name}'. Available: {available}")
    
    splitter_class = AVAILABLE_SPLITTERS[splitter_name]
    return splitter_class(**kwargs)


def create_splitter(splitter_name: str, **kwargs):
    """Alias of `get_splitter` for a more factory-like API."""
    return get_splitter(splitter_name, **kwargs)


def list_available_splitters(include_aliases: bool = False):
    """List available splitter names.

    Args:
        include_aliases: Whether to include alias names such as `mmseqs2` and `hydra`.
    """
    names = sorted(AVAILABLE_SPLITTERS.keys())
    if include_aliases:
        names.extend(sorted(SPLITTER_ALIASES.keys()))
    return names


def compare_all_splitters(
    sequences, 
    labels=None, 
    embeddings=None,
    splitter_configs=None,
    output_dir=None
):
    """
    Convenience function to compare all available splitters.
    
    Args:
        sequences: List of sequences to split
        labels: Optional list of labels
        embeddings: Optional embedding matrix
        splitter_configs: Optional dict of configs for each splitter
        output_dir: Optional directory to save results
        
    Returns:
        SplitComparisonReport with results for all splitters
        
    Example:
        >>> comparison = compare_all_splitters(
        ...     sequences=fasta_list,
        ...     labels=labels,
        ...     splitter_configs={
        ...         "cdhit": {"identity": 0.8},
        ...         "mmseqs": {"identity": 0.8}
        ...     }
        ... )
        >>> ranking = comparison.get_ranking()
        >>> print(f"Best splitter: {ranking[0][0]}")
    """
    if splitter_configs is None:
        splitter_configs = {}
    
    # Initialize analyzer
    analyzer = SplitAnalyzer(
        sequences=sequences,
        labels=labels, 
        embeddings=embeddings
    )
    
    # Generate splits with different strategies
    strategies = {}
    
    for splitter_name, splitter_class in AVAILABLE_SPLITTERS.items():
        try:
            config = splitter_configs.get(splitter_name, {})
            splitter = splitter_class(**config)
            
            split_indices = splitter.get_split_indices(sequences)
            strategies[splitter_name] = split_indices
            
        except Exception as e:
            print(f"Warning: Failed to create splits with {splitter_name}: {e}")
    
    # Compare strategies
    comparison = analyzer.compare_split_strategies(strategies)
    
    # Save results if output directory provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        analyzer.save_analysis_results(
            comparison,
            os.path.join(output_dir, "splitter_comparison.json")
        )
        
        # Save comparison table
        df = comparison.to_dataframe()
        df.to_csv(os.path.join(output_dir, "splitter_comparison.csv"), index=False)
        
        # Generate comparison plot
        try:
            fig = comparison.plot_comparison()
            fig.savefig(os.path.join(output_dir, "splitter_comparison.png"))
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to save comparison plot: {e}")
    
    return comparison
