from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union
import os

import numpy as np
import pandas as pd


from pepbenchmark.neg_sampler.neg_meta import NEG_POOL_MAP, read_dataset_sequences
from pepbenchmark.similarity.similarity import compute_similarity_matrix
from pepbenchmark.utils.logging import get_logger
# Delay importing the clustering module to improve import speed.
# from pepbenchmark.cluster import cluster_sequences


logger = get_logger(__name__)


class SamplingPoolManager:
    """Negative sampling pool manager.

    Features:
    - Add or remove sequences from built-in datasets, user input, or CSV files.
    - Process sequences by length filtering, exact deduplication, and similarity-based redundancy removal.
    - Export the current pool to CSV.

    Design notes:
    - **Explicit over implicit**: remove the mixed `data` parameter and use clear `dataset_names`/`sequences`/`csv_file_path` entry points.
    - **Controllable granularity**: `filter_by_length`, `deduplicate`, and `remove_redundancy` can be used independently; `process_pool` is only a convenience wrapper.
    - **Immutable input, mutable pool**: normalize all input first (trim and uppercase), then merge into `_neg_pool` (`set[str]`).
    """

    # ---------------------------- Construction and properties ----------------------------

    def __init__(
        self,
        *,
        include_datasets: Optional[Sequence[str]] = None,
        exclude_datasets: Optional[Sequence[str]] = None,
        include_sequences: Optional[Sequence[str]] = None,
        include_csv_file: Optional[Union[str, Path]] = None,
        include_csv_column: str = "sequence",

    ) -> None:
        """Initialize the negative sample pool.

        Args:
            include_datasets: Dataset names to include at startup.
            exclude_datasets: Dataset names to remove from the pool at startup.
            include_sequences: User-provided sequences to include at startup.
            include_csv_file: CSV file path to load sequences from at startup.
            include_csv_column: Sequence column name in the CSV, default is "sequence".
        """
        self._neg_pool: Set[str] = set()

        # Add first.
        if include_datasets:
            self.add_sequences(dataset_names=include_datasets)
        if include_sequences:
            self.add_sequences(sequences=list(include_sequences))
        if include_csv_file:
            self.add_sequences(csv_file_path=include_csv_file, sequence_column=include_csv_column)

        # Remove afterward.
        if exclude_datasets:
            self.remove_sequences(dataset_names=exclude_datasets)


    def add_sequences(
        self,
        *,
        dataset_names: Optional[Union[str, Sequence[str]]] = None,
        sequences: Optional[Sequence[str]] = None,
        csv_file_path: Optional[Union[str, Path]] = None,
        sequence_column: str = "sequence",
    ) -> None:
        """Add sequences to the negative pool (**without processing**).

        Only does three things: load → clean and normalize → merge into the set.

        Args:
            dataset_names: Built-in dataset name or list of names.
            sequences: Sequence list provided directly.
            csv_file_path: CSV path (column specified by `sequence_column`).
            sequence_column: Sequence column name in the CSV.

        Examples:
            manager.add_sequences(dataset_names=["bbp", "cpp"]) 
            manager.add_sequences(sequences=["ACD", "EFG"]) 
            manager.add_sequences(csv_file_path="seqs.csv", sequence_column="seq")
        """
        all_sequences: List[str] = []

        if dataset_names is not None:
            names = self._ensure_list(dataset_names)
            for name in names:
                all_sequences.extend(self._load_dataset_sequences(name))
                logger.info("Loaded sequences from datasets: %s", name)

        # Direct sequences.
        if sequences is not None:
            # Handle DataFrame input.
            if hasattr(sequences, 'iloc'):  # Check whether this is a DataFrame or Series.
                if hasattr(sequences, 'columns') and 'sequence' in sequences.columns:
                    # DataFrame with a 'sequence' column.
                    seq_list = sequences['sequence'].tolist()
                    all_sequences.extend(seq_list)

                elif hasattr(sequences, 'tolist'):
                    # Series input.
                    seq_list = sequences.tolist()
                    all_sequences.extend(seq_list)

                else:
                    seq_list = list(sequences)
                    all_sequences.extend(seq_list)

            else:
                # Handle regular lists and iterables.
                seq_list = list(sequences)
                all_sequences.extend(seq_list)
                logger.info("Added %d user-provided sequences", len(seq_list))

        # CSV
        if csv_file_path:
            loaded = self._load_csv_sequences(csv_file_path, sequence_column)
            all_sequences.extend(loaded)
            logger.info("Loaded %d sequences from CSV '%s'", len(loaded), csv_file_path)

        if not all_sequences:
            logger.warning("No sequences were added")
            return

        # Normalize and merge into the pool.
        normalized = all_sequences
        before = len(self._neg_pool)
        self._neg_pool.update(normalized)
        after = len(self._neg_pool)
        logger.info("Added %d raw sequences. Pool size: %d -> %d (+%d)", len(all_sequences), before, after, after - before)


    def remove_sequences(
        self,
        *,
        dataset_names: Optional[Union[str, Sequence[str]]] = None,
        sequences: Optional[Sequence[str]] = None,
        csv_file_path: Optional[Union[str, Path]] = None,
        sequence_column: str = "sequence",
    ) -> None:
        """Remove sequences from the negative pool in batches.

        - Supports collecting sequences to remove from datasets, direct input, or CSV, then normalizes them and applies a set difference.
        """
        if not self._neg_pool:
            logger.warning("Sampling pool is empty, nothing to remove")
            return

        to_remove: List[str] = []

        if dataset_names is not None:
            names = self._ensure_list(dataset_names)
            for name in names:
                to_remove.extend(self._load_dataset_sequences(name,flag="pos"))
                logger.info("Loaded sequences to remove from dataset '%s'", name)

        if sequences is not None:
            # Handle DataFrame input.
            if hasattr(sequences, 'iloc'):  # Check whether this is a DataFrame or Series.
                if hasattr(sequences, 'columns') and 'sequence' in sequences.columns:
                    # DataFrame with a 'sequence' column.
                    to_remove.extend(sequences['sequence'].tolist())
                elif hasattr(sequences, 'tolist'):
                    # Series input.
                    to_remove.extend(sequences.tolist())
                else:
                    to_remove.extend(list(sequences))
            else:
                # Handle regular lists and iterables.
                to_remove.extend(list(sequences))

        if csv_file_path:
            loaded = self._load_csv_sequences(csv_file_path, sequence_column)
            to_remove.extend(loaded)

        if not to_remove:
            logger.warning("No sequences specified for removal")
            return
        before = len(self._neg_pool)
        to_remove_set = set(to_remove)
        removed = len(self._neg_pool & to_remove_set)
        self._neg_pool.difference_update(to_remove_set)
        after = len(self._neg_pool)
        logger.info("Removed %d sequences from pool. Pool size: %d -> %d (-%d)", removed, before, after, before - after)
    # ---------------------------- Public API: processing (split atomic operations) ----------------------------

    def filter_by_length(self, *, min_length: Optional[int] = None, max_length: Optional[int] = None) -> None:
        """Filter the current pool by sequence length.

        Args:
            min_length: Minimum length (None means no lower bound).
            max_length: Maximum length (None means no upper bound).
        """
        if not self._neg_pool:
            logger.warning("Sampling pool is empty, nothing to filter")
            return

        if min_length is None and max_length is None:
            logger.info("No length constraints provided; skipping length filter")
            return

        before = len(self._neg_pool)
        kept: Set[str] = set()
        for seq in self._neg_pool:
            n = len(seq)
            if (min_length is None or n >= min_length) and (max_length is None or n <= max_length):
                kept.add(seq)
        self._neg_pool = kept
        after = len(self._neg_pool)
        logger.info("Length filter (min=%s, max=%s): %d -> %d", min_length, max_length, before, after)

    def remove_redundancy(
        self,
        *,
        method: str = "mmseqs2",
        threshold: float = 0.9,
        coverage_threshold: Optional[float] = None,
        similarity_method: str = "levenshtein",
        **kwargs
    ) -> None:
        """Remove redundancy from the negative pool by clustering and keeping one representative per cluster.

        Args:
            method: Clustering method ("mmseqs", "cdhit", "connected", "hierarchical").
            threshold: Similarity threshold; sequences above it are clustered together.
            coverage_threshold: Coverage threshold (only for alignment-based methods).
            similarity_method: Similarity calculation method when not using external tools.
            **kwargs: Extra arguments passed to the clustering method.
        """
        if not self._neg_pool:
            logger.warning("Sampling pool is empty, nothing to remove redundancy from")
            return

        sequences = list(self._neg_pool)
        before = len(sequences)

        if before == 1:
            logger.info("Only one sequence in pool, no redundancy to remove")
            return

        logger.info("Starting redundancy removal with method '%s', threshold=%.3f", method, threshold)

        try:
            # Delay importing the clustering module to improve import speed.
            from pepbenchmark.cluster import cluster_sequences
            
            # Prepare parameters for different methods.
            clustering_kwargs = kwargs.copy()
            
            # For external tools (mmseqs, cdhit), use raw sequences directly.
            if method in ["mmseqs", "mmseqs2", "cdhit"]:
                # Handle parameter mapping and conflicts.
                if method in ["mmseqs", "mmseqs2"]:
                    # MMseqs2 uses the identity parameter.
                    clustering_kwargs["identity"] = threshold
                    # If the user provided c, use it; otherwise use coverage_threshold.
                    if "c" not in clustering_kwargs and coverage_threshold is not None:
                        clustering_kwargs["c"] = coverage_threshold
                    
                    # Drop parameters unsupported by MMseqs2.
                    mmseqs2_valid_params = {
                        'identity', 'min_seq_id', 'c', 'cov_mode', 'alignment_mode', 
                        'seq_id_mode', 's', 'kmer_per_seq', 'cluster_mode', 'threads',
                        'min_cluster_size'
                    }
                    # Remove invalid parameters.
                    invalid_params = set(clustering_kwargs.keys()) - mmseqs2_valid_params
                    for param in invalid_params:
                        removed_value = clustering_kwargs.pop(param)
                        logger.debug(f"Removed invalid MMseqs2 parameter '{param}' = {removed_value}")
                        
                elif method == "cdhit":
                    # CD-HIT uses the c parameter as the similarity threshold.
                    if "c" not in clustering_kwargs:
                        clustering_kwargs["c"] = threshold
                    # If coverage_threshold is provided, map it to the appropriate parameter.
                    if coverage_threshold is not None and "aL" not in clustering_kwargs:
                        clustering_kwargs["aL"] = coverage_threshold
                    
                    # Drop parameters unsupported by CD-HIT.
                    cdhit_valid_params = {
                        'c', 'aL', 'aS', 'G', 't', 'n', 'd', 'l', 'g', 'T', 'M',
                        'min_cluster_size'
                    }
                    # Remove invalid parameters.
                    invalid_params = set(clustering_kwargs.keys()) - cdhit_valid_params
                    for param in invalid_params:
                        removed_value = clustering_kwargs.pop(param)
                        logger.debug(f"Removed invalid CD-HIT parameter '{param}' = {removed_value}")
                
                cluster_result = cluster_sequences(
                    sequences=sequences,
                    method=method,
                    **clustering_kwargs
                )
            else:
                # Other methods require a similarity matrix.
                logger.info("Computing similarity matrix for %d sequences", before)
                similarity_matrix = compute_similarity_matrix(
                    sequences,
                    sequences,
                    input_type="sequence",
                    method=similarity_method,
                    mode="full",
                    show_progress=True
                )

                clustering_kwargs["similarity_matrix"] = similarity_matrix
                clustering_kwargs["threshold"] = threshold
                
                cluster_result = cluster_sequences(
                    sequences=sequences,
                    method=method,
                    **clustering_kwargs
                )

            # Get representative sequences.
            representative_indices = cluster_result.get_cluster_representatives()
            representative_sequences = [sequences[idx] for idx in representative_indices.values()]

            # Update the negative pool.
            self._neg_pool = set(representative_sequences)
            after = len(self._neg_pool)

            logger.info(
                "Redundancy removal completed: %d -> %d sequences (%.1f%% reduction, %d clusters)",
                before, after, (before - after) / before * 100, len(representative_indices)
            )

        except Exception as e:
            logger.error("Redundancy removal failed: %s", str(e))
            raise

    def remove_similar_to_positives(
        self,
        positive_sequences: Sequence[str],
        *,
        threshold: float = 0.8,
        similarity_method: str = "levenshtein",
        coverage_threshold: Optional[float] = None,
        batch_size: int = 1000,
        use_mmseqs: bool = False,
        **kwargs
    ) -> None:
        """Remove sequences from the negative pool that are too similar to positives.

        Args:
            positive_sequences: Positive sequence list.
            threshold: Similarity threshold; negatives above it are removed.
            similarity_method: Similarity method when use_mmseqs=False.
            coverage_threshold: Coverage threshold (only for alignment-based methods).
            batch_size: Batch size to avoid excessive memory usage when use_mmseqs=False.
            use_mmseqs: Whether to use MMseqs2 for efficient similarity search (recommended for large datasets).
            **kwargs: Extra arguments passed to similarity calculation or MMseqs2.
        """
        if not self._neg_pool:
            logger.warning("Sampling pool is empty, nothing to filter")
            return

        if not positive_sequences:
            logger.warning("No positive sequences provided, skipping similarity filter")
            return

        positive_sequences = list(positive_sequences)
        negative_sequences = list(self._neg_pool)
        before = len(negative_sequences)

        logger.info(
            "Starting similarity filter: %d negative vs %d positive sequences (threshold=%.3f)",
            before, len(positive_sequences), threshold
        )

        sequences_to_remove = set()

        try:
            if use_mmseqs:
                # Use MMseqs2 for efficient similarity search.
                sequences_to_remove = self._remove_similar_with_mmseqs(
                    negative_sequences, positive_sequences, threshold, coverage_threshold, **kwargs
                )
            else:
                # Use the traditional similarity-matrix method.
                sequences_to_remove = self._remove_similar_with_matrix(
                    negative_sequences, positive_sequences, threshold, similarity_method, 
                    batch_size, **kwargs
                )

            # Remove similar sequences from the negative pool.
            removed_count = len(sequences_to_remove)
            self._neg_pool.difference_update(sequences_to_remove)
            after = len(self._neg_pool)

            logger.info(
                "Similarity filter completed: %d -> %d sequences (removed %d, %.1f%% reduction)",
                before, after, removed_count, removed_count / before * 100 if before > 0 else 0
            )

        except Exception as e:
            logger.error("Similarity filtering failed: %s", str(e))
            raise

    def _remove_similar_with_mmseqs(
        self,
        negative_sequences: List[str],
        positive_sequences: List[str],
        threshold: float,
        coverage_threshold: Optional[float] = None,
        **kwargs
    ) -> Set[str]:
        """Search for similar sequences using MMseqs2."""
        import tempfile
        import os
        from pepbenchmark.cluster.utils import save_fasta
        
        sequences_to_remove = set()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save positive samples as the database.
            pos_fasta = os.path.join(temp_dir, "positives.fasta")
            save_fasta(positive_sequences, pos_fasta)
            
            # Save negative samples as query sequences.
            neg_fasta = os.path.join(temp_dir, "negatives.fasta")
            save_fasta(negative_sequences, neg_fasta)
            
            # Create the MMseqs2 databases.
            pos_db = os.path.join(temp_dir, "pos_db")
            neg_db = os.path.join(temp_dir, "neg_db")
            result_db = os.path.join(temp_dir, "result")
            result_tsv = os.path.join(temp_dir, "result.tsv")
            
            import subprocess
            
            # Create the positive-sample database.
            cmd_create_pos = ["mmseqs", "createdb", pos_fasta, pos_db]
            logger.info("Creating positive database: %s", " ".join(cmd_create_pos))
            subprocess.run(cmd_create_pos, check=True, capture_output=True)
            
            # Create the negative-sample database.
            cmd_create_neg = ["mmseqs", "createdb", neg_fasta, neg_db]
            logger.info("Creating negative database: %s", " ".join(cmd_create_neg))
            subprocess.run(cmd_create_neg, check=True, capture_output=True)
            
            # Run the search: negative samples are searched against positive samples.
            search_cmd = [
                "mmseqs", "search", neg_db, pos_db, result_db, temp_dir,
                "--min-seq-id", str(threshold),
            ]
            
            # Add coverage parameters.
            if coverage_threshold is not None:
                search_cmd.extend(["-c", str(coverage_threshold)])
                
            # Add any additional MMseqs2 parameters.
            for key, value in kwargs.items():
                if key.startswith("mmseqs_"):
                    param_name = key[7:].replace("_", "-")
                    search_cmd.extend([f"--{param_name}", str(value)])
            
            logger.info("Running MMseqs2 search: %s", " ".join(search_cmd))
            subprocess.run(search_cmd, check=True, capture_output=True)
            
            # Convert the results to TSV.
            cmd_tsv = ["mmseqs", "convertalis", neg_db, pos_db, result_db, result_tsv]
            logger.info("Converting results to TSV: %s", " ".join(cmd_tsv))
            subprocess.run(cmd_tsv, check=True, capture_output=True)
            
            # Parse the results and identify similar negative samples.
            if os.path.exists(result_tsv) and os.path.getsize(result_tsv) > 0:
                with open(result_tsv, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            neg_id = parts[0]  # Query sequence ID (seq0, seq1, ...)
                            pos_id = parts[1]  # Target sequence ID
                            identity = float(parts[2])  # Sequence identity
                            
                            if identity >= threshold:
                                # Retrieve the actual sequence from the sequence ID.
                                neg_idx = int(neg_id.replace("seq", ""))
                                if 0 <= neg_idx < len(negative_sequences):
                                    neg_seq = negative_sequences[neg_idx]
                                    sequences_to_remove.add(neg_seq)
                                    logger.debug("MMseqs2: Removing negative sequence (identity=%.3f): %s", 
                                               identity, neg_seq[:50])
            
            logger.info("MMseqs2 search found %d sequences to remove", len(sequences_to_remove))
            
        return sequences_to_remove

    def _remove_similar_with_matrix(
        self,
        negative_sequences: List[str],
        positive_sequences: List[str],
        threshold: float,
        similarity_method: str,
        batch_size: int,
        **kwargs
    ) -> Set[str]:
        """Use the similarity-matrix method to find similar sequences (original implementation)."""
        sequences_to_remove = set()
        
        # Process in batches to control memory usage.
        for i in range(0, len(negative_sequences), batch_size):
            batch_negatives = negative_sequences[i:i + batch_size]
            logger.info("Processing batch %d/%d (%d sequences)",
                       i // batch_size + 1,
                       (len(negative_sequences) + batch_size - 1) // batch_size,
                       len(batch_negatives))

            # Compute similarities between the current batch of negatives and all positives.
            similarity_matrix = compute_similarity_matrix(
                batch_negatives,
                positive_sequences,
                input_type="sequence",
                method=similarity_method,
                mode="full",
                show_progress=False,
                **kwargs
            )

            # Check whether each negative sample is too similar to any positive sample.
            for neg_idx, neg_seq in enumerate(batch_negatives):
                max_similarity = similarity_matrix[neg_idx, :].max()
                if max_similarity >= threshold:
                    sequences_to_remove.add(neg_seq)
                    logger.debug("Removing negative sequence (similarity=%.3f): %s", max_similarity, neg_seq[:50])
        
        return sequences_to_remove


    # ---------------------------- Public API: query / persistence ----------------------------

    def get_sampling_pool(self) -> List[str]:
        """Get the current negative pool as a copied list."""
        return list(self._neg_pool)

    def get_pool_size(self) -> int:
        """Get the size of the negative pool."""
        return len(self._neg_pool)

    def clear_pool(self) -> None:
        """Clear the negative pool."""
        self._neg_pool.clear()
        logger.info("Sampling pool cleared")

    def save_pool(self, output_path: Union[str, Path]) -> None:
        """Save the current pool to a CSV file.

        Args:
            output_path: Output file path
        """
        df = pd.DataFrame({"sequence": sorted(self._neg_pool)})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %d sequences to '%s'", len(self._neg_pool), output_path)

    def get_available_datasets(self) -> List[str]:
        """Return available dataset names by directly exposing `NEG_POOL_MAP.keys()`."""
        try:
            from neg_meta import NEG_POOL_MAP  # type: ignore
            return list(NEG_POOL_MAP.keys())
        except Exception:
            logger.warning("NEG_POOL_MAP is not available in import path")
            return []

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Return the info dictionary for the specified dataset, or None if it does not exist."""
        try:
            from neg_meta import NEG_POOL_MAP  # type: ignore
            return NEG_POOL_MAP.get(dataset_name)
        except Exception:
            logger.warning("NEG_POOL_MAP is not available in import path")
            return None

    # ---------------------------- Private helpers ----------------------------

    @staticmethod
    def _ensure_list(names: Union[str, Sequence[str]]) -> List[str]:
        return [names] if isinstance(names, str) else list(names)

    @staticmethod
    def _load_csv_sequences(csv_file_path: Union[str, Path], sequence_column: str) -> List[str]:
        """Read the sequence column from a CSV file and return the raw string list."""
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:  # pragma: no cover
            logger.error("Failed to read CSV '%s': %s", csv_file_path, e)
            raise
        if sequence_column not in df.columns:
            raise ValueError(
                f"Column '{sequence_column}' not found in CSV. Available columns: {list(df.columns)}"
            )
        values = df[sequence_column].dropna().astype(str).tolist()
        return values

    @staticmethod
    def _load_dataset_sequences(dataset_name: str,flag="all") -> List[str]:
        """Load sequences from a built-in dataset without normalization.

        Requires external definitions for:
            - `neg_meta.NEG_POOL_MAP`
            - `neg_meta.read_dataset_sequences(dataset_name, dataset_info)`
        """

        if dataset_name not in NEG_POOL_MAP:
            available = list(NEG_POOL_MAP.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")

        seqs = read_dataset_sequences(dataset_name,flag=flag)
        if seqs is None or len(seqs) == 0:
            seqs = []
        return seqs


if __name__ == "__main__":
    # Test only when run directly.
    manager = SamplingPoolManager(include_datasets=["bbp", "cpp"])
    print("Initial pool size:", manager.get_pool_size())
    
    manager.filter_by_length(max_length=50)
    print("After length filter:", manager.get_pool_size())
    
    # Test internal redundancy removal.
    manager.remove_redundancy(method="mmseqs", threshold=0.9)
    print("After redundancy removal:", manager.get_pool_size())
    
    # Test similarity filtering against positive samples with MMseqs2 (recommended).
    example_positives = ["PEPTIDESEQ", "ANOTHERSEQ", "TESTSEQUENCE"]
    manager.remove_similar_to_positives(
        example_positives, 
        threshold=0.8, 
        use_mmseqs=True,  # Use MMseqs2 for efficient searching.
        coverage_threshold=0.8
    )
    print("After removing similar to positives (MMseqs2):", manager.get_pool_size())
    
    manager.save_pool("negatives.csv")
    print("Pool saved to 'negatives.csv'")
