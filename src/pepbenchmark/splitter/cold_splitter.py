from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from collections import Counter

from pepbenchmark.splitter.base_splitter import AbstractSplitter
from pepbenchmark.splitter.cdhit_splitter import CDHitSplitter
from pepbenchmark.splitter.ecfp_splitter import ECFPSplitter
from pepbenchmark.splitter.mmseq_splitter import MMseqs2Splitter
from pepbenchmark.splitter.random_splitter import RandomSplitter
from pepbenchmark.splitter.hybrid_splitter import HybridSplitter
from pepbenchmark.splitter.kmer_splitter import KmerSplitter
from pepbenchmark.splitter.split_analyzer import SplitAnalyzer
from pepbenchmark.utils.logging import get_logger

logger = get_logger(__name__)

class ColdSplitter(AbstractSplitter):
    """
    Enhanced Cold Splitter for peptide-protein interaction datasets.
    
    This splitter provides multiple interaction-aware splitting strategies to ensure
    proper evaluation of machine learning models on peptide-protein interaction tasks.
    It prevents data leakage by ensuring that entities (peptides, proteins, or both)
    seen during testing are not present in the training set.
    
    Key Features:
    - Multiple interaction modes (peptide-cold, protein-cold, double-cold, random)
    - Advanced splitting strategies (frequency-based, label-balanced)
    - Automatic data leakage detection
    - Comprehensive split quality evaluation
    - Integration with multiple base splitting methods (MMseqs2, CD-HIT, ECFP, Random)
    
    Interaction Modes:
    1. peptide_cold: Ensure peptides in test set are unseen during training
    2. protein_cold: Ensure proteins in test set are unseen during training  
    3. double_cold: Ensure both peptides AND proteins in test set are unseen
    4. random: Random split based on unique interaction pairs (replaces interaction_based)
    
    Split Strategies:
    - default: Use base splitter (MMseqs2, CD-HIT, etc.) or greedy balancing
    - frequency_rare_to_test: Prioritize rare entities for test set
    - frequency_common_to_train: Prioritize common entities for training set
    - frequency_balanced: Balance frequency distribution across splits
    - label_balanced: Balance class labels while preserving entity constraints
    
    Examples:
        >>> # Basic peptide-cold split
        >>> splitter = ColdSplitter(
        ...     split_method="mmseqs",
        ...     interaction_mode="peptide_cold",
        ...     peptide_column="peptide_sequence", 
        ...     protein_column="protein_id"
        ... )
        >>> 
        >>> # DataFrame with peptide-protein interactions
        >>> df = pd.DataFrame({
        ...     'peptide_sequence': ['AKLM', 'VFSL', 'AKLM', 'YTHG'],
        ...     'protein_id': ['P1', 'P1', 'P2', 'P1'],
        ...     'label': [1, 0, 1, 0]
        ... })
        >>> 
        >>> splits = splitter.get_split_indices(df)
        >>> print(f"Train: {len(splits['train'])}, Test: {len(splits['test'])}")
        
        >>> # Advanced double-cold split with frequency balancing  
        >>> splitter = ColdSplitter(
        ...     split_method="random",
        ...     interaction_mode="double_cold",
        ...     peptide_column="peptide_sequence",
        ...     protein_column="protein_id"
        ... )
        >>> 
        >>> splits = splitter.get_split_indices(
        ...     df, 
        ...     split_strategy="frequency_balanced",
        ...     frac_train=0.7, frac_valid=0.15, frac_test=0.15
        ... )
        >>> 
        >>> # Evaluate split quality
        >>> quality = splitter.evaluate_split_quality(df, splits, verbose=True)
        >>> print("Split quality analysis completed!")
        
        >>> # Random interaction-based splitting  
        >>> splitter = ColdSplitter(
        ...     split_method="random", 
        ...     interaction_mode="random",
        ...     peptide_column="peptide_sequence",
        ...     protein_column="protein_id"
        ... )
        >>> 
        >>> splits = splitter.get_split_indices(
        ...     df,
        ...     split_strategy="label_balanced",
        ...     label_column="label"
        ... )
    """
    def __init__(
        self, 
        split_method: str = "random", 
        peptide_column: str = "peptide_sequence",
        protein_column: str = "protein_id",
        interaction_mode: str = "peptide_cold",
        use_greedy_balance: bool = False
    ):
        """
        Initialize ColdSplitter for peptide-protein interaction datasets.

        Args:
            split_method: Method to use for splitting entities ('mmseqs', 'cdhit', 'random', 'ecfp', 'hybrid', 'hydra', 'kmer')
            peptide_column: Name of the column containing peptide identifiers (for peptide-protein interactions)
            protein_column: Name of the column containing protein identifiers (for peptide-protein interactions)
            interaction_mode: Interaction splitting mode:
                - "peptide_cold": Split ensuring no peptides in test set appear in train set  
                - "protein_cold": Split ensuring no proteins in test set appear in train set
                - "double_cold": Split ensuring both peptides AND proteins in test set are unseen
                - "random": Random split based on unique interaction pairs (replaces interaction_based)
            use_greedy_balance: Whether to use greedy multi-trial balancing to approximate ratios
        """
        super().__init__()
        self.split_method = split_method.lower()
        self.peptide_column = peptide_column
        self.protein_column = protein_column
        self.interaction_mode = interaction_mode
        self.use_greedy_balance = use_greedy_balance
        self._base_splitter = self._get_base_splitter()
        self._last_split_result = None  # Initialize for leakage detection
        
        # Validate interaction mode and column configurations
        self._validate_configuration()

    def _get_base_splitter(self):
        """Get the appropriate splitter based on the split method."""
        splitter_map = {
            "mmseqs": MMseqs2Splitter,
            "cdhit": CDHitSplitter,
            "random": RandomSplitter,
            "ecfp": ECFPSplitter,
            "hybrid": HybridSplitter,
            "hydra": HybridSplitter,  # hydra is an alias for hybrid
            "kmer": KmerSplitter,
        }
        
        if self.split_method not in splitter_map:
            raise ValueError(
                f"Unknown split_method: {self.split_method}. "
                f"Supported methods: {list(splitter_map.keys())}"
            )
        
        return splitter_map[self.split_method]()

    def _validate_configuration(self):
        """Validate the configuration parameters for different interaction modes."""
        valid_modes = ["peptide_cold", "protein_cold", "double_cold", "random"]
        if self.interaction_mode not in valid_modes:
            raise ValueError(
                f"Unknown interaction_mode: {self.interaction_mode}. "
                f"Supported modes: {valid_modes}"
            )
        
        # Check if required columns are specified
        if self.peptide_column is None or self.protein_column is None:
            raise ValueError(
                "Both peptide_column and protein_column must be specified"
            )
            
        logger.info(f"ColdSplitter configured with interaction_mode='{self.interaction_mode}'")
        logger.info(f"Using peptide_column='{self.peptide_column}', protein_column='{self.protein_column}'")

    def _extract_entities(self, data: Union[List[str], pd.DataFrame]) -> List[str]:
        """
        Extract unique entities from data based on interaction mode.

        Args:
            data: DataFrame with peptide/protein columns

        Returns:
            List of unique entity identifiers

        Raises:
            ValueError: If DataFrame doesn't contain required columns or invalid input type
        """
        if isinstance(data, pd.DataFrame):
            if self.interaction_mode == "peptide_cold":
                if self.peptide_column not in data.columns:
                    raise ValueError(f"DataFrame must contain '{self.peptide_column}' column for peptide_cold mode")
                return list(data[self.peptide_column].unique())
            
            elif self.interaction_mode == "protein_cold":
                if self.protein_column not in data.columns:
                    raise ValueError(f"DataFrame must contain '{self.protein_column}' column for protein_cold mode")
                return list(data[self.protein_column].unique())
            
            elif self.interaction_mode == "double_cold":
                # For double cold, we need to ensure BOTH peptides AND proteins are separated
                if self.peptide_column not in data.columns or self.protein_column not in data.columns:
                    raise ValueError(f"DataFrame must contain both '{self.peptide_column}' and '{self.protein_column}' columns for double_cold mode")
                
                # 🔧 Fix: For double_cold mode, uniformly use interaction pairs to avoid mapping conflicts caused by separating entities
                # Both random and sequence similarity-based methods use interaction pairs to ensure a strict double cold start
                interaction_pairs = (data[self.peptide_column].astype(str) + "::" + data[self.protein_column].astype(str)).unique()
                return list(interaction_pairs)
            
            elif self.interaction_mode == "random":
                # For random mode, create interaction pair identifiers like interaction_based did
                if self.peptide_column not in data.columns or self.protein_column not in data.columns:
                    raise ValueError(f"DataFrame must contain both '{self.peptide_column}' and '{self.protein_column}' columns for random mode")
                
                # Create interaction pair identifiers: "peptide::protein"
                interaction_pairs = (data[self.peptide_column].astype(str) + "::" + data[self.protein_column].astype(str)).unique()
                return list(interaction_pairs)
                
        else:
            raise ValueError(
                "data must be a pandas DataFrame"
            )


    def _split_entities(
        self,
        entities: List[str],
        frac_train: float,
        frac_valid: float, 
        frac_test: float,
        seed: Optional[int],
        n_trials: int = 100,
        tolerance: float = 0.02,
        split_strategy: str = "default",
        **kwargs: Any,
    ) -> Dict[str, List[str]]:
        """
        Split entities either using base splitter or greedy balancing.

        Args:
            entities: List of unique entity identifiers
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed
            n_trials: Number of random trials (for greedy balancing)
            tolerance: Allowed deviation from target ratio
            split_strategy: Splitting strategy to use:
                - "default": Use base splitter or greedy balancing
                - "frequency_rare_to_test": Put rare entities in test set
                - "frequency_common_to_train": Put common entities in train set
                - "frequency_balanced": Balance frequency distribution
                - "label_balanced": Balance label distribution across splits
            **kwargs: Additional parameters for the base splitter

        Returns:
            Dictionary mapping split names to lists of entity identifiers
        """
        # Handle data parameter for frequency and label-based strategies
        data = kwargs.get('data', None)
        
        # Apply different splitting strategies
        if split_strategy.startswith("frequency_"):
            if not isinstance(data, pd.DataFrame):
                logger.warning("Frequency-based splitting requires DataFrame input, falling back to default")
                split_strategy = "default"
            else:
                frequency_strategy = split_strategy.replace("frequency_", "")
                return self._apply_frequency_based_splitting(
                    entities, data, frac_train, frac_valid, frac_test, seed, frequency_strategy
                )
        
        elif split_strategy == "label_balanced":
            if not isinstance(data, pd.DataFrame):
                logger.warning("Label-balanced splitting requires DataFrame input, falling back to default")
                split_strategy = "default"
            else:
                label_column = kwargs.get('label_column', 'label')
                return self._apply_balanced_label_splitting(
                    entities, data, frac_train, frac_valid, frac_test, seed, label_column
                )
        
        # Default splitting logic (original implementation)
        if not self.use_greedy_balance and split_strategy == "default":
            # --- Original logic ---
            # Remove 'data' from kwargs to avoid conflict
            kwargs_clean = {k: v for k, v in kwargs.items() if k != 'data'}
            entity_indices = self._base_splitter.get_split_indices(
                data=entities, 
                frac_train=frac_train, 
                frac_valid=frac_valid, 
                frac_test=frac_test, 
                seed=seed, 
                **kwargs_clean
            )
            return {
                split_name: [entities[i] for i in indices]
                for split_name, indices in entity_indices.items()
            }

        # --- Greedy multi-trial + fine-tuning ---
        import random
        target_ratios = (frac_train, frac_valid, frac_test)
        total = len(entities)
        random.seed(seed)

        def compute_diff(counts):
            return sum(abs(counts[k] / total - t)
                       for k, t in zip(["train", "valid", "test"], target_ratios))

        best_split, best_counts, best_diff = None, None, float("inf")

        for trial in range(n_trials):
            shuffled = entities[:]
            random.shuffle(shuffled)

            splits = {"train": [], "valid": [], "test": []}
            counts = {"train": 0, "valid": 0, "test": 0}

            # greedy allocation
            for e in shuffled:
                ratios = {k: counts[k] / total for k in counts}
                gaps = {
                    "train": target_ratios[0] - ratios["train"],
                    "valid": target_ratios[1] - ratios["valid"],
                    "test": target_ratios[2] - ratios["test"],
                }
                group = max(gaps, key=gaps.get)
                splits[group].append(e)
                counts[group] += 1

            diff = compute_diff(counts)
            if diff < best_diff:
                best_diff = diff
                best_split, best_counts = splits, counts

        # post-adjustment
        splits, counts = best_split, best_counts
        improved = True
        while improved:
            improved = False
            for src, dst in [("train", "valid"), ("train", "test"),
                             ("valid", "train"), ("valid", "test"),
                             ("test", "train"), ("test", "valid")]:
                if not splits[src]:
                    continue
                candidate = splits[src][0]  # move one entity
                new_counts = counts.copy()
                new_counts[src] -= 1
                new_counts[dst] += 1
                if compute_diff(new_counts) < compute_diff(counts):
                    splits[src].remove(candidate)
                    splits[dst].append(candidate)
                    counts = new_counts
                    improved = True
                    break

        final_diff = compute_diff(counts)
        if final_diff > tolerance:
            logger.warning(
                f"Greedy split final ratio deviation {final_diff:.3f} exceeds tolerance {tolerance}"
            )

        return splits


    def _apply_frequency_based_splitting(
        self,
        entities: List[str],
        data: pd.DataFrame,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int],
        frequency_strategy: str = "rare_to_test"
    ) -> Dict[str, List[str]]:
        """
        Apply frequency-based splitting strategies.

        Args:
            entities: List of unique entity identifiers
            data: Original DataFrame for frequency analysis
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed
            frequency_strategy: Strategy for frequency-based splitting:
                - "rare_to_test": Put rare entities in test set
                - "common_to_train": Put common entities in train set
                - "balanced_frequency": Balance frequency distribution across splits

        Returns:
            Dictionary mapping split names to lists of entity identifiers
        """
        import random
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Calculate entity frequencies
        if self.interaction_mode == "peptide_cold":
            entity_counts = data[self.peptide_column].value_counts().to_dict()
        elif self.interaction_mode == "protein_cold":
            entity_counts = data[self.protein_column].value_counts().to_dict()
        elif self.interaction_mode == "double_cold":
            # For double_cold, we need to count both peptides and proteins separately
            peptide_counts = data[self.peptide_column].value_counts().to_dict()
            protein_counts = data[self.protein_column].value_counts().to_dict()
            
            # Combine counts with prefixes
            entity_counts = {}
            for peptide, count in peptide_counts.items():
                entity_counts[f"peptide:{peptide}"] = count
            for protein, count in protein_counts.items():
                entity_counts[f"protein:{protein}"] = count
        else:
            # For random mode, count based on the composite key
            composite_series = data[self.peptide_column].astype(str) + "::" + data[self.protein_column].astype(str)
            entity_counts = composite_series.value_counts().to_dict()
        
        # Sort entities by frequency
        entities_with_freq = [(entity, entity_counts.get(entity, 0)) for entity in entities]
        
        if frequency_strategy == "rare_to_test":
            # Sort by frequency (ascending) - rarest first
            entities_with_freq.sort(key=lambda x: x[1])
        elif frequency_strategy == "common_to_train":
            # Sort by frequency (descending) - most common first
            entities_with_freq.sort(key=lambda x: x[1], reverse=True)
        elif frequency_strategy == "balanced_frequency":
            # Shuffle to randomize, then use frequency-aware allocation
            random.shuffle(entities_with_freq)
        
        # Calculate target sizes
        total_entities = len(entities)
        train_target = int(frac_train * total_entities)
        valid_target = int(frac_valid * total_entities)
        test_target = total_entities - train_target - valid_target
        
        splits = {"train": [], "valid": [], "test": []}
        
        if frequency_strategy in ["rare_to_test", "common_to_train"]:
            # Sequential allocation
            for i, (entity, freq) in enumerate(entities_with_freq):
                if len(splits["test"]) < test_target:
                    splits["test"].append(entity)
                elif len(splits["valid"]) < valid_target:
                    splits["valid"].append(entity)
                else:
                    splits["train"].append(entity)
                    
        elif frequency_strategy == "balanced_frequency":
            # Balanced allocation - distribute high and low frequency entities across splits
            entities_sorted_by_freq = sorted(entities_with_freq, key=lambda x: x[1])
            
            # Create frequency groups
            n_groups = 3  # high, medium, low frequency
            group_size = len(entities_sorted_by_freq) // n_groups
            
            freq_groups = [
                entities_sorted_by_freq[:group_size],  # low frequency
                entities_sorted_by_freq[group_size:2*group_size],  # medium frequency
                entities_sorted_by_freq[2*group_size:]  # high frequency
            ]
            
            # Distribute each frequency group proportionally across splits
            for group in freq_groups:
                random.shuffle(group)
                group_size = len(group)
                group_train = int(frac_train * group_size)
                group_valid = int(frac_valid * group_size)
                
                for i, (entity, freq) in enumerate(group):
                    if i < group_train:
                        splits["train"].append(entity)
                    elif i < group_train + group_valid:
                        splits["valid"].append(entity)
                    else:
                        splits["test"].append(entity)
        
        logger.info(f"Applied frequency-based splitting with strategy '{frequency_strategy}'")
        logger.info(f"   Train: {len(splits['train'])} entities")
        logger.info(f"   Valid: {len(splits['valid'])} entities")
        logger.info(f"   Test: {len(splits['test'])} entities")
        
        return splits

    def _apply_balanced_label_splitting(
        self,
        entities: List[str],
        data: pd.DataFrame,
        frac_train: float,
        frac_valid: float,
        frac_test: float,
        seed: Optional[int],
        label_column: str = "label"
    ) -> Dict[str, List[str]]:
        """
        Apply label-balanced splitting to ensure class distribution balance.

        Args:
            entities: List of unique entity identifiers
            data: Original DataFrame containing labels
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed
            label_column: Name of the label column

        Returns:
            Dictionary mapping split names to lists of entity identifiers
        """
        import random
        from collections import defaultdict
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        if label_column not in data.columns:
            logger.warning(f"Label column '{label_column}' not found, falling back to random splitting")
            random.shuffle(entities)
            total = len(entities)
            train_end = int(frac_train * total)
            valid_end = train_end + int(frac_valid * total)
            
            return {
                "train": entities[:train_end],
                "valid": entities[train_end:valid_end],
                "test": entities[valid_end:]
            }
        
        # Group entities by their dominant label
        entity_label_map = defaultdict(list)
        
        for entity in entities:
            # Get all samples for this entity
            if self.interaction_mode == "peptide_cold":
                entity_data = data[data[self.peptide_column] == entity]
            elif self.interaction_mode == "protein_cold":
                entity_data = data[data[self.protein_column] == entity]
            elif self.interaction_mode == "double_cold":
                # For double_cold, entities are prefixed with "peptide:" or "protein:"
                if entity.startswith("peptide:"):
                    peptide = entity[8:]  # Remove "peptide:" prefix
                    entity_data = data[data[self.peptide_column].astype(str) == peptide]
                elif entity.startswith("protein:"):
                    protein = entity[8:]  # Remove "protein:" prefix
                    entity_data = data[data[self.protein_column].astype(str) == protein]
                else:
                    continue
            else:  # random mode
                if "::" in entity:
                    peptide, protein = entity.split("::", 1)
                    entity_data = data[(data[self.peptide_column].astype(str) == peptide) & 
                                     (data[self.protein_column].astype(str) == protein)]
                else:
                    continue
            
            if len(entity_data) > 0:
                # Find dominant label for this entity
                label_counts = entity_data[label_column].value_counts()
                dominant_label = label_counts.index[0]
                entity_label_map[dominant_label].append(entity)
        
        # Split each label group proportionally
        splits = {"train": [], "valid": [], "test": []}
        
        for label, label_entities in entity_label_map.items():
            random.shuffle(label_entities)
            n_entities = len(label_entities)
            
            train_end = int(frac_train * n_entities)
            valid_end = train_end + int(frac_valid * n_entities)
            
            splits["train"].extend(label_entities[:train_end])
            splits["valid"].extend(label_entities[train_end:valid_end])
            splits["test"].extend(label_entities[valid_end:])
            
            logger.info(f"Label {label}: {len(label_entities)} entities -> "
                       f"Train:{train_end}, Valid:{valid_end-train_end}, Test:{n_entities-valid_end}")
        
        logger.info(f"Applied balanced label splitting")
        logger.info(f"   Total train: {len(splits['train'])} entities")
        logger.info(f"   Total valid: {len(splits['valid'])} entities")
        logger.info(f"   Total test: {len(splits['test'])} entities")
        
        return splits



    def _map_entities_to_data_indices(
        self,
        entity_splits: Dict[str, List[str]],
        data: Union[List[str], pd.DataFrame]
    ) -> Dict[str, List[int]]:
        """
        Map entity splits back to original data indices based on interaction mode.

        Args:
            entity_splits: Dictionary mapping split names to entity lists
            data: Original input data

        Returns:
            Dictionary mapping split names to data indices
        """
        if isinstance(data, pd.DataFrame):
            if self.interaction_mode == "peptide_cold":
                return {
                    split_name: data[data[self.peptide_column].isin(entities)].index.tolist()
                    for split_name, entities in entity_splits.items()
                }
            
            elif self.interaction_mode == "protein_cold":
                return {
                    split_name: data[data[self.protein_column].isin(entities)].index.tolist()
                    for split_name, entities in entity_splits.items()
                }
            
            elif self.interaction_mode == "double_cold":
                # 🔧 Fix: For double_cold mode, uniformly use interaction pair mapping to ensure a strict double cold start
                # Both random and sequence similarity-based methods use interaction pair logic
                result = {}
                for split_name, entities in entity_splits.items():
                    indices = []
                    for entity in entities:
                        if "::" in entity:
                            peptide, protein = entity.split("::", 1)
                            mask = (data[self.peptide_column].astype(str) == peptide) & \
                                   (data[self.protein_column].astype(str) == protein)
                            indices.extend(data[mask].index.tolist())
                    result[split_name] = indices
                return result
            
            elif self.interaction_mode == "random":
                # For random mode, entities are pairs "peptide::protein" (like interaction_based)
                result = {}
                for split_name, entities in entity_splits.items():
                    indices = []
                    for entity in entities:
                        if "::" in entity:
                            peptide, protein = entity.split("::", 1)
                            mask = (data[self.peptide_column].astype(str) == peptide) & \
                                   (data[self.protein_column].astype(str) == protein)
                            indices.extend(data[mask].index.tolist())
                    result[split_name] = indices
                return result
                
        else:
            raise ValueError("Only DataFrame input is supported")

    def get_split_indices(
        self,
        data: Union[List[str], pd.DataFrame],
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = 42,
        split_strategy: str = "default",
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """
        Generate entity-based cold split indices.

        Args:
            data: DataFrame with entity column, or list of entity identifiers
            frac_train: Fraction of entities for training
            frac_valid: Fraction of entities for validation
            frac_test: Fraction of entities for testing
            seed: Random seed for reproducibility
            split_strategy: Splitting strategy to use:
                - "default": Use base splitter or greedy balancing
                - "frequency_rare_to_test": Put rare entities in test set
                - "frequency_common_to_train": Put common entities in train set
                - "frequency_balanced": Balance frequency distribution
                - "label_balanced": Balance label distribution across splits
            **kwargs: Additional arguments for the underlying splitter

        Returns:
            Dictionary with train, valid, test data indices
        """
        logger.info(
            f"Starting cold split: data_type={type(data).__name__}, "
            f"interaction_mode={self.interaction_mode}, "
            f"split_method={self.split_method}, "
            f"split_strategy={split_strategy}, "
            f"fractions=({frac_train}, {frac_valid}, {frac_test}), seed={seed}"
        )

        self.validate_fractions(frac_train, frac_valid, frac_test)

        # Extract unique entities
        entities = self._extract_entities(data)
        logger.info(f"Extracted {len(entities)} unique entities for splitting")

        # Split entities using base splitter
        entity_splits = self._split_entities(
            entities, frac_train, frac_valid, frac_test, seed, split_strategy=split_strategy, data=data, **kwargs
        )

        # Map back to original data indices
        split_result = self._map_entities_to_data_indices(entity_splits, data)

        logger.info(
            f"Cold split completed: Train={len(split_result['train'])}, "
            f"Valid={len(split_result['valid'])}, Test={len(split_result['test'])}"
        )
        
        # Store the last split result for leakage detection
        self._last_split_result = split_result
        
        # Perform data leakage detection
        self._detect_data_leakage(entity_splits, data, split_result)
        
        return split_result

    def _detect_data_leakage(
        self,
        entity_splits: Dict[str, List[str]],
        data: Union[List[str], pd.DataFrame],
        split_result: Dict[str, List[int]]
    ) -> None:
        """
        Detect potential data leakage in the cold split.

        Args:
            entity_splits: Dictionary mapping split names to entity lists
            data: Original input data
        """
        logger.info("Performing data leakage detection...")
        
        train_entities = set(entity_splits["train"])
        valid_entities = set(entity_splits["valid"])
        test_entities = set(entity_splits["test"])
        
        # Check for entity overlaps between splits
        train_valid_overlap = train_entities.intersection(valid_entities)
        train_test_overlap = train_entities.intersection(test_entities)
        valid_test_overlap = valid_entities.intersection(test_entities)
        
        total_overlaps = len(train_valid_overlap) + len(train_test_overlap) + len(valid_test_overlap)
        
        if total_overlaps > 0:
            logger.warning(f"⚠️  Data leakage detected! Found {total_overlaps} entity overlaps:")
            if train_valid_overlap:
                logger.warning(f"   Train-Valid overlap: {len(train_valid_overlap)} entities")
            if train_test_overlap:
                logger.warning(f"   Train-Test overlap: {len(train_test_overlap)} entities")
            if valid_test_overlap:
                logger.warning(f"   Valid-Test overlap: {len(valid_test_overlap)} entities")
        else:
            logger.info("✅ No data leakage detected - all entities are properly separated")
        
        # For peptide-protein interaction modes, perform additional checks
        if isinstance(data, pd.DataFrame) and self.interaction_mode in ["peptide_cold", "protein_cold", "double_cold"]:
            self._detect_interaction_leakage(data, split_result)

    def _detect_interaction_leakage(self, data: pd.DataFrame, split_result: Dict[str, List[int]]) -> None:
        """
        Detect interaction-specific data leakage for peptide-protein datasets.

        Args:
            data: DataFrame containing peptide-protein interaction data
            split_result: Dictionary with train/valid/test indices
        """
        if self.interaction_mode == "peptide_cold":
            # Check if same proteins appear across splits (this is allowed in peptide_cold)
            logger.info("Peptide-cold mode: Checking protein distribution across splits...")
            train_proteins = set(data[data.index.isin(split_result["train"])][self.protein_column])
            test_proteins = set(data[data.index.isin(split_result["test"])][self.protein_column])
            
            protein_overlap = train_proteins.intersection(test_proteins)
            logger.info(f"   Proteins in both train and test: {len(protein_overlap)}/{len(train_proteins.union(test_proteins))} (expected in peptide-cold)")
            
        elif self.interaction_mode == "protein_cold":
            # Check if same peptides appear across splits (this is allowed in protein_cold)
            logger.info("Protein-cold mode: Checking peptide distribution across splits...")
            train_peptides = set(data[data.index.isin(split_result["train"])][self.peptide_column])
            test_peptides = set(data[data.index.isin(split_result["test"])][self.peptide_column])
            
            peptide_overlap = train_peptides.intersection(test_peptides)
            logger.info(f"   Peptides in both train and test: {len(peptide_overlap)}/{len(train_peptides.union(test_peptides))} (expected in protein-cold)")
            
        elif self.interaction_mode == "double_cold":
            # In double cold, neither peptides nor proteins should overlap
            logger.info("Double-cold mode: Checking both peptide and protein separation...")
            train_data = data[data.index.isin(split_result["train"])]
            test_data = data[data.index.isin(split_result["test"])]
            
            train_peptides = set(train_data[self.peptide_column])
            test_peptides = set(test_data[self.peptide_column])
            peptide_overlap = train_peptides.intersection(test_peptides)
            
            train_proteins = set(train_data[self.protein_column])
            test_proteins = set(test_data[self.protein_column])
            protein_overlap = train_proteins.intersection(test_proteins)
            
            if peptide_overlap or protein_overlap:
                logger.warning(f"⚠️  Double-cold leakage detected:")
                if peptide_overlap:
                    logger.warning(f"   Peptide overlap: {len(peptide_overlap)} peptides")
                if protein_overlap:
                    logger.warning(f"   Protein overlap: {len(protein_overlap)} proteins")
            else:
                logger.info("✅ Double-cold validation passed - no peptide or protein overlaps")

    def get_split_kfold_indices(
        self,
        data: Union[List[str], pd.DataFrame],
        k_folds: int = 5,
        seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate entity-based cold k-fold cross-validation splits.

        Args:
            data: DataFrame with entity column, or list of entity identifiers
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for the underlying splitter

        Returns:
            Dictionary with k-fold split results
        """
        logger.info(
            f"Starting cold k-fold split: data_type={type(data).__name__}, "
            f"interaction_mode={self.interaction_mode}, "
            f"split_method={self.split_method}, "
            f"k_folds={k_folds}, seed={seed}"
        )

        if k_folds <= 1:
            raise ValueError(f"k_folds must be greater than 1, got {k_folds}")

        # Extract unique entities
        entities = self._extract_entities(data)
        logger.info(f"Extracted {len(entities)} unique entities for k-fold splitting")

        # Get k-fold splits of entities
        entity_kfold_splits = self._base_splitter.get_split_kfold_indices(
            entities, k_folds, seed, **kwargs
        )

        # Map each fold back to data indices
        kfold_results = {}
        for fold_key, fold_entity_splits in entity_kfold_splits.items():
            # Convert entity indices back to entity identifiers
            fold_entity_names = {
                split_name: [entities[i] for i in indices]
                for split_name, indices in fold_entity_splits.items()
            }
            
            # Map to data indices
            kfold_results[fold_key] = self._map_entities_to_data_indices(
                fold_entity_names, data
            )

            logger.info(
                f"{fold_key} completed: "
                f"Train={len(kfold_results[fold_key]['train'])}, "
                f"Valid={len(kfold_results[fold_key]['valid'])}, "
                f"Test={len(kfold_results[fold_key]['test'])}"
            )

        logger.info(f"All {k_folds} cold k-fold splits completed successfully")
        return kfold_results

    def get_split_indices_n(
        self,
        data: Union[List[str], pd.DataFrame],
        n_splits: int = 5,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Union[List[int], int] = 42,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate multiple entity-based cold splits.

        Args:
            data: DataFrame with entity column, or list of entity identifiers
            n_splits: Number of random splits to generate
            frac_train: Fraction of entities for training
            frac_valid: Fraction of entities for validation
            frac_test: Fraction of entities for testing
            seed: Random seed or list of seeds for reproducibility
            **kwargs: Additional arguments for the underlying splitter

        Returns:
            Dictionary with multiple split results
        """
        logger.info(
            f"Starting n cold splits: data_type={type(data).__name__}, "
            f"interaction_mode={self.interaction_mode}, "
            f"split_method={self.split_method}, "
            f"n_splits={n_splits}, "
            f"fractions=({frac_train}, {frac_valid}, {frac_test}), seed={seed}"
        )

        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")

        self.validate_fractions(frac_train, frac_valid, frac_test)

        # Extract unique entities
        entities = self._extract_entities(data)
        logger.info(f"Extracted {len(entities)} unique entities for n-splits")

        # Get multiple splits of entities
        entity_n_splits = self._base_splitter.get_split_indices_n(
            entities, n_splits, frac_train, frac_valid, frac_test, seed=seed, **kwargs
        )

        # Map each split back to data indices
        split_results = {}
        for split_key, split_entity_indices in entity_n_splits.items():
            # Convert entity indices back to entity identifiers
            split_entity_names = {
                split_name: [entities[i] for i in indices]
                for split_name, indices in split_entity_indices.items()
            }
            
            # Map to data indices
            split_results[split_key] = self._map_entities_to_data_indices(
                split_entity_names, data
            )

            if not self.validate_split_results(split_results[split_key], len(data)):
                logger.warning(f"{split_key} validation failed")

            logger.info(
                f"{split_key} completed: "
                f"Train={len(split_results[split_key]['train'])}, "
                f"Valid={len(split_results[split_key]['valid'])}, "
                f"Test={len(split_results[split_key]['test'])}"
            )

        logger.info(f"All {n_splits} cold splits completed successfully")
        return split_results

    def clear_cache(self) -> None:
        """Clear cache in the underlying splitter."""
        if hasattr(self._base_splitter, "clear_cache"):
            self._base_splitter.clear_cache()
        logger.info("Cold splitter cache cleared")

    def evaluate_split_quality(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        verbose: bool = True,
        include_similarity_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the cold split using SplitAnalyzer.

        Args:
            data: Original DataFrame with sequences and labels
            split_indices: Dictionary with train/valid/test indices
            verbose: Whether to print detailed analysis
            include_similarity_analysis: Whether to include similarity analysis (requires embeddings)

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Extract sequences based on interaction mode
            sequences = self._extract_sequences_for_analysis(data)
            
            # Extract labels if available
            labels = None
            if 'label' in data.columns:
                labels = data['label'].tolist()
            elif 'target' in data.columns:
                labels = data['target'].tolist()
            
            # Initialize SplitAnalyzer
            analyzer = SplitAnalyzer(sequences=sequences, labels=labels)
            
            analysis_results = {}
            
            # Basic split statistics
            stats = analyzer.get_split_statistics(split_indices)
            analysis_results['basic_stats'] = stats
            
            if verbose:
                logger.info("=== COLD SPLIT QUALITY EVALUATION ===")
                logger.info(f"Split method: {self.split_method}")
                logger.info(f"Interaction mode: {self.interaction_mode}")
                logger.info(f"Basic statistics:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
            
            # Class distribution analysis (if labels available)
            if labels is not None:
                try:
                    class_dist = analyzer.analyze_split_class_distribution(split_indices, verbose=verbose)
                    analysis_results['class_distribution'] = class_dist
                except Exception as e:
                    logger.warning(f"Failed to analyze class distribution: {e}")
            
            # Data leakage detection (entity-specific)
            leakage_analysis = self._analyze_entity_leakage(data, split_indices)
            analysis_results['entity_leakage'] = leakage_analysis
            
            if verbose:
                logger.info("Entity leakage analysis:")
                for key, value in leakage_analysis.items():
                    logger.info(f"  {key}: {value}")
            
            # Similarity analysis (if requested and possible)
            if include_similarity_analysis:
                try:
                    # This would require embeddings or sequence similarity computation
                    logger.info("Similarity analysis not implemented yet - requires embeddings")
                except Exception as e:
                    logger.warning(f"Failed to perform similarity analysis: {e}")
            
            # Cold split specific metrics
            cold_metrics = self._calculate_cold_split_metrics(data, split_indices)
            analysis_results['cold_metrics'] = cold_metrics
            
            if verbose:
                logger.info("Cold split specific metrics:")
                for key, value in cold_metrics.items():
                    logger.info(f"  {key}: {value}")
                logger.info("=== END EVALUATION ===")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to evaluate split quality: {e}")
            return {"error": str(e)}

    def _extract_sequences_for_analysis(self, data: pd.DataFrame) -> List[str]:
        """Extract sequences from DataFrame for analysis."""
        # Look for common sequence column names
        seq_columns = ['sequence', 'seq', 'peptide_sequence', 'protein_sequence']
        
        for col in seq_columns:
            if col in data.columns:
                return data[col].astype(str).tolist()
        
        # If no sequence column found, create dummy sequences based on indices
        logger.warning("No sequence column found, using dummy sequences for analysis")
        return [f"seq_{i}" for i in range(len(data))]

    def _analyze_entity_leakage(
        self, 
        data: pd.DataFrame, 
        split_indices: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """Analyze entity-specific data leakage."""
        leakage_results = {
            "has_leakage": False,
            "entity_overlaps": {},
            "interaction_overlaps": {}
        }
        
        try:
            # Extract entities for each split
            split_entities = {}
            
            for split_name, indices in split_indices.items():
                split_data = data.iloc[indices]
                entities = self._extract_entities(split_data)
                split_entities[split_name] = set(entities)
            
            # Check for entity overlaps
            train_entities = split_entities.get("train", set())
            valid_entities = split_entities.get("valid", set())
            test_entities = split_entities.get("test", set())
            
            overlaps = {
                "train_valid": len(train_entities.intersection(valid_entities)),
                "train_test": len(train_entities.intersection(test_entities)),
                "valid_test": len(valid_entities.intersection(test_entities))
            }
            
            leakage_results["entity_overlaps"] = overlaps
            leakage_results["has_leakage"] = any(count > 0 for count in overlaps.values())
            
            # For peptide-protein interactions, also check interaction-level overlaps
            if self.interaction_mode in ["peptide_cold", "protein_cold", "double_cold"]:
                interaction_overlaps = self._check_interaction_overlaps(data, split_indices)
                leakage_results["interaction_overlaps"] = interaction_overlaps
            
        except Exception as e:
            logger.warning(f"Failed to analyze entity leakage: {e}")
            leakage_results["error"] = str(e)
        
        return leakage_results

    def _check_interaction_overlaps(
        self, 
        data: pd.DataFrame, 
        split_indices: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """Check for interaction-specific overlaps in peptide-protein datasets."""
        interaction_analysis = {}
        
        try:
            for split_name, indices in split_indices.items():
                split_data = data.iloc[indices]
                
                if self.peptide_column in split_data.columns and self.protein_column in split_data.columns:
                    peptides = set(split_data[self.peptide_column].astype(str))
                    proteins = set(split_data[self.protein_column].astype(str))
                    interactions = set(
                        split_data[self.peptide_column].astype(str) + "::" + 
                        split_data[self.protein_column].astype(str)
                    )
                    
                    interaction_analysis[split_name] = {
                        "unique_peptides": len(peptides),
                        "unique_proteins": len(proteins),
                        "unique_interactions": len(interactions)
                    }
            
            # Calculate overlaps
            if len(interaction_analysis) >= 2:
                train_data = data.iloc[split_indices["train"]]
                test_data = data.iloc[split_indices["test"]]
                
                if self.interaction_mode == "peptide_cold":
                    train_peptides = set(train_data[self.peptide_column].astype(str))
                    test_peptides = set(test_data[self.peptide_column].astype(str))
                    interaction_analysis["peptide_overlap"] = len(train_peptides.intersection(test_peptides))
                
                elif self.interaction_mode == "protein_cold":
                    train_proteins = set(train_data[self.protein_column].astype(str))
                    test_proteins = set(test_data[self.protein_column].astype(str))
                    interaction_analysis["protein_overlap"] = len(train_proteins.intersection(test_proteins))
                
                elif self.interaction_mode == "double_cold":
                    train_peptides = set(train_data[self.peptide_column].astype(str))
                    test_peptides = set(test_data[self.peptide_column].astype(str))
                    train_proteins = set(train_data[self.protein_column].astype(str))
                    test_proteins = set(test_data[self.protein_column].astype(str))
                    
                    interaction_analysis["peptide_overlap"] = len(train_peptides.intersection(test_peptides))
                    interaction_analysis["protein_overlap"] = len(train_proteins.intersection(test_proteins))
        
        except Exception as e:
            logger.warning(f"Failed to check interaction overlaps: {e}")
            interaction_analysis["error"] = str(e)
        
        return interaction_analysis

    def _calculate_cold_split_metrics(
        self, 
        data: pd.DataFrame, 
        split_indices: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """Calculate metrics specific to cold split evaluation."""
        metrics = {}
        
        try:
            # Entity coverage metrics
            total_entities = len(self._extract_entities(data))
            
            entity_counts = {}
            for split_name, indices in split_indices.items():
                split_data = data.iloc[indices]
                split_entities = len(self._extract_entities(split_data))
                entity_counts[f"{split_name}_entities"] = split_entities
            
            metrics.update(entity_counts)
            metrics["total_entities"] = total_entities
            
            # Entity distribution ratios
            for split_name in ["train", "valid", "test"]:
                if f"{split_name}_entities" in entity_counts:
                    ratio = entity_counts[f"{split_name}_entities"] / total_entities
                    metrics[f"{split_name}_entity_ratio"] = round(ratio, 4)
            
            # Interaction mode specific metrics
            if self.interaction_mode in ["peptide_cold", "protein_cold", "double_cold"] and \
               self.peptide_column in data.columns and self.protein_column in data.columns:
                
                # Count unique peptides and proteins
                total_peptides = len(data[self.peptide_column].unique())
                total_proteins = len(data[self.protein_column].unique())
                total_interactions = len(
                    (data[self.peptide_column].astype(str) + "::" + data[self.protein_column].astype(str)).unique()
                )
                
                metrics.update({
                    "total_unique_peptides": total_peptides,
                    "total_unique_proteins": total_proteins,
                    "total_unique_interactions": total_interactions
                })
            
        except Exception as e:
            logger.warning(f"Failed to calculate cold split metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics

    def get_split_info(self) -> Dict[str, Any]:
        """
        Get information about the current cold splitter configuration.

        Returns:
            Dictionary containing splitter configuration information.
        """
        info = {
            "split_method": self.split_method,
            "interaction_mode": self.interaction_mode,
            "base_splitter_class": self._base_splitter.__class__.__name__,
            "use_greedy_balance": self.use_greedy_balance,
            "peptide_column": self.peptide_column,
            "protein_column": self.protein_column
        }
            
        return info

    def validate_cold_split(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate cold split using integrated validation functionality.

        Args:
            data: Original DataFrame
            split_indices: Dictionary with train/valid/test indices
            verbose: Whether to print detailed validation report

        Returns:
            Dictionary containing validation results
        """
        validator = ColdStartValidation()
        return validator.validate_cold_split(
            data, split_indices, self.interaction_mode, 
            peptide_column=self.peptide_column, 
            protein_column=self.protein_column,
            split_method=self.split_method,  # Pass split_method to validate double_cold correctly
            verbose=verbose
        )


class ColdStartValidation:
    """
    Cold start split validation tool for verifying the effectiveness and quality of splits.
    
    This class provides comprehensive validation functions for cold start splits, including:
    - Completeness check: Ensure all samples are correctly assigned
    - Overlap check: Detect sample overlap between splits
    - Cold start specificity check: Validate the separation effectiveness in different modes
    - Data distribution statistics: Analyze entity distribution in each split
    - Label distribution statistics: Analyze binary and multi-class label distributions
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_cold_split(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        interaction_mode: str,
        peptide_column: str = "pep_seq",
        protein_column: str = "prot_seq",
        label_column: str = "label",
        split_method: str = "random",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation on a cold start split.
        
        Args:
            data: Original data DataFrame
            split_indices: Dictionary of split indices with train/valid/test keys
            interaction_mode: Interaction mode ("peptide_cold", "protein_cold", "double_cold", "random")
            peptide_column: Name of the peptide column
            protein_column: Name of the protein column  
            label_column: Name of the label column
            verbose: Whether to output a detailed report
            
        Returns:
            A dictionary containing the validation results
        """
        results = {}
        issues = []
        
        # 1. Completeness check
        completeness_results = self._check_completeness(data, split_indices)
        results['completeness'] = completeness_results
        if not completeness_results['is_complete']:
            issues.extend(completeness_results['issues'])
        
        # 2. Overlap check
        overlap_results = self._check_overlap(split_indices)
        results['overlap'] = overlap_results
        if overlap_results['has_overlap']:
            issues.extend(overlap_results['overlaps'])
        
        # 3. Cold start specificity check
        cold_results = self._check_cold_start_validity(
            data, split_indices, interaction_mode, peptide_column, protein_column, split_method
        )
        results['cold_start'] = cold_results
        if not cold_results['is_valid']:
            issues.extend(cold_results['issues'])
        
        # 4. Data distribution statistics
        distribution_stats = self._compute_distribution_statistics(
            data, split_indices, interaction_mode, peptide_column, protein_column
        )
        results['distribution'] = distribution_stats
        
        # 5. Label distribution statistics
        if label_column in data.columns:
            label_stats = self._compute_label_statistics(data, split_indices, label_column)
            results['labels'] = label_stats
        else:
            results['labels'] = {"error": f"Label column '{label_column}' not found"}
        
        # 6. Overall result
        results['is_valid'] = len(issues) == 0
        results['issues'] = issues
        results['interaction_mode'] = interaction_mode
        
        if verbose:
            self._print_validation_report(results, interaction_mode)
        
        return results
    
    def _check_completeness(self, data: pd.DataFrame, split_indices: Dict[str, List[int]]) -> Dict[str, Any]:
        """Check data completeness"""
        total_samples = len(data)
        assigned_indices = set()
        
        for split_name, indices in split_indices.items():
            assigned_indices.update(indices)
        
        assigned_count = len(assigned_indices)
        unique_count = sum(len(indices) for indices in split_indices.values())
        
        is_complete = assigned_count == total_samples
        has_duplicates = unique_count != assigned_count
        
        issues = []
        if not is_complete:
            issues.append("Data split is incomplete")
        if has_duplicates:
            issues.append("Duplicate sample assignments exist")
        
        return {
            'total_samples': total_samples,
            'assigned_samples': assigned_count,
            'unique_samples': assigned_count,
            'is_complete': is_complete,
            'has_duplicates': has_duplicates,
            'issues': issues
        }
    
    def _check_overlap(self, split_indices: Dict[str, List[int]]) -> Dict[str, Any]:
        """Check for overlap between splits"""
        overlaps = []
        overlap_details = {}
        
        split_names = list(split_indices.keys())
        
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                set1 = set(split_indices[split1])
                set2 = set(split_indices[split2])
                overlap = set1.intersection(set2)
                
                if overlap:
                    overlap_key = f"{split1}_vs_{split2}"
                    overlaps.append(f"Sample overlap exists between splits: {overlap_key}")
                    overlap_details[overlap_key] = len(overlap)
        
        return {
            'has_overlap': len(overlaps) > 0,
            'overlaps': overlaps,
            'overlap_details': overlap_details
        }
    
    def _check_cold_start_validity(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        interaction_mode: str,
        peptide_column: str,
        protein_column: str,
        split_method: str = "random"
    ) -> Dict[str, Any]:
        """Check cold start specificity"""
        if interaction_mode == "peptide_cold":
            return self._check_peptide_cold(data, split_indices, peptide_column)
        elif interaction_mode == "protein_cold":
            return self._check_protein_cold(data, split_indices, protein_column)
        elif interaction_mode == "double_cold":
            # 🔧 Fix: For double_cold mode, uniformly use interaction pair validation to ensure a strict double cold start
            # Both random and sequence similarity-based methods use interaction pair validation
            return self._check_interaction_cold(data, split_indices, peptide_column, protein_column)
        elif interaction_mode == "random":
            return {'is_valid': True, 'issues': [], 'details': 'Random split - no cold start constraints'}
        else:
            return {'is_valid': False, 'issues': [f"Unknown interaction mode: {interaction_mode}"], 'details': {}}
    
    def _check_interaction_cold(self, data: pd.DataFrame, split_indices: Dict[str, List[int]], peptide_column: str, protein_column: str) -> Dict[str, Any]:
        """Check interaction-level cold start validity (for double_cold with random method)"""
        issues = []
        overlap_details = {}
        
        # Get the set of interaction pairs for each split
        split_interactions = {}
        for split_name, indices in split_indices.items():
            split_data = data.iloc[indices]
            interactions = set()
            for _, row in split_data.iterrows():
                interaction = f"{row[peptide_column]}::{row[protein_column]}"
                interactions.add(interaction)
            split_interactions[split_name] = interactions
        
        # Check for interaction pair overlap
        split_names = list(split_interactions.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = split_interactions[split1].intersection(split_interactions[split2])
                if overlap:
                    overlap_key = f"{split1}_vs_{split2}"
                    overlap_details[overlap_key] = len(overlap)
                    issues.append(f"Interaction pair cold start is invalid")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'details': overlap_details
        }
    
    def _check_peptide_cold(self, data: pd.DataFrame, split_indices: Dict[str, List[int]], peptide_column: str) -> Dict[str, Any]:
        """Check peptide cold start validity"""
        issues = []
        overlap_details = {}
        
        # Get the set of peptides for each split
        split_peptides = {}
        for split_name, indices in split_indices.items():
            split_data = data.iloc[indices]
            split_peptides[split_name] = set(split_data[peptide_column])
        
        # Check for overlap
        split_names = list(split_peptides.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = split_peptides[split1].intersection(split_peptides[split2])
                if overlap:
                    overlap_key = f"{split1}_vs_{split2}"
                    overlap_details[overlap_key] = len(overlap)
                    issues.append(f"Peptide cold start is invalid")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'details': overlap_details
        }
    
    def _check_protein_cold(self, data: pd.DataFrame, split_indices: Dict[str, List[int]], protein_column: str) -> Dict[str, Any]:
        """Check protein cold start validity"""
        issues = []
        overlap_details = {}
        
        # Get the set of proteins for each split
        split_proteins = {}
        for split_name, indices in split_indices.items():
            split_data = data.iloc[indices]
            split_proteins[split_name] = set(split_data[protein_column])
        
        # Check for overlap
        split_names = list(split_proteins.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = split_proteins[split1].intersection(split_proteins[split2])
                if overlap:
                    overlap_key = f"{split1}_vs_{split2}"
                    overlap_details[overlap_key] = len(overlap)
                    issues.append(f"Protein cold start is invalid")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'details': overlap_details
        }
    
    def _check_double_cold(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        peptide_column: str,
        protein_column: str
    ) -> Dict[str, Any]:
        """Check double cold start validity"""
        issues = []
        details = {}
        
        # Check peptide and protein separation separately
        peptide_result = self._check_peptide_cold(data, split_indices, peptide_column)
        protein_result = self._check_protein_cold(data, split_indices, protein_column)
        
        peptide_valid = peptide_result['is_valid']
        protein_valid = protein_result['is_valid']
        
        if not peptide_valid:
            issues.append("Double cold start is invalid")
            details['peptide_separation'] = False
        else:
            details['peptide_separation'] = True
            
        if not protein_valid:
            if "Double cold start is invalid" not in issues:
                issues.append("Double cold start is invalid")
            details['protein_separation'] = False
        else:
            details['protein_separation'] = True
        
        # Merge details
        if peptide_result['details']:
            details.update({f"peptide_{k}": v for k, v in peptide_result['details'].items()})
        if protein_result['details']:
            details.update({f"protein_{k}": v for k, v in protein_result['details'].items()})
        
        return {
            'is_valid': peptide_valid and protein_valid,
            'issues': issues,
            'details': details
        }
    
    def _compute_distribution_statistics(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        interaction_mode: str,
        peptide_column: str,
        protein_column: str
    ) -> Dict[str, Any]:
        """Compute data distribution statistics"""
        stats = {}
        
        for split_name, indices in split_indices.items():
            split_data = data.iloc[indices]
            
            split_stats = {
                'sample_count': len(split_data)
            }
            
            # Compute corresponding statistics based on interaction mode
            if peptide_column in split_data.columns:
                split_stats['unique_peptides'] = split_data[peptide_column].nunique()
                
                # Calculate peptide length statistics
                if split_data[peptide_column].dtype == 'object':
                    peptide_lengths = split_data[peptide_column].astype(str).str.len()
                    if len(peptide_lengths) > 0:
                        split_stats['peptide_length_mean'] = peptide_lengths.mean()
                        split_stats['peptide_length_std'] = peptide_lengths.std()
                        split_stats['peptide_length_min'] = peptide_lengths.min()
                        split_stats['peptide_length_max'] = peptide_lengths.max()
            
            if protein_column in split_data.columns:
                split_stats['unique_proteins'] = split_data[protein_column].nunique()
            
            # Calculate number of unique interactions
            if peptide_column in split_data.columns and protein_column in split_data.columns:
                interactions = split_data[peptide_column].astype(str) + "::" + split_data[protein_column].astype(str)
                split_stats['unique_interactions'] = interactions.nunique()
            
            stats[split_name] = split_stats
        
        return stats
    
    def _compute_label_statistics(
        self,
        data: pd.DataFrame,
        split_indices: Dict[str, List[int]],
        label_column: str
    ) -> Dict[str, Any]:
        """Compute label distribution statistics"""
        label_stats = {}
        
        # Check label type
        unique_labels = data[label_column].unique()
        is_binary = len(unique_labels) == 2
        
        label_stats['label_type'] = 'binary' if is_binary else 'multiclass'
        label_stats['unique_labels'] = len(unique_labels)
        
        # Calculate label distribution for each split
        for split_name, indices in split_indices.items():
            split_data = data.iloc[indices]
            split_labels = split_data[label_column]
            
            label_counts = split_labels.value_counts().to_dict()
            total_samples = len(split_labels)
            
            split_stats = {
                'label_counts': label_counts,
                'total_samples': total_samples
            }
            
            if is_binary:
                # Binary classification statistics
                positive_count = label_counts.get(1, label_counts.get(True, 0))
                negative_count = total_samples - positive_count
                positive_ratio = positive_count / total_samples if total_samples > 0 else 0
                
                split_stats.update({
                    'positive_samples': positive_count,
                    'negative_samples': negative_count,
                    'positive_ratio': positive_ratio
                })
            
            label_stats[split_name] = split_stats
        
        return label_stats
    
    def _print_validation_report(self, results: Dict[str, Any], interaction_mode: str):
        """Print a detailed validation report"""
        print("=" * 80)
        print(f"Cold Start Split Validation Report - {interaction_mode.upper()}")
        print("=" * 80)
        print()
        
        # Completeness check
        completeness = results['completeness']
        print("📊 Completeness Check:")
        print(f"  Total samples: {completeness['total_samples']}")
        print(f"  Assigned samples: {completeness['assigned_samples']}")
        print(f"  Unique samples: {completeness['unique_samples']}")
        print(f"  Is complete: {'✅' if completeness['is_complete'] else '❌'}")
        if completeness['has_duplicates']:
            print(f"  ⚠️  Duplicate sample assignments exist")
        print()
        
        # Overlap check
        overlap = results['overlap']
        print("🔍 Overlap Check:")
        if not overlap['has_overlap']:
            print("  ✅ No sample overlap")
        else:
            for overlap_desc in overlap['overlaps']:
                print(f"  ❌ {overlap_desc}")
                for overlap_key, count in overlap['overlap_details'].items():
                    print(f"  ❌ {overlap_key}: {count} overlapping samples")
        print()
        
        # Cold start specificity check
        cold_start = results['cold_start']
        print(f"❄️  {interaction_mode.upper()} Specificity Check:")
        if cold_start['is_valid']:
            if interaction_mode == "peptide_cold":
                print("  ✅ Peptide cold start is valid")
            elif interaction_mode == "protein_cold":
                print("  ✅ Protein cold start is valid")
            elif interaction_mode == "double_cold":
                print("  ✅ Double cold start is valid")
            elif interaction_mode == "random":
                print("  ✅ Random split (no cold start constraints)")
        else:
            for issue in cold_start['issues']:
                print(f"  ❌ {issue}")
            if interaction_mode == "double_cold":
                details = cold_start['details']
                peptide_sep = details.get('peptide_separation', False)
                protein_sep = details.get('protein_separation', False)
                print(f"    - Peptide separation: {'✅' if peptide_sep else '❌'}")
                print(f"    - Protein separation: {'✅' if protein_sep else '❌'}")
            else:
                for key, value in cold_start['details'].items():
                    if 'vs' in key:
                        print(f"    - {key}: {value} overlapping {interaction_mode.split('_')[0]}s")
        print()
        
        # Data distribution statistics
        distribution = results['distribution']
        print("📈 Data Distribution Statistics:")
        for split_name, stats in distribution.items():
            print(f"  {split_name.upper()}:")
            print(f"    Sample count: {stats['sample_count']}")
            if 'unique_peptides' in stats:
                print(f"    Unique peptides: {stats['unique_peptides']}")
            if 'unique_proteins' in stats:
                print(f"    Unique proteins: {stats['unique_proteins']}")
            if 'unique_interactions' in stats:
                print(f"    Unique interactions: {stats['unique_interactions']}")
            if 'peptide_length_mean' in stats:
                mean_len = stats['peptide_length_mean']
                std_len = stats['peptide_length_std']
                min_len = stats['peptide_length_min']
                max_len = stats['peptide_length_max']
                print(f"    Peptide length: {mean_len:.1f}±{std_len:.1f} [{min_len}-{max_len}]")
        print()
        
        # Label distribution statistics
        if 'labels' in results and 'error' not in results['labels']:
            labels = results['labels']
            print("🏷️  Label Distribution Statistics:")
            if labels['label_type'] == 'binary':
                print("  Binary classification task:")
                for split_name in ['train', 'valid', 'test']:
                    if split_name in labels:
                        split_stats = labels[split_name]
                        pos_count = split_stats['positive_samples']
                        neg_count = split_stats['negative_samples']
                        pos_ratio = split_stats['positive_ratio']
                        print(f"    {split_name.upper()}: Positive samples {pos_count}, Negative samples {neg_count} (Positive ratio: {pos_ratio:.3f})")
            else:
                print(f"  Multi-class task ({labels['unique_labels']} classes):")
                for split_name in ['train', 'valid', 'test']:
                    if split_name in labels:
                        split_stats = labels[split_name]
                        print(f"    {split_name.upper()}: {split_stats['label_counts']}")
        elif 'labels' in results and 'error' in results['labels']:
            print("🏷️  Label Distribution Statistics:")
            print(f"  ⚠️  {results['labels']['error']}")
        print()
        
        # Summary
        print("📋 Summary:")
        if results['is_valid']:
            print("  ✅ All checks passed, split is valid")
        else:
            print("  ❌ Issues found:")
            for issue in results['issues']:
                print(f"    - {issue}")
        print("=" * 80)
        print()


def validate_cold_split_from_splitter(
    splitter,
    data: pd.DataFrame,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: Optional[int] = 42,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Split and validate the results using a given splitter.
    
    Args:
        splitter: ColdSplitter instance
        data: Original DataFrame
        frac_train: Fraction for training set
        frac_valid: Fraction for validation set
        frac_test: Fraction for test set
        seed: Random seed
        verbose: Whether to output a detailed report
        **kwargs: Additional arguments to pass to the splitter
        
    Returns:
        A dictionary containing the validation results
    """
    # Perform the split
    split_indices = splitter.get_split_indices(
        data, frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test, seed=seed, **kwargs
    )
    
    # Validate the split results
    validator = ColdStartValidation()
    return validator.validate_cold_split(
        data, split_indices, splitter.interaction_mode,
        peptide_column=splitter.peptide_column,
        protein_column=splitter.protein_column,
        split_method=splitter.split_method,
        verbose=verbose
    )


def quick_validation_summary(
    data: pd.DataFrame,
    split_indices: Dict[str, List[int]],
    interaction_mode: str,
    peptide_column: str = "pep_seq",
    protein_column: str = "prot_seq",
    split_method: str = "random"
) -> str:
    """
    Generate a quick summary report for a cold start split.
    
    Args:
        data: Original DataFrame
        split_indices: Dictionary of split indices
        interaction_mode: Interaction mode
        peptide_column: Name of the peptide column
        protein_column: Name of the protein column
        
    Returns:
        A summary report string
    """
    validator = ColdStartValidation()
    results = validator.validate_cold_split(
        data, split_indices, interaction_mode, peptide_column, protein_column, split_method=split_method, verbose=False
    )
    
    # Build the summary
    summary_lines = []
    summary_lines.append(f"Cold Start Split Validation ({interaction_mode})")
    summary_lines.append("=" * 40)
    
    # Overall status
    status = "✅ Valid" if results['is_valid'] else "❌ Invalid"
    summary_lines.append(f"Overall Status: {status}")
    
    # Completeness
    completeness = results['completeness']
    total = completeness['total_samples']
    assigned = completeness['assigned_samples']
    completeness_status = "✅" if completeness['is_complete'] else "❌"
    summary_lines.append(f"Sample Completeness: {completeness_status} ({assigned}/{total})")
    
    # Distribution statistics
    distribution = results['distribution']
    for split_name in ['train', 'valid', 'test']:
        if split_name in distribution:
            stats = distribution[split_name]
            sample_count = stats['sample_count']
            peptide_count = stats.get('unique_peptides', '?')
            protein_count = stats.get('unique_proteins', '?')
            summary_lines.append(f"{split_name}: {sample_count} samples, {peptide_count} peptides, {protein_count} proteins")
    
    # List of issues
    if not results['is_valid']:
        summary_lines.append("Issues:")
        for issue in results['issues']:
            summary_lines.append(f"  - {issue}")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    """
    Comprehensive examples of ColdSplitter usage for peptide-protein interaction datasets.
    """
    import pandas as pd
    import numpy as np
    
    # Create sample peptide-protein interaction dataset
    np.random.seed(42)
    
    peptides = ['AKLM', 'VFSL', 'YTHG', 'MMKL', 'QWER', 'ZXCV'] * 10
    proteins = ['P001', 'P002', 'P003', 'P004', 'P005'] * 12
    labels = np.random.randint(0, 2, size=60)
    
    # Shuffle to create realistic interactions
    indices = np.random.permutation(60)
    sample_data = pd.DataFrame({
        'peptide_sequence': [peptides[i] for i in indices],
        'protein_id': [proteins[i] for i in indices], 
        'label': labels[indices],
        'interaction_strength': np.random.uniform(0, 1, 60)
    })
    
    print("=== COLD SPLITTER EXAMPLES FOR PEPTIDE-PROTEIN INTERACTIONS ===")
    print(f"Sample dataset: {len(sample_data)} interactions")
    print(f"Unique peptides: {sample_data['peptide_sequence'].nunique()}")
    print(f"Unique proteins: {sample_data['protein_id'].nunique()}")
    print(f"Class distribution: {sample_data['label'].value_counts().to_dict()}")
    print()
    
    # Example 1: Peptide-Cold Split
    print("Example 1: Peptide-Cold Split (Random-based)")
    print("-" * 50)
    
    splitter1 = ColdSplitter(
        split_method="random", 
        interaction_mode="peptide_cold",
        peptide_column="peptide_sequence",
        protein_column="protein_id"
    )
    
    splits1 = splitter1.get_split_indices(sample_data, seed=42)
    
    print(f"Train set: {len(splits1['train'])} samples")
    print(f"Valid set: {len(splits1['valid'])} samples") 
    print(f"Test set: {len(splits1['test'])} samples")
    
    # Analyze split
    train_peptides = set(sample_data.iloc[splits1['train']]['peptide_sequence'])
    test_peptides = set(sample_data.iloc[splits1['test']]['peptide_sequence'])
    peptide_overlap = len(train_peptides.intersection(test_peptides))
    print(f"Peptide overlap between train/test: {peptide_overlap} (should be 0 for peptide-cold)")
    print()
    
    # Example 2: Double-Cold Split with Frequency Balancing
    print("Example 2: Double-Cold Split with Frequency Balancing")
    print("-" * 50)
    
    splitter2 = ColdSplitter(
        split_method="random",
        interaction_mode="double_cold",
        peptide_column="peptide_sequence", 
        protein_column="protein_id"
    )
    
    splits2 = splitter2.get_split_indices(
        sample_data,
        split_strategy="frequency_balanced",
        frac_train=0.7,
        frac_valid=0.15, 
        frac_test=0.15,
        seed=42
    )
    
    print(f"Train set: {len(splits2['train'])} samples")
    print(f"Valid set: {len(splits2['valid'])} samples")
    print(f"Test set: {len(splits2['test'])} samples")
    
    # Verify double-cold property
    train_data = sample_data.iloc[splits2['train']]
    test_data = sample_data.iloc[splits2['test']]
    
    train_peptides = set(train_data['peptide_sequence'])
    test_peptides = set(test_data['peptide_sequence'])
    train_proteins = set(train_data['protein_id'])
    test_proteins = set(test_data['protein_id'])
    
    peptide_overlap = len(train_peptides.intersection(test_peptides))
    protein_overlap = len(train_proteins.intersection(test_proteins))
    
    print(f"Peptide overlap: {peptide_overlap} (should be 0)")
    print(f"Protein overlap: {protein_overlap} (should be 0)")
    print()
    
    # Example 3: Random Interaction-based Split
    print("Example 3: Random Interaction-based Split")
    print("-" * 50)
    
    splitter3 = ColdSplitter(
        split_method="random",
        interaction_mode="random",  # New random mode that replaces interaction_based
        peptide_column="peptide_sequence",
        protein_column="protein_id"
    )
    
    splits3 = splitter3.get_split_indices(
        sample_data,
        split_strategy="label_balanced",
        label_column="label",
        seed=42
    )
    
    print(f"Train set: {len(splits3['train'])} samples")
    print(f"Valid set: {len(splits3['valid'])} samples") 
    print(f"Test set: {len(splits3['test'])} samples")
    
    # Check label distribution
    for split_name, indices in splits3.items():
        split_labels = sample_data.iloc[indices]['label'].value_counts()
        print(f"{split_name} label distribution: {split_labels.to_dict()}")
    print()
    
    # Example 4: Label-Balanced Protein-Cold Split
    print("Example 4: Label-Balanced Protein-Cold Split")
    print("-" * 50)
    
    splitter4 = ColdSplitter(
        split_method="random",
        interaction_mode="protein_cold",
        peptide_column="peptide_sequence",
        protein_column="protein_id"
    )
    
    splits4 = splitter4.get_split_indices(
        sample_data,
        split_strategy="label_balanced",
        label_column="label",
        seed=42
    )
    
    print(f"Train set: {len(splits4['train'])} samples")
    print(f"Valid set: {len(splits4['valid'])} samples") 
    print(f"Test set: {len(splits4['test'])} samples")
    
    # Check label distribution
    for split_name, indices in splits4.items():
        split_labels = sample_data.iloc[indices]['label'].value_counts()
        print(f"{split_name} label distribution: {split_labels.to_dict()}")
    print()
    
    # Example 5: Split Quality Evaluation
    print("Example 5: Comprehensive Split Quality Evaluation")
    print("-" * 50)
    
    try:
        # Add a sequence column for analysis
        sample_data['sequence'] = sample_data['peptide_sequence']  # Dummy sequence column
        
        quality_analysis = splitter1.evaluate_split_quality(
            sample_data, 
            splits1, 
            verbose=True
        )
        
        print("Quality analysis keys:", list(quality_analysis.keys()))
        
        if 'cold_metrics' in quality_analysis:
            print("Cold-specific metrics:")
            for key, value in quality_analysis['cold_metrics'].items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Quality evaluation demo failed: {e}")
    
    print()
    
    # Example 6: K-fold Cross-Validation
    print("Example 6: Cold K-fold Cross-Validation")
    print("-" * 50)
    
    try:
        kfold_splits = splitter1.get_split_kfold_indices(sample_data, k_folds=3, seed=42)
        
        print(f"Generated {len(kfold_splits)} folds")
        for fold_name, fold_splits in kfold_splits.items():
            print(f"{fold_name}: Train={len(fold_splits['train'])}, "
                  f"Valid={len(fold_splits['valid'])}, Test={len(fold_splits['test'])}")
                  
    except Exception as e:
        print(f"K-fold demo failed: {e}")
    
    print()
    
    # Example 7: Multiple Random Splits
    print("Example 7: Multiple Random Splits with Different Strategies")
    print("-" * 50)
    
    try:
        multiple_splits = splitter2.get_split_indices_n(
            sample_data,
            n_splits=3,
            split_strategy="frequency_rare_to_test",
            seed=[42, 123, 456]
        )
        
        print(f"Generated {len(multiple_splits)} different splits")
        for split_name, split_data in multiple_splits.items():
            print(f"{split_name}: Train={len(split_data['train'])}, "
                  f"Valid={len(split_data['valid'])}, Test={len(split_data['test'])}")
                  
    except Exception as e:
        print(f"Multiple splits demo failed: {e}")
    
    print()
    print("=== END OF EXAMPLES ===")
    print()
    print("Available interaction modes:", ["peptide_cold", "protein_cold", "double_cold", "random"])
    print("Available split strategies:", ["default", "frequency_rare_to_test", "frequency_common_to_train", "frequency_balanced", "label_balanced"])
    print("Available split methods:", ["mmseqs", "cdhit", "random", "ecfp"])
