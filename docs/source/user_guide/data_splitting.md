# Data Splitting

Data splitting is a crucial step in machine learning that determines how your dataset is divided into training, validation, and test sets. PepBenchmark provides multiple splitting strategies to handle different scenarios and requirements, particularly for peptide sequence data where sequence similarity can lead to data leakage.

## Overview

PepBenchmark offers a flexible splitting framework with the following key components:

- **Base Classes**: `BaseSplitter` (interface) and `AbstractSplitter` (common functionality)
- **Concrete Splitters**: `RandomSplitter`, `MMseqs2Splitter`, and extensible for future splitters
- **Multiple Split Types**: Single splits, k-fold cross-validation, and multiple random splits
- **I/O Support**: Save and load split results in JSON or numpy formats

## Splitting Methods

All splitters provide three main methods:

1. **`get_split_indices()`**: Generate a single train/valid/test split
2. **`get_split_kfold_indices()`**: Generate k-fold cross-validation splits
3. **`get_split_indices_n()`**: Generate multiple random splits with different seeds

### Result Key Naming Conventions

- `get_split_indices()`: Returns `{"train": [...], "valid": [...], "test": [...]}`
- `get_split_kfold_indices()`: Returns `{"fold_0": {...}, "fold_1": {...}, ...}`
- `get_split_indices_n()`: Returns `{"seed_0": {...}, "seed_1": {...}, ...}`

## Random Splitting

Random splitting performs completely random data division without considering sequence relationships.

### Basic Usage

```python
from pepbenchmark.splitter.random_splitter import RandomSplitter

# Initialize splitter
splitter = RandomSplitter()

# Single split
split_result = splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)

# K-fold cross-validation
kfold_splits = splitter.get_split_kfold_indices(
    data=sequences,
    k_folds=5,
    seed=42
)

# Multiple random splits
multiple_splits = splitter.get_split_indices_n(
    data=sequences,
    n_splits=10,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)
```

### When to Use Random Splitting

- **Quick prototyping**: Fast and simple splitting for initial experiments
- **Independent samples**: When data points are truly independent
- **Baseline comparison**: Establishing baseline performance metrics
- **Large diverse datasets**: When sequence similarity is not a major concern

### Advantages and Limitations

**Advantages:**
- Fast and computationally efficient
- Reproducible with fixed seeds
- Simple to understand and implement
- Works with any data type

**Limitations:**
- May cause data leakage with similar sequences
- Not suitable for sequence-based tasks requiring homology awareness
- Can lead to overly optimistic performance estimates

## MMseqs2 Homology-Aware Splitting

MMseqs2 splitting uses sequence clustering to ensure similar sequences are placed in the same split, preventing data leakage due to sequence homology.

### Basic Usage

```python
from pepbenchmark.splitter.homo_splitter import MMseqs2Splitter

# Initialize splitter
splitter = MMseqs2Splitter()

# Single homology-aware split
split_result = splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    identity=0.25,  # Sequence identity threshold
    seed=42,
    # Optional MMseqs2 parameters
    coverage=0.8,
    sensitivity=7.5,
    threads=4
)

# K-fold with homology awareness
kfold_splits = splitter.get_split_kfold_indices(
    data=sequences,
    k_folds=5,
    identity=0.25,
    seed=42
)
```

### MMseqs2 Parameters

The splitter accepts various MMseqs2 parameters for fine-tuning clustering:

- `identity`: Sequence identity threshold (default: 0.25)
- `coverage`: Coverage threshold (default: 0.8)
- `sensitivity`: Search sensitivity (default: 7.5)
- `threads`: Number of CPU threads to use
- `cov_mode`: Coverage mode (0, 1, 2, or 3)
- `alignment_mode`: Alignment mode
- `seq_id_mode`: Sequence identity mode
- `mask`: Masking mode

### Caching Mechanism

The MMseqs2Splitter implements intelligent caching:

```python
# First call runs MMseqs2 clustering
split1 = splitter.get_split_indices(data, identity=0.25)

# Subsequent calls with same data and parameters use cached results
split2 = splitter.get_split_indices(data, identity=0.25)  # Uses cache

# Clear cache if needed
splitter.clear_cache()

# Get clustering information
cluster_info = splitter.get_cluster_info()
```

### When to Use MMseqs2 Splitting

- **Sequence-based tasks**: When working with protein/peptide sequences
- **Avoiding data leakage**: When sequence similarity could inflate performance
- **Realistic evaluation**: For more conservative performance estimates
- **Homology-aware ML**: When sequence relationships matter

### Advantages and Limitations

**Advantages:**
- Prevents data leakage from sequence similarity
- More realistic performance estimates
- Configurable clustering parameters
- Intelligent caching for efficiency
- Detailed clustering statistics

**Limitations:**
- Computationally more expensive than random splitting
- Requires MMseqs2 installation
- May result in imbalanced splits
- Depends on clustering quality

## Validation and Statistics

All splitters provide built-in validation and statistics:

```python
# Validate split results
is_valid = splitter.validate_split_results(split_result, len(data))

# Get split statistics
stats = splitter.get_split_statistics(split_result)
print(f"Train size: {stats['train_size']} ({stats['train_fraction']:.2%})")
print(f"Valid size: {stats['valid_size']} ({stats['valid_fraction']:.2%})")
print(f"Test size: {stats['test_size']} ({stats['test_fraction']:.2%})")
```

## Saving and Loading Splits

Split results can be saved and loaded for reproducibility:

```python
# Save splits
splitter.save_split_results(split_result, "my_split.json", format="json")
splitter.save_split_results(split_result, "my_split.npz", format="numpy")

# Load splits
loaded_split = splitter.load_split_results("my_split.json", format="json")
```

## Best Practices

### Choosing the Right Splitter

1. **For sequence data**: Use MMseqs2Splitter to avoid homology bias
2. **For non-sequence data**: RandomSplitter is usually sufficient
3. **For quick testing**: RandomSplitter for faster iteration
4. **For publication**: MMseqs2Splitter for more rigorous evaluation

### Parameter Selection

1. **Identity threshold**: Lower values (0.2-0.3) for stricter clustering
2. **Split ratios**: Adjust based on dataset size and task requirements
3. **K-fold**: Use 5-10 folds depending on dataset size
4. **Seeds**: Use fixed seeds for reproducibility

### Validation Strategy

1. **Always validate**: Check split completeness and overlaps
2. **Monitor statistics**: Ensure splits match expected fractions
3. **Save splits**: Store splits for reproducible experiments
4. **Cross-validate**: Use k-fold for robust performance estimates

## Extending the Framework

The splitting framework is designed for easy extension. To add new splitters:

1. **Inherit from AbstractSplitter**: Get common functionality for free
2. **Implement required methods**: `get_split_indices()` and `get_split_kfold_indices()`
3. **Add specific parameters**: Customize for your splitting strategy
4. **Maintain conventions**: Follow the naming and return format conventions

### Example Custom Splitter

```python
from pepbenchmark.splitter.base_splitter import AbstractSplitter

class MyCustomSplitter(AbstractSplitter):
    """Custom splitter implementation."""
    
    def get_split_indices(self, data, **kwargs):
        # Your custom splitting logic here
        pass
    
    def get_split_kfold_indices(self, data, **kwargs):
        # Your custom k-fold logic here
        pass
```

## Future Splitters

The framework is designed to accommodate future splitting strategies such as:

- **Stratified splitting**: Maintain class distribution across splits
- **Temporal splitting**: Time-based splits for time series data
- **Hierarchical splitting**: Multi-level splitting strategies
- **Domain-aware splitting**: Consider functional or structural domains
- **Phylogenetic splitting**: Evolutionary relationship-based splitting

## Troubleshooting

### Common Issues

1. **MMseqs2 not found**: Ensure MMseqs2 is installed and in PATH
2. **Memory issues**: Reduce dataset size or adjust MMseqs2 parameters
3. **Imbalanced splits**: Check clustering results and adjust parameters
4. **Slow performance**: Use caching and appropriate thread counts

### Performance Tips

1. **Use caching**: Reuse clustering results when possible
2. **Adjust threads**: Set appropriate thread count for your system
3. **Batch processing**: Process multiple splits efficiently
4. **Monitor resources**: Check memory and CPU usage during clustering
