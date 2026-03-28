# Cluster Module - Peptide Clustering and Redundancy Removal

The `cluster` module provides a unified interface for sequence clustering with multiple backend implementations, enabling easy grouping of similar peptides and removal of redundant sequences from datasets.

## Overview

Sequence clustering is a fundamental preprocessing step that:
- **Groups similar sequences** by identity or similarity threshold
- **Identifies redundant sequences** for removal or aggregation
- **Reduces dataset size** while preserving diversity
- **Prepares data** for model training (splitting by clusters preserves sequence diversity)

The module abstracts away differences between clustering algorithms through a factory pattern and unified result container, allowing seamless method switching with identical interfaces.

## Module Architecture

### Core Components

#### 1. **ClusterFactory** (`factory.py`)
- **Purpose**: Factory pattern for creating clusterer instances
- **Key Methods**:
  - `ClusterFactory.create_clusterer(method, config)`: Main factory method
  - `ClusterFactory.list_available_methods()`: Show registered backends
  - Individual creators: `create_cdhit()`, `create_mmseqs2()`, etc.
- **Benefit**: Unified creation interface regardless of backend

#### 2. **AbstractClusterer Interface** (`interfaces.py`)
```python
class AbstractClusterer:
    def cluster_sequences(sequences, **kwargs) -> UnifiedClusterResult:
        """Cluster input sequences and return unified result"""
```
- **Purpose**: Abstract base class all implementations must follow
- **Benefit**: Consistent behavior and API across all methods
- **Required Method**: `cluster_sequences()` returning `UnifiedClusterResult`

#### 3. **ClusterConfig** (`interfaces.py`)
- **Purpose**: Configuration container for clustering parameters
- **Key Fields**:
  - `method`: "cdhit", "mmseqs2", "motif", "similarity", "molecular"
  - `threshold`: Sequence identity threshold (0.0-1.0)
  - `coverage`: Coverage threshold (typically 0.8-1.0)
  - Additional method-specific parameters
- **Benefit**: Type-safe configuration with defaults

#### 4. **UnifiedClusterResult** (`interfaces.py`)
- **Purpose**: Standardized output format from all clustering methods
- **Key Attributes**:
  - `clusters`: List[List[int]] - cluster membership indices
  - `representatives`: List[int] - representative sequence per cluster
  - `centers`: Optional[List[str]] - consensus/representative sequences
  - `similarities`: Optional[np.ndarray] - pairwise distances
  - `metadata`: Dict - method-specific information
- **Key Methods**:
  - `.cluster_count()`: Number of clusters formed
  - `.cluster_distribution()`: Statistics on cluster sizes
  - `.get_cluster(cluster_id)`: Get sequences in specific cluster
  - `.get_representatives(include_members)`: Extract cluster representatives
  - `.summary_stats()`: Summary statistics as dict

### Clustering Implementations

#### 1. **CD-HIT** (`cdhit_cluster.py`)
- **Method**: Fast greedy clustering by sequence identity
- **Pros**: Fast, widely used, straightforward
- **Cons**: Order-dependent results, no overlap control
- **Best For**: Quick preprocessing, large datasets
- **Parameters**:
  - `threshold` (default: 0.9): Sequence identity threshold
  - `word_size` (default: 5): Word size for clustering
- **Output**: Fast deterministic clusters

#### 2. **MMseqs2** (`mmseqs_cluster.py`)
- **Method**: Sensitive clustering using k-mer matching and alignment
- **Pros**: Fast, highly sensitive, better quality than CD-HIT
- **Cons**: More complex setup, higher memory use
- **Best For**: High-quality clustering, strict thresholds
- **Parameters**:
  - `threshold` (default: 0.9): Sequence identity threshold
  - `coverage` (default: 0.8): Query/target coverage threshold
  - `e_value` (default: 1e-3): E-value threshold
- **Output**: Higher sensitivity clusters with coverage control

#### 3. **Motif-based** (`motif_cluster.py`)
- **Method**: Clustering based on conserved motif patterns
- **Pros**: Domain-aware clustering, biologically meaningful
- **Cons**: Requires motif predictions, slower
- **Best For**: Domain-aware analysis, structural grouping
- **Parameters**:
  - `motif_type`: "iupred" or "secondary"
  - `min_length`: Minimum motif length
- **Output**: Clusters based on conserved regions

#### 4. **Similarity-based** (`similarity_cluster.py`)
- **Method**: Clustering from precomputed similarity matrix
- **Pros**: Flexible, can use custom similarity metrics
- **Cons**: Requires pre-computed matrix (memory intensive)
- **Best For**: Custom similarity metrics, benchmark comparisons
- **Parameters**:
  - `similarity_matrix`: Precomputed (N, N) matrix
  - `threshold`: Similarity cutoff (0.0-1.0)
- **Output**: Communities from similarity graph

#### 5. **Molecular** (`molecular_cluster.py`)
- **Method**: ECFP fingerprint-based molecular clustering
- **Pros**: Chemical structure-aware, complementary to sequence
- **Cons**: Requires RDKit, structure-dependent
- **Best For**: Structure-property modeling, chemical similarity
- **Parameters**:
  - `smiles_list`: SMILES strings per sequence
  - `threshold`: Tanimoto similarity cutoff
- **Output**: Structure-based clusters

## Public API

### Main Entry Points

```python
from pepbenchmark.cluster import cluster_sequences, remove_redundancy

# Basic clustering
result = cluster_sequences(
    sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY", ...],
    method="cdhit",  # or "mmseqs2", "motif", "similarity", "molecular"
    threshold=0.9
)

# Redundancy removal with representative selection
representatives = remove_redundancy(
    sequences=sequences,
    method="cdhit",
    threshold=0.9,
    strategy="first"  # or "longest", "central", "all"
)
```

### Factory Functions

```python
from pepbenchmark.cluster import (
    create_clusterer,
    create_cdhit_clusterer,
    create_mmseqs2_clusterer,
    create_motif_clusterer,
    create_similarity_clusterer,
    create_molecular_clusterer,
    list_available_methods
)

# Method 1: Generic factory
clusterer = create_clusterer(method="mmseqs2", threshold=0.9, coverage=0.8)
result = clusterer.cluster_sequences(sequences)

# Method 2: Specific creators
cdhit = create_cdhit_clusterer(threshold=0.9, word_size=5)
result = cdhit.cluster_sequences(sequences)

# Check available methods
methods = list_available_methods()
# Output: ["cdhit", "mmseqs2", "motif", "similarity", "molecular"]
```

### Core Classes

```python
from pepbenchmark.cluster import (
    AbstractClusterer,
    ClusterConfig,
    ClusterFactory,
    UnifiedClusterResult
)

# Direct instantiation
clusterer = ClusterFactory.create_clusterer(method="similarity", similarity_threshold=0.9)
result = clusterer.cluster_sequences(sequences, similarity_matrix=sim_matrix)

# Inspect results
print(f"Total clusters: {result.total_clusters}")
stats = result.get_statistics()
print(stats)
```

## Method Selection Guide

### Quick Decision Tree

1. **Speed critical?** → CD-HIT
2. **High quality clusters needed?** → MMseqs2
3. **Domain/motif aware?** → Motif-based
4. **Custom similarity metric?** → Similarity-based
5. **Structure important?** → Molecular

### Detailed Comparison

| Method | Speed | Quality | Memory | Setup | Use Case |
|--------|-------|---------|--------|-------|----------|
| CD-HIT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Low | Simple | Large datasets, quick preprocessing |
| MMseqs2 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | Moderate | Production clustering, strict thresholds |
| Motif | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Complex | Domain analysis, structural grouping |
| Similarity | ⭐⭐ | ⭐⭐⭐⭐ | High | Complex | Custom metrics, research/comparison |
| Molecular | ⭐⭐⭐ | ⭐⭐⭐⭐ | High | Complex | Structure-aware modeling |

## Workflow Examples

### Example 1: Basic Preprocessing
```python
from pepbenchmark.cluster import cluster_sequences

# Remove redundancy at 90% identity
sequences = [...]  # Your peptide list
result = cluster_sequences(sequences, method="cdhit", threshold=0.9)

# Get one representative per cluster
representatives = [sequences[i] for i in result.representatives]
print(f"Reduced {len(sequences)} sequences to {len(representatives)} clusters")
```

### Example 2: Quality-focused Clustering
```python
from pepbenchmark.cluster import create_mmseqs2_clusterer

clusterer = create_mmseqs2_clusterer(threshold=0.95, coverage=0.9)
result = clusterer.cluster_sequences(sequences)

# Analyze cluster distribution
stats = result.summary_stats()
print(f"Largest cluster: {stats['max_cluster_size']} sequences")
print(f"Average cluster size: {stats['mean_cluster_size']:.2f}")
```

### Example 3: Method Comparison
```python
from pepbenchmark.cluster import list_available_methods, create_clusterer

methods = list_available_methods()
results = {}

for method in methods:
    try:
        clusterer = create_clusterer(method=method, threshold=0.9)
        results[method] = clusterer.cluster_sequences(sequences)
    except Exception as e:
        print(f"Method {method} not available: {e}")

# Compare results
for method, result in results.items():
    print(f"{method}: {result.cluster_count()} clusters")
```

### Example 4: Redundancy Removal with Strategy
```python
from pepbenchmark.cluster import remove_redundancy

# Get longest sequence per cluster (better for model training)
reps_longest = remove_redundancy(
    sequences=sequences,
    method="cdhit",
    threshold=0.9,
    strategy="longest"
)

# Get central sequence (most similar to cluster members)
reps_central = remove_redundancy(
    sequences=sequences,
    method="mmseqs2",
    threshold=0.95,
    strategy="central"
)
```

## Integration with Other Modules

### With Similarity Module
```python
from pepbenchmark.similarity import SimilarityAnalyzer
from pepbenchmark.cluster import create_similarity_clusterer

# Compute similarity matrix
analyzer = SimilarityAnalyzer(sequences)
sim_matrix = analyzer.compute_matrix()

# Use for custom clustering
clusterer = create_similarity_clusterer(
    similarity_matrix=sim_matrix,
    threshold=0.8
)
result = clusterer.cluster_sequences(sequences)
```

### With Redundancy Module
```python
from pepbenchmark.cluster import cluster_sequences
from pepbenchmark.redundancy import RedundancyAnalyzer

# First, understand redundancy
sequences = [...]
sim_matrix = ...  # from similarity module
analyzer = RedundancyAnalyzer(sequences, sim_matrix)
report = analyzer.compute_metrics(thresholds=(0.7, 0.8, 0.9))

# Then cluster at recommended threshold
threshold = report.recommendation.suggested_threshold
result = cluster_sequences(sequences, method="cdhit", threshold=threshold)
```

## Common Patterns

### Pattern 1: Iterative Threshold Selection
```python
from pepbenchmark.cluster import create_cdhit_clusterer

sequences = [...]
for threshold in [0.85, 0.90, 0.95, 0.99]:
    clusterer = create_cdhit_clusterer(threshold=threshold)
    result = clusterer.cluster_sequences(sequences)
    print(f"Threshold {threshold}: {result.cluster_count()} clusters")
```

### Pattern 2: Representative Selection Comparison
```python
from pepbenchmark.cluster import cluster_sequences

result = cluster_sequences(sequences, method="mmseqs2")

# Different representative strategies
first = result.get_representatives(strategy="first")
longest = result.get_representatives(strategy="longest")
central = result.get_representatives(strategy="central")

print(f"First: {len(first)} representatives")
print(f"Longest: {len(longest)} representatives")
print(f"Central: {len(central)} representatives")
```

### Pattern 3: Method Fallback Chain
```python
from pepbenchmark.cluster import list_available_methods, create_clusterer

def safe_cluster(sequences, threshold=0.9):
    """Try methods in order, fallback to next if unavailable"""
    preferred_methods = ["mmseqs2", "cdhit", "similarity"]
    
    for method in preferred_methods:
        try:
            clusterer = create_clusterer(method=method, threshold=threshold)
            return clusterer.cluster_sequences(sequences), method
        except ImportError:
            continue
    
    raise RuntimeError("No clustering method available")

result, method_used = safe_cluster(sequences)
print(f"Clustered with {method_used}")
```

## Performance Considerations

### Speed Benchmarks (on 1000 sequences)
- **CD-HIT**: ~0.5s (fastest)
- **MMseqs2**: ~2s
- **Similarity-based**: ~30s (requires matrix computation)
- **Motif-based**: ~5s (depends on motif prediction)
- **Molecular**: ~3s (requires SMILES/structure)

### Memory Usage
- **CD-HIT**: O(n²) in worst case, typically O(n log n)
- **MMseqs2**: O(n) with k-mer indexing
- **Similarity-based**: O(n²) for full matrix storage
- **Motif-based**: O(n) for predictions
- **Molecular**: O(n) for fingerprints

### Recommendations
- **< 1000 sequences**: Any method works; use best quality available
- **1000-10000 sequences**: Use CD-HIT or MMseqs2; avoid similarity matrix
- **> 10000 sequences**: Use CD-HIT; consider batch processing
- **Memory constrained**: CD-HIT > MMseqs2 > Molecular > Motif > Similarity

## Troubleshooting

### ImportError: CD-HIT/MMseqs2 not found
- **Issue**: Clustering method not installed or not in PATH
- **Solution**: Install via conda (`conda install cd-hit mmseqs2`) or use alternative method

### "No clustering method available"
- **Issue**: All clustering backends failed
- **Solution**: Check installed packages; use similarity-based with precomputed matrix as fallback

### Slow performance with large datasets
- **Issue**: Memory or computation bottleneck
- **Solution**: 
  - Use CD-HIT for speed
  - Process in batches
  - Use less strict threshold
  - Try different word_size/parameters

### Method-specific unavailable
- **Issue**: "ImportError: ModuleNotFoundError" for specific method
- **Solution**: That method's dependencies not installed; use alternative method

## API Reference

### Main Functions

#### `cluster_sequences()`
```python
def cluster_sequences(
    sequences: List[str],
    method: str = "cdhit",
    threshold: float = 0.9,
    **kwargs
) -> UnifiedClusterResult:
    """Cluster sequences using specified method"""
```

#### `remove_redundancy()`
```python
def remove_redundancy(
    sequences: List[str],
    method: str = "cdhit",
    threshold: float = 0.9,
    cluster_result: Optional[UnifiedClusterResult] = None,
    strategy: str = "first",
    **kwargs
) -> List[str]:
    """Remove redundant sequences, return representatives"""
```

### Configuration

#### `ClusterConfig`
- `method`: str - Clustering method name
- `threshold`: float - Identity/similarity threshold (0-1)
- `coverage`: Optional[float] - Coverage threshold (MMseqs2)
- `word_size`: Optional[int] - Word size (CD-HIT)
- Additional method-specific parameters in kwargs

### Result Container

#### `UnifiedClusterResult`
- **Attributes**:
  - `clusters: List[List[int]]` - Cluster membership
  - `representatives: List[int]` - Representative indices
  - `centers: Optional[List[str]]` - Consensus sequences
  - `similarities: Optional[np.ndarray]` - Distance matrix
  - `metadata: Dict` - Method-specific info
- **Methods**:
  - `cluster_count() -> int`
  - `cluster_distribution() -> Dict[int, int]`
  - `get_cluster(cluster_id: int) -> List[int]`
  - `get_representatives(strategy: str) -> List[int]`
  - `summary_stats() -> Dict`

## Related Modules

- **[similarity module](../similarity/README.md)**: Compute sequence similarity matrices and pairwise distances
- **[redundancy module](../redundancy/README.md)**: Analyze dataset redundancy and recommend preprocessing thresholds
- **Dataset Manager**: Apply clustering to prepare train/test splits

## References

- **CD-HIT**: Li & Godzik (2006). Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences.
- **MMseqs2**: Steinegger & Söding (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.
- **Clustering principles**: Comprehensive guide in similarity and redundancy modules
