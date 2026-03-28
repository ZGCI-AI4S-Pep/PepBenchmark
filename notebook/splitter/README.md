# Splitter Module

`pepbenchmark.splitter` provides multiple data splitting strategies for train/validation/test partitioning and split quality analysis.

It covers two major use cases:
- **Standard random / rule-based splits**: such as `RandomSplitter` and `ColdSplitter`
- **Leakage-aware splits based on clustering results**: such as `MMseqs2Splitter`, `CDHitSplitter`, `UnifiedClusterSplitter`, and `UnifiedResultSplitter`

## Module Structure

### 1. Base Classes
- `BaseSplitter`
- `AbstractSplitter`
- `AbstractClusteringSplitter`

These classes define the shared interface:
- `get_split_indices(...)`
- `get_split_indices_n(...)`
- `get_split_kfold_indices(...)`

### 2. Concrete Splitters
- `RandomSplitter`: the simplest random split
- `ColdSplitter`: cold-start / entity-aware split
- `MMseqs2Splitter`: split after MMseqs2-based clustering
- `CDHitSplitter`: split after CD-HIT-based clustering
- `ECFPSplitter`: split based on molecular fingerprint similarity
- `MotifSplitter`: split based on motif clustering
- `KmerSplitter`: split based on k-mer clustering
- `HybridSplitter`: hybrid clustering split

### 3. Unified Result-Driven Splitters
This is currently the recommended set of entry points:

- `UnifiedClusterSplitter`: split using an existing `UnifiedClusterResult`
- `UnifiedResultSplitter`: generate single-run, repeated, or k-fold splits directly from a cluster result
- `split_from_clustering_result(...)`: convenience function

### 4. Analysis and Validation
- `SplitAnalyzer`
- `analyze_split_class_distribution(...)`
- `analyze_cross_dataset_similarity(...)`
- `detect_potential_data_leakage(...)`

## Recommended Entry Points

### A. Simplest Option: Random Split

```python
from pepbenchmark.splitter import RandomSplitter

splitter = RandomSplitter()
splits = splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42,
)
```

### B. Factory-Based Construction

```python
from pepbenchmark.splitter import create_splitter, list_available_splitters

print(list_available_splitters(include_aliases=True))

splitter = create_splitter("random")
splits = splitter.get_split_indices(sequences, seed=42)
```

### C. Split from Existing Clustering Results (Recommended)

```python
from pepbenchmark.cluster import create_clusterer
from pepbenchmark.splitter import UnifiedResultSplitter

clusterer = create_clusterer(method="similarity", similarity_threshold=0.9)
cluster_result = clusterer.cluster_sequences(sequences, similarity_matrix=sim_matrix)

splitter = UnifiedResultSplitter(random_seed=42)
splits = splitter.get_split_indices(
    data=sequences,
    cluster_result=cluster_result,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
)
```

## Public API

```python
from pepbenchmark.splitter import (
    BaseSplitter,
    AbstractSplitter,
    AbstractClusteringSplitter,
    RandomSplitter,
    ColdSplitter,
    MMseqs2Splitter,
    CDHitSplitter,
    ECFPSplitter,
    MotifSplitter,
    KmerSplitter,
    HybridSplitter,
    UnifiedClusterSplitter,
    UnifiedResultSplitter,
    SplitAnalyzer,
    get_splitter,
    create_splitter,
    list_available_splitters,
    split_from_clustering_result,
)
```

## When to Use Each Splitter

- **For a baseline**: `RandomSplitter`
- **To avoid entity leakage / cold start**: `ColdSplitter`
- **To reuse an existing clustering result**: `UnifiedClusterSplitter` / `UnifiedResultSplitter`
- **To focus on homologous-sequence leakage**: `MMseqs2Splitter` / `CDHitSplitter`
- **To focus on motif or k-mer patterns**: `MotifSplitter` / `KmerSplitter`
- **For molecular tasks**: `ECFPSplitter`

## API Summary

### `list_available_splitters()`
Returns the names of currently available splitters.

### `get_splitter(name)` / `create_splitter(name)`
Create a splitter by name, with alias support:
- `mmseqs2` -> `mmseqs`
- `hydra` -> `hybrid`

### `UnifiedResultSplitter`
Best suited when:
- clustering has already been performed
- you want to try multiple split strategies on the same cluster result
- you want to generate repeated splits or k-fold partitions

## Notes

- Some splitters depend on external tools such as MMseqs2 and CD-HIT.
- Some splitters depend on additional libraries such as RDKit or motif-related modules.
- Prefer the public API over importing concrete implementations from individual files.

## Minimal Example

```python
from pepbenchmark.splitter import RandomSplitter

sequences = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
splitter = RandomSplitter()
result = splitter.get_split_indices(sequences, seed=42)

print({k: len(v) for k, v in result.items()})
```

## Related Modules

- [cluster/README.md](../cluster/README.md): cluster first, then split
- [similarity/README.md](../similarity/README.md): provides matrices for similarity-based splitting
- [redundancy/README.md](../redundancy/README.md): assess redundancy first, then design split thresholds
