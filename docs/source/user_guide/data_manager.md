# Dataset Management

Use the dataset-manager package when you want to load official benchmark artifacts instead of rebuilding features from scratch.

## Main Entry Points

- `SinglePeptideDatasetManager` for single-peptide datasets.
- `PPIDatasetManager` for protein-peptide interaction datasets.

## Typical Usage

```python
from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager

manager = SinglePeptideDatasetManager(
    "ace_inhibitory",
    official_feature_names=["fasta", "label"],
    dataset_dir="PepBenchData/PepBenchData-50",
)
```

Use `set_user_feature()` when you want to inject your own feature arrays after loading an official base dataset.
