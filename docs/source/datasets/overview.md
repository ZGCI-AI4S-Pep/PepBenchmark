# Dataset Overview

PepBenchmark provides a rich collection of standardized peptide datasets covering various biological activities and application scenarios. All datasets are carefully curated and validated to ensure data quality and consistency.

## Dataset Classification

### By Task Type

- **Single-instance prediction**: Predict whether a peptide possesses a specific property (e.g., antibacterial activity, cell penetration, etc.)
- **Multi-instance prediction**: Predict properties involving two input entities, such as protein-peptide interactions or binding affinities between peptides and proteins.
- **Generation**: Generate novel peptide sequences with desired properties.

### By Application Scenario

- **ADME**: Drug metabolism and pharmacokinetics related datasets
- **Therapeutic-AMP**: Therapeutic antimicrobial peptide datasets
- **Therapeutic-Other**: Other therapeutic peptide datasets
- **Tox**: Toxicity and safety related datasets
- **Interaction**: Peptide-protein interaction datasets

### By Peptide Type
- **Natural**: Datasets containing peptides composed only of canonical (natural) amino acids. These datasets are suitable for studying and developing peptide drugs and functional peptides based on natural amino acids.
- **Non-natural**: Datasets containing peptides with non-canonical (non-natural) amino acids (such as D-amino acids, modified residues, or artificially designed sequences).

## Dataset Features

### Data Quality Assurance
- **Deduplication**: All datasets undergo strict removal of duplicate sequences
- **Quality Validation**: Sequence format and label consistency verification
- **Standardization**: Unified data format and naming conventions
- **Complete Documentation**: Each dataset has detailed source and processing descriptions

### Data Distribution Characteristics

```python
# View basic statistical information of datasets
from pepbenchmark.metadata import get_dataset_statistics

stats = get_dataset_statistics()
print("Dataset Statistics:")
for category, datasets in stats.items():
    print(f"\n{category}:")
    for dataset_key, info in datasets.items():
        print(f"  {dataset_key}: {info['num_samples']} samples, "
              f"average length {info['avg_length']:.1f}, "
              f"standard deviation {info['length_std']:.1f}")
```

## Data Access and Usage

### Quick Loading

```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# Load any dataset
dataset = SingleTaskDatasetManager(dataset_name="antibacterial", official_feature_names=["fasta", "label"])
print(f"Dataset size: {len(dataset)}")

# Get sequences and labels
sequences = dataset.get_official_feature("fasta")
labels = dataset.get_official_feature("label")
```

### Batch Analysis

```python
from pepbenchmark.metadata import natural_binary_keys
from pepbenchmark.utils.analysis import DatasetAnalyzer

# Analyze all natural peptide binary classification datasets
analyzer = DatasetAnalyzer()
for dataset_key in natural_binary_keys:
    analysis = analyzer.analyze(dataset_key)
    print(f"{dataset_key}: {analysis.summary()}")
```

### Contribution Guidelines
We welcome community contributions of new datasets or improvements to existing datasets. Please refer to the [Contribution Guide](../contributing.md) to learn how to participate.

## Citations and Acknowledgments

When using PepBenchmark datasets, please cite the corresponding original data sources and this project:

```bibtex
@article{pepbenchmark2024,
  title={PepBenchmark: A Comprehensive Benchmark for Peptide},
  author={Your Team},
  journal={Journal Name},
  year={2024}
}
```

Each dataset page contains specific citation information. Please ensure proper citation of original data sources.

## Next Steps
- View detailed information on specific dataset categories
- Learn [official data loading](../user_guide/data_loading.md) techniques
- Understand how to [build custom datasets](../construct_dataset.md)
- Explore [example applications](../examples/)
