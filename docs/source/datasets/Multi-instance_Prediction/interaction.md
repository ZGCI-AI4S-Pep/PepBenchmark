# Protein Interaction Datasets

Protein interaction datasets cover the prediction of interactions between peptides and proteins, including peptide-protein binding site prediction and binding affinity prediction tasks.

## Dataset Overview

Protein interaction datasets primarily focus on interactions between peptides and proteins, providing important predictive capabilities for drug target discovery and protein function research.

### Dataset Classification

- **Peptide-Protein Interaction**: Predict interaction between peptides and proteins
- **Binding Affinity**: Predict binding affinity between peptides and proteins

## Detailed Datasets

### Peptide-Protein Interaction (PpI)

Predicts binding sites between peptides and proteins, which is significant for drug target discovery.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 9,499
- **Sequence Length Range**: -
- **Average Length**: -
- **Positive Sample Ratio**: -
- **Positive Sample Average Similarity**: -
- **High Similarity Ratio**: -
- **Application Scenarios**: Drug target discovery, protein function research
- **Data Source**: De novo design of peptide binders to conformationally diverse targets with contrastive language modeling, PepNN: a deep attention model for the identification of peptide binding sites

**Usage Example:**
```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# Load peptide-protein interaction dataset
dataset = SingleTaskDatasetManager(dataset_name="PpI", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
ppi_labels = dataset.get_official_feature("label")
```

### Peptide-Protein Binding Affinity (Pep_PI)

Predicts binding affinity between peptides and proteins, providing quantitative assessment of interaction strength.

**Dataset Features:**
- **Task Type**: Regression
- **Total Samples**: 1,806
- **Sequence Length Range**: -
- **Average Length**: -
- **Median Length**: -
- **Label Mean**: -
- **Label Median**: -
- **Label Standard Deviation**: -
- **Label Range**: -
- **Application Scenarios**: Binding affinity assessment, drug design
- **Data Source**: PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids

**Usage Example:**
```python
# Load peptide-protein binding affinity dataset
dataset = SingleTaskDatasetManager(dataset_name="Pep_PI", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
binding_affinity_values = dataset.get_official_feature("label")
```

## Data Statistical Analysis

### Dataset Statistical Charts

```python
import matplotlib.pyplot as plt
import pandas as pd

# Protein interaction dataset statistics
interaction_datasets = {
    'PpI': {'samples': 9499, 'avg_length': None, 'pos_ratio': None},
    'Pep_PI': {'samples': 1806, 'avg_length': None, 'pos_ratio': None}
}

# Create statistical charts
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Sample count comparison
datasets = list(interaction_datasets.keys())
samples = [interaction_datasets[d]['samples'] for d in datasets]
ax.bar(datasets, samples)
ax.set_title('Protein Interaction Dataset Sample Counts')
ax.set_ylabel('Sample Count')

plt.tight_layout()
plt.show()
```

## Application Scenarios

### Drug Development
Protein interaction datasets play an important role in drug development:
- Drug target discovery
- Peptide drug design
- Protein function research
- Drug-target interaction prediction

### Bioinformatics
- Protein function annotation
- Protein network analysis
- Evolutionary relationship studies

### Structural Biology
- Protein structure prediction
- Binding site analysis
- Molecular docking studies

## Interaction Mechanisms

### Binding Site Identification
- **Sequence Patterns**: Identify sequence features of binding sites
- **Structural Features**: Analyze structural features of binding sites
- **Evolutionary Conservation**: Study evolutionary conservation of binding sites

### Binding Affinity
- **Molecular Recognition**: Molecular recognition mechanisms between peptides and proteins
- **Binding Strength**: Quantitative assessment of binding strength
- **Specificity**: Assessment of binding specificity

## Data Quality Assurance

### Data Preprocessing
- **Sequence Validation**: Ensure all sequence formats are correct
- **Label Consistency**: Verify correspondence between labels and sequences
- **Deduplication**: Remove duplicate sequences
- **Quality Filtering**: Filter low-quality data

### Data Standardization
- **Unified Format**: All datasets use standard formats
- **Label Encoding**: Binary classification tasks use 0/1 encoding, regression tasks use continuous values
- **Sequence Length**: Record minimum, maximum, and average lengths

## Citation Information

When using protein interaction datasets, please cite the following literature:

```bibtex
@article{denovodesign2023,
  title={De novo design of peptide binders to conformationally diverse targets with contrastive language modeling},
  author={...},
  journal={...},
  year={2023}
}

@article{pepnn2023,
  title={PepNN: a deep attention model for the identification of peptide binding sites},
  author={...},
  journal={...},
  year={2023}
}

@article{pepland2023,
  title={PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids},
  author={...},
  journal={...},
  year={2023}
}
```

## Next Steps
- View [Therapeutic Antimicrobial Peptide Datasets](therapeutic_amp.md)
- Learn about [Other Therapeutic Peptide Datasets](therapeutic_other.md)
- Explore [Toxicity and Safety Datasets](tox.md)
