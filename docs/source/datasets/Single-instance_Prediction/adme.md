# ADME Datasets

ADME (Absorption, Distribution, Metabolism, Excretion) datasets cover the pharmacokinetic properties of peptide drugs in vivo, including absorption, distribution, metabolism, and excretion-related prediction tasks.

## Dataset Overview

ADME datasets primarily focus on the pharmacokinetic properties of peptide drugs, providing important predictive capabilities for drug development.

### Dataset Classification

- **Cell Penetration**: Assess peptide cell membrane penetration ability
- **Blood-Brain Barrier Penetration**: Predict peptide ability to cross the blood-brain barrier
- **Anti-fouling Properties**: Assess peptide anti-fouling characteristics

## Detailed Datasets

### Cell Penetrating Peptides (cpp)

Assesses peptide cell membrane penetration ability, which is significant for drug delivery system design.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 1,914
- **Sequence Length Range**: 3-61
- **Average Length**: 17.03
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.641
- **High Similarity Ratio**: 0.034
- **Application Scenarios**: Drug delivery, cell biology research
- **Data Source**: PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids

**Usage Example:**
```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# Load cell penetrating peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="cpp", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
penetration_labels = dataset.get_official_feature("label")
```

### Blood-Brain Barrier Penetration (bbp)

Predicts peptide ability to cross the blood-brain barrier, which is crucial for central nervous system drug development.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 644
- **Sequence Length Range**: 2-82
- **Average Length**: 14.40
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.535
- **High Similarity Ratio**: 0.006
- **Application Scenarios**: Central nervous system drug development
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load blood-brain barrier penetration dataset
dataset = SingleTaskDatasetManager(dataset_name="bbp", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
bbp_labels = dataset.get_official_feature("label")
```

### Anti-fouling Properties (nonfouling)

Assesses peptide anti-fouling characteristics, which is significant for biomaterial surface modification and medical device development.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 7,110
- **Sequence Length Range**: 5-11
- **Average Length**: 6.11
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.629
- **High Similarity Ratio**: 0.000
- **Application Scenarios**: Biomaterials, medical devices
- **Data Source**: PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction

**Usage Example:**
```python
# Load anti-fouling properties dataset
dataset = SingleTaskDatasetManager(dataset_name="nonfouling", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
nonfouling_labels = dataset.get_official_feature("label")
```

## Data Statistical Analysis

### Dataset Statistical Charts

```python
import matplotlib.pyplot as plt
import pandas as pd

# ADME dataset statistics
adme_datasets = {
    'cpp': {'samples': 1914, 'avg_length': 17.03, 'pos_ratio': 0.5},
    'bbp': {'samples': 644, 'avg_length': 14.40, 'pos_ratio': 0.5},
    'nonfouling': {'samples': 7110, 'avg_length': 6.11, 'pos_ratio': 0.5}
}

# Create statistical charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Sample count comparison
datasets = list(adme_datasets.keys())
samples = [adme_datasets[d]['samples'] for d in datasets]
ax1.bar(datasets, samples)
ax1.set_title('ADME Dataset Sample Counts')
ax1.set_ylabel('Sample Count')

# Average length comparison
avg_lengths = [adme_datasets[d]['avg_length'] for d in datasets]
ax2.bar(datasets, avg_lengths)
ax2.set_title('ADME Dataset Average Sequence Lengths')
ax2.set_ylabel('Average Length')

plt.tight_layout()
plt.show()
```

## Application Scenarios

### Drug Development
ADME datasets play an important role in the drug development process, helping researchers:
- Assess candidate peptide cell penetration ability
- Predict drug ability to cross the blood-brain barrier
- Optimize drug delivery system design

### Biomaterial Development
Anti-fouling peptide datasets provide important references for biomaterial surface modification:
- Medical device surface functionalization
- Biosensor development
- Tissue engineering material design

## Data Quality Assurance

### Data Preprocessing
- **Sequence Validation**: Ensure all sequence formats are correct
- **Label Consistency**: Verify correspondence between labels and sequences
- **Deduplication**: Remove duplicate sequences
- **Quality Filtering**: Filter low-quality data

### Data Standardization
- **Unified Format**: All datasets use FASTA format
- **Label Encoding**: Binary classification tasks use 0/1 encoding
- **Sequence Length**: Record minimum, maximum, and average lengths

## Citation Information

When using ADME datasets, please cite the following literature:

```bibtex
@article{pepland2023,
  title={PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids},
  author={...},
  journal={...},
  year={2023}
}

@article{autopeptideml2023,
  title={AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors},
  author={...},
  journal={...},
  year={2023}
}

@article{peptidebert2023,
  title={PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction},
  author={...},
  journal={...},
  year={2023}
}
```

## Next Steps
- View [Therapeutic Antimicrobial Peptide Datasets](therapeutic_amp.md)
- Learn about [Other Therapeutic Peptide Datasets](therapeutic_other.md)
- Explore [Toxicity and Safety Datasets](tox.md)
