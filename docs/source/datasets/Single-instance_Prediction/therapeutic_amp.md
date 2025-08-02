# Therapeutic Antimicrobial Peptide Datasets

Therapeutic antimicrobial peptide datasets cover various antimicrobial activity prediction tasks, including antibacterial, antifungal, antiviral, antiparasitic, and other different types of microbial activities.

## Dataset Overview

Therapeutic antimicrobial peptide datasets primarily focus on antimicrobial activity of peptide drugs, providing important predictive capabilities for antimicrobial drug development.

### Dataset Classification

- **Antibacterial Activity**: Predict peptide antibacterial activity
- **Antifungal Activity**: Predict peptide antifungal activity
- **Antiviral Activity**: Predict peptide antiviral activity
- **Antiparasitic Activity**: Predict peptide antiparasitic activity
- **Minimum Inhibitory Concentration**: Predict peptide minimum inhibitory concentration

## Detailed Datasets

### Antibacterial Peptides (antibacterial)

Predicts peptide antibacterial activity, which is significant for developing novel antimicrobial drugs.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 27,256
- **Sequence Length Range**: 2-150
- **Average Length**: 24.39
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.691
- **High Similarity Ratio**: 0.093
- **Application Scenarios**: Antimicrobial drug development, infectious disease treatment
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# Load antibacterial peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antibacterial", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antibacterial_labels = dataset.get_official_feature("label")
```

### Antifungal Peptides (antifungal)

Predicts peptide antifungal activity, which is significant for fungal infection treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 14,564
- **Sequence Length Range**: 2-148
- **Average Length**: 34.30
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.657
- **High Similarity Ratio**: 0.058
- **Application Scenarios**: Fungal infection treatment, agricultural applications
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load antifungal peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antifungal", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antifungal_labels = dataset.get_official_feature("label")
```

### Antiviral Peptides (antiviral)

Predicts peptide antiviral activity, which is significant for viral infection treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 7,308
- **Sequence Length Range**: 2-137
- **Average Length**: 22.70
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.620
- **High Similarity Ratio**: 0.057
- **Application Scenarios**: Viral infection treatment, vaccine development
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load antiviral peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antiviral", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antiviral_labels = dataset.get_official_feature("label")
```

### Antimicrobial Peptides (antimicrobial)

Predicts peptide broad-spectrum antimicrobial activity, including activity against multiple microorganisms.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 50,412
- **Sequence Length Range**: 2-150
- **Average Length**: 27.44
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.662
- **High Similarity Ratio**: 0.097
- **Application Scenarios**: Broad-spectrum antimicrobial drug development
- **Data Source**: -

**Usage Example:**
```python
# Load antimicrobial peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antimicrobial", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antimicrobial_labels = dataset.get_official_feature("label")
```

### Antiparasitic Peptides (antiparasitic)

Predicts peptide antiparasitic activity, which is significant for parasitic infection treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 7,662
- **Sequence Length Range**: 2-206
- **Average Length**: 36.94
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.639
- **High Similarity Ratio**: 0.021
- **Application Scenarios**: Parasitic infection treatment, tropical disease treatment
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load antiparasitic peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antiparasitic", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antiparasitic_labels = dataset.get_official_feature("label")
```

## Regression Datasets

### E.coli Minimum Inhibitory Concentration (E.coli_mic)

Predicts peptide minimum inhibitory concentration against E. coli, which is significant for assessing antimicrobial activity strength.

**Dataset Features:**
- **Task Type**: Regression
- **Total Samples**: 3,280
- **Sequence Length Range**: 2-190
- **Average Length**: 23.34
- **Median Length**: 20
- **Label Mean**: 1.082
- **Label Median**: 1.079
- **Label Standard Deviation**: 0.764
- **Label Range**: -1.155-3.220
- **Application Scenarios**: Antimicrobial activity strength assessment
- **Data Source**: Deep learning regression model for antimicrobial peptide design

**Usage Example:**
```python
# Load E.coli minimum inhibitory concentration dataset
dataset = SingleTaskDatasetManager(dataset_name="E.coli_mic", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
mic_values = dataset.get_official_feature("label")
```

### P.aeruginosa Minimum Inhibitory Concentration (P.aeruginosa_mic)

Predicts peptide minimum inhibitory concentration against P. aeruginosa.

**Dataset Features:**
- **Task Type**: Regression
- **Total Samples**: 1,608
- **Sequence Length Range**: 2-190
- **Average Length**: 22.36
- **Median Length**: 20
- **Label Mean**: 1.097
- **Label Median**: 1.079
- **Label Standard Deviation**: 0.692
- **Label Range**: -0.863-3.015
- **Application Scenarios**: Antimicrobial activity strength assessment
- **Data Source**: Deep learning regression model for antimicrobial peptide design

**Usage Example:**
```python
# Load P.aeruginosa minimum inhibitory concentration dataset
dataset = SingleTaskDatasetManager(dataset_name="P.aeruginosa_mic", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
mic_values = dataset.get_official_feature("label")
```

### S.aureus Minimum Inhibitory Concentration (S.aureus_mic)

Predicts peptide minimum inhibitory concentration against S. aureus.

**Dataset Features:**
- **Task Type**: Regression
- **Total Samples**: 2,887
- **Sequence Length Range**: 2-190
- **Average Length**: 22.80
- **Median Length**: 20
- **Label Mean**: 1.087
- **Label Median**: 1.032
- **Label Standard Deviation**: 0.748
- **Label Range**: -1.127-3.255
- **Application Scenarios**: Antimicrobial activity strength assessment
- **Data Source**: Deep learning regression model for antimicrobial peptide design

**Usage Example:**
```python
# Load S.aureus minimum inhibitory concentration dataset
dataset = SingleTaskDatasetManager(dataset_name="S.aureus_mic", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
mic_values = dataset.get_official_feature("label")
```

## Data Statistical Analysis

### Dataset Statistical Charts

```python
import matplotlib.pyplot as plt
import pandas as pd

# Therapeutic antimicrobial peptide dataset statistics
amp_datasets = {
    'antibacterial': {'samples': 27256, 'avg_length': 24.39, 'pos_ratio': 0.5},
    'antifungal': {'samples': 14564, 'avg_length': 34.30, 'pos_ratio': 0.5},
    'antiviral': {'samples': 7308, 'avg_length': 22.70, 'pos_ratio': 0.5},
    'antimicrobial': {'samples': 50412, 'avg_length': 27.44, 'pos_ratio': 0.5},
    'antiparasitic': {'samples': 7662, 'avg_length': 36.94, 'pos_ratio': 0.5}
}

# Create statistical charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Sample count comparison
datasets = list(amp_datasets.keys())
samples = [amp_datasets[d]['samples'] for d in datasets]
ax1.bar(datasets, samples)
ax1.set_title('Therapeutic Antimicrobial Peptide Dataset Sample Counts')
ax1.set_ylabel('Sample Count')
ax1.tick_params(axis='x', rotation=45)

# Average length comparison
avg_lengths = [amp_datasets[d]['avg_length'] for d in datasets]
ax2.bar(datasets, avg_lengths)
ax2.set_title('Therapeutic Antimicrobial Peptide Dataset Average Sequence Lengths')
ax2.set_ylabel('Average Length')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Application Scenarios

### Drug Development
Therapeutic antimicrobial peptide datasets play an important role in antimicrobial drug development:
- Novel antimicrobial drug discovery
- Drug-resistant strain treatment strategies
- Broad-spectrum antimicrobial drug design

### Infectious Disease Treatment
- Bacterial infection treatment
- Fungal infection treatment
- Viral infection treatment
- Parasitic infection treatment

### Agricultural Applications
- Crop protection
- Animal health
- Food safety

## Data Quality Assurance

### Data Preprocessing
- **Sequence Validation**: Ensure all sequence formats are correct
- **Label Consistency**: Verify correspondence between labels and sequences
- **Deduplication**: Remove duplicate sequences
- **Quality Filtering**: Filter low-quality data

### Data Standardization
- **Unified Format**: All datasets use FASTA format
- **Label Encoding**: Binary classification tasks use 0/1 encoding, regression tasks use continuous values
- **Sequence Length**: Record minimum, maximum, and average lengths

## Citation Information

When using therapeutic antimicrobial peptide datasets, please cite the following literature:

```bibtex
@article{autopeptideml2023,
  title={AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors},
  author={...},
  journal={...},
  year={2023}
}

@article{deeplearningregression2023,
  title={Deep learning regression model for antimicrobial peptide design},
  author={...},
  journal={...},
  year={2023}
}
```

## Next Steps
- View [Other Therapeutic Peptide Datasets](therapeutic_other.md)
- Learn about [ADME Datasets](adme.md)
- Explore [Toxicity and Safety Datasets](tox.md)
