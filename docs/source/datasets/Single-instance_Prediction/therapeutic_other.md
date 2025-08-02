# Other Therapeutic Peptide Datasets

Other therapeutic peptide datasets cover various therapeutic activities beyond antimicrobial activity, including anticancer, antidiabetic, antioxidant, anti-inflammatory, anti-aging, neuropeptide, quorum sensing, ACE inhibition, DPP-IV inhibition, and various other biological activities.

## Dataset Overview

Other therapeutic peptide datasets primarily focus on other therapeutic activities of peptide drugs, providing important predictive capabilities for various disease treatments.

### Dataset Classification

- **Anticancer Activity**: Predict peptide anticancer activity
- **Antidiabetic Activity**: Predict peptide antidiabetic activity
- **Antioxidant Activity**: Predict peptide antioxidant activity
- **Anti-inflammatory Activity**: Predict peptide anti-inflammatory activity
- **Anti-aging Activity**: Predict peptide anti-aging activity
- **Neuropeptide Activity**: Predict peptide neuropeptide activity
- **Quorum Sensing**: Predict peptide quorum sensing activity
- **ACE Inhibition**: Predict peptide ACE inhibitory activity
- **DPP-IV Inhibition**: Predict peptide DPP-IV inhibitory activity

## Detailed Datasets

### Anticancer Peptides (anticancer)

Predicts peptide anticancer activity, which is significant for cancer treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 12,364
- **Sequence Length Range**: 2-145
- **Average Length**: 28.11
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.647
- **High Similarity Ratio**: 0.045
- **Application Scenarios**: Cancer treatment, tumor-targeted therapy
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# Load anticancer peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="anticancer", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
anticancer_labels = dataset.get_official_feature("label")
```

### Antidiabetic Peptides (antidiabetic)

Predicts peptide antidiabetic activity, which is significant for diabetes treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 2,606
- **Sequence Length Range**: 2-46
- **Average Length**: 9.29
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.607
- **High Similarity Ratio**: 0.008
- **Application Scenarios**: Diabetes treatment, blood glucose regulation
- **Data Source**: Peptidepedia, Discovery of potential antidiabetic peptides using deep learning

**Usage Example:**
```python
# Load antidiabetic peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antidiabetic", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antidiabetic_labels = dataset.get_official_feature("label")
```

### Antioxidant Peptides (antioxidant)

Predicts peptide antioxidant activity, which is significant for oxidative stress-related disease treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 2,248
- **Sequence Length Range**: 2-69
- **Average Length**: 8.21
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.586
- **High Similarity Ratio**: 0.008
- **Application Scenarios**: Oxidative stress treatment, aging-related diseases
- **Data Source**: Peptidepedia

**Usage Example:**
```python
# Load antioxidant peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antioxidant", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antioxidant_labels = dataset.get_official_feature("label")
```

### Anti-inflammatory Peptides (antiinflamatory)

Predicts peptide anti-inflammatory activity, which is significant for inflammatory disease treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 7,644
- **Sequence Length Range**: 2-107
- **Average Length**: 16.62
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.531
- **High Similarity Ratio**: 0.002
- **Application Scenarios**: Inflammatory disease treatment, immune regulation
- **Data Source**: Peptidepedia

**Usage Example:**
```python
# Load anti-inflammatory peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antiinflamatory", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antiinflamatory_labels = dataset.get_official_feature("label")
```

### Anti-aging Peptides (antiaging)

Predicts peptide anti-aging activity, which is significant for aging-related disease treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 550
- **Sequence Length Range**: 2-80
- **Average Length**: 10.01
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.562
- **High Similarity Ratio**: 0.000
- **Application Scenarios**: Aging-related disease treatment, cosmetic medicine
- **Data Source**: Peptidepedia

**Usage Example:**
```python
# Load anti-aging peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="antiaging", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
antiaging_labels = dataset.get_official_feature("label")
```

### Neuropeptides (neuropeptide)

Predicts peptide neuropeptide activity, which is significant for nervous system disease treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 8,628
- **Sequence Length Range**: 2-149
- **Average Length**: 18.27
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.674
- **High Similarity Ratio**: 0.026
- **Application Scenarios**: Nervous system disease treatment, neural regulation
- **Data Source**: Peptidepedia, AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load neuropeptide dataset
dataset = SingleTaskDatasetManager(dataset_name="neuropeptide", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
neuropeptide_labels = dataset.get_official_feature("label")
```

### Quorum Sensing Peptides (quorum_sensing)

Predicts peptide quorum sensing activity, which is significant for bacterial communication and biofilm formation research.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 470
- **Sequence Length Range**: 3-48
- **Average Length**: 9.91
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.666
- **High Similarity Ratio**: 0.021
- **Application Scenarios**: Bacterial communication research, biofilm control
- **Data Source**: Peptidepedia, AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load quorum sensing peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="quorum_sensing", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
quorum_sensing_labels = dataset.get_official_feature("label")
```

### ACE Inhibitory Peptides (ace_inhibitory)

Predicts peptide ACE inhibitory activity, which is significant for hypertension treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 4,356
- **Sequence Length Range**: 2-81
- **Average Length**: 7.19
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.675
- **High Similarity Ratio**: 0.012
- **Application Scenarios**: Hypertension treatment, cardiovascular diseases
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load ACE inhibitory peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="ace_inhibitory", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
ace_inhibitory_labels = dataset.get_official_feature("label")
```

### DPP-IV Inhibitory Peptides (dppiv_inhibitors)

Predicts peptide DPP-IV inhibitory activity, which is significant for diabetes treatment.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 1,168
- **Sequence Length Range**: 2-33
- **Average Length**: 5.74
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.626
- **High Similarity Ratio**: 0.021
- **Application Scenarios**: Diabetes treatment, blood glucose regulation
- **Data Source**: AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load DPP-IV inhibitory peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="dppiv_inhibitors", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
dppiv_inhibitors_labels = dataset.get_official_feature("label")
```

### TTCA Peptides (ttca)

Predicts peptide tricarboxylic acid cycle-related activity, which is significant for metabolic regulation.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 1,150
- **Sequence Length Range**: 8-20
- **Average Length**: 9.24
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.569
- **High Similarity Ratio**: 0.014
- **Application Scenarios**: Metabolic regulation, energy metabolism
- **Data Source**: Peptidepedia, AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load TTCA peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="ttca", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
ttca_labels = dataset.get_official_feature("label")
```

## Data Statistical Analysis

### Dataset Statistical Charts

```python
import matplotlib.pyplot as plt
import pandas as pd

# Other therapeutic peptide dataset statistics
therapeutic_datasets = {
    'anticancer': {'samples': 12364, 'avg_length': 28.11, 'pos_ratio': 0.5},
    'antidiabetic': {'samples': 2606, 'avg_length': 9.29, 'pos_ratio': 0.5},
    'antioxidant': {'samples': 2248, 'avg_length': 8.21, 'pos_ratio': 0.5},
    'antiinflamatory': {'samples': 7644, 'avg_length': 16.62, 'pos_ratio': 0.5},
    'antiaging': {'samples': 550, 'avg_length': 10.01, 'pos_ratio': 0.5},
    'neuropeptide': {'samples': 8628, 'avg_length': 18.27, 'pos_ratio': 0.5},
    'quorum_sensing': {'samples': 470, 'avg_length': 9.91, 'pos_ratio': 0.5},
    'ace_inhibitory': {'samples': 4356, 'avg_length': 7.19, 'pos_ratio': 0.5},
    'dppiv_inhibitors': {'samples': 1168, 'avg_length': 5.74, 'pos_ratio': 0.5},
    'ttca': {'samples': 1150, 'avg_length': 9.24, 'pos_ratio': 0.5}
}

# Create statistical charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Sample count comparison
datasets = list(therapeutic_datasets.keys())
samples = [therapeutic_datasets[d]['samples'] for d in datasets]
ax1.bar(datasets, samples)
ax1.set_title('Other Therapeutic Peptide Dataset Sample Counts')
ax1.set_ylabel('Sample Count')
ax1.tick_params(axis='x', rotation=45)

# Average length comparison
avg_lengths = [therapeutic_datasets[d]['avg_length'] for d in datasets]
ax2.bar(datasets, avg_lengths)
ax2.set_title('Other Therapeutic Peptide Dataset Average Sequence Lengths')
ax2.set_ylabel('Average Length')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Application Scenarios

### Disease Treatment
Other therapeutic peptide datasets play an important role in various disease treatments:
- Cancer treatment and tumor targeting
- Diabetes treatment and blood glucose regulation
- Inflammatory disease treatment
- Nervous system disease treatment
- Cardiovascular disease treatment

### Drug Development
- Novel therapeutic peptide drug discovery
- Targeted drug design
- Personalized treatment plans

### Biomedical Research
- Disease mechanism research
- Drug mechanism of action research
- Biomarker discovery

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

When using other therapeutic peptide datasets, please cite the following literature:

```bibtex
@article{peptidepedia2023,
  title={Peptidepedia: A comprehensive database of peptide bioactivity},
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

@article{antidiabetic2023,
  title={Discovery of potential antidiabetic peptides using deep learning},
  author={...},
  journal={...},
  year={2023}
}
```

## Next Steps
- View [Therapeutic Antimicrobial Peptide Datasets](therapeutic_amp.md)
- Learn about [ADME Datasets](adme.md)
- Explore [Toxicity and Safety Datasets](tox.md)
