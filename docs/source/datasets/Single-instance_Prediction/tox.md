# Toxicity and Safety Datasets

Toxicity and safety datasets cover toxicity assessment and safety prediction of peptide drugs, including hemolytic activity, toxicity, allergenicity, anti-mammalian cell activity, and various other toxicity-related activities.

## Dataset Overview

Toxicity and safety datasets primarily focus on toxicity assessment of peptide drugs, providing important predictive capabilities for drug safety evaluation.

### Dataset Classification

- **Hemolytic Activity**: Predict peptide hemolytic activity
- **Toxicity**: Predict general toxicity of peptides
- **Allergenicity**: Predict peptide allergen activity
- **Anti-mammalian Cell**: Predict peptide toxicity to mammalian cells

## Detailed Datasets

### Hemolytic Peptides (hemolytic)

Predicts peptide hemolytic activity, which is significant for assessing peptide drug toxicity to red blood cells.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 4,040
- **Sequence Length Range**: 2-50
- **Average Length**: 19.49
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.630
- **High Similarity Ratio**: 0.054
- **Application Scenarios**: Drug safety assessment, red blood cell toxicity evaluation
- **Data Source**: Peptidepedia, Hemolytik2, PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction

**Usage Example:**
```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# Load hemolytic peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="hemolytic", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
hemolytic_labels = dataset.get_official_feature("label")
```

### Hemolytic Peptide HC50 (hemolytic_hc50)

Predicts peptide hemolytic concentration, providing quantitative hemolytic activity assessment.

**Dataset Features:**
- **Task Type**: Regression
- **Total Samples**: 1,699
- **Sequence Length Range**: 6-39
- **Unit**: μM
- **Average Length**: 18.79
- **Median Length**: 18
- **Label Mean**: 117.22
- **Label Median**: 95.6
- **Label Standard Deviation**: 105.87
- **Label Range**: 0.19-474.8
- **Application Scenarios**: Hemolytic activity strength assessment, drug safety evaluation
- **Data Source**: Prediction of hemolytic peptides and their hemolytic concentration

**Usage Example:**
```python
# Load hemolytic peptide HC50 dataset
dataset = SingleTaskDatasetManager(dataset_name="hemolytic_hc50", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
hc50_values = dataset.get_official_feature("label")
```

### Toxic Peptides (toxicity)

Predicts general toxicity of peptides, which is significant for assessing overall safety of peptide drugs.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 6,158
- **Sequence Length Range**: 5-138
- **Average Length**: 33.80
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.583
- **High Similarity Ratio**: 0.024
- **Application Scenarios**: Drug safety assessment, toxicity prediction
- **Data Source**: Peptidepedia, AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors

**Usage Example:**
```python
# Load toxic peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="toxicity", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
toxicity_labels = dataset.get_official_feature("label")
```

### Allergenic Peptides (allergen)

Predicts peptide allergen activity, which is significant for assessing immune safety of peptide drugs.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 2,728
- **Sequence Length Range**: 4-150
- **Average Length**: 41.08
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.596
- **High Similarity Ratio**: 0.006
- **Application Scenarios**: Allergen prediction, immune safety assessment
- **Data Source**: Peptidepedia

**Usage Example:**
```python
# Load allergenic peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="allergen", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
allergen_labels = dataset.get_official_feature("label")
```

### Anti-mammalian Cell Peptides (anti_mammalian_cell)

Predicts peptide toxicity to mammalian cells, which is significant for assessing peptide drug toxicity to normal cells.

**Dataset Features:**
- **Task Type**: Binary Classification
- **Total Samples**: 10,956
- **Sequence Length Range**: 2-133
- **Average Length**: 18.21
- **Positive Sample Ratio**: 0.5
- **Positive Sample Average Similarity**: 0.709
- **High Similarity Ratio**: 0.118
- **Application Scenarios**: Cell toxicity assessment, drug safety evaluation
- **Data Source**: Peptidepedia

**Usage Example:**
```python
# Load anti-mammalian cell peptide dataset
dataset = SingleTaskDatasetManager(dataset_name="anti_mammalian_cell", official_feature_names=["fasta", "label"])
sequences = dataset.get_official_feature("fasta")
anti_mammalian_cell_labels = dataset.get_official_feature("label")
```

## Data Statistical Analysis

### Dataset Statistical Charts

```python
import matplotlib.pyplot as plt
import pandas as pd

# Toxicity and safety dataset statistics
toxicity_datasets = {
    'hemolytic': {'samples': 4040, 'avg_length': 19.49, 'pos_ratio': 0.5},
    'toxicity': {'samples': 6158, 'avg_length': 33.80, 'pos_ratio': 0.5},
    'allergen': {'samples': 2728, 'avg_length': 41.08, 'pos_ratio': 0.5},
    'anti_mammalian_cell': {'samples': 10956, 'avg_length': 18.21, 'pos_ratio': 0.5}
}

# Create statistical charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Sample count comparison
datasets = list(toxicity_datasets.keys())
samples = [toxicity_datasets[d]['samples'] for d in datasets]
ax1.bar(datasets, samples)
ax1.set_title('Toxicity and Safety Dataset Sample Counts')
ax1.set_ylabel('Sample Count')
ax1.tick_params(axis='x', rotation=45)

# Average length comparison
avg_lengths = [toxicity_datasets[d]['avg_length'] for d in datasets]
ax2.bar(datasets, avg_lengths)
ax2.set_title('Toxicity and Safety Dataset Average Sequence Lengths')
ax2.set_ylabel('Average Length')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Application Scenarios

### Drug Safety Assessment
Toxicity and safety datasets play an important role in the drug development process:
- Candidate drug toxicity assessment
- Red blood cell toxicity prediction
- Cell toxicity assessment
- Immune safety assessment

### Drug Screening
- High-throughput toxicity screening
- Safety prediction model development
- Toxicity mechanism research

### Regulatory Compliance
- Drug approval safety assessment
- Clinical trial safety prediction
- Post-marketing safety monitoring

## Toxicity Mechanisms

### Hemolytic Mechanisms
- **Membrane Disruption**: Direct disruption of red blood cell membrane structure
- **Ion Channels**: Formation of ion channels leading to cell death
- **Membrane Permeability**: Increased cell membrane permeability

### Cell Toxicity Mechanisms
- **Apoptosis**: Induction of programmed cell death
- **Necrosis**: Leading to cell necrosis
- **Metabolic Interference**: Interference with normal cell metabolism

### Immune Toxicity Mechanisms
- **Allergic Reactions**: Induction of allergic reactions
- **Immune Activation**: Over-activation of the immune system
- **Inflammatory Response**: Induction of inflammatory responses

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

When using toxicity and safety datasets, please cite the following literature:

```bibtex
@article{peptidepedia2023,
  title={Peptidepedia: A comprehensive database of peptide bioactivity},
  author={...},
  journal={...},
  year={2023}
}

@article{hemolytik22023,
  title={Hemolytik2: A comprehensive database of hemolytic peptides},
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

@article{autopeptideml2023,
  title={AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors},
  author={...},
  journal={...},
  year={2023}
}

@article{hemolyticprediction2023,
  title={Prediction of hemolytic peptides and their hemolytic concentration},
  author={...},
  journal={...},
  year={2023}
}
```

## Next Steps
- View [Therapeutic Antimicrobial Peptide Datasets](therapeutic_amp.md)
- Learn about [Other Therapeutic Peptide Datasets](therapeutic_other.md)
- Explore [ADME Datasets](adme.md)
