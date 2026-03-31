# PepBenchmark

Peptide therapeutics are widely regarded as the *third generation of drugs*. However, progress in peptide Machine Learning (ML) has been hindered by the lack of standardized benchmarks.

We present **PepBenchmark**, a comprehensive framework that unifies datasets, preprocessing pipelines, and evaluation protocols for peptide drug discovery.

PepBenchmark consists of three major components:

1. **PepBenchData**  
   A well-curated dataset collection including:
   - 29 canonical peptide datasets  
   - 6 non-canonical peptide datasets  
   - Organized into 7 task groups  


2. **PepBenchPipeline**  
   A standardized preprocessing pipeline for:
   - Data cleaning  
   - Dataset construction  
   - Data splitting  
   - Feature transformation  
   This ensures consistency and avoids common issues in ad hoc pipelines.

3. **PepBenchLeaderboard**  
   A unified evaluation protocol with strong baselines across four model families:
   - Fingerprint-based (FP)
   - Graph Neural Network-based (GNN)
   - Protein Language Model-based (PLM)
   - SMILES-based models

For more details, please refer to our paper:  
**PepBenchmark: A Standardized Benchmark for Peptide Machine Learning**

---

## ⚠️ Important Update

We use:
- `run_seeds = [42, 43, 44, 45, 46]` → controls training randomness  
- `fold_seeds = [0, 1, 2, 3, 4]` → controls dataset splitting randomness  

Each experiment pairs:
```
(run_seed=42, fold_seed=0)
(run_seed=43, fold_seed=1)
(run_seed=44, fold_seed=2)
(run_seed=45, fold_seed=3)
(run_seed=46, fold_seed=4)
```

Therefore, performance variance reflects:
- Model training randomness
- Data split variability

We observed that **fold_seed significantly affects results** on some datasets.

👉 Future update:  
We will provide experiments with **fixed data splits (same fold_seed)** and varying `run_seed` to better evaluate training stability.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate pepbenchmark
python -m pip install -e .
```

---

## Repository Structure

```
src/pepbenchmark       # Core library
docs/source            # Documentation (Sphinx)
notebook               # Usage examples
PepBenchData           # Benchmark datasets
tests                  # Unit tests
```

---

## Package Overview

- `pepbenchmark.analyze` – sequence & SMILES analysis  
- `pepbenchmark.cluster` – clustering for deduplication & splitting  
- `pepbenchmark.dataset_manager` – dataset loaders  
- `pepbenchmark.model` – baseline models  
- `pepbenchmark.neg_sampler` – negative sample generation  
- `pepbenchmark.parser` – FASTA, HELM, BiLN parsing  
- `pepbenchmark.pep_utils` – feature engineering tools  
- `pepbenchmark.redundancy` – deduplication utilities  
- `pepbenchmark.similarity` – similarity computation  
- `pepbenchmark.splitter` – data splitting strategies  
- `pepbenchmark.utils` – logging & reproducibility  

---

## Basic Usage

### 1. Load Dataset

Datasets can be automatically downloaded from HuggingFace.

```python
from pepbenchmark.metadata import DatasetGroup, DatasetType, list_datasets

print(list_datasets())
print(len(list_datasets()))

for group in DatasetGroup:
    print(group, len(list_datasets(group=group)))

for dataset_type in DatasetType:
    print(dataset_type, len(list_datasets(dataset_type=dataset_type)))
```

```python
from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager

manager = SinglePeptideDatasetManager(
    "ace_inhibitory",
    official_feature_names=["fasta", "label"],
    dataset_dir="../PepBenchData/PepBenchData-50",
)

sequences = manager.get_feature("fasta")
labels = manager.get_feature("label")

splits = manager.set_official_split_indices(
    split_type="hybrid_split",
    fold_seed=0
)

print(len(splits["train"]))
print(len(splits["valid"]))
print(len(splits["test"]))
```

---

### 2. Train Model

We currently support four model families:
- FP-based
- GNN-based
- PLM-based
- SMILES-based

Example (Random Forest):

```python
from pepbenchmark.model.fp_model import build_fp_model
from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager

dataset = SinglePeptideDatasetManager(
    "ace_inhibitory",
    official_feature_names=["fasta", "ecfp6", "label"],
    dataset_dir="../PepBenchData/PepBenchData-50",
)

runner = build_fp_model(
    "rf",
    model_name="rf_test",
    dataset=dataset,
    task_type="binary_classification",
    fingerprint_type="ecfp6",
    n_estimators=100,
)

multi_results = runner.run_multi(
    run_seeds=[42, 43, 44, 45, 46],
    fold_seeds=[0, 1, 2, 3, 4],
    split_type="hybrid_split",
    base_dir="./results",
)

print(multi_results.to_dataframe())
summary = multi_results.get_summary_stats()
print(summary["test_roc-auc"])
```

---

### 3. Feature Generation

Supported transformations:
- FASTA / SMILES / HELM / BiLN conversions  
- Embedding generation via HuggingFace models  

```python
from pepbenchmark.pep_utils import convert

print(convert.FormatTransform.supported_transforms())

fasta2emb = convert.Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
embedding = fasta2emb(["ALAGGGPCR", "ALLLG"])

print(embedding.shape)  # (640,)
```

```python
helm2fasta = convert.Helm2Fasta()
helm2smiles = convert.Helm2Smiles()

helm = "PEPTIDE1{[d(N->O)Gly(allyl)].P.I.[meV].[meA].[bAla]}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$V2.0"

print(helm2fasta(helm))
print(helm2smiles(helm))
```

---

## PepBenchData

Datasets are automatically downloaded from HuggingFace:
https://huggingface.co/datasets/jiahuizhang/PepBenchData

Alternatively, manually download and place under project root.

⚠️ Notes:
- Data is processed for prediction tasks:
  - Deduplication reduces dataset size
  - Negative sampling may introduce UniRef sequences (unverified)

If you prefer raw data:
https://huggingface.co/datasets/jiahuizhang/PepBenchData_raw

---

## PepBenchPipeline

For custom dataset construction, use:
- Negative sampling  
- Data splitting  
- Feature generation  
Example pipeline (see `notebook/pipeline`):


---

## PepBenchLeaderboard

Reproducible experiments are available in:
```
notebook/model
```

---


## Pretrained Models

### ESM-150M Fine-tuned Model

We provide a peptide-specific fine-tuned ESM-150M model.

Training details:
https://github.com/ZGCI-AI4S-Pep/peptide-esm
---

## Non-Canonical Peptide Generation

We trained a model that:
- Inputs canonical peptides  
- Generates non-canonical peptides  

Used for improving negative sampling in non-canonical datasets.

---

## Citation

If you use PepBenchmark, please cite:

```bibtex
@inproceedings{zhang2026pepbenchmark,
  title={PepBenchmark: A Standardized Benchmark for Peptide Machine Learning},
  author={Zhang, Jiahui and Wang, Rouyi and Zhou, Kuangqi and Xiao, Tianshu and Zhu, Lingyan and Min, Yaosen and Wang, Yang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=NskQgtSdll}
}
```
