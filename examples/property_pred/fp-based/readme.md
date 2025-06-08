## Project Title

A brief description of your project.

---

## Table of Contents

- [Project Title](#project-title)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Fingerprint Generation](#fingerprint-generation)
- [Training and Evaluation](#training-and-evaluation)
  - [Single Run](#single-run)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Batch Experiments](#batch-experiments)
- [Supported Models and Splits](#supported-models-and-splits)
- [Scripts](#scripts)
- [License](#license)

---

## Features

* Generate molecular fingerprints (ECFP, FCFP)
* Train multiple ML models (RF, AdaBoost, GBDT, KNN, SVM, XGBoost, LightGBM)
* Support homology-based and random dataset splits
* Automated hyperparameter tuning with Optuna
* Batch execution scripts for reproducibility

---

## Prerequisites

* [Conda](https://docs.conda.io/en/latest/)
* Python 3.10+

---

## Environment Setup

```bash
# Create and activate a new conda environment
conda create --name fp_env python=3.10 -y
conda activate fp_env

# Install dependencies
conda install scikit-learn rdkit numpy pandas -y
```

---

## Fingerprint Generation

Generate molecular fingerprints for each dataset before training.

```bash
python convert.py \
  --fp_type ecfp \   # fingerprint type: ecfp or fcfp
  --radius 3 \       # radius for circular fingerprints
  --nbits 2048        # number of bits in fingerprint vector
```

* **`--fp_type`**: `ecfp` | `fcfp`
* **`--radius`**: integer, e.g., 2 or 3
* **`--nbits`**: integer, e.g., 1024 or 2048

The script will save the resulting fingerprint vectors (e.g., `*.npy` or `*.csv`) under a designated `data/fingerprints/` directory.

---

## Training and Evaluation

### Single Run

Train a single model on a chosen dataset, split, and fingerprint configuration:

```bash
python fp.py \
  --task BBP_APML \                        # dataset name
  --split_type Homology_based_split \       # split method
  --model rf \                               # model: rf, adaboost, gradboost, knn, svm, xgboost, lightgbm
  --split_index random1 \                    # split index: random1–random5
  --fp_type ecfp \                           # fingerprint type
  --nbits 2048 \                             # fingerprint bit size
  --radius 3                                 # fingerprint radius
```

The results (performance metrics, model artifacts) will be saved in `checkpoints/<task>/<split_type>/<model>/<split_index>/`.

### Hyperparameter Tuning

Add `--tune` and specify number of trials to automatically optimize hyperparameters with Optuna:

```bash
python fp.py \
  --task BBP_APML \
  --split_type Homology_based_split \
  --model rf \
  --split_index random1 \
  --fp_type ecfp \
  --nbits 2048 \
  --radius 3 \
  --tune \                       # enable hyperparameter tuning
  --n_trials 20                   # number of Optuna trials
```

Tuned hyperparameters and best model will be stored under `checkpoints/<task>/<split_type>/<model>`.

### Batch Experiments

Run all models on one dataset with 5 random splits:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Run all datasets, all models, all splits:

```bash
chmod +x run_all.sh
./run_all.sh
```

These scripts loop over tasks, split types, split indices, and models, automating fingerprint generation, training, and evaluation.

---

## Supported Models and Splits

| Split Type             | Description             |
| ---------------------- | ----------------------- |
| `Homology_based_split` | Sequence/homology-based |
| `Random_split`         | Random stratified split |

| Model     | Library      |
| --------- | ------------ |
| rf        | scikit-learn |
| adaboost  | scikit-learn |
| gradboost | scikit-learn |
| knn       | scikit-learn |
| svm       | scikit-learn |
| xgboost   | XGBoost      |
| lightgbm  | LightGBM     |

---

## Scripts

* **`convert.py`**: Generate fingerprints
* **`fp.py`**: Train/evaluate models, with optional tuning
* **`run_experiments.sh`**: Batch run one dataset across models and splits
* **`run_all.sh`**: Batch run all datasets, models, splits

Each script includes `--help` for detailed arguments.

```bash
python convert.py --help
python fp.py --help
```

---

## License

Specify your project’s license here.
