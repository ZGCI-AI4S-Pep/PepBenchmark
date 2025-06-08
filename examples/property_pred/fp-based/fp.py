import random
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AtomPairs import Pairs, Torsions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import argparse
import os
import json
from pepbenchmark.metadata import DATASET_MAP

# External libraries for xgboost, lightgbm, and optuna tuning
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

import numpy as np
from rdkit.Chem import rdFingerprintGenerator as rfp
from utils import load_data, featurize, set_seed, FP_Converter

def parse_args():
    parser = argparse.ArgumentParser(description="fp-based property prediction with multi-model Optuna tuning and loading best params")
    parser.add_argument('--task', type=str, required=True, choices=list(DATASET_MAP.keys()), help="Task name from the dataset map")
    parser.add_argument('--model', type=str, default='rf', choices=['rf','adaboost','gradboost','knn','svm','xgboost','lightgbm'], help="Model to train or tune")
    parser.add_argument('--split_type', type=str, default="Random_split", choices=["Random_split", "Homology_based_split"], help="Split type")
    parser.add_argument('--split_index', type=str, default="random1", choices=["random1","random2","random3","random4","random5"], help="Split index")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--output_dir', type=str, default=None, help="Path to save trained model and parameters")
    parser.add_argument('--fp_type', type=str, default="ecfp", help="Fingerprint type")
    parser.add_argument('--nbits', type=int, default=2048, help="Number of bits in the fingerprint")
    parser.add_argument('--radius', type=int, default=3, help="Radius for fingerprint generation")
    parser.add_argument('--tune', action='store_true', help='Whether to perform hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Arguments: {args}")
    set_seed(args.random_seed)

    # Define available model configs
    model_configs = {
        'rf': RandomForestClassifier(random_state=args.random_seed),
        'adaboost': AdaBoostClassifier(random_state=args.random_seed),
        'gradboost': GradientBoostingClassifier(random_state=args.random_seed),
        'knn': KNeighborsClassifier(),
        'svm': SVC(probability=True, random_state=args.random_seed),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=args.random_seed),
        'lightgbm': LGBMClassifier(random_state=args.random_seed)
    }

    
    
    # Build file paths and load data



    meta = DATASET_MAP[args.task]
    base_dir = meta['path']
    print(f"Loading data for task '{args.task}' split '{args.split_type}/{args.split_index}'...")
    train_df = pd.read_csv(os.path.join(base_dir, args.split_type, args.split_index, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(base_dir, args.split_type, args.split_index, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, args.split_type, args.split_index, 'test.csv'))
    print(f"Data loaded: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    

    # train_seqs, train_labels = train_df['sequence'], train_df['label']
    # valid_seqs, valid_labels = valid_df['sequence'], valid_df['label']
    # test_seqs,  test_labels  = test_df['sequence'], test_df['label']


    

    save_name = f"{args.fp_type}_nbits{args.nbits}_radius{args.radius}"
    precomputed_path = os.path.join(base_dir, save_name,"combine.csv")
    if os.path.exists(precomputed_path):
        print(f"Using precomputed fingerprint file: {precomputed_path}")
        all_data = pd.read_csv(precomputed_path)
        
        train_df = all_data[all_data['sequence'].isin(train_df['sequence'])]
        valid_df = all_data[all_data['sequence'].isin(valid_df['sequence'])]
        test_df = all_data[all_data['sequence'].isin(test_df['sequence'])]

        train_features = [np.array([float(x) for x in fp.strip('[]').replace(',', ' ').split()]) for fp in train_df['fp'].tolist()]
        valid_features = [np.array([float(x) for x in fp.strip('[]').replace(',', ' ').split()]) for fp in valid_df['fp'].tolist()]
        test_features = [np.array([float(x) for x in fp.strip('[]').replace(',', ' ').split()]) for fp in test_df['fp'].tolist()]

    else:

        # Featurization
        print(f"Featurizing with {args.fp_type} (nbits={args.nbits}, radius={args.radius})...")
        conv = FP_Converter(type=args.fp_type, nbits=args.nbits, radius=args.radius)
        train_features = featurize(train_df["sequence"], conv)
        valid_features = featurize(valid_df["sequence"], conv)
        test_features  = featurize(test_df["sequence"], conv)
        print("Featurization complete.")
    train_labels = train_df['label'].tolist()
    valid_labels = valid_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    # Prepare output directory
    if args.output_dir is None:
        args.output_dir = os.path.join('checkpoints', args.task, args.split_type, args.model, f"{args.fp_type}_{args.nbits}_{args.radius}", args.split_index)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    params_path = os.path.join(args.output_dir, f'best_params_{args.model}.json')
    # Load or init model
    if not args.tune and os.path.exists(params_path):
        best_params = json.load(open(params_path))
        print(f"Loaded best params for {args.model} from {params_path}: {best_params}")
        model = model_configs[args.model].__class__(**best_params, random_state=args.random_seed)
    else:
        model = model_configs[args.model]
        best_params = None
        print(f"Initialized model {args.model} with default params: {model.get_params()}")

    # Hyperparameter tuning
    if args.tune:
        print(f"Starting Optuna tuning for model '{args.model}' with {args.n_trials} trials...")
        def objective(trial):
            # Define search space per model
            if args.model == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
                clf = RandomForestClassifier(**params, random_state=args.random_seed, n_jobs=-1)
            elif args.model == 'adaboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0)
                }
                clf = AdaBoostClassifier(**params, random_state=args.random_seed)
            elif args.model == 'gradboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
                }
                clf = GradientBoostingClassifier(**params, random_state=args.random_seed)
            elif args.model == 'knn':
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
                }
                clf = KNeighborsClassifier(**params)
            elif args.model == 'svm':
                kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
                params = {
                    'C': trial.suggest_loguniform('C', 1e-3, 10),
                    'kernel': kernel,
                    'gamma': trial.suggest_loguniform('gamma', 1e-4, 1.0),
                    'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
                }
                clf = SVC(**params, probability=True, random_state=args.random_seed, max_iter=int(1e4))
            elif args.model == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 1.0)
                }
                clf = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=args.random_seed)
            elif args.model == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                }
                clf = LGBMClassifier(**params, random_state=args.random_seed, verbose=-1)
            else:
                raise ValueError(f"Unknown model {args.model}")

            clf.fit(train_features, train_labels)
            preds = clf.predict_proba(valid_features)[:,1]
            auc = roc_auc_score(valid_labels, preds)
            return auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials)
        best_params = study.best_params
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved best params for {args.model}: {best_params}")
        model = model_configs[args.model].__class__(**best_params, random_state=args.random_seed)

    # Train and evaluate
    print(f"Training model '{args.model}'...")
    model.fit(train_features, train_labels)
    print("Training complete. Evaluating...")
    results = []
    for name, X, y in [('Train', train_features, train_labels), ('Validation', valid_features, valid_labels), ('Test', test_features, test_labels)]:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
        metrics = {
            'accuracy': accuracy_score(y, preds),
            'precision': precision_score(y, preds, zero_division=0),
            'recall': recall_score(y, preds, zero_division=0),
            'f1': f1_score(y, preds, zero_division=0),
            'roc_auc': roc_auc_score(y, probs) if probs is not None else None
        }
        print(f"{name} set metrics: {metrics}")
        results.append({**{'Model': args.model, 'Fingerprint': args.fp_type, 'nbits': args.nbits, 'radius': args.radius, 'Set': name}, **metrics})

    # Save outputs
    model_path = os.path.join(args.output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    metrics_path = os.path.join(args.output_dir, 'metrics.csv')
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print("Done.")
    
#python fp.py --task BBP_APML --split_type  Homology_based_split --fp_type ecfp --nbits 2048 --radius 3