import argparse
import json
import os

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager
from pepbenchmark.utils.seed import set_seed
from pepbenchmark.evaluator import evaluate_classification
from xgboost import XGBClassifier
from pepbenchmark.raw_data import DATASET_MAP
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="fp-based hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(DATASET_MAP.keys()),
        help="Task name from the dataset map",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "adaboost", "gradboost", "knn", "svm", "xgboost", "lightgbm"],
        help="Model to tune",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="mmseqs2_split",
        choices=["random_split", "mmseqs2_split"],
        help="Split type",
    )
    parser.add_argument(
        "--fold_seed",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Fold seed for split",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./checkpoints',
        help="Path to save best parameters",
    )
    parser.add_argument("--fp_type", type=str, default="ecfp6", help="Fingerprint type",choices=["ecfp6", "ecfp4","RDKit","MACCS", "TopologicalTorsion", "AtomPair"])
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    return parser.parse_args()


def create_objective(model_name, X_train, y_train, X_valid, y_valid, random_seed):
    """Create objective function for Optuna optimization"""
    
    def objective(trial):
        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }
            clf = RandomForestClassifier(
                **params, random_state=random_seed, n_jobs=-1
            )
        elif model_name == "adaboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 1e-3, 1.0
                ),
            }
            clf = AdaBoostClassifier(**params, random_state=random_seed)
        elif model_name == "gradboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 1e-3, 1.0
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
            clf = GradientBoostingClassifier(
                **params, random_state=random_seed
            )
        elif model_name == "knn":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
            }
            clf = KNeighborsClassifier(**params)
        elif model_name == "svm":
            kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
            params = {
                "C": trial.suggest_loguniform("C", 1e-3, 10),
                "kernel": kernel,
                "gamma": trial.suggest_loguniform("gamma", 1e-4, 1.0),
                "degree": (
                    trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
                ),
            }
            clf = SVC(
                **params,
                probability=True,
                random_state=random_seed,
                max_iter=int(1e4),
            )
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 1e-3, 0.3
                ),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-6, 1.0),
            }
            clf = XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=random_seed,
            )
        elif model_name == "lightgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 1e-3, 0.3
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
            }
            clf = LGBMClassifier(
                **params, random_state=random_seed, verbose=-1
            )
        else:
            raise ValueError(f"Unknown model {model_name}")

        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        return auc
    
    return objective


def create_final_model(model_name, best_params, random_seed):
    """Create final model with best parameters"""
    if model_name == "rf":
        return RandomForestClassifier(**best_params, random_state=random_seed, n_jobs=-1)
    elif model_name == "adaboost":
        return AdaBoostClassifier(**best_params, random_state=random_seed)
    elif model_name == "gradboost":
        return GradientBoostingClassifier(**best_params, random_state=random_seed)
    elif model_name == "knn":
        return KNeighborsClassifier(**best_params)
    elif model_name == "svm":
        return SVC(**best_params, probability=True, random_state=random_seed, max_iter=int(1e4))
    elif model_name == "xgboost":
        return XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=random_seed,
        )
    elif model_name == "lightgbm":
        return LGBMClassifier(**best_params, random_state=random_seed, verbose=-1)
    else:
        raise ValueError(f"Unknown model {model_name}")


if __name__ == "__main__":
    args = parse_args()
    print(f"Arguments: {args}")
    set_seed(args.random_seed)

    # Load dataset
    dataset = SingleTaskDatasetManager(
        dataset_name=args.task,
        official_feature_names=["fasta", args.fp_type, "label"],
        force_download=False,
    )
    dataset.set_official_split_indices(split_type=args.split_type, fold_seed=args.fold_seed)

    # Get features
    train_features, valid_features, test_features = dataset.get_train_val_test_features(format="dict")
    X_train = train_features[f"official_{args.fp_type}"]
    y_train = train_features["official_label"]
    X_valid = valid_features[f"official_{args.fp_type}"]
    y_valid = valid_features["official_label"]
    X_test = test_features[f"official_{args.fp_type}"]
    y_test = test_features["official_label"]
    
    # Prepare output directory
    args.output_dir = os.path.join(
        args.output_dir,
        args.task,
        args.split_type,
        args.model,
        str(args.fold_seed),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Hyperparameter tuning
    print(
        f"Starting Optuna tuning for model '{args.model}' with {args.n_trials} trials..."
    )
    
    objective = create_objective(args.model, X_train, y_train, X_valid, y_valid, args.random_seed)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    best_params = study.best_params
    best_auc = study.best_value
    
    # Save best parameters
    params_path = os.path.join(args.output_dir, f"best_params_{args.model}.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best params for {args.model}: {best_params}")
    print(f"Saved best params to {params_path}")
    
    # Save tuning results
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(args.output_dir, f"tuning_trials_{args.model}.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"Tuning trials saved to {trials_path}")
    
    print("Hyperparameter tuning completed.")
    
    # Train final model with best parameters and evaluate
    print(f"\nTraining final model with best parameters...")
    final_model = create_final_model(args.model, best_params, args.random_seed)
    final_model.fit(X_train, y_train)
    
    print("Training complete. Evaluating...")
    results = []
    for name, X, y in [
        ("Train", X_train, y_train),
        ("Validation", X_valid, y_valid),
        ("Test", X_test, y_test),
    ]:
        preds = final_model.predict(X)
        probs = (
            final_model.predict_proba(X)[:, 1] if hasattr(final_model, "predict_proba") else None
        )

        metrics = evaluate_classification(y_true=y, y_pred=preds, y_score=probs)
        print(f"{name} set metrics: {metrics}")
        results.append(
            {
                **{
                    "Split":name,
                    "Model": args.model,
                    "Fingerprint": args.fp_type,
                },
                **metrics,
            }
        )

    # Save final model and metrics
    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(final_model, model_path)
    print(f"Final model saved to {model_path}")
    
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    print("Hyperparameter tuning and final model training completed.")

# python fp_tune.py --task AV_APML --split_type random_split --fp_type ecfp6 --model rf --n_trials 100
