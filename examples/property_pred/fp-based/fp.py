import argparse
import json
import os

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager
from pepbenchmark.utils.seed import set_seed
from xgboost import XGBClassifier
from pepbenchmark.raw_data import DATASET_MAP

from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP
from pepbenchmark.evaluator import evaluate_classification
from pepbenchmark.utils.seed import set_seed
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="fp-based property prediction"
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
        help="Model to train",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="mmseqs2_split",
        choices=["random_split", "mmseqs2_split", "cdhit_split"],
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
        help="Path to save trained model and parameters",
    )
    parser.add_argument("--fp_type", type=str, default="ecfp6", help="Fingerprint type",choices=["ecfp6", "ecfp4","RDKit","MACCS", "TopologicalTorsion", "AtomPair"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Arguments: {args}")
    set_seed(args.random_seed)

    # Define available model configs
    model_configs = {
        "rf": RandomForestClassifier(random_state=args.random_seed),
        "adaboost": AdaBoostClassifier(random_state=args.random_seed),
        "gradboost": GradientBoostingClassifier(random_state=args.random_seed),
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True, random_state=args.random_seed),
        "xgboost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=args.random_seed,
        ),
        "lightgbm": LGBMClassifier(random_state=args.random_seed),
    }

    dataset = SingleTaskDatasetManager(
        dataset_name=args.task,
        official_feature_names=["fasta", args.fp_type, "label"],
        force_download=False,
    )
    dataset.set_official_split_indices(split_type=args.split_type, fold_seed=args.fold_seed)

    
    
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

    params_path = os.path.join(args.output_dir, f"best_params_{args.model}.json")
    # Load model with best params if available
    if os.path.exists(params_path):
        best_params = json.load(open(params_path))
        print(f"Loaded best params for {args.model} from {params_path}: {best_params}")
        model = model_configs[args.model].__class__(
            **best_params, random_state=args.random_seed
        )
    else:
        model = model_configs[args.model]
        print(
            f"Initialized model {args.model} with default params: {model.get_params()}"
        )

    # Train and evaluate
    print(f"Training model '{args.model}'...")
    model.fit(X_train, y_train)
    print("Training complete. Evaluating...")
    results = []
    for name, X, y in [
        ("Train", X_train, y_train),
        ("Validation", X_valid, y_valid),
        ("Test", X_test, y_test),
    ]:
        preds = model.predict(X)
        probs = (
            model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
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

    # Save outputs
    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print("Done.")

# python fp.py --task AV_APML --split_type random_split --fp_type ecfp6
