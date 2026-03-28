from dataclasses import dataclass, asdict
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from pepbenchmark.dataset_manager.single_dataset import SinglePeptideDatasetManager
from pepbenchmark.evaluator import (
    Classification_Metric,
    Regression_Metric,
    evaluate_classification,
    evaluate_regression,
)
from pepbenchmark.utils.logging import get_logger
from pepbenchmark.utils.seed import set_seed

logger = get_logger()


# =========================
# Result Containers
# =========================
@dataclass
class ModelResults:
    train_metrics: Dict[str, float]
    valid_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time: float
    train_samples: int
    valid_samples: int
    test_samples: int
    random_seed: int
    fold_seed: int
    model_name: str


@dataclass
class MultiRunResults:
    results: List[ModelResults]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for res in self.results:
            row = {
                "random_seed": res.random_seed,
                "fold_seed": res.fold_seed,
                "model_name": res.model_name,
                "train_samples": res.train_samples,
                "valid_samples": res.valid_samples,
                "test_samples": res.test_samples,
                "training_time": res.training_time,
            }

            for prefix, metrics in (
                ("train", res.train_metrics),
                ("valid", res.valid_metrics),
                ("test", res.test_metrics),
            ):
                for metric_name, metric_value in metrics.items():
                    row[f"{prefix}_{metric_name}"] = metric_value

            rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_stats(self) -> Dict[str, Any]:
        df = self.to_dataframe()
        summary: Dict[str, Any] = {}

        for col in df.columns:
            if not col.startswith(("train_", "valid_", "test_")):
                continue
            if not np.issubdtype(df[col].dtype, np.number):
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            summary[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        return summary

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)

        df = self.to_dataframe()
        summary = self.get_summary_stats()

        df.to_csv(os.path.join(save_dir, "multi_run_results.csv"), index=False)

        with open(os.path.join(save_dir, "multi_run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


# =========================
# Base Model
# =========================
class FPModel:
    """Fingerprint-based ML base model."""

    model_type: Optional[str] = None

    def __init__(
        self,
        model_name: str,
        dataset: Optional[SinglePeptideDatasetManager] = None,
        task_type: str = "binary_classification",
        fingerprint_type: str = "ecfp6",
        metrics: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        **model_params,
    ):
        self.model_name = model_name
        self.model_type = model_type or self.model_type or model_name
        self.dataset = dataset
        self.task_type = task_type
        self.fingerprint_type = fingerprint_type
        self.model_params = model_params

        self.experiment_dir: Optional[str] = None
        self.estimator: Optional[Any] = None
        self.training_time: float = 0.0
        self.random_seed: Optional[int] = None

        if metrics is not None:
            self.metrics = metrics
        elif task_type == "regression":
            self.metrics = Regression_Metric
        elif task_type == "binary_classification":
            self.metrics = Classification_Metric
        else:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                "Currently only supports 'binary_classification' and 'regression'."
            )

    # -------------------------
    # To be implemented by subclasses
    # -------------------------
    def build_estimator(self) -> Any:
        raise NotImplementedError

    # -------------------------
    # IO / Paths
    # -------------------------
    def get_experiment_dir(
        self,
        base_dir: str,
        split_type: str,
        fold_seed: int,
        run_seed: Optional[int] = None,
    ) -> str:
        dataset_name = self.dataset.dataset_name if self.dataset is not None else "unknown_dataset"
        path = os.path.join(base_dir, dataset_name, self.model_name, split_type, f"fold_{fold_seed}")
        if run_seed is not None:
            path = os.path.join(path, f"run_{run_seed}")
        return path

    def save_model(self, name: str = "model.joblib") -> None:
        if self.estimator is None:
            raise ValueError("Estimator is not trained yet.")
        if self.experiment_dir is None:
            raise ValueError("experiment_dir is not set.")

        path = os.path.join(self.experiment_dir, name)
        joblib.dump(self.estimator, path)
        logger.info(f"[SAVE MODEL] {self.model_name} saved to {path}")

    def save_results(self, results: ModelResults) -> None:
        if self.experiment_dir is None:
            raise ValueError("experiment_dir is not set.")

        logger.info(f"[SAVE RESULTS] Saving results at {self.experiment_dir}")

        config = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "fingerprint_type": self.fingerprint_type,
            "metrics": list(self.metrics) if hasattr(self.metrics, "__iter__") else self.metrics,
            "input_model_params": self.model_params,
            "resolved_model_params": (
                self.estimator.get_params() if self.estimator is not None and hasattr(self.estimator, "get_params") else {}
            ),
            "dataset_name": self.dataset.dataset_name if self.dataset else None,
        }

        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(self.experiment_dir, "metrics.json"), "w") as f:
            json.dump(asdict(results), f, indent=2)

    # -------------------------
    # Train / Predict / Eval
    # -------------------------
    def train(self, train_features: Dict[str, Any]) -> None:
        logger.info(f"[BUILD MODEL] {self.model_name}")
        self.estimator = self.build_estimator()

        X_train = train_features[self.fingerprint_type]
        y_train = train_features["label"]

        start = time.time()
        self.estimator.fit(X_train, y_train)
        self.training_time = time.time() - start

        logger.info(f"[TRAIN DONE] {self.model_name} in {self.training_time:.2f}s")
        self.save_model()

    def _predict(self, X: Any) -> Tuple[np.ndarray, Any]:
        if self.estimator is None:
            raise ValueError("Estimator is not trained or loaded.")

        preds = self.estimator.predict(X)
        probs = None

        if hasattr(self.estimator, "predict_proba"):
            try:
                probs = self.estimator.predict_proba(X)
                if self.task_type == "binary_classification":
                    probs = probs[:, 1]
            except Exception as e:
                logger.warning(f"[PREDICT_PROBA FAILED] {e}")

        if probs is None and hasattr(self.estimator, "decision_function"):
            from scipy.special import expit

            scores = self.estimator.decision_function(X)
            probs = expit(scores)

        if probs is None:
            probs = preds

        return preds, probs

    def evaluate(self, features: Dict[str, Any], split: str = "test") -> Dict[str, float]:
        X = features[self.fingerprint_type]
        y = features["label"]
        preds, probs = self._predict(X)

        if self.task_type == "binary_classification":
            metrics = evaluate_classification(y_true=y, y_pred=preds, y_score=probs)
        elif self.task_type == "regression":
            metrics = evaluate_regression(y_true=y, y_pred=preds)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        logger.info(f"[METRICS] {split}: {metrics}")
        return metrics

    # -------------------------
    # Run
    # -------------------------
    def run(
        self,
        run_seed: int = 42,
        fold_seed: int = 0,
        split_type: str = "random_split",
        base_dir: str = "./results",
    ) -> ModelResults:
        if self.dataset is None:
            raise ValueError("dataset cannot be None in run().")

        logger.info(
            f"[RUN] {self.model_name} | seed={run_seed}, fold={fold_seed}, split={split_type}"
        )

        self.random_seed = run_seed
        set_seed(run_seed)

        self.dataset.set_official_split_indices(split_type=split_type, fold_seed=fold_seed)
        train_features, valid_features, test_features = self.dataset.get_train_val_test_features("dict")

        self.experiment_dir = self.get_experiment_dir(
            base_dir=base_dir,
            split_type=split_type,
            fold_seed=fold_seed,
            run_seed=run_seed,
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.train(train_features)

        train_metrics = self.evaluate(train_features, split="train")
        valid_metrics = self.evaluate(valid_features, split="valid")
        test_metrics = self.evaluate(test_features, split="test")

        results = ModelResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            training_time=self.training_time,
            train_samples=len(train_features[self.fingerprint_type]),
            valid_samples=len(valid_features[self.fingerprint_type]),
            test_samples=len(test_features[self.fingerprint_type]),
            random_seed=run_seed,
            fold_seed=fold_seed,
            model_name=self.model_name,
        )

        self.save_results(results)
        return results

    def clone(self) -> "FPModel":
        model_cls: Type[FPModel] = self.__class__
        return model_cls(
            model_name=self.model_name,
            model_type=self.model_type,
            dataset=self.dataset,
            task_type=self.task_type,
            fingerprint_type=self.fingerprint_type,
            metrics=self.metrics,
            **self.model_params,
        )

    def run_multi(
        self,
        run_seeds: Optional[List[int]] = None,
        fold_seeds: Optional[List[int]] = None,
        split_type: str = "random_split",
        base_dir: str = "./results",
    ) -> MultiRunResults:
        run_seeds = run_seeds or [42, 43, 44, 45, 46]
        fold_seeds = fold_seeds or [0, 1, 2, 3, 4]

        if len(run_seeds) != len(fold_seeds):
            logger.warning(
                "[RUN_MULTI] run_seeds and fold_seeds length mismatch, truncating to min length"
            )
            min_len = min(len(run_seeds), len(fold_seeds))
            run_seeds = run_seeds[:min_len]
            fold_seeds = fold_seeds[:min_len]

        all_results: List[ModelResults] = []

        for run_seed, fold_seed in zip(run_seeds, fold_seeds):
            logger.info(f"[RUN_MULTI] running seed={run_seed}, fold={fold_seed}")

            fresh_model = self.clone()
            result = fresh_model.run(
                run_seed=run_seed,
                fold_seed=fold_seed,
                split_type=split_type,
                base_dir=base_dir,
            )
            all_results.append(result)

        logger.info(f"[RUN_MULTI DONE] total runs: {len(all_results)}")
        return MultiRunResults(results=all_results)

    # -------------------------
    # Load
    # -------------------------
    @classmethod
    def load_model(cls, model_path: str, **kwargs) -> "FPModel":
        if model_path.endswith(".joblib"):
            estimator = joblib.load(model_path)
            base_dir = os.path.dirname(model_path)
        else:
            estimator = joblib.load(os.path.join(model_path, "model.joblib"))
            base_dir = model_path

        with open(os.path.join(base_dir, "config.json"), "r") as f:
            config = json.load(f)

        model_type = config.get("model_type", config["model_name"])
        model_cls = MODEL_REGISTRY.get(model_type, cls)

        inst = model_cls(
            model_name=config["model_name"],
            model_type=model_type,
            task_type=config.get("task_type", "binary_classification"),
            fingerprint_type=config.get("fingerprint_type", "ecfp6"),
            **config.get("input_model_params", config.get("resolved_model_params", {})),
            **kwargs,
        )
        inst.estimator = estimator
        inst.experiment_dir = base_dir
        return inst


# =========================
# Reusable sklearn-like subclass
# =========================
class SklearnLikeFPModel(FPModel):
    classifier_cls: Optional[Type[Any]] = None
    regressor_cls: Optional[Type[Any]] = None

    def default_params(self) -> Dict[str, Any]:
        return {}

    def build_estimator(self) -> Any:
        params = dict(self.default_params())
        params.update(self.model_params)

        if self.task_type == "binary_classification":
            estimator_cls = self.classifier_cls
        elif self.task_type == "regression":
            estimator_cls = self.regressor_cls
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        if estimator_cls is None:
            raise ValueError(
                f"{self.__class__.__name__} does not support task_type={self.task_type}"
            )

        try:
            from inspect import signature

            sig = signature(estimator_cls.__init__)
            if "random_state" in sig.parameters:
                params.setdefault("random_state", getattr(self, "random_seed", 42))
        except Exception:
            pass

        return estimator_cls(**params)


# =========================
# Concrete Models
# =========================
class RFModel(SklearnLikeFPModel):
    model_type = "rf"
    classifier_cls = RandomForestClassifier
    regressor_cls = RandomForestRegressor


class AdaBoostModel(SklearnLikeFPModel):
    model_type = "adaboost"
    classifier_cls = AdaBoostClassifier
    regressor_cls = AdaBoostRegressor


class GradBoostModel(SklearnLikeFPModel):
    model_type = "gradboost"
    classifier_cls = GradientBoostingClassifier
    regressor_cls = GradientBoostingRegressor


class KNNModel(SklearnLikeFPModel):
    model_type = "knn"
    classifier_cls = KNeighborsClassifier
    regressor_cls = KNeighborsRegressor


class SVMModel(SklearnLikeFPModel):
    model_type = "svm"
    classifier_cls = SVC
    regressor_cls = SVR

    def default_params(self) -> Dict[str, Any]:
        if self.task_type == "binary_classification":
            return {"probability": True}
        return {}


class XGBoostModel(SklearnLikeFPModel):
    model_type = "xgboost"
    classifier_cls = XGBClassifier
    regressor_cls = XGBRegressor

    def default_params(self) -> Dict[str, Any]:
        if self.task_type == "binary_classification":
            return {
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
        return {}


class LightGBMModel(SklearnLikeFPModel):
    model_type = "lightgbm"
    classifier_cls = LGBMClassifier
    regressor_cls = LGBMRegressor


# =========================
# Registry / Factory
# =========================
MODEL_REGISTRY: Dict[str, Type[FPModel]] = {
    "rf": RFModel,
    "adaboost": AdaBoostModel,
    "gradboost": GradBoostModel,
    "knn": KNNModel,
    "svm": SVMModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


def build_fp_model(model_type: str, **kwargs) -> FPModel:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_name = kwargs.pop("model_name", model_type)
    return MODEL_REGISTRY[model_type](
        model_name=model_name,
        model_type=model_type,
        **kwargs,
    )