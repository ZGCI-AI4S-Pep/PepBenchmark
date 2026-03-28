from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from pepbenchmark.model.gnn_model import GNNPredictor
except Exception:
    GNNPredictor = None

try:
    from torch_geometric.data import Batch as PyGBatch
except ImportError:
    try:
        from torch_geometric.data import Batch as PyGBatch
    except ImportError:
        PyGBatch = None

from pepbenchmark.dataset_manager.ppi_dataset import PPIDatasetManager
from pepbenchmark.evaluator import (
    Classification_Metric,
    Regression_Metric,
    evaluate_classification,
    evaluate_regression,
)
from pepbenchmark.utils.logging import get_logger
from pepbenchmark.utils.seed import set_seed

logger = get_logger()


@dataclass
class PPIRuntimeSettings:
    local_files_only: bool = False
    cache_dir: Optional[str] = None
    local_model_dir: Optional[str] = None
    dataloader_num_workers: int = 0


@dataclass
class PPITrainingSettings:
    epochs: int = 30
    learning_rate: float = 1e-4
    batch_size: int = 32
    early_stopping_patience: int = 5


@dataclass
class PPIConfig:
    protein_model: str = "facebook/esm2_t30_150M_UR50D"
    protein_encoding: str = "plm"
    peptide_encoding: str = "plm"
    peptide_model: Optional[str] = None
    protein_feature_key: Optional[str] = None
    peptide_feature_key: Optional[str] = None
    fusion_method: str = "concat"
    hidden_dim: int = 300
    freeze_encoders: bool = False
    freeze_protein_encoder: Optional[bool] = None
    freeze_peptide_encoder: Optional[bool] = None
    mlp_hidden_dim: Optional[int] = None
    mlp_dropout: float = 0.1
    task_type: str = "binary_classification"
    max_protein_len: Optional[int] = None
    max_peptide_len: Optional[int] = None
    base_dir: str = "./results"
    seed: int = 42
    metrics: Optional[List[str]] = None
    model_name: Optional[str] = None
    runtime: PPIRuntimeSettings = field(default_factory=PPIRuntimeSettings)
    training: PPITrainingSettings = field(default_factory=PPITrainingSettings)
    model_params: Dict[str, Any] = field(default_factory=dict)

    def resolved_peptide_model(self) -> str:
        if self.peptide_model:
            return self.peptide_model
        if self.peptide_encoding == "smiles":
            return "seyonec/ChemBERTa-zinc-base-v1"
        return self.protein_model

    def resolved_metrics(self) -> List[str]:
        if self.metrics is not None:
            return list(self.metrics)
        if self.task_type == "regression":
            return list(Regression_Metric)
        return list(Classification_Metric)

    def resolved_model_name(self) -> str:
        if self.model_name:
            return self.model_name

        protein_name = self.protein_model.replace("/", "_")
        peptide_name = self.resolved_peptide_model().replace("/", "_")
        parts = [protein_name, self.protein_encoding]
        if peptide_name != protein_name:
            parts.append(peptide_name)
        parts.extend([self.peptide_encoding, self.fusion_method])
        return "_".join(parts)


@dataclass
class PPIDataContext:
    label_key: str = "label"
    protein_sequence_key: Optional[str] = None
    peptide_sequence_key: Optional[str] = None
    protein_feature_key: Optional[str] = None
    peptide_feature_key: Optional[str] = None
    protein_encoding_mode: str = "plm"
    peptide_encoding_mode: str = "plm"
    protein_feature_dim: Optional[int] = None
    protein_embedding_dim: Optional[int] = None
    peptide_feature_dim: Optional[int] = None


@dataclass
class PPIModelResults:
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
    split_type: str


@dataclass
class PPIMultiRunResults:
    results: List[PPIModelResults]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for res in self.results:
            row = {
                "random_seed": res.random_seed,
                "fold_seed": res.fold_seed,
                "model_name": res.model_name,
                "split_type": res.split_type,
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
        for column in df.columns:
            if not column.startswith(("train_", "valid_", "test_")):
                continue
            if not np.issubdtype(df[column].dtype, np.number):
                continue

            series = df[column].dropna()
            if series.empty:
                continue

            summary[column] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        return summary

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.to_dataframe().to_csv(os.path.join(save_dir, "multi_run_results.csv"), index=False)
        with open(os.path.join(save_dir, "multi_run_summary.json"), "w", encoding="utf-8") as file:
            json.dump(self.get_summary_stats(), file, indent=2, ensure_ascii=False)


@dataclass
class RunContext:
    split_type: str
    fold_seed: int
    run_seed: int
    experiment_dir: str


class ExperimentManager:
    def __init__(self, config: PPIConfig, dataset: Optional[PPIDatasetManager]):
        self.config = config
        self.dataset = dataset

    def get_experiment_dir(self, split_type: str, fold_seed: int, run_seed: int) -> str:
        dataset_name = self.dataset.dataset_name if self.dataset is not None else "unknown_dataset"
        return os.path.join(
            self.config.base_dir,
            dataset_name,
            self.config.resolved_model_name(),
            split_type,
            f"fold_{fold_seed}_seed_{run_seed}",
        )

    def create_run_context(self, split_type: str, fold_seed: int, run_seed: int) -> RunContext:
        experiment_dir = self.get_experiment_dir(split_type, fold_seed, run_seed)
        os.makedirs(experiment_dir, exist_ok=True)
        return RunContext(
            split_type=split_type,
            fold_seed=fold_seed,
            run_seed=run_seed,
            experiment_dir=experiment_dir,
        )

    def save_model(self, model: nn.Module, experiment_dir: str, name: str = "model.pt") -> None:
        model_path = os.path.join(experiment_dir, name)
        torch.save(model.state_dict(), model_path)
        logger.info(f"[SAVE MODEL] saved to {model_path}")

    def save_results(
        self,
        experiment_dir: str,
        results: PPIModelResults,
        data_context: Optional[PPIDataContext],
    ) -> None:
        payload = {
            "ppi_config": asdict(self.config),
            "data_context": asdict(data_context) if data_context is not None else None,
            "resolved_metrics": self.config.resolved_metrics(),
            "dataset_name": self.dataset.dataset_name if self.dataset else None,
        }

        with open(os.path.join(experiment_dir, "config.json"), "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

        with open(os.path.join(experiment_dir, "metrics.json"), "w", encoding="utf-8") as file:
            json.dump(asdict(results), file, indent=2, ensure_ascii=False)


def _to_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.to(dtype=dtype)
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        return tensor if dtype is None else tensor.to(dtype=dtype)
    if isinstance(value, (list, tuple)):
        tensor = torch.tensor(value)
        return tensor if dtype is None else tensor.to(dtype=dtype)
    tensor = torch.tensor(value)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def ppi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    protein_data: Dict[str, List[Any]] = {}
    peptide_data: Dict[str, List[Any]] = {}
    protein_graph_data: List[Any] = []
    peptide_graph_data: List[Any] = []
    labels: List[torch.Tensor] = []

    for item in batch:
        labels.append(_to_tensor(item["labels"]))
        for key, value in item.items():
            if key == "labels":
                continue
            if key.startswith("protein_"):
                if key == "protein_features" and hasattr(value, "x") and hasattr(value, "edge_index"):
                    protein_graph_data.append(value)
                    continue
                protein_data.setdefault(key, []).append(value)
                continue
            if key == "peptide_features" and hasattr(value, "x") and hasattr(value, "edge_index"):
                peptide_graph_data.append(value)
                continue
            peptide_data.setdefault(key, []).append(value)

    batch_data: Dict[str, Any] = {}
    for key, values in protein_data.items():
        batch_data[key] = torch.stack([_to_tensor(value) for value in values])

    if protein_graph_data:
        if PyGBatch is None:
            raise ImportError("PyTorch Geometric is required for graph batching.")
        batch_data["protein_features"] = PyGBatch.from_data_list(protein_graph_data)

    if peptide_graph_data:
        if PyGBatch is None:
            raise ImportError("PyTorch Geometric is required for graph batching.")
        batch_data["peptide_features"] = PyGBatch.from_data_list(peptide_graph_data)

    for key, values in peptide_data.items():
        if key == "peptide_features" and peptide_graph_data:
            continue
        batch_data[key] = torch.stack([_to_tensor(value) for value in values])

    batch_data["labels"] = torch.stack(labels)
    return batch_data


class PPIDataset(Dataset):
    def __init__(
        self,
        protein_sequences: List[str],
        peptide_sequences: List[str],
        labels: List[Any],
        protein_tokenizer: Optional[Any] = None,
        peptide_tokenizer: Optional[Any] = None,
        peptide_features: Optional[Dict[str, Any]] = None,
        protein_features: Optional[Dict[str, Any]] = None,
        max_protein_len: int = 1024,
        max_peptide_len: int = 200,
        encoding_mode: str = "plm",
        protein_encoding_mode: str = "plm",
    ):
        self.protein_sequences = protein_sequences
        self.peptide_sequences = peptide_sequences
        self.labels = labels
        self.protein_tokenizer = protein_tokenizer
        self.peptide_tokenizer = peptide_tokenizer
        self.peptide_features = peptide_features or {}
        self.protein_features = protein_features or {}
        self.max_protein_len = max_protein_len
        self.max_peptide_len = max_peptide_len
        self.encoding_mode = encoding_mode
        self.protein_encoding_mode = protein_encoding_mode

    @staticmethod
    def _encode_tokenized_sequence(
        sequence: str,
        tokenizer: Any,
        max_length: int,
        space_separated: bool,
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        if tokenizer is None:
            raise ValueError(f"{prefix}_tokenizer is required for {prefix} tokenized encoding")
        encoded = tokenizer(
            " ".join(sequence) if space_separated else sequence,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {f"{prefix}_{key}": value.squeeze(0) for key, value in encoded.items()}

    @staticmethod
    def _build_feature_item(
        features: Dict[str, Any],
        feature_key: str,
        idx: int,
        prefix: str,
        encoding_mode: str,
    ) -> Dict[str, Any]:
        value = features.get(feature_key)
        if value is None:
            raise ValueError(f"{feature_key} feature is required for {prefix} {encoding_mode} mode")
        sample = value[idx]
        if encoding_mode == "gnn":
            return {f"{prefix}_features": sample}
        return {f"{prefix}_features": _to_tensor(sample, dtype=torch.float32)}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        protein_seq = self.protein_sequences[idx] if self.protein_sequences else ""
        peptide_seq = self.peptide_sequences[idx] if self.peptide_sequences else ""
        label = self.labels[idx]

        if self.protein_encoding_mode in {"plm", "smiles"}:
            protein_item = self._encode_tokenized_sequence(
                sequence=protein_seq,
                tokenizer=self.protein_tokenizer,
                max_length=self.max_protein_len,
                space_separated=self.protein_encoding_mode == "plm",
                prefix="protein",
            )
        elif self.protein_encoding_mode in {"embedding", "fp", "gnn"}:
            protein_item = self._build_feature_item(
                features=self.protein_features,
                feature_key=f"protein_{self.protein_encoding_mode}",
                idx=idx,
                prefix="protein",
                encoding_mode=self.protein_encoding_mode,
            )
        else:
            raise ValueError(f"Unsupported protein encoding mode: {self.protein_encoding_mode}")

        if self.encoding_mode in {"plm", "smiles"}:
            peptide_item = self._encode_tokenized_sequence(
                sequence=peptide_seq,
                tokenizer=self.peptide_tokenizer,
                max_length=self.max_peptide_len,
                space_separated=self.encoding_mode == "plm",
                prefix="peptide",
            )
        elif self.encoding_mode in {"fp", "embedding", "gnn"}:
            peptide_item = self._build_feature_item(
                features=self.peptide_features,
                feature_key=f"peptide_{self.encoding_mode}",
                idx=idx,
                prefix="peptide",
                encoding_mode=self.encoding_mode,
            )
        else:
            raise ValueError(f"Unsupported peptide encoding mode: {self.encoding_mode}")

        label_dtype = torch.float32 if isinstance(label, (float, np.floating)) else None
        item = {**protein_item, **peptide_item, "labels": _to_tensor(label, dtype=label_dtype)}
        return item


class FeatureProjectionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, two_layer: bool = True):
        super().__init__()
        if two_layer:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        self.dropout = nn.Dropout(0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(features))


class GraphFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.LazyLinear(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features: Any) -> torch.Tensor:
        if not hasattr(features, "x"):
            raise ValueError("Graph encoder requires PyG Data/Batch with attribute 'x'.")

        x = features.x.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if hasattr(features, "batch"):
            batch_index = features.batch
            num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
            pooled = []
            for graph_idx in range(num_graphs):
                graph_nodes = x[batch_index == graph_idx]
                pooled.append(graph_nodes.mean(dim=0) if graph_nodes.numel() > 0 else torch.zeros(x.size(-1), device=x.device))
            graph_features = torch.stack(pooled) if pooled else torch.zeros((0, x.size(-1)), device=x.device)
        else:
            graph_features = x.mean(dim=0, keepdim=True)

        return self.dropout(self.projection(graph_features))


class BackboneFeatureExtractor(nn.Module):
    def __init__(self, mode: str, backbone: Optional[nn.Module], freeze_backbone: bool = False):
        super().__init__()
        self.mode = mode
        self.backbone = backbone
        if self.backbone is not None and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, **inputs: Any) -> torch.Tensor:
        if self.mode in {"plm", "smiles"}:
            if self.backbone is None:
                raise ValueError(f"backbone is required for {self.mode} mode")
            outputs = self.backbone(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
            )
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            return hidden_states[:, 0, :]

        if self.mode == "gnn":
            graph_batch = inputs.get("features")
            if graph_batch is None:
                raise ValueError("Graph input is required for gnn mode")

            if self.backbone is None:
                if not hasattr(graph_batch, "x"):
                    raise ValueError("Graph batch must have attribute 'x' in gnn mode")
                x = graph_batch.x.float()
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                if hasattr(graph_batch, "batch"):
                    batch_index = graph_batch.batch
                    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
                    pooled = []
                    for graph_idx in range(num_graphs):
                        graph_nodes = x[batch_index == graph_idx]
                        pooled.append(
                            graph_nodes.mean(dim=0)
                            if graph_nodes.numel() > 0
                            else torch.zeros(x.size(-1), device=x.device)
                        )
                    return torch.stack(pooled) if pooled else torch.zeros((0, x.size(-1)), device=x.device)
                return x.mean(dim=0, keepdim=True)

            if hasattr(self.backbone, "node_encoder") and hasattr(self.backbone, "pool"):
                h_node = self.backbone.node_encoder(graph_batch)
                return self.backbone.pool(h_node, graph_batch.batch)

            gnn_out = self.backbone(graph_batch)
            if isinstance(gnn_out, tuple):
                gnn_out = gnn_out[0]
            return gnn_out

        features = inputs.get("features")
        if features is None:
            raise ValueError(f"features are required for {self.mode} mode")
        return features.float()


class FlexibleEncoder(nn.Module):
    def __init__(
        self,
        encoding_mode: str,
        hidden_dim: int,
        backbone: Optional[nn.Module] = None,
        feature_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoding_mode = encoding_mode
        self.extractor = BackboneFeatureExtractor(
            mode=encoding_mode,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
        )

        if encoding_mode == "fp" and feature_dim is not None:
            self.projector = FeatureProjectionEncoder(input_dim=feature_dim, hidden_dim=hidden_dim, two_layer=True)
        elif encoding_mode == "embedding" and feature_dim is not None:
            self.projector = FeatureProjectionEncoder(input_dim=feature_dim, hidden_dim=hidden_dim, two_layer=False)
        else:
            self.projector = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def forward(self, **inputs: Any) -> torch.Tensor:
        features = self.extractor(**inputs)
        return self.projector(features)


class ProteinEncoder(FlexibleEncoder):
    def __init__(
        self,
        encoding_mode: str,
        hidden_dim: int,
        backbone: Optional[nn.Module] = None,
        feature_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__(
            encoding_mode=encoding_mode,
            hidden_dim=hidden_dim,
            backbone=backbone,
            feature_dim=feature_dim,
            freeze_backbone=freeze_encoder,
            dropout=dropout,
        )


class PeptideEncoder(nn.Module):
    def __init__(
        self,
        encoding_mode: str,
        hidden_dim: int,
        backbone: Optional[nn.Module] = None,
        feature_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoding_mode = encoding_mode
        self.encoder = FlexibleEncoder(
            encoding_mode=encoding_mode,
            hidden_dim=hidden_dim,
            backbone=backbone,
            feature_dim=feature_dim,
            freeze_backbone=freeze_encoder,
            dropout=dropout,
        )

    def forward(self, **inputs: Any) -> torch.Tensor:
        if self.encoding_mode in {"plm", "smiles"}:
            return self.encoder(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )
        return self.encoder(inputs.get("features"))


class FusionLayer(nn.Module):
    def __init__(
        self,
        protein_dim: int = 256,
        peptide_dim: int = 256,
        fusion_method: str = "concat",
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim

        if fusion_method == "concat":
            fusion_dim = protein_dim + peptide_dim
            self.projection = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        elif fusion_method == "add":
            if protein_dim != peptide_dim:
                raise ValueError("Protein and peptide dimensions must match for add fusion")
            self.projection = nn.Sequential(
                nn.Linear(protein_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        elif fusion_method == "attention":
            fusion_dim = max(protein_dim, peptide_dim)
            self.protein_proj = nn.Linear(protein_dim, fusion_dim) if protein_dim != fusion_dim else nn.Identity()
            self.peptide_proj = nn.Linear(peptide_dim, fusion_dim) if peptide_dim != fusion_dim else nn.Identity()
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.projection = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, protein_features: torch.Tensor, peptide_features: torch.Tensor) -> torch.Tensor:
        if self.fusion_method == "concat":
            fused = torch.cat([protein_features, peptide_features], dim=-1)
        elif self.fusion_method == "add":
            fused = protein_features + peptide_features
        else:
            protein_proj = self.protein_proj(protein_features).unsqueeze(1)
            peptide_proj = self.peptide_proj(peptide_features).unsqueeze(1)
            sequence = torch.cat([protein_proj, peptide_proj], dim=1)
            attended, _ = self.attention(sequence, sequence, sequence)
            fused = attended.mean(dim=1)
        return self.projection(fused)


class PPIModel(nn.Module):
    def __init__(
        self,
        protein_encoder: nn.Module,
        peptide_encoder: nn.Module,
        fusion_layer: FusionLayer,
        task_type: str = "binary_classification",
        num_classes: int = 2,
        mlp_hidden_dim: Optional[int] = None,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.peptide_encoder = peptide_encoder
        self.fusion_layer = fusion_layer
        self.task_type = task_type

        output_dim = 1 if task_type in {"binary_classification", "regression"} else num_classes
        mlp_hidden = mlp_hidden_dim or max(fusion_layer.hidden_dim // 2, 64)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_layer.hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        protein_inputs = {
            key.replace("protein_", ""): value
            for key, value in batch.items()
            if key.startswith("protein_")
        }
        peptide_inputs = {
            key.replace("peptide_", ""): value
            for key, value in batch.items()
            if key.startswith("peptide_")
        }

        protein_features = self.protein_encoder(**protein_inputs)
        peptide_features = self.peptide_encoder(**peptide_inputs)
        fused_features = self.fusion_layer(protein_features, peptide_features)
        return self.classifier(fused_features)


class PPIRunner:
    model_type: str = "ppi"

    def __init__(
        self,
        config: PPIConfig,
        dataset: Optional[PPIDatasetManager] = None,
        protein_backbone: Optional[nn.Module] = None,
        peptide_backbone: Optional[nn.Module] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.experiment_manager = ExperimentManager(config=config, dataset=dataset)
        self.custom_protein_backbone = protein_backbone
        self.custom_peptide_backbone = peptide_backbone

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PPIModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.protein_tokenizer: Optional[Any] = None
        self.peptide_tokenizer: Optional[Any] = None
        self.training_history: List[Dict[str, Any]] = []
        self.training_time: float = 0.0
        self.data_context: Optional[PPIDataContext] = None

        logger.info(f"[PPI] Using device: {self.device}")

    def clone(self) -> "PPIRunner":
        return PPIRunner(
            config=copy.deepcopy(self.config),
            dataset=self.dataset,
            protein_backbone=self.custom_protein_backbone,
            peptide_backbone=self.custom_peptide_backbone,
        )

    def _resolve_model_source(self, model_name: str) -> str:
        local_dir = self.config.runtime.local_model_dir
        if local_dir:
            candidate = os.path.join(local_dir, model_name.replace("/", "_"))
            if os.path.exists(candidate):
                return candidate
        return model_name

    def _load_hf_model_and_tokenizer(self, model_name: str) -> Tuple[Any, Any]:
        model_source = self._resolve_model_source(model_name)
        blocked_keys = {
            "protein_backbone_params",
            "peptide_backbone_params",
            "gnn_backbone_params",
            "protein_gnn_backbone_params",
            "peptide_gnn_backbone_params",
        }
        kwargs = {k: v for k, v in self.config.model_params.items() if k not in blocked_keys}
        kwargs["local_files_only"] = self.config.runtime.local_files_only
        if self.config.runtime.cache_dir:
            kwargs["cache_dir"] = self.config.runtime.cache_dir

        model = AutoModel.from_pretrained(model_source, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=self.config.runtime.local_files_only,
            cache_dir=self.config.runtime.cache_dir,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return model, tokenizer

    def _load_hf_tokenizer(self, model_name: str) -> Any:
        model_source = self._resolve_model_source(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=self.config.runtime.local_files_only,
            cache_dir=self.config.runtime.cache_dir,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return tokenizer

    @staticmethod
    def _find_first_existing(features: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        return next((candidate for candidate in candidates if candidate in features), None)

    @staticmethod
    def _infer_feature_dim(values: Any) -> int:
        sample = values[0]
        if isinstance(sample, torch.Tensor):
            return int(sample.shape[-1]) if sample.dim() > 0 else 1
        if isinstance(sample, np.ndarray):
            return int(sample.shape[-1]) if sample.ndim > 0 else 1
        if hasattr(sample, "shape") and len(sample.shape) > 0:
            return int(sample.shape[-1])
        if hasattr(sample, "__len__") and not isinstance(sample, str):
            return len(sample)
        return 1

    @staticmethod
    def _normalize_task_type(task_type: Optional[str]) -> str:
        if task_type in {None, "classification", "binary", "binary_classification"}:
            return "binary_classification"
        return task_type

    def _resolve_freeze_flag(self, side: str) -> bool:
        if side == "protein" and self.config.freeze_protein_encoder is not None:
            return self.config.freeze_protein_encoder
        if side == "peptide" and self.config.freeze_peptide_encoder is not None:
            return self.config.freeze_peptide_encoder
        return self.config.freeze_encoders

    def _build_gnn_backbone(self, side: str) -> nn.Module:
        if GNNPredictor is None:
            raise ImportError("GNN backbone requested but pepbenchmark.model.gnn_model.GNNPredictor is unavailable")

        specific_key = f"{side}_gnn_backbone_params"
        side_key = f"{side}_backbone_params"
        backbone_params = dict(self.config.model_params.get("gnn_backbone_params", {}))
        backbone_params.update(self.config.model_params.get(side_key, {}))
        backbone_params.update(self.config.model_params.get(specific_key, {}))

        gnn = GNNPredictor(
            num_tasks=1,
            num_layer=backbone_params.get("num_layer", 5),
            emb_dim=backbone_params.get("emb_dim", self.config.hidden_dim),
            gnn_type=backbone_params.get("gnn_type", "gin"),
            virtual_node=backbone_params.get("virtual_node", True),
            residual=backbone_params.get("residual", False),
            drop_ratio=backbone_params.get("drop_ratio", 0.5),
            JK=backbone_params.get("JK", "last"),
            graph_pooling=backbone_params.get("graph_pooling", "mean"),
        )
        return gnn

    def _resolve_side_feature_key(
        self,
        side: str,
        encoding: str,
        train_features: Dict[str, Any],
        explicit_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[int]]:
        if encoding in {"plm", "smiles"}:
            return None, None

        if explicit_key is not None:
            if explicit_key not in train_features:
                raise ValueError(f"{side}_feature_key={explicit_key} not found in features")
            dim = None if encoding == "gnn" else self._infer_feature_dim(train_features[explicit_key])
            return explicit_key, dim

        if side == "protein":
            fp_candidates = ["prot_ecfp6", "prot_ecfp4", "official_prot_ecfp6", "official_prot_ecfp4"]
            gnn_candidates = ["prot_graph", "protein_graph", "official_prot_graph"]
            embed_prefix = ("prot_", "official_prot_")
        else:
            fp_candidates = ["pep_ecfp6", "pep_ecfp4", "official_pep_ecfp6", "official_pep_ecfp4"]
            gnn_candidates = ["pep_graph", "graph", "official_pep_graph"]
            embed_prefix = ("pep_", "official_pep_")

        selected_key: Optional[str] = None
        if encoding == "fp":
            selected_key = self._find_first_existing(train_features, fp_candidates)
        elif encoding == "gnn":
            selected_key = self._find_first_existing(train_features, gnn_candidates)
        elif encoding == "embedding":
            for key in train_features:
                if key.startswith(embed_prefix) and "embedding" in key:
                    selected_key = key
                    break
        else:
            raise ValueError(f"Unsupported {side} encoding: {encoding}")

        if selected_key is None:
            raise ValueError(f"{side} feature field not found for encoding {encoding}.")

        dim = None if encoding == "gnn" else self._infer_feature_dim(train_features[selected_key])
        return selected_key, dim

    def _create_data_context(
        self,
        train_features: Dict[str, Any],
        valid_features: Dict[str, Any],
        test_features: Dict[str, Any],
    ) -> PPIDataContext:
        context = PPIDataContext()
        context.label_key = self._find_first_existing(train_features, ["label", "official_label"]) or "label"

        context.protein_encoding_mode = self.config.protein_encoding
        context.peptide_encoding_mode = self.config.peptide_encoding

        if self.config.protein_encoding == "plm":
            context.protein_sequence_key = self._find_first_existing(
                train_features,
                ["prot_fasta", "protein_sequence", "protein_fasta", "protein", "target_sequence", "official_prot_fasta"],
            )
        elif self.config.protein_encoding == "smiles":
            context.protein_sequence_key = self._find_first_existing(
                train_features,
                ["prot_smiles", "protein_smiles", "smiles", "official_prot_smiles"],
            )
        else:
            context.protein_feature_key, context.protein_feature_dim = self._resolve_side_feature_key(
                side="protein",
                encoding=self.config.protein_encoding,
                train_features=train_features,
                explicit_key=self.config.protein_feature_key,
            )
            context.protein_embedding_dim = context.protein_feature_dim

        if self.config.peptide_encoding == "plm":
            context.peptide_sequence_key = self._find_first_existing(
                train_features,
                ["pep_fasta", "peptide_sequence", "peptide_fasta", "peptide", "official_pep_fasta"],
            )
        elif self.config.peptide_encoding == "smiles":
            context.peptide_sequence_key = self._find_first_existing(
                train_features,
                ["pep_smiles", "peptide_smiles", "smiles", "official_pep_smiles"],
            )
        else:
            context.peptide_feature_key, context.peptide_feature_dim = self._resolve_side_feature_key(
                side="peptide",
                encoding=self.config.peptide_encoding,
                train_features=train_features,
                explicit_key=self.config.peptide_feature_key,
            )

        if self.config.protein_encoding in {"plm", "smiles"} and context.protein_sequence_key is None:
            raise ValueError(f"Protein sequence field not found for encoding {self.config.protein_encoding}.")
        if self.config.protein_encoding in {"fp", "embedding", "gnn"} and context.protein_feature_key is None:
            raise ValueError(f"Protein feature field not found for encoding {self.config.protein_encoding}.")
        if self.config.peptide_encoding in {"plm", "smiles"} and context.peptide_sequence_key is None:
            raise ValueError(f"Peptide sequence field not found for encoding {self.config.peptide_encoding}.")
        if self.config.peptide_encoding in {"fp", "embedding", "gnn"} and context.peptide_feature_key is None:
            raise ValueError(f"Peptide feature field not found for encoding {self.config.peptide_encoding}.")

        self._auto_set_max_lengths(
            train_features=train_features,
            valid_features=valid_features,
            test_features=test_features,
            context=context,
        )
        return context

    def _auto_set_max_lengths(
        self,
        train_features: Dict[str, Any],
        valid_features: Dict[str, Any],
        test_features: Dict[str, Any],
        context: PPIDataContext,
    ) -> None:
        if self.config.max_protein_len is None and context.protein_sequence_key is not None:
            all_proteins = list(train_features[context.protein_sequence_key])
            all_proteins.extend(valid_features[context.protein_sequence_key])
            all_proteins.extend(test_features[context.protein_sequence_key])
            lengths = [len(seq) for seq in all_proteins if seq]
            if lengths:
                self.config.max_protein_len = min(int(np.percentile(lengths, 95)) + 5, 2048)

        if self.config.max_peptide_len is None and context.peptide_sequence_key is not None:
            all_peptides = list(train_features[context.peptide_sequence_key])
            all_peptides.extend(valid_features[context.peptide_sequence_key])
            all_peptides.extend(test_features[context.peptide_sequence_key])
            lengths = [len(seq) for seq in all_peptides if seq]
            if lengths:
                self.config.max_peptide_len = min(int(np.percentile(lengths, 95)) + 5, 512)

        if self.config.max_protein_len is None:
            self.config.max_protein_len = 1024
        if self.config.max_peptide_len is None:
            self.config.max_peptide_len = 200

    def _get_num_classes(self, labels: List[Any]) -> int:
        if self.config.task_type == "regression":
            return 1
        if self.config.task_type == "multi_class_classification":
            return len(set(labels))
        return 2

    def _build_model(self, context: PPIDataContext, train_features: Dict[str, Any]) -> PPIModel:
        logger.info(f"[BUILD MODEL] {self.config.resolved_model_name()}")

        protein_backbone = self.custom_protein_backbone
        if protein_backbone is None and self.config.protein_encoding in {"plm", "smiles"}:
            protein_backbone, self.protein_tokenizer = self._load_hf_model_and_tokenizer(self.config.protein_model)
        elif self.config.protein_encoding in {"plm", "smiles"} and self.protein_tokenizer is None:
            self.protein_tokenizer = self._load_hf_tokenizer(self.config.protein_model)
        elif protein_backbone is None and self.config.protein_encoding == "gnn":
            protein_backbone = self._build_gnn_backbone(side="protein")

        peptide_backbone = self.custom_peptide_backbone
        peptide_model_name = self.config.resolved_peptide_model()
        if peptide_backbone is None and self.config.peptide_encoding in {"plm", "smiles"}:
            peptide_backbone, self.peptide_tokenizer = self._load_hf_model_and_tokenizer(peptide_model_name)
        elif self.config.peptide_encoding in {"plm", "smiles"} and self.peptide_tokenizer is None:
            self.peptide_tokenizer = self._load_hf_tokenizer(peptide_model_name)
        elif peptide_backbone is None and self.config.peptide_encoding == "gnn":
            peptide_backbone = self._build_gnn_backbone(side="peptide")

        protein_encoder = ProteinEncoder(
            encoding_mode=self.config.protein_encoding,
            hidden_dim=self.config.hidden_dim,
            backbone=protein_backbone,
            feature_dim=context.protein_feature_dim,
            freeze_encoder=self._resolve_freeze_flag("protein"),
            dropout=self.config.mlp_dropout,
        )
        peptide_encoder = PeptideEncoder(
            encoding_mode=self.config.peptide_encoding,
            hidden_dim=self.config.hidden_dim,
            backbone=peptide_backbone,
            feature_dim=context.peptide_feature_dim,
            freeze_encoder=self._resolve_freeze_flag("peptide"),
            dropout=self.config.mlp_dropout,
        )

        fusion_layer = FusionLayer(
            protein_dim=self.config.hidden_dim,
            peptide_dim=self.config.hidden_dim,
            fusion_method=self.config.fusion_method,
            hidden_dim=self.config.hidden_dim,
        )

        labels = train_features[context.label_key]
        model = PPIModel(
            protein_encoder=protein_encoder,
            peptide_encoder=peptide_encoder,
            fusion_layer=fusion_layer,
            task_type=self.config.task_type,
            num_classes=self._get_num_classes(labels),
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            mlp_dropout=self.config.mlp_dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
        )
        if self.config.task_type == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.config.task_type == "multi_class_classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")

        return model

    def _build_split_dataset(self, features: Dict[str, Any], context: PPIDataContext) -> PPIDataset:
        labels = features[context.label_key]
        protein_sequences = features[context.protein_sequence_key] if context.protein_sequence_key else []
        peptide_sequences = features[context.peptide_sequence_key] if context.peptide_sequence_key else []

        protein_features = {}
        if context.protein_feature_key is not None:
            protein_features[f"protein_{context.protein_encoding_mode}"] = features[context.protein_feature_key]

        peptide_features = {}
        if context.peptide_feature_key is not None:
            peptide_features[f"peptide_{context.peptide_encoding_mode}"] = features[context.peptide_feature_key]

        return PPIDataset(
            protein_sequences=protein_sequences,
            peptide_sequences=peptide_sequences,
            labels=labels,
            protein_tokenizer=self.protein_tokenizer,
            peptide_tokenizer=self.peptide_tokenizer,
            peptide_features=peptide_features,
            protein_features=protein_features,
            max_protein_len=self.config.max_protein_len or 1024,
            max_peptide_len=self.config.max_peptide_len or 200,
            encoding_mode=context.peptide_encoding_mode,
            protein_encoding_mode=context.protein_encoding_mode,
        )

    def _build_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        use_graph_collate = self.config.peptide_encoding == "gnn" or self.config.protein_encoding == "gnn"
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.runtime.dataloader_num_workers,
            collate_fn=ppi_collate_fn if use_graph_collate else None,
        )

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif hasattr(value, "to") and hasattr(value, "x") and hasattr(value, "edge_index"):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.criterion is None:
            raise ValueError("criterion has not been initialized")
        if self.config.task_type == "binary_classification":
            return self.criterion(outputs.squeeze(-1), labels.float())
        if self.config.task_type == "multi_class_classification":
            return self.criterion(outputs, labels.long())
        return self.criterion(outputs.squeeze(-1), labels.float())

    def _train_epoch(self, dataloader: DataLoader) -> float:
        if self.model is None or self.optimizer is None:
            raise ValueError("model and optimizer must be initialized before training")

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            labels = batch.pop("labels")

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self._compute_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _evaluate_epoch(self, dataloader: DataLoader) -> float:
        if self.model is None:
            raise ValueError("model must be initialized before evaluation")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                labels = batch.pop("labels")
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _predict_loader(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("model must be initialized before prediction")

        self.model.eval()
        predictions: List[Any] = []
        probabilities: List[Any] = []
        labels_all: List[Any] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                labels = batch.pop("labels")
                outputs = self.model(batch)

                if self.config.task_type == "binary_classification":
                    logits = outputs.squeeze(-1)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    probabilities.extend(probs.cpu().numpy())
                    predictions.extend(preds.cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())
                elif self.config.task_type == "multi_class_classification":
                    probs = torch.softmax(outputs, dim=-1)
                    preds = torch.argmax(outputs, dim=-1)
                    probabilities.extend(probs.cpu().numpy())
                    predictions.extend(preds.cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())
                else:
                    preds = outputs.squeeze(-1)
                    probabilities.extend(preds.cpu().numpy())
                    predictions.extend(preds.cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())

        return np.array(labels_all), np.array(predictions), np.array(probabilities)

    def _evaluate_features(
        self,
        features: Dict[str, Any],
        context: PPIDataContext,
        split_name: str,
    ) -> Dict[str, float]:
        dataset = self._build_split_dataset(features, context)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        y_true, y_pred, y_score = self._predict_loader(dataloader)

        if self.config.task_type == "regression":
            metrics = evaluate_regression(y_true=y_true, y_pred=y_pred)
        else:
            score = y_score if self.config.task_type == "multi_class_classification" else y_score.reshape(-1)
            metrics = evaluate_classification(y_true=y_true, y_pred=y_pred, y_score=score)

        logger.info(f"[METRICS] {split_name}: {metrics}")
        return metrics

    def train_once(
        self,
        train_features: Dict[str, Any],
        valid_features: Dict[str, Any],
        test_features: Dict[str, Any],
        run_context: RunContext,
    ) -> Tuple[PPIDataContext, float]:
        context = self._create_data_context(train_features, valid_features, test_features)
        self.data_context = context
        self.model = self._build_model(context, train_features)

        train_dataset = self._build_split_dataset(train_features, context)
        valid_dataset = self._build_split_dataset(valid_features, context)
        train_loader = self._build_dataloader(train_dataset, shuffle=True)
        valid_loader = self._build_dataloader(valid_dataset, shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.training_history = []

        logger.info(f"[TRAIN START] {self.config.resolved_model_name()}")
        start = time.time()
        for epoch in range(self.config.training.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._evaluate_epoch(valid_loader)
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict()) if self.model is not None else None
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"[EARLY STOPPING] at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"[EPOCH {epoch + 1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        if self.model is not None and best_state is not None:
            self.model.load_state_dict(best_state)

        training_time = time.time() - start
        self.training_time = training_time
        logger.info(f"[TRAIN DONE] {self.config.resolved_model_name()} in {training_time:.2f}s")

        if self.model is None:
            raise ValueError("model was not built successfully")

        self.experiment_manager.save_model(self.model, run_context.experiment_dir)
        with open(os.path.join(run_context.experiment_dir, "training_history.json"), "w", encoding="utf-8") as file:
            json.dump(self.training_history, file, indent=2, ensure_ascii=False)

        return context, training_time

    def run(
        self,
        run_seed: Optional[int] = None,
        fold_seed: int = 0,
        split_type: Optional[str] = None,
    ) -> PPIModelResults:
        if self.dataset is None:
            raise ValueError("dataset cannot be None in run().")

        effective_run_seed = self.config.seed if run_seed is None else run_seed
        set_seed(effective_run_seed)

        if hasattr(self.dataset, "resolve_split_type"):
            resolved_split_type = self.dataset.resolve_split_type(split_type)  # type: ignore[attr-defined]
        else:
            split_map = getattr(self.dataset, "OFFICIAL_SPLIT_MAP", {}) or {}
            if not isinstance(split_map, dict) or len(split_map) == 0:
                resolved_split_type = "random_split" if split_type is None else split_type
            elif split_type is None:
                resolved_split_type = (
                    "protein_hybrid_cold_split"
                    if "protein_hybrid_cold_split" in split_map
                    else next(iter(split_map.keys()))
                )
            elif split_type in split_map:
                resolved_split_type = split_type
            else:
                alias_map = {
                    "random_split": "protein_random_cold_split",
                    "ecfp_split": "protein_kmer_cold_split",
                    "kmer_split": "protein_kmer_cold_split",
                    "mmseqs_split": "protein_mmseqs_cold_split",
                    "hybrid_split": "protein_hybrid_cold_split",
                    "pep_random_split": "peptide_random_cold_split",
                    "pep_mmseqs_split": "peptide_mmseqs_cold_split",
                    "pep_kmer_split": "peptide_kmer_cold_split",
                    "pep_hybrid_split": "peptide_hybrid_cold_split",
                }
                candidate = alias_map.get(split_type)
                if candidate in split_map:
                    logger.info(f"[PPI] mapped split_type '{split_type}' -> '{candidate}'")
                    resolved_split_type = candidate
                else:
                    available = ", ".join(sorted(split_map.keys()))
                    raise ValueError(
                        f"Unsupported split_type '{split_type}'. Available official split types: {available}"
                    )
        logger.info(
            f"[RUN] model={self.config.resolved_model_name()} | "
            f"seed={effective_run_seed}, fold={fold_seed}, split={resolved_split_type}"
        )

        self.dataset.set_official_split_indices(split_type=resolved_split_type, fold_seed=fold_seed)
        train_features, valid_features, test_features = self.dataset.get_train_val_test_features("dict")

        if not train_features or len(next(iter(train_features.values()))) == 0:
            raise ValueError(f"Resolved split '{resolved_split_type}' produced an empty train split.")
        if not valid_features or len(next(iter(valid_features.values()))) == 0:
            raise ValueError(f"Resolved split '{resolved_split_type}' produced an empty valid split.")
        if not test_features or len(next(iter(test_features.values()))) == 0:
            raise ValueError(f"Resolved split '{resolved_split_type}' produced an empty test split.")

        run_context = self.experiment_manager.create_run_context(
            split_type=resolved_split_type,
            fold_seed=fold_seed,
            run_seed=effective_run_seed,
        )

        context, training_time = self.train_once(
            train_features=train_features,
            valid_features=valid_features,
            test_features=test_features,
            run_context=run_context,
        )

        train_metrics = self._evaluate_features(train_features, context, "train")
        valid_metrics = self._evaluate_features(valid_features, context, "valid")
        test_metrics = self._evaluate_features(test_features, context, "test")

        results = PPIModelResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            training_time=training_time,
            train_samples=len(train_features[context.label_key]),
            valid_samples=len(valid_features[context.label_key]),
            test_samples=len(test_features[context.label_key]),
            random_seed=effective_run_seed,
            fold_seed=fold_seed,
            model_name=self.config.resolved_model_name(),
            split_type=resolved_split_type,
        )

        self.experiment_manager.save_results(
            experiment_dir=run_context.experiment_dir,
            results=results,
            data_context=context,
        )
        return results

    def run_multi(
        self,
        run_seeds: Optional[List[int]] = None,
        fold_seeds: Optional[List[int]] = None,
        split_type: Optional[str] = None,
    ) -> PPIMultiRunResults:
        run_seeds = run_seeds or [42, 43, 44, 45, 46]
        fold_seeds = fold_seeds or [0, 1, 2, 3, 4]

        if len(run_seeds) != len(fold_seeds):
            logger.warning("[RUN_MULTI] run_seeds and fold_seeds length mismatch, truncating to min length")
            min_len = min(len(run_seeds), len(fold_seeds))
            run_seeds = run_seeds[:min_len]
            fold_seeds = fold_seeds[:min_len]

        all_results: List[PPIModelResults] = []
        for current_run_seed, current_fold_seed in zip(run_seeds, fold_seeds):
            logger.info(f"[RUN_MULTI] running seed={current_run_seed}, fold={current_fold_seed}")
            fresh_runner = self.clone()
            result = fresh_runner.run(
                run_seed=current_run_seed,
                fold_seed=current_fold_seed,
                split_type=split_type,
            )
            all_results.append(result)

        logger.info(f"[RUN_MULTI DONE] total runs: {len(all_results)}")
        return PPIMultiRunResults(results=all_results)

    def predict(self, features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")

        if self.data_context is None:
            self.data_context = self._create_data_context(features, features, features)

        dataset = self._build_split_dataset(features, self.data_context)
        dataloader = self._build_dataloader(dataset, shuffle=False)
        _, preds, probs = self._predict_loader(dataloader)
        return preds, probs

    def evaluate(self, features: Dict[str, Any], split: str = "test") -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        if self.data_context is None:
            self.data_context = self._create_data_context(features, features, features)
        return self._evaluate_features(features, self.data_context, split)

    @classmethod
    def load_model(
        cls,
        model_path: str,
        dataset: Optional[PPIDatasetManager] = None,
    ) -> "PPIRunner":
        if model_path.endswith(".pt"):
            base_dir = os.path.dirname(model_path)
            model_file = model_path
        else:
            base_dir = model_path
            model_file = os.path.join(base_dir, "model.pt")

        config_path = os.path.join(base_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json under: {base_dir}")

        with open(config_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        raw_config = payload.get("ppi_config", {})
        runtime = PPIRuntimeSettings(**raw_config.get("runtime", {}))
        training = PPITrainingSettings(**raw_config.get("training", {}))
        config = PPIConfig(
            protein_model=raw_config.get("protein_model", "facebook/esm2_t30_150M_UR50D"),
            protein_encoding=raw_config.get("protein_encoding", "plm"),
            peptide_encoding=raw_config.get("peptide_encoding", "plm"),
            peptide_model=raw_config.get("peptide_model"),
            protein_feature_key=raw_config.get("protein_feature_key"),
            peptide_feature_key=raw_config.get("peptide_feature_key"),
            fusion_method=raw_config.get("fusion_method", "concat"),
            hidden_dim=raw_config.get("hidden_dim", 256),
            freeze_encoders=raw_config.get("freeze_encoders", False),
            freeze_protein_encoder=raw_config.get("freeze_protein_encoder"),
            freeze_peptide_encoder=raw_config.get("freeze_peptide_encoder"),
            mlp_hidden_dim=raw_config.get("mlp_hidden_dim"),
            mlp_dropout=raw_config.get("mlp_dropout", 0.1),
            task_type=raw_config.get("task_type", "binary_classification"),
            max_protein_len=raw_config.get("max_protein_len"),
            max_peptide_len=raw_config.get("max_peptide_len"),
            base_dir=raw_config.get("base_dir", "./results"),
            seed=raw_config.get("seed", 42),
            metrics=raw_config.get("metrics"),
            model_name=raw_config.get("model_name"),
            runtime=runtime,
            training=training,
            model_params=raw_config.get("model_params", {}),
        )
        runner = PPIRunner(config=config, dataset=dataset)

        raw_context = payload.get("data_context")
        if raw_context is None:
            raise ValueError("Saved model is missing data_context; cannot reconstruct architecture.")
        context = PPIDataContext(**raw_context)
        context.protein_encoding_mode = config.protein_encoding
        context.peptide_encoding_mode = config.peptide_encoding
        if context.protein_feature_dim is None and context.protein_embedding_dim is not None:
            context.protein_feature_dim = context.protein_embedding_dim
        runner.data_context = context

        dummy_features = {context.label_key: [0, 1]}
        protein_dim = context.protein_feature_dim or context.protein_embedding_dim
        if context.protein_feature_key and protein_dim:
            dummy_features[context.protein_feature_key] = np.zeros((2, protein_dim), dtype=np.float32)
        if context.peptide_feature_key and context.peptide_feature_dim:
            dummy_features[context.peptide_feature_key] = np.zeros((2, context.peptide_feature_dim), dtype=np.float32)
        if context.protein_sequence_key:
            dummy_features[context.protein_sequence_key] = ["AA", "BB"]
        if context.peptide_sequence_key:
            dummy_features[context.peptide_sequence_key] = ["CC", "DD"]
        if (
            (context.peptide_feature_key and config.peptide_encoding == "gnn")
            or (context.protein_feature_key and config.protein_encoding == "gnn")
        ):
            raise ValueError("Loading GNN-based PPI model without dataset context is not yet supported.")

        runner.model = runner._build_model(context, dummy_features)
        runner.model.load_state_dict(torch.load(model_file, map_location=runner.device))
        runner.model.to(runner.device)
        runner.model.eval()
        return runner


class PPI(PPIRunner):
    def __init__(
        self,
        dataset: Optional[PPIDatasetManager] = None,
        **kwargs: Any,
    ):
        protein_backbone = kwargs.pop("protein_backbone", None)
        peptide_backbone = kwargs.pop("peptide_backbone", None)
        config = PPIConfig(
            protein_model=kwargs.pop("protein_model", "facebook/esm2_t30_150M_UR50D"),
            protein_encoding=kwargs.pop("protein_encoding", "plm"),
            peptide_encoding=kwargs.pop("peptide_encoding", "plm"),
            peptide_model=kwargs.pop("peptide_model", None),
            protein_feature_key=kwargs.pop("protein_feature_key", None),
            peptide_feature_key=kwargs.pop("peptide_feature_key", None),
            fusion_method=kwargs.pop("fusion_method", "concat"),
            hidden_dim=kwargs.pop("hidden_dim", 256),
            freeze_encoders=kwargs.pop("freeze_encoders", False),
            freeze_protein_encoder=kwargs.pop("freeze_protein_encoder", None),
            freeze_peptide_encoder=kwargs.pop("freeze_peptide_encoder", None),
            mlp_hidden_dim=kwargs.pop("mlp_hidden_dim", None),
            mlp_dropout=kwargs.pop("mlp_dropout", 0.1),
            task_type=kwargs.pop("task_type", "binary_classification"),
            max_protein_len=kwargs.pop("max_protein_len", None),
            max_peptide_len=kwargs.pop("max_peptide_len", None),
            base_dir=kwargs.pop("base_dir", "./results"),
            seed=kwargs.pop("seed", 42),
            metrics=kwargs.pop("metrics", None),
            model_name=kwargs.pop("model_name", None),
            runtime=PPIRuntimeSettings(
                local_files_only=kwargs.pop("local_files_only", False),
                cache_dir=kwargs.pop("cache_dir", None),
                local_model_dir=kwargs.pop("local_model_dir", None),
                dataloader_num_workers=kwargs.pop("dataloader_num_workers", 0),
            ),
            training=PPITrainingSettings(
                epochs=kwargs.pop("epochs", 30),
                learning_rate=kwargs.pop("learning_rate", 1e-4),
                batch_size=kwargs.pop("batch_size", 32),
                early_stopping_patience=kwargs.pop("early_stopping_patience", 5),
            ),
            model_params=kwargs,
        )
        super().__init__(
            config=config,
            dataset=dataset,
            protein_backbone=protein_backbone,
            peptide_backbone=peptide_backbone,
        )


MODEL_REGISTRY: Dict[str, Type[PPI]] = {
    "ppi": PPI,
}


def build_ppi_model(
    model_type: str = "ppi",
    dataset: Optional[PPIDatasetManager] = None,
    **kwargs: Any,
) -> PPIRunner:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    if dataset is not None:
        metadata = getattr(dataset, "dataset_metadata", None)
        if isinstance(metadata, dict):
            dataset_task_type = metadata.get("type")
        else:
            dataset_task_type = getattr(metadata, "type", None)

        if dataset_task_type is not None:
            dataset_task_type = str(dataset_task_type)

        if "task_type" not in kwargs and dataset_task_type is not None:
            kwargs["task_type"] = PPIRunner._normalize_task_type(dataset_task_type)
        elif "task_type" in kwargs and dataset_task_type is not None:
            requested_task = PPIRunner._normalize_task_type(kwargs["task_type"])
            inferred_task = PPIRunner._normalize_task_type(dataset_task_type)
            if requested_task != inferred_task:
                logger.warning(
                    f"[BUILD_PPI_MODEL] requested task_type={requested_task} but dataset metadata suggests {inferred_task}"
                )

    return MODEL_REGISTRY[model_type](dataset=dataset, **kwargs)
