# gnn_model.py
# Copyright ZGCA
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass, asdict
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GlobalAttention,
    MessagePassing,
    Set2Set,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import degree

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
# Optional OGB encoders
# =========================
try:
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
except ImportError:
    logger.warning("OGB not found. Using simple atom/bond encoders.")

    class AtomEncoder(torch.nn.Module):
        def __init__(self, emb_dim: int):
            super().__init__()
            self.emb_dim = emb_dim
            self.atom_embedding = torch.nn.Embedding(119, emb_dim)
            self.linear: Optional[torch.nn.Linear] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                return self.atom_embedding(x.long())
            if x.dim() == 2 and x.size(1) == 1:
                return self.atom_embedding(x.squeeze(-1).long())

            if self.linear is None:
                self.linear = torch.nn.Linear(x.size(-1), self.emb_dim).to(x.device)
            return self.linear(x.float())

    class BondEncoder(torch.nn.Module):
        def __init__(self, emb_dim: int):
            super().__init__()
            self.emb_dim = emb_dim
            self.bond_embedding = torch.nn.Embedding(6, emb_dim)
            self.linear: Optional[torch.nn.Linear] = None

        def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
            if edge_attr.dim() == 1:
                return self.bond_embedding(edge_attr.long())
            if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                return self.bond_embedding(edge_attr.squeeze(-1).long())

            if self.linear is None:
                self.linear = torch.nn.Linear(edge_attr.size(-1), self.emb_dim).to(edge_attr.device)
            return self.linear(edge_attr.float())


# =========================
# Result Containers
# =========================
@dataclass
class GNNRunResults:
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
    epochs_trained: int
    best_val_loss: float


@dataclass
class GNNMultiRunResults:
    results: List[GNNRunResults]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for res in self.results:
            row = {
                "random_seed": res.random_seed,
                "fold_seed": res.fold_seed,
                "model_name": res.model_name,
                "training_time": res.training_time,
                "epochs_trained": res.epochs_trained,
                "best_val_loss": res.best_val_loss,
                "train_samples": res.train_samples,
                "valid_samples": res.valid_samples,
                "test_samples": res.test_samples,
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
        self.to_dataframe().to_csv(os.path.join(save_dir, "multi_run_results.csv"), index=False)
        with open(os.path.join(save_dir, "multi_run_summary.json"), "w") as f:
            json.dump(self.get_summary_stats(), f, indent=2)


# =========================
# Convolution Layers
# =========================
class GINConv(MessagePassing):
    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_embedding = self.bond_encoder(edge_attr)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
        root = F.relu(x + self.root_emb.weight) / deg.view(-1, 1)
        return out + root

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


# =========================
# Node Encoders
# =========================
class BaseNodeEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float = 0.5,
        JK: str = "last",
        residual: bool = False,
        gnn_type: str = "gin",
    ):
        super().__init__()

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.gnn_type = gnn_type

        self.atom_encoder = AtomEncoder(emb_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layer):
            self.convs.append(self._build_conv())
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def _build_conv(self) -> torch.nn.Module:
        if self.gnn_type == "gin":
            return GINConv(self.emb_dim)
        if self.gnn_type == "gcn":
            return GCNConv(self.emb_dim)
        if self.gnn_type == "gat":
            return GATConv(self.emb_dim, self.emb_dim // 6, heads=6, dropout=self.drop_ratio)
        if self.gnn_type == "transformer":
            return TransformerConv(
                self.emb_dim,
                self.emb_dim // 2,
                heads=2,
                dropout=self.drop_ratio,
                edge_dim=3,
            )
        raise ValueError(f"Undefined GNN type: {self.gnn_type}")

    def _apply_conv(
        self,
        layer_idx: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if self.gnn_type == "transformer":
            return self.convs[layer_idx](h, edge_index, edge_attr.float())
        return self.convs[layer_idx](h, edge_index, edge_attr)

    def _finalize_jk(self, h_list: List[torch.Tensor]) -> torch.Tensor:
        if self.JK == "last":
            return h_list[-1]
        if self.JK == "sum":
            out = 0
            for h in h_list:
                out = out + h
            return out
        raise ValueError(f"Unsupported JK strategy: {self.JK}")


class StandardNodeEncoder(BaseNodeEncoder):
    def forward(self, batched_data: Any) -> torch.Tensor:
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h = self._apply_conv(layer, h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        return self._finalize_jk(h_list)


class VirtualNodeEncoder(BaseNodeEncoder):
    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float = 0.5,
        JK: str = "last",
        residual: bool = False,
        gnn_type: str = "gin",
    ):
        if gnn_type not in {"gin", "gcn"}:
            raise ValueError("Virtual node encoder currently supports only 'gin' and 'gcn'.")

        super().__init__(
            num_layer=num_layer,
            emb_dim=emb_dim,
            drop_ratio=drop_ratio,
            JK=JK,
            residual=residual,
            gnn_type=gnn_type,
        )

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for _ in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                )
            )

    def forward(self, batched_data: Any) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        num_graphs = int(batch[-1].item()) + 1
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(num_graphs, dtype=torch.long, device=edge_index.device)
        )

        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            h = self._apply_conv(layer, h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            if layer < self.num_layer - 1:
                virtualnode_update = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                transformed = self.mlp_virtualnode_list[layer](virtualnode_update)
                transformed = F.dropout(transformed, self.drop_ratio, training=self.training)

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + transformed
                else:
                    virtualnode_embedding = transformed

        return self._finalize_jk(h_list)


# =========================
# Graph-level Predictor
# =========================
class GNNPredictor(torch.nn.Module):
    def __init__(
        self,
        num_tasks: int,
        num_layer: int = 5,
        emb_dim: int = 300,
        gnn_type: str = "gin",
        virtual_node: bool = True,
        residual: bool = False,
        drop_ratio: float = 0.5,
        JK: str = "last",
        graph_pooling: str = "mean",
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if virtual_node:
            self.node_encoder = VirtualNodeEncoder(
                num_layer=num_layer,
                emb_dim=emb_dim,
                drop_ratio=drop_ratio,
                JK=JK,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.node_encoder = StandardNodeEncoder(
                num_layer=num_layer,
                emb_dim=emb_dim,
                drop_ratio=drop_ratio,
                JK=JK,
                residual=residual,
                gnn_type=gnn_type,
            )

        self.pool = self._build_pool(graph_pooling, emb_dim)
        output_dim = 2 * emb_dim if graph_pooling == "set2set" else emb_dim
        self.graph_pred_linear = torch.nn.Linear(output_dim, num_tasks)

    @staticmethod
    def _build_pool(graph_pooling: str, emb_dim: int) -> Any:
        if graph_pooling == "sum":
            return global_add_pool
        if graph_pooling == "mean":
            return global_mean_pool
        if graph_pooling == "max":
            return global_max_pool
        if graph_pooling == "attention":
            return GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                )
            )
        if graph_pooling == "set2set":
            return Set2Set(emb_dim, processing_steps=2)
        raise ValueError(f"Invalid graph pooling type: {graph_pooling}")

    def forward(self, batched_data: Any) -> torch.Tensor:
        h_node = self.node_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)


# =========================
# Trainer
# =========================
class GNNTrainer:
    model_type = "gnn"

    def __init__(
        self,
        model_name: str = "gin",
        dataset: Optional[Any] = None,
        task_type: Optional[str] = None,
        epochs: int = 30,
        metrics: Optional[List[str]] = None,
        num_layer: int = 5,
        emb_dim: int = 300,
        gnn_type: str = "gin",
        virtual_node: bool = True,
        residual: bool = False,
        drop_ratio: float = 0.5,
        JK: str = "last",
        graph_pooling: str = "mean",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: Optional[str] = None,
        early_stopping_patience: int = 10,
        weight_decay: float = 0.0,
        **extra_config,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.task_type = task_type

        self.graph_feature_key = "graph"
        self.label_key = "label"

        self.epochs = epochs
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.gnn_type = gnn_type
        self.virtual_node = virtual_node
        self.residual = residual
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.graph_pooling = graph_pooling
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.extra_config = extra_config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics

        self.model: Optional[GNNPredictor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[torch.nn.Module] = None
        self.training_time: float = 0.0
        self.training_history: List[Dict[str, Any]] = []
        self.experiment_dir: Optional[str] = None
        self.random_seed: Optional[int] = None
        self.num_tasks: Optional[int] = None

        logger.info(f"[GNN] Using device: {self.device}")

    def clone(self) -> "GNNTrainer":
        return self.__class__(
            model_name=self.model_name,
            dataset=self.dataset,
            task_type=self.task_type,
            epochs=self.epochs,
            metrics=copy.deepcopy(self.metrics),
            num_layer=self.num_layer,
            emb_dim=self.emb_dim,
            gnn_type=self.gnn_type,
            virtual_node=self.virtual_node,
            residual=self.residual,
            drop_ratio=self.drop_ratio,
            JK=self.JK,
            graph_pooling=self.graph_pooling,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
            weight_decay=self.weight_decay,
            **copy.deepcopy(self.extra_config),
        )

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

    def infer_task_type(self) -> str:
        if self.task_type in {"binary_classification", "regression"}:
            return self.task_type

        if self.dataset is None:
            self.task_type = "binary_classification"
            return self.task_type

        try:
            labels = self.dataset.get_official_feature("label")
            numeric_labels = []
            for label in labels:
                try:
                    if isinstance(label, (int, np.integer)):
                        numeric_labels.append(int(label))
                    elif isinstance(label, (float, np.floating)):
                        numeric_labels.append(float(label))
                    else:
                        numeric_labels.append(float(label))
                except Exception:
                    self.task_type = "binary_classification"
                    return self.task_type

            unique_values = set(numeric_labels)
            all_binary_like = unique_values.issubset({0, 1})

            if len(unique_values) == 2 and all_binary_like:
                self.task_type = "binary_classification"
            else:
                self.task_type = "regression"

            logger.info(f"[GNN] Inferred task type: {self.task_type}")
            return self.task_type
        except Exception as e:
            logger.warning(f"[GNN] Could not infer task type: {e}. Defaulting to binary_classification.")
            self.task_type = "binary_classification"
            return self.task_type

    def infer_num_tasks(self) -> int:
        task_type = self.infer_task_type()
        self.num_tasks = 1 if task_type in {"binary_classification", "regression"} else None
        return self.num_tasks

    def get_metrics(self) -> List[str]:
        if self.metrics is not None:
            return self.metrics
        if self.infer_task_type() == "regression":
            return list(Regression_Metric)
        return list(Classification_Metric)

    def _build_criterion(self) -> torch.nn.Module:
        task_type = self.infer_task_type()
        if task_type == "binary_classification":
            return torch.nn.BCEWithLogitsLoss()
        if task_type == "regression":
            return torch.nn.MSELoss()
        raise ValueError(f"Unsupported task type: {task_type}")

    def _format_labels(self, labels: Any, device: torch.device) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels_tensor = labels.to(device)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        return labels_tensor.float()

    @staticmethod
    def _get_pyg_dataloader_cls():
        try:
            from torch_geometric.loader import DataLoader as PyGDataLoader
            return PyGDataLoader
        except ImportError:
            try:
                from torch_geometric.data import DataLoader as PyGDataLoader
                return PyGDataLoader
            except ImportError as e:
                raise ImportError("PyTorch Geometric is not installed.") from e

    def _validate_graph_features(self) -> None:
        if self.dataset is None:
            raise ValueError("[GNN] Dataset not provided.")

        train_features, _, _ = self.dataset.get_train_val_test_features("dict")
        if self.graph_feature_key not in train_features:
            raise ValueError(f"[GNN] Graph features '{self.graph_feature_key}' not found in dataset.")

    def _pack_graphs_with_labels(self, graphs: List[Any], labels: List[Any]) -> List[Any]:
        packed = []
        for graph, label in zip(graphs, labels):
            graph_copy = copy.deepcopy(graph)
            graph_copy.y = torch.tensor([label], dtype=torch.float32) if not isinstance(label, torch.Tensor) else label
            packed.append(graph_copy)
        return packed

    def _prepare_dataloaders(self) -> Tuple[Any, Any, Any]:
        self._validate_graph_features()

        train_features, valid_features, test_features = self.dataset.get_train_val_test_features("dict")

        train_graphs = self._pack_graphs_with_labels(
            train_features[self.graph_feature_key],
            train_features[self.label_key],
        )
        valid_graphs = self._pack_graphs_with_labels(
            valid_features[self.graph_feature_key],
            valid_features[self.label_key],
        )
        test_graphs = self._pack_graphs_with_labels(
            test_features[self.graph_feature_key],
            test_features[self.label_key],
        )

        PyGDataLoader = self._get_pyg_dataloader_cls()
        train_loader = PyGDataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        valid_loader = PyGDataLoader(valid_graphs, batch_size=self.batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_graphs, batch_size=self.batch_size, shuffle=False)

        logger.info(
            f"[DATA LOADERS] Train={len(train_graphs)}, Valid={len(valid_graphs)}, Test={len(test_graphs)}"
        )
        return train_loader, valid_loader, test_loader

    def build_model(self) -> GNNPredictor:
        num_tasks = self.infer_num_tasks()
        model = GNNPredictor(
            num_tasks=num_tasks,
            num_layer=self.num_layer,
            emb_dim=self.emb_dim,
            gnn_type=self.gnn_type,
            virtual_node=self.virtual_node,
            residual=self.residual,
            drop_ratio=self.drop_ratio,
            JK=self.JK,
            graph_pooling=self.graph_pooling,
        ).to(self.device)

        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.criterion = self._build_criterion()

        logger.info(f"[BUILD MODEL DONE] {self.model_name} | num_tasks={num_tasks}")
        return model

    def save_model(self, name: str = "model.pt") -> None:
        if self.model is None:
            raise ValueError("Model is not built yet.")
        if self.experiment_dir is None:
            raise ValueError("experiment_dir is not set.")

        model_path = os.path.join(self.experiment_dir, name)
        torch.save(self.model.state_dict(), model_path)

        config = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "task_type": self.infer_task_type(),
            "epochs": self.epochs,
            "metrics": self.get_metrics(),
            "num_layer": self.num_layer,
            "emb_dim": self.emb_dim,
            "gnn_type": self.gnn_type,
            "virtual_node": self.virtual_node,
            "residual": self.residual,
            "drop_ratio": self.drop_ratio,
            "JK": self.JK,
            "graph_pooling": self.graph_pooling,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "weight_decay": self.weight_decay,
            "num_tasks": self.num_tasks,
            "extra_config": self.extra_config,
            "dataset_name": self.dataset.dataset_name if self.dataset else None,
        }

        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_model(
        cls,
        model_path: str,
        dataset: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "GNNTrainer":
        if model_path.endswith(".pt"):
            model_file = model_path
            base_dir = os.path.dirname(model_path)
        else:
            model_file = os.path.join(model_path, "model.pt")
            base_dir = model_path

        config_path = os.path.join(base_dir, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        inst = cls(
            model_name=config["model_name"],
            dataset=dataset,
            task_type=config.get("task_type"),
            epochs=config.get("epochs", 30),
            metrics=config.get("metrics"),
            num_layer=config.get("num_layer", 5),
            emb_dim=config.get("emb_dim", 300),
            gnn_type=config.get("gnn_type", "gin"),
            virtual_node=config.get("virtual_node", True),
            residual=config.get("residual", False),
            drop_ratio=config.get("drop_ratio", 0.5),
            JK=config.get("JK", "last"),
            graph_pooling=config.get("graph_pooling", "mean"),
            learning_rate=config.get("learning_rate", 1e-3),
            batch_size=config.get("batch_size", 32),
            device=device,
            early_stopping_patience=config.get("early_stopping_patience", 10),
            weight_decay=config.get("weight_decay", 0.0),
            **config.get("extra_config", {}),
            **kwargs,
        )
        inst.num_tasks = config.get("num_tasks", 1)
        inst.build_model()
        inst.model.load_state_dict(torch.load(model_file, map_location=inst.device))
        inst.experiment_dir = base_dir
        return inst

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        task_type = self.infer_task_type()
        if task_type == "binary_classification":
            return self.criterion(logits.view(-1), labels.view(-1).float())
        if task_type == "regression":
            return self.criterion(logits.view(-1), labels.view(-1).float())
        raise ValueError(f"Unsupported task type: {task_type}")

    def train_epoch(self, data_loader: Any) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            try:
                batch = batch.to(self.device)
                labels = self._format_labels(batch.y, self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = self._compute_loss(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.warning(f"[TRAIN EPOCH] Error in batch processing: {e}")
                continue

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate_loss(self, data_loader: Any) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch = batch.to(self.device)
                    labels = self._format_labels(batch.y, self.device)
                    logits = self.model(batch)
                    loss = self._compute_loss(logits, labels)

                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    logger.warning(f"[EVALUATE LOSS] Error in batch processing: {e}")
                    continue

        return total_loss / num_batches if num_batches > 0 else 0.0

    def predict_loader(self, data_loader: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        scores = []
        labels_all = []

        task_type = self.infer_task_type()

        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch = batch.to(self.device)
                    logits = self.model(batch)
                    labels = batch.y.detach().cpu().view(-1).numpy()

                    if task_type == "binary_classification":
                        probs = torch.sigmoid(logits.view(-1))
                        preds = (probs > 0.5).long()

                        predictions.extend(preds.cpu().numpy().tolist())
                        scores.extend(probs.cpu().numpy().tolist())
                        labels_all.extend(labels.tolist())

                    elif task_type == "regression":
                        values = logits.view(-1).cpu().numpy()
                        predictions.extend(values.tolist())
                        scores.extend(values.tolist())
                        labels_all.extend(labels.tolist())

                    else:
                        raise ValueError(f"Unsupported task type: {task_type}")
                except Exception as e:
                    logger.warning(f"[PREDICT] Error in batch processing: {e}")
                    continue

        return np.array(labels_all), np.array(predictions), np.array(scores)

    def evaluate_loader(self, data_loader: Any, split: str) -> Dict[str, float]:
        y_true, y_pred, y_score = self.predict_loader(data_loader)
        task_type = self.infer_task_type()

        if task_type == "binary_classification":
            metrics = evaluate_classification(y_true=y_true, y_pred=y_pred, y_score=y_score)
        elif task_type == "regression":
            metrics = evaluate_regression(y_true=y_true, y_pred=y_pred)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        logger.info(f"[METRICS] {split}: {metrics}")
        return metrics

    def fit(self) -> Dict[str, Any]:
        if self.dataset is None:
            raise ValueError("dataset cannot be None in fit().")

        train_loader, valid_loader, _ = self._prepare_dataloaders()
        self.build_model()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []

        logger.info(
            f"[GNN TRAINING] Starting for {self.epochs} epochs with patience={self.early_stopping_patience}"
        )

        start_time = time.time()

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate_loss(valid_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"[EPOCH {epoch + 1}/{self.epochs}] "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"[EARLY STOPPING] Stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.training_time = time.time() - start_time
        self.training_history = [
            {"epoch": i + 1, "train_loss": train_losses[i], "val_loss": val_losses[i]}
            for i in range(len(train_losses))
        ]

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
            "history": self.training_history,
        }

    def run(
        self,
        run_seed: int = 42,
        fold_seed: int = 0,
        split_type: str = "random_split",
        base_dir: str = "./results",
    ) -> GNNRunResults:
        if self.dataset is None:
            raise ValueError("dataset cannot be None in run().")

        logger.info(f"[RUN] {self.model_name} | seed={run_seed}, fold={fold_seed}, split={split_type}")

        self.random_seed = run_seed
        set_seed(run_seed)

        self.dataset.set_official_split_indices(split_type=split_type, fold_seed=fold_seed)

        self.experiment_dir = self.get_experiment_dir(
            base_dir=base_dir,
            split_type=split_type,
            fold_seed=fold_seed,
            run_seed=run_seed,
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        train_loader, valid_loader, test_loader = self._prepare_dataloaders()
        fit_info = self.fit()

        train_metrics = self.evaluate_loader(train_loader, split="train")
        valid_metrics = self.evaluate_loader(valid_loader, split="valid")
        test_metrics = self.evaluate_loader(test_loader, split="test")

        results = GNNRunResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            training_time=self.training_time,
            train_samples=len(train_loader.dataset),
            valid_samples=len(valid_loader.dataset),
            test_samples=len(test_loader.dataset),
            random_seed=run_seed,
            fold_seed=fold_seed,
            model_name=self.model_name,
            epochs_trained=fit_info["epochs_trained"],
            best_val_loss=float(fit_info["best_val_loss"]),
        )

        self.save_model()
        with open(os.path.join(self.experiment_dir, "metrics.json"), "w") as f:
            json.dump(asdict(results), f, indent=2)
        with open(os.path.join(self.experiment_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)

        return results

    def run_multi(
        self,
        run_seeds: Optional[List[int]] = None,
        fold_seeds: Optional[List[int]] = None,
        split_type: str = "random_split",
        base_dir: str = "./results",
    ) -> GNNMultiRunResults:
        run_seeds = run_seeds or [42, 43, 44, 45, 46]
        fold_seeds = fold_seeds or [0, 1, 2, 3, 4]

        if len(run_seeds) != len(fold_seeds):
            logger.warning("[RUN_MULTI] run_seeds and fold_seeds length mismatch, truncating.")
            min_len = min(len(run_seeds), len(fold_seeds))
            run_seeds = run_seeds[:min_len]
            fold_seeds = fold_seeds[:min_len]

        all_results = []
        for run_seed, fold_seed in zip(run_seeds, fold_seeds):
            fresh_model = self.clone()
            result = fresh_model.run(
                run_seed=run_seed,
                fold_seed=fold_seed,
                split_type=split_type,
                base_dir=base_dir,
            )
            all_results.append(result)

        return GNNMultiRunResults(results=all_results)


GRAPH_MODEL_REGISTRY: Dict[str, Type[GNNTrainer]] = {
    "gnn": GNNTrainer,
    "gin": GNNTrainer,
    "gcn": GNNTrainer,
    "gat": GNNTrainer,
    "transformer": GNNTrainer,
}


def build_graph_model(model_type: str, **kwargs) -> GNNTrainer:
    if model_type not in GRAPH_MODEL_REGISTRY:
        raise ValueError(f"Unsupported graph model type: {model_type}")

    model_name = kwargs.pop("model_name", model_type)

    if "gnn_type" not in kwargs and model_type in {"gin", "gcn", "gat", "transformer"}:
        kwargs["gnn_type"] = model_type

    return GRAPH_MODEL_REGISTRY[model_type](
        model_name=model_name,
        **kwargs,
    )