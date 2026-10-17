"""Frozen-backbone, second-stage training for R3H-Net.

Stage 1 is the unchanged ``MultiSourceRiskModel``.  Its account representation
is extracted once and cached.  Stage 2 trains only the repetition/graph head.
The default inductive protocol constructs separate train, train+validation,
and full test graph views, so held-out accounts are absent from message passing
and thread hyperedges until their evaluation phase.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from multisource.config import ModelConfig, TrainConfig
from multisource.data import (MultiSourceDataset, collate_accounts, load_accounts,
                              load_splits, move_batch, split_hash)
from multisource.evaluation import classification_metrics, select_threshold
from multisource.model import MultiSourceRiskModel
from multisource.url_features import load_url_cache

from .gnn import R3HConfig, R3HNetwork


REPETITION_FEATURES = (
    "exact_duplicate_ratio",
    "normalized_duplicate_ratio",
    "max_duplicate_cluster_share",
    "cross_target_repeat_fraction",
    "cross_post_repeat_fraction",
)
EDGE_LOG1P_FEATURES = (
    "comment_count", "top_level_count", "reply_count", "post_fallback_count",
    "distinct_posts", "duration_days",
)
INCIDENCE_LOG1P_FEATURES = ("comment_count",)
THREAD_FEATURES = ("comment_count", "reply_count", "commenter_count", "participant_count")
THREAD_LOG1P_FEATURES = THREAD_FEATURES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _names(value: np.ndarray) -> tuple[str, ...]:
    return tuple(str(x) for x in value.tolist())


def _indices(names: Iterable[str], selected: Iterable[str]) -> list[int]:
    names = tuple(names)
    missing = set(selected).difference(names)
    if missing:
        raise ValueError(f"Missing graph features: {sorted(missing)}")
    return [names.index(name) for name in selected]


@dataclass
class FeatureStandardizer:
    names: tuple[str, ...]
    log1p_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, names: Iterable[str],
            log1p_names: Iterable[str] = ()) -> "FeatureStandardizer":
        names, log1p_names = tuple(names), tuple(log1p_names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Feature matrix and feature names are inconsistent")
        transformed = cls._prepare(values, names, log1p_names)
        if len(transformed):
            mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
            scale[scale < 1e-6] = 1.0
        else:
            mean = np.zeros(len(names), np.float32)
            scale = np.ones(len(names), np.float32)
        return cls(names, log1p_names, mean, scale)

    @staticmethod
    def _prepare(values: np.ndarray, names: tuple[str, ...],
                 log1p_names: tuple[str, ...]) -> np.ndarray:
        result = np.nan_to_num(np.asarray(values, dtype=np.float32), copy=True)
        for index in _indices(names, log1p_names):
            result[:, index] = np.log1p(np.clip(result[:, index], 0, None))
        return result

    def transform(self, values: np.ndarray) -> np.ndarray:
        prepared = self._prepare(values, self.names, self.log1p_names)
        return ((prepared - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names), "log1p_names": list(self.log1p_names),
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
        }


@dataclass
class RawGraph:
    node_ids: np.ndarray
    labels: np.ndarray
    repetition_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_feature_names: tuple[str, ...]
    thread_count: int
    incidence_index: np.ndarray
    incidence_features: np.ndarray
    incidence_feature_names: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "RawGraph":
        with np.load(path, allow_pickle=False) as archive:
            node_names = _names(archive["node_feature_names"])
            repetition = archive["node_features"][:, _indices(node_names, REPETITION_FEATURES)]
            graph = cls(
                node_ids=archive["node_ids"].astype(str),
                labels=archive["labels"].astype(np.int64),
                repetition_features=repetition.astype(np.float32),
                edge_index=archive["edge_index"].astype(np.int64),
                edge_features=archive["edge_features"].astype(np.float32),
                edge_feature_names=_names(archive["edge_feature_names"]),
                thread_count=len(archive["thread_ids"]),
                incidence_index=archive["incidence_index"].astype(np.int64),
                incidence_features=archive["incidence_features"].astype(np.float32),
                incidence_feature_names=_names(archive["incidence_feature_names"]),
            )
        graph.validate()
        return graph

    def validate(self) -> None:
        count = len(self.node_ids)
        if count != len(set(self.node_ids.tolist())):
            raise ValueError("graph_arrays.npz contains duplicate node IDs")
        if self.labels.shape != (count,) or not set(np.unique(self.labels)).issubset({0, 1}):
            raise ValueError("Graph labels must contain one binary label per node")
        if self.repetition_features.shape != (count, len(REPETITION_FEATURES)):
            raise ValueError("Unexpected repetition feature shape")
        if self.edge_index.shape[0] != 2 or self.edge_index.shape[1] != len(self.edge_features):
            raise ValueError("Edge index and feature rows are inconsistent")
        if self.incidence_index.shape[0] != 2 or self.incidence_index.shape[1] != len(self.incidence_features):
            raise ValueError("Incidence index and feature rows are inconsistent")
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= count):
            raise ValueError("Edge index is outside the node range")
        if self.incidence_index.size:
            if self.incidence_index[0].min() < 0 or self.incidence_index[0].max() >= count:
                raise ValueError("Incidence node index is outside the node range")
            if self.incidence_index[1].min() < 0 or self.incidence_index[1].max() >= self.thread_count:
                raise ValueError("Incidence thread index is outside the thread range")


@dataclass
class GraphView:
    global_nodes: np.ndarray
    labels: np.ndarray
    content: np.ndarray
    repetition_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    thread_features: np.ndarray
    incidence_index: np.ndarray
    incidence_features: np.ndarray

    def statistics(self) -> dict[str, int]:
        return {
            "nodes": len(self.global_nodes), "edges": self.edge_index.shape[1],
            "threads": len(self.thread_features), "incidences": self.incidence_index.shape[1],
        }


def induced_view(graph: RawGraph, content: np.ndarray,
                 allowed_nodes: np.ndarray | list[int]) -> GraphView:
    """Create a node-induced directed graph and reconstruct its thread hypergraph."""
    allowed = np.unique(np.asarray(allowed_nodes, dtype=np.int64))
    if not len(allowed):
        raise ValueError("Cannot construct an empty graph view")
    if allowed.min() < 0 or allowed.max() >= len(graph.node_ids):
        raise ValueError("Allowed node index is outside the graph")
    remap = np.full(len(graph.node_ids), -1, dtype=np.int64)
    remap[allowed] = np.arange(len(allowed), dtype=np.int64)

    edge_keep = (remap[graph.edge_index[0]] >= 0) & (remap[graph.edge_index[1]] >= 0)
    selected_edges = graph.edge_index[:, edge_keep]
    edge_index = remap[selected_edges] if selected_edges.size else np.empty((2, 0), np.int64)
    edge_features = graph.edge_features[edge_keep]

    incidence_keep = remap[graph.incidence_index[0]] >= 0
    selected_incidence = graph.incidence_index[:, incidence_keep]
    incidence_features = graph.incidence_features[incidence_keep]
    if selected_incidence.shape[1]:
        used_threads = np.unique(selected_incidence[1])
        thread_remap = np.full(graph.thread_count, -1, dtype=np.int64)
        thread_remap[used_threads] = np.arange(len(used_threads), dtype=np.int64)
        local_nodes = remap[selected_incidence[0]]
        local_threads = thread_remap[selected_incidence[1]]
        incidence_index = np.stack([local_nodes, local_threads])

        comment_col = graph.incidence_feature_names.index("comment_count")
        reply_col = graph.incidence_feature_names.index("reply_fraction")
        author_col = graph.incidence_feature_names.index("role_author")
        comments = incidence_features[:, comment_col]
        thread_comment_count = np.bincount(
            local_threads, weights=comments, minlength=len(used_threads)
        )
        thread_reply_count = np.bincount(
            local_threads, weights=comments * incidence_features[:, reply_col],
            minlength=len(used_threads),
        )
        thread_commenters = np.bincount(
            local_threads, weights=(incidence_features[:, author_col] < 0.5),
            minlength=len(used_threads),
        )
        unique_pairs = np.unique(np.stack([local_threads, local_nodes], axis=1), axis=0)
        thread_participants = np.bincount(unique_pairs[:, 0], minlength=len(used_threads))
        thread_features = np.stack([
            thread_comment_count, thread_reply_count, thread_commenters, thread_participants,
        ], axis=1).astype(np.float32)
    else:
        incidence_index = np.empty((2, 0), np.int64)
        thread_features = np.empty((0, len(THREAD_FEATURES)), np.float32)

    return GraphView(
        global_nodes=allowed, labels=graph.labels[allowed], content=content[allowed],
        repetition_features=graph.repetition_features[allowed], edge_index=edge_index,
        edge_features=edge_features, thread_features=thread_features,
        incidence_index=incidence_index, incidence_features=incidence_features,
    )


def fit_standardizers(train_view: GraphView, graph: RawGraph) -> dict[str, FeatureStandardizer]:
    return {
        "repetition": FeatureStandardizer.fit(
            train_view.repetition_features, REPETITION_FEATURES
        ),
        "edge": FeatureStandardizer.fit(
            train_view.edge_features, graph.edge_feature_names, EDGE_LOG1P_FEATURES
        ),
        "thread": FeatureStandardizer.fit(
            train_view.thread_features, THREAD_FEATURES, THREAD_LOG1P_FEATURES
        ),
        "incidence": FeatureStandardizer.fit(
            train_view.incidence_features, graph.incidence_feature_names,
            INCIDENCE_LOG1P_FEATURES,
        ),
    }


def transform_view(view: GraphView,
                   standardizers: dict[str, FeatureStandardizer]) -> GraphView:
    return replace(
        view,
        repetition_features=standardizers["repetition"].transform(view.repetition_features),
        edge_features=standardizers["edge"].transform(view.edge_features),
        thread_features=standardizers["thread"].transform(view.thread_features),
        incidence_features=standardizers["incidence"].transform(view.incidence_features),
    )


@dataclass
class Stage2Config:
    seed: int = 42
    hidden_dim: int = 128
    dropout: float = 0.2
    graph_dropout: float = 0.15
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda:0"
    amp: bool = True
    protocol: str = "inductive"
    pos_weight: bool = True
    grad_clip: float = 5.0
    threshold_strategy: str = "max_f1"
    target_recall: float = 0.8
    use_repetition: bool = True
    use_directed: bool = True
    use_hypergraph: bool = True

    def validate(self) -> None:
        if self.protocol not in {"inductive", "transductive"}:
            raise ValueError("protocol must be inductive or transductive")
        if min(self.hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError("hidden_dim, epochs, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.threshold_strategy not in {"max_f1", "max_precision_at_recall"}:
            raise ValueError("Unsupported threshold strategy")
        if not 0 <= self.target_recall <= 1:
            raise ValueError("target_recall must be in [0, 1]")
        if not 0 <= self.dropout < 1 or not 0 <= self.graph_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_backbone_representations(
    checkpoint_path: Path,
    embedding_dir: Path,
    url_cache: Path | None,
    splits_path: Path,
    node_ids: np.ndarray,
    output_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract deterministic Stage-1 account vectors, or validate and reuse a cache."""
    checkpoint_sha = _sha256_file(checkpoint_path)
    splits = load_splits(splits_path)
    current_split_sha = split_hash(splits)
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as archive:
            cached_ids = archive["node_ids"].astype(str)
            if not np.array_equal(cached_ids, node_ids.astype(str)):
                raise ValueError("Content cache node order differs from graph_arrays.npz")
            if str(archive["checkpoint_sha256"].item()) != checkpoint_sha:
                raise ValueError("Content cache was extracted from a different Stage-1 checkpoint")
            if str(archive["split_sha256"].item()) != current_split_sha:
                raise ValueError("Content cache was extracted with a different split")
            values = archive["representations"].astype(np.float32)
        return values, {
            "path": str(output_path.resolve()), "reused": True,
            "checkpoint_sha256": checkpoint_sha, "split_sha256": current_split_sha,
        }

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("split_sha256") != current_split_sha:
        raise ValueError("Stage-1 checkpoint was trained with a different split")
    model_config = ModelConfig(**checkpoint["model_config"])
    train_config = TrainConfig(**checkpoint["train_config"])
    saved_url_manifest = checkpoint.get("url_preprocessor")
    if model_config.use_url and saved_url_manifest is not None and url_cache is None:
        raise ValueError("The Stage-1 checkpoint was trained with a URL cache; provide --url-cache")
    # URL-disabled ablation checkpoints may still record the suite's shared
    # cache manifest. Do not load that unused cache during extraction.
    effective_url_cache = url_cache if model_config.use_url else None
    shared_urls = load_url_cache(effective_url_cache) if effective_url_cache else None
    if effective_url_cache:
        manifest = shared_urls[1]
        if manifest["split_sha256"] != current_split_sha:
            raise ValueError("URL cache was fitted with a different split")
        if saved_url_manifest is not None and manifest != saved_url_manifest:
            raise ValueError("URL cache differs from the preprocessor saved in the Stage-1 checkpoint")
        if model_config.use_url and saved_url_manifest is None:
            raise ValueError(
                "Stage-1 was trained without a URL cache; supplying one now would change its behavior"
            )
    dataset = MultiSourceDataset(
        embedding_dir, [str(x) for x in node_ids], training=False, seed=train_config.seed,
        max_posts=train_config.max_posts, max_comments=train_config.max_comments,
        max_urls=train_config.max_urls, url_cache=effective_url_cache, url_data=shared_urls,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed, generator=generator,
        collate_fn=partial(collate_accounts, url_max_bytes=model_config.url_max_bytes),
    )
    model = MultiSourceRiskModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.requires_grad_(False)
    representations, observed_ids = [], []
    with torch.no_grad():
        for batch in loader:
            observed_ids.extend(batch["account_ids"])
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                output = model(batch)
            representations.append(output["account_representation"].float().cpu().numpy())
    if observed_ids != [str(x) for x in node_ids]:
        raise RuntimeError("Stage-1 extraction changed account order")
    values = np.concatenate(representations).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, node_ids=node_ids.astype(str), representations=values,
            checkpoint_sha256=np.asarray(checkpoint_sha),
            split_sha256=np.asarray(current_split_sha),
        )
    temporary.replace(output_path)
    return values, {
        "path": str(output_path.resolve()), "reused": False,
        "checkpoint_sha256": checkpoint_sha, "split_sha256": current_split_sha,
    }


def _to_device(view: GraphView, device: torch.device) -> dict[str, torch.Tensor]:
    def tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)
    return {
        "content_representation": tensor(view.content, torch.float32),
        "repetition_features": tensor(view.repetition_features, torch.float32),
        "edge_index": tensor(view.edge_index, torch.long),
        "edge_features": tensor(view.edge_features, torch.float32),
        "thread_features": tensor(view.thread_features, torch.float32),
        "incidence_index": tensor(view.incidence_index, torch.long),
        "incidence_features": tensor(view.incidence_features, torch.float32),
        "labels": tensor(view.labels, torch.float32),
    }


def _local_mask(view: GraphView, global_indices: np.ndarray) -> np.ndarray:
    return np.isin(view.global_nodes, global_indices)


def _loss(logits: torch.Tensor, labels: torch.Tensor,
          pos_weight: torch.Tensor | None) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def _evaluate_view(model: R3HNetwork, view: GraphView, selected: np.ndarray,
                   device: torch.device, amp: bool,
                   pos_weight: torch.Tensor | None) -> tuple[float, list[int], list[float], np.ndarray]:
    model.eval()
    local = _local_mask(view, selected)
    if not local.any():
        raise ValueError("Evaluation selection is absent from its graph view")
    tensors = _to_device(view, device)
    mask = torch.as_tensor(local, dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
    ):
        output = model(**{key: value for key, value in tensors.items() if key != "labels"})
        logits = output["account_logit"][mask]
        labels = tensors["labels"][mask]
        loss = _loss(logits, labels, pos_weight)
    probabilities = torch.sigmoid(logits.float()).cpu().tolist()
    truth = labels.long().cpu().tolist()
    return float(loss.cpu()), truth, probabilities, view.global_nodes[local]


def _validate_graph_against_inputs(graph: RawGraph, splits: dict[str, list[str]],
                                   embedding_dir: Path) -> dict[str, np.ndarray]:
    by_id = {str(account["account_id"]): int(account["label"])
             for account in load_accounts(embedding_dir)}
    split_ids = [aid for name in ("train", "validation", "test") for aid in splits[name]]
    if set(split_ids) != set(graph.node_ids.tolist()):
        missing_graph = set(split_ids).difference(graph.node_ids.tolist())
        missing_split = set(graph.node_ids.tolist()).difference(split_ids)
        raise ValueError(
            "Graph/split account sets differ: "
            f"missing_from_graph={len(missing_graph)}, missing_from_split={len(missing_split)}"
        )
    node_lookup = {aid: index for index, aid in enumerate(graph.node_ids.tolist())}
    for index, aid in enumerate(graph.node_ids.tolist()):
        if aid not in by_id or by_id[aid] != int(graph.labels[index]):
            raise ValueError(f"Graph label disagrees with embedding metadata for account {aid}")
    return {
        name: np.asarray([node_lookup[aid] for aid in splits[name]], dtype=np.int64)
        for name in ("train", "validation", "test")
    }


def train_stage2(
    checkpoint_path: Path,
    embedding_dir: Path,
    splits_path: Path,
    graph_arrays: Path,
    output_dir: Path,
    config: Stage2Config,
    *,
    url_cache: Path | None = None,
    content_cache: Path | None = None,
    extraction_batch_size: int = 16,
    extraction_num_workers: int = 4,
) -> dict[str, Any]:
    """Train R3H-Net without updating or overwriting the Stage-1 model."""
    config.validate()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output_dir}")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    _seed_everything(config.seed)
    device = torch.device(config.device)
    splits = load_splits(splits_path)
    graph = RawGraph.load(graph_arrays)
    split_indices = _validate_graph_against_inputs(graph, splits, embedding_dir)
    output_dir.mkdir(parents=True)
    cache_path = content_cache or output_dir / "stage1_representations.npz"
    content, cache_metadata = extract_backbone_representations(
        checkpoint_path, embedding_dir, url_cache, splits_path, graph.node_ids,
        cache_path, config.device, extraction_batch_size, extraction_num_workers, config.amp,
    )
    if content.shape[0] != len(graph.node_ids):
        raise ValueError("Stage-1 cache has the wrong number of accounts")

    train_nodes = split_indices["train"]
    validation_nodes = split_indices["validation"]
    test_nodes = split_indices["test"]
    fit_view = induced_view(graph, content, train_nodes)
    standardizers = fit_standardizers(fit_view, graph)
    if config.protocol == "inductive":
        raw_views = {
            "train": fit_view,
            "validation": induced_view(
                graph, content, np.concatenate([train_nodes, validation_nodes])
            ),
            "test": induced_view(
                graph, content, np.concatenate([train_nodes, validation_nodes, test_nodes])
            ),
        }
        views = {
            name: transform_view(view, standardizers) for name, view in raw_views.items()
        }
    else:
        full = induced_view(
            graph, content, np.concatenate([train_nodes, validation_nodes, test_nodes])
        )
        raw_views = {"train": full, "validation": full, "test": full}
        transformed_full = transform_view(full, standardizers)
        views = {"train": transformed_full, "validation": transformed_full,
                 "test": transformed_full}
    view_statistics = {key: value.statistics() for key, value in raw_views.items()}
    del raw_views, fit_view

    model_config = R3HConfig(
        content_dim=content.shape[1], hidden_dim=config.hidden_dim,
        repetition_dim=len(REPETITION_FEATURES), edge_dim=graph.edge_features.shape[1],
        thread_dim=len(THREAD_FEATURES), incidence_dim=graph.incidence_features.shape[1],
        dropout=config.dropout, graph_dropout=config.graph_dropout,
        use_repetition=config.use_repetition, use_directed=config.use_directed,
        use_hypergraph=config.use_hypergraph,
    )
    model = R3HNetwork(model_config).to(device)
    labels = graph.labels[train_nodes]
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    pos_weight = (
        torch.tensor(negatives / max(1, positives), dtype=torch.float32, device=device)
        if config.pos_weight else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = config.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    metadata = {
        "stage2": config.to_dict(), "r3h": asdict(model_config),
        "stage1_checkpoint": str(checkpoint_path.resolve()),
        "embedding_dir": str(embedding_dir.resolve()),
        "url_cache": str(url_cache.resolve()) if url_cache else None,
        "splits": str(splits_path.resolve()), "graph_arrays": str(graph_arrays.resolve()),
        "graph_sha256": _sha256_file(graph_arrays), "content_cache": cache_metadata,
        "feature_standardizers": {key: value.to_dict() for key, value in standardizers.items()},
        "view_statistics": view_statistics,
        "protocol_note": (
            "train-only / train+validation / full induced views"
            if config.protocol == "inductive" else
            "one full unlabeled topology view with split-specific supervised masks"
        ),
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "splits.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_score, best_epoch, stale = -float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_view = views["train"]
        tensors = _to_device(train_view, device)
        train_mask = torch.as_tensor(
            _local_mask(train_view, train_nodes), dtype=torch.bool, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            output = model(**{key: value for key, value in tensors.items() if key != "labels"})
            train_loss = _loss(
                output["account_logit"][train_mask], tensors["labels"][train_mask], pos_weight
            )
        if scaler.is_enabled():
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        train_loss_value = float(train_loss.detach().cpu())
        # Do not retain the large training hypergraph autograd buffers while
        # transferring the validation graph to the GPU.
        del output, tensors, train_mask, train_loss
        validation_loss, val_y, val_p, _ = _evaluate_view(
            model, views["validation"], validation_nodes, device, config.amp, pos_weight
        )
        validation_pr_auc = (
            float(average_precision_score(val_y, val_p))
            if len(set(val_y)) == 2 else float("nan")
        )
        score = validation_pr_auc if np.isfinite(validation_pr_auc) else -validation_loss
        row = {
            "epoch": epoch, "train_loss": train_loss_value,
            "validation_loss": validation_loss, "validation_pr_auc": validation_pr_auc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "r3h_config": asdict(model_config),
                "stage2_config": config.to_dict(), "epoch": epoch,
                "validation_pr_auc": validation_pr_auc, "threshold": None,
                "threshold_details": None, "metadata": metadata,
            }, output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= config.patience:
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_loss, val_y, val_p, _ = _evaluate_view(
        model, views["validation"], validation_nodes, device, config.amp, pos_weight
    )
    threshold, threshold_details = select_threshold(
        val_y, val_p, config.threshold_strategy, config.target_recall
    )
    checkpoint["threshold"] = threshold
    checkpoint["threshold_details"] = threshold_details
    torch.save(checkpoint, output_dir / "best_model.pt")

    test_loss, test_y, test_p, test_global_nodes = _evaluate_view(
        model, views["test"], test_nodes, device, config.amp, pos_weight
    )
    test_ids = [str(graph.node_ids[index]) for index in test_global_nodes]
    result = {
        "best_epoch": best_epoch, "validation_selection_metric": "pr_auc",
        "best_validation_pr_auc": checkpoint["validation_pr_auc"],
        "validation_loss": validation_loss, "threshold_selection": threshold_details,
        "test_loss": test_loss, "test": classification_metrics(test_y, test_p, threshold),
        "protocol": config.protocol, "view_statistics": metadata["view_statistics"],
        "branch_configuration": {
            "repetition": config.use_repetition, "directed": config.use_directed,
            "hypergraph": config.use_hypergraph,
        },
        "device": str(device), "history": history,
    }
    (output_dir / "test_predictions.jsonl").write_text("".join(
        json.dumps({
            "account_id": aid, "label": int(label), "probability": probability,
            "prediction": int(probability >= threshold),
        }) + "\n"
        for aid, label, probability in zip(test_ids, test_y, test_p)
    ), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
