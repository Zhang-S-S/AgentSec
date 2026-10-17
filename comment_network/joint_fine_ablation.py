"""End-to-end mini-batch training for the parallel R3H fusion model.

No Stage-1 representation cache is created. Each optimization step samples a
local directed/comment-thread subgraph, encodes target and context accounts
with shared multisource weights, performs graph/hypergraph message passing, and
updates the complete model from the final target-account loss.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multisource.model_fine_ablation import (FineAblationModelConfig,
                                             FineAblationRiskModel)
from multisource.data import (MultiSourceDataset, collate_accounts, load_accounts,
                              load_splits, split_hash)
from multisource.evaluation import classification_metrics, select_threshold
from multisource.url_features import load_url_cache

from .gnn_fine_ablation import JointR3HConfig, JointR3HFusionModel
from .stage2 import (EDGE_LOG1P_FEATURES, INCIDENCE_LOG1P_FEATURES,
                     REPETITION_FEATURES, THREAD_FEATURES, THREAD_LOG1P_FEATURES,
                     FeatureStandardizer, GraphView, RawGraph, induced_view)


TOPOLOGY_FEATURES = (
    "out_degree", "in_degree", "out_strength", "in_strength", "self_strength",
    "reciprocal_ratio", "target_hhi", "target_entropy", "pagerank", "active",
)
TOPOLOGY_LOG1P_FEATURES = (
    "out_degree", "in_degree", "out_strength", "in_strength", "self_strength",
)


@dataclass
class JointTrainConfig:
    seed: int = 42
    target_batch_size: int = 16
    max_posts: int = 128
    max_comments: int = 128
    max_urls: int = 64
    context_max_posts: int = 32
    context_max_comments: int = 32
    context_max_urls: int = 16
    max_in_neighbors: int = 4
    max_out_neighbors: int = 4
    max_threads: int = 2
    max_thread_participants: int = 4
    epochs: int = 30
    patience: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    baseline_aux_weight: float = 0.0
    device: str = "cuda:0"
    num_workers: int = 4
    amp: bool = True
    pos_weight: bool = True
    threshold_strategy: str = "max_f1"
    target_recall: float = 0.8
    graph_dropout: float = 0.15
    initial_gate_bias: float = -2.5
    use_content: bool = True
    use_repetition: bool = True
    use_directed: bool = True
    use_hypergraph: bool = True
    directed_mode: str = "both"
    use_topology: bool = True
    use_edge_features: bool = True

    def validate(self) -> None:
        positive = (
            "target_batch_size", "max_posts", "max_comments", "max_urls",
            "context_max_posts", "context_max_comments", "context_max_urls",
            "epochs", "patience",
        )
        if any(getattr(self, name) < 1 for name in positive):
            raise ValueError(f"These settings must be positive: {positive}")
        limits = ("max_in_neighbors", "max_out_neighbors", "max_threads",
                  "max_thread_participants")
        if any(getattr(self, name) < 0 for name in limits):
            raise ValueError(f"Sampling limits must be non-negative: {limits}")
        if self.max_thread_participants == 1:
            raise ValueError("max_thread_participants must be zero or at least two")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.baseline_aux_weight < 0:
            raise ValueError("baseline_aux_weight must be non-negative")
        if self.threshold_strategy not in {"max_f1", "max_precision_at_recall"}:
            raise ValueError("Unsupported threshold strategy")
        if not 0 <= self.target_recall <= 1:
            raise ValueError("target_recall must be in [0, 1]")
        if not 0 <= self.graph_dropout < 1:
            raise ValueError("graph_dropout must be in [0, 1)")
        if not (self.use_content or self.use_repetition
                or self.use_directed or self.use_hypergraph):
            raise ValueError("At least one content or relational branch must be enabled")
        if not self.use_content and self.baseline_aux_weight:
            raise ValueError("baseline_aux_weight requires the content branch")
        if self.directed_mode not in {"both", "incoming", "outgoing"}:
            raise ValueError(f"Unsupported directed_mode: {self.directed_mode}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JointGraphView:
    graph: GraphView
    topology_features: np.ndarray
    role_author_cutoff: float = 0.5

    def statistics(self) -> dict[str, int]:
        return self.graph.statistics()


def compute_topology_features(view: GraphView, comment_count_column: int) -> np.ndarray:
    """Compute label-free directed topology statistics entirely in memory."""
    size = len(view.global_nodes)
    output = np.zeros((size, len(TOPOLOGY_FEATURES)), dtype=np.float32)
    if not view.edge_index.shape[1]:
        return output
    source, target = view.edge_index
    weights = np.clip(view.edge_features[:, comment_count_column], 0, None).astype(np.float64)
    weights = np.where(weights > 0, weights, 1.0)
    nonself = source != target
    src, dst, weight = source[nonself], target[nonself], weights[nonself]
    out_degree = np.bincount(src, minlength=size).astype(np.float64)
    in_degree = np.bincount(dst, minlength=size).astype(np.float64)
    out_strength = np.bincount(src, weights=weight, minlength=size)
    in_strength = np.bincount(dst, weights=weight, minlength=size)
    self_strength = np.bincount(
        source[~nonself], weights=weights[~nonself], minlength=size
    )

    if len(src):
        keys = src.astype(np.int64) * size + dst
        reverse = dst.astype(np.int64) * size + src
        reciprocal = np.isin(keys, reverse, assume_unique=False)
        reciprocal_degree = np.bincount(src[reciprocal], minlength=size)
    else:
        reciprocal_degree = np.zeros(size)
    reciprocal_ratio = reciprocal_degree / np.maximum(out_degree, 1)

    hhi = np.zeros(size, dtype=np.float64)
    entropy = np.zeros(size, dtype=np.float64)
    if len(src):
        order = np.argsort(src, kind="stable")
        ordered_src, ordered_weight = src[order], weight[order]
        starts = np.flatnonzero(np.r_[True, ordered_src[1:] != ordered_src[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            node = int(ordered_src[start])
            probability = ordered_weight[start:end] / ordered_weight[start:end].sum()
            hhi[node] = np.square(probability).sum()
            entropy[node] = -(probability * np.log(probability + 1e-12)).sum()

    pagerank = np.full(size, 1 / max(1, size), dtype=np.float64)
    damping = 0.85
    if len(src):
        transition = weight / np.maximum(out_strength[src], 1e-12)
        dangling = out_strength == 0
        for _ in range(100):
            incoming = np.bincount(
                dst, weights=pagerank[src] * transition, minlength=size
            )
            updated = ((1 - damping) / size + damping * (
                incoming + pagerank[dangling].sum() / size
            ))
            if np.abs(updated - pagerank).sum() < 1e-10:
                pagerank = updated
                break
            pagerank = updated

    output[:] = np.stack([
        out_degree, in_degree, out_strength, in_strength, self_strength,
        reciprocal_ratio, hhi, entropy, pagerank,
        ((out_degree + in_degree + self_strength) > 0).astype(np.float64),
    ], axis=1).astype(np.float32)
    return output


def _view_local_indices(view: GraphView, global_indices: np.ndarray) -> np.ndarray:
    lookup = {int(global_index): local for local, global_index in enumerate(view.global_nodes)}
    missing = [int(index) for index in global_indices if int(index) not in lookup]
    if missing:
        raise ValueError(f"Target node is absent from graph view: {missing[0]}")
    return np.asarray([lookup[int(index)] for index in global_indices], dtype=np.int64)


class LocalSubgraphSampler:
    """One-hop directed and account-thread neighborhood sampler."""

    def __init__(self, view: JointGraphView, *, max_in_neighbors: int,
                 max_out_neighbors: int, max_threads: int,
                 max_thread_participants: int, training: bool) -> None:
        self.view = view
        self.training = training
        self.max_in_neighbors = max_in_neighbors
        self.max_out_neighbors = max_out_neighbors
        self.max_threads = max_threads
        self.max_thread_participants = max_thread_participants
        graph = view.graph
        size, thread_count = len(graph.global_nodes), len(graph.thread_features)
        self.in_edges: list[list[int]] = [[] for _ in range(size)]
        self.out_edges: list[list[int]] = [[] for _ in range(size)]
        for edge_id, (source, target) in enumerate(graph.edge_index.T.tolist()):
            self.out_edges[int(source)].append(edge_id)
            self.in_edges[int(target)].append(edge_id)
        self.node_incidences: list[list[int]] = [[] for _ in range(size)]
        self.thread_incidences: list[list[int]] = [[] for _ in range(thread_count)]
        for incidence_id, (node, thread) in enumerate(graph.incidence_index.T.tolist()):
            self.node_incidences[int(node)].append(incidence_id)
            self.thread_incidences[int(thread)].append(incidence_id)

    def _choose(self, values: Sequence[int], limit: int,
                scores: np.ndarray | None = None) -> list[int]:
        values = list(dict.fromkeys(int(x) for x in values))
        if limit == 0:
            return []
        if len(values) <= limit:
            return values
        if self.training:
            chosen = np.random.choice(np.asarray(values), size=limit, replace=False)
            return [int(x) for x in chosen]
        if scores is None:
            return sorted(values)[:limit]
        return sorted(values, key=lambda value: (-float(scores[value]), value))[:limit]

    def sample(self, targets: Sequence[int]) -> dict[str, np.ndarray]:
        graph = self.view.graph
        target_order = list(dict.fromkeys(int(x) for x in targets))
        target_set = set(target_order)
        selected_edges: set[int] = set()
        selected_threads: set[int] = set()
        selected_incidences: set[int] = set()
        nodes = set(target_order)
        edge_scores = graph.edge_features[:, 0] if len(graph.edge_features) else None
        thread_scores = graph.thread_features[:, 0] if len(graph.thread_features) else None
        author_column = 0

        for target in target_order:
            for edge_id in self._choose(
                self.in_edges[target], self.max_in_neighbors, edge_scores
            ) + self._choose(
                self.out_edges[target], self.max_out_neighbors, edge_scores
            ):
                selected_edges.add(edge_id)
                nodes.update(int(x) for x in graph.edge_index[:, edge_id])

            incident_rows = self.node_incidences[target]
            candidate_threads = [int(graph.incidence_index[1, row]) for row in incident_rows]
            for thread in self._choose(candidate_threads, self.max_threads, thread_scores):
                selected_threads.add(thread)
                rows = self.thread_incidences[thread]
                participants = list(dict.fromkeys(
                    int(graph.incidence_index[0, row]) for row in rows
                ))
                required = {target}
                required.update(
                    int(graph.incidence_index[0, row]) for row in rows
                    if graph.incidence_features[row, author_column]
                    > self.view.role_author_cutoff
                )
                if self.max_thread_participants and len(required) < self.max_thread_participants:
                    remaining = [node for node in participants if node not in required]
                    slots = self.max_thread_participants - len(required)
                    if self.training or len(remaining) <= slots:
                        selected_participants = self._choose(remaining, slots)
                    else:
                        participant_score: dict[int, float] = {}
                        for row in rows:
                            node = int(graph.incidence_index[0, row])
                            participant_score[node] = max(
                                participant_score.get(node, -float("inf")),
                                float(graph.incidence_features[row, 1]),
                            )
                        selected_participants = sorted(
                            remaining,
                            key=lambda node: (-participant_score.get(node, 0.0), node),
                        )[:slots]
                    required.update(selected_participants)
                elif self.max_thread_participants:
                    # Always retain the target; unusually many authors are
                    # deterministically truncated after it.
                    required = {target, *sorted(required.difference({target}))[
                        :self.max_thread_participants - 1
                    ]}
                else:
                    required.update(participants)
                nodes.update(required)
                selected_incidences.update(
                    row for row in rows
                    if int(graph.incidence_index[0, row]) in required
                )

        context_order = sorted(nodes.difference(target_set))
        node_order = target_order + context_order
        node_remap = {node: local for local, node in enumerate(node_order)}
        edge_ids = np.asarray(sorted(selected_edges), dtype=np.int64)
        if len(edge_ids):
            edge_index = np.asarray([
                [node_remap[int(graph.edge_index[0, edge])] for edge in edge_ids],
                [node_remap[int(graph.edge_index[1, edge])] for edge in edge_ids],
            ], dtype=np.int64)
            edge_features = graph.edge_features[edge_ids]
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_features = np.empty((0, graph.edge_features.shape[1]), dtype=np.float32)

        thread_order = sorted(selected_threads)
        thread_remap = {thread: local for local, thread in enumerate(thread_order)}
        incidence_ids = np.asarray(sorted(selected_incidences), dtype=np.int64)
        if len(incidence_ids):
            incidence_index = np.asarray([
                [node_remap[int(graph.incidence_index[0, row])] for row in incidence_ids],
                [thread_remap[int(graph.incidence_index[1, row])] for row in incidence_ids],
            ], dtype=np.int64)
            incidence_features = graph.incidence_features[incidence_ids]
        else:
            incidence_index = np.empty((2, 0), dtype=np.int64)
            incidence_features = np.empty(
                (0, graph.incidence_features.shape[1]), dtype=np.float32
            )
        thread_features = (
            graph.thread_features[np.asarray(thread_order, dtype=np.int64)]
            if thread_order else
            np.empty((0, graph.thread_features.shape[1]), dtype=np.float32)
        )
        node_ids = np.asarray(node_order, dtype=np.int64)
        return {
            "view_nodes": node_ids,
            "global_nodes": graph.global_nodes[node_ids],
            "target_count": np.asarray(len(target_order), dtype=np.int64),
            "topology_features": self.view.topology_features[node_ids],
            "repetition_features": graph.repetition_features[node_ids],
            "edge_index": edge_index,
            "edge_features": edge_features,
            "thread_features": thread_features,
            "incidence_index": incidence_index,
            "incidence_features": incidence_features,
        }


class _TargetDataset(Dataset):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = [int(x) for x in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return self.indices[index]


class JointBatchCollator:
    def __init__(self, sampler: LocalSubgraphSampler,
                 target_content: MultiSourceDataset | None,
                 context_content: MultiSourceDataset | None,
                 url_max_bytes: int, *, use_content: bool = True,
                 account_ids: np.ndarray | None = None) -> None:
        self.sampler = sampler
        self.target_content = target_content
        self.context_content = context_content
        self.url_max_bytes = url_max_bytes
        self.use_content = use_content
        self.account_ids = account_ids
        if use_content and (target_content is None or context_content is None):
            raise ValueError("Content datasets are required when use_content=True")
        if not use_content and account_ids is None:
            raise ValueError("account_ids are required for a content-free collator")

    def __call__(self, target_view_nodes: list[int]) -> dict[str, Any]:
        sample = self.sampler.sample(target_view_nodes)
        global_nodes = sample.pop("global_nodes")
        target_count = int(sample.pop("target_count"))
        if self.use_content:
            assert self.target_content is not None and self.context_content is not None
            target_rows = [self.target_content[int(index)]
                           for index in global_nodes[:target_count]]
            context_rows = [self.context_content[int(index)]
                            for index in global_nodes[target_count:]]
            target_content: dict[str, Any] = collate_accounts(
                target_rows, self.url_max_bytes
            )
            context_content: dict[str, Any] | None = (
                collate_accounts(context_rows, self.url_max_bytes) if context_rows else None
            )
            target_account_ids = [row["account_id"] for row in target_rows]
            target_labels = torch.tensor(
                [row["label"] for row in target_rows], dtype=torch.float32
            )
        else:
            assert self.account_ids is not None
            local_targets = np.asarray(sample["view_nodes"][:target_count], dtype=np.int64)
            target_content, context_content = {}, None
            target_account_ids = [
                str(self.account_ids[int(index)]) for index in global_nodes[:target_count]
            ]
            target_labels = torch.as_tensor(
                self.sampler.view.graph.labels[local_targets], dtype=torch.float32
            )
        result: dict[str, Any] = {
            "target_content": target_content,
            "context_content": context_content,
            "target_count": target_count,
            "target_account_ids": target_account_ids,
            "target_labels": target_labels,
        }
        for key, value in sample.items():
            dtype = torch.long if key in {"view_nodes", "edge_index", "incidence_index"} else torch.float32
            result[key] = torch.as_tensor(value, dtype=dtype)
        return result


def _worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _validate_inputs(graph: RawGraph, splits: dict[str, list[str]],
                     accounts: list[dict]) -> dict[str, np.ndarray]:
    account_labels = {str(row["account_id"]): int(row["label"]) for row in accounts}
    node_lookup = {str(aid): index for index, aid in enumerate(graph.node_ids)}
    split_ids = [aid for name in ("train", "validation", "test") for aid in splits[name]]
    if set(split_ids) != set(node_lookup):
        raise ValueError("The graph node set must exactly match the split account set")
    for index, aid in enumerate(graph.node_ids):
        if aid not in account_labels or account_labels[aid] != int(graph.labels[index]):
            raise ValueError(f"Graph label mismatch for account {aid}")
    return {
        name: np.asarray([node_lookup[aid] for aid in splits[name]], dtype=np.int64)
        for name in ("train", "validation", "test")
    }


def _prepare_views(graph: RawGraph, split_indices: dict[str, np.ndarray]
                   ) -> tuple[dict[str, JointGraphView], dict[str, Any]]:
    dummy = np.zeros((len(graph.node_ids), 1), dtype=np.float32)
    train_nodes = split_indices["train"]
    validation_nodes = split_indices["validation"]
    test_nodes = split_indices["test"]
    raw = {
        "train": induced_view(graph, dummy, train_nodes),
        "validation": induced_view(
            graph, dummy, np.concatenate([train_nodes, validation_nodes])
        ),
        "test": induced_view(
            graph, dummy, np.concatenate([train_nodes, validation_nodes, test_nodes])
        ),
    }
    comment_column = graph.edge_feature_names.index("comment_count")
    raw_topology = {
        name: compute_topology_features(view, comment_column) for name, view in raw.items()
    }
    standardizers = {
        "topology": FeatureStandardizer.fit(
            raw_topology["train"], TOPOLOGY_FEATURES, TOPOLOGY_LOG1P_FEATURES
        ),
        "repetition": FeatureStandardizer.fit(
            raw["train"].repetition_features, REPETITION_FEATURES
        ),
        "edge": FeatureStandardizer.fit(
            raw["train"].edge_features, graph.edge_feature_names, EDGE_LOG1P_FEATURES
        ),
        "thread": FeatureStandardizer.fit(
            raw["train"].thread_features, THREAD_FEATURES, THREAD_LOG1P_FEATURES
        ),
        "incidence": FeatureStandardizer.fit(
            raw["train"].incidence_features, graph.incidence_feature_names,
            INCIDENCE_LOG1P_FEATURES,
        ),
    }
    result = {}
    for name, view in raw.items():
        transformed = GraphView(
            global_nodes=view.global_nodes, labels=view.labels, content=view.content,
            repetition_features=standardizers["repetition"].transform(
                view.repetition_features
            ),
            edge_index=view.edge_index,
            edge_features=standardizers["edge"].transform(view.edge_features),
            thread_features=standardizers["thread"].transform(view.thread_features),
            incidence_index=view.incidence_index,
            incidence_features=standardizers["incidence"].transform(
                view.incidence_features
            ),
        )
        author_column = graph.incidence_feature_names.index("role_author")
        incidence_scaler = standardizers["incidence"]
        author_cutoff = float(
            (0.5 - incidence_scaler.mean[author_column])
            / incidence_scaler.scale[author_column]
        )
        result[name] = JointGraphView(
            transformed, standardizers["topology"].transform(raw_topology[name]),
            role_author_cutoff=author_cutoff,
        )
    return result, {name: scaler.to_dict() for name, scaler in standardizers.items()}


def _build_loader(view: JointGraphView, target_global_indices: np.ndarray,
                  embedding_dir: Path, url_cache: Path | None,
                  url_data: tuple[dict[str, list[dict]], dict] | None,
                  account_ids: np.ndarray,
                  model_config: FineAblationModelConfig,
                  train_config: JointTrainConfig,
                  *, training: bool) -> DataLoader:
    target_local = _view_local_indices(view.graph, target_global_indices)
    all_account_ids = [str(x) for x in account_ids]
    if train_config.use_content:
        target_content: MultiSourceDataset | None = MultiSourceDataset(
            embedding_dir, all_account_ids, training=training, seed=train_config.seed,
            max_posts=train_config.max_posts, max_comments=train_config.max_comments,
            max_urls=train_config.max_urls, url_cache=url_cache, url_data=url_data,
        )
        context_content: MultiSourceDataset | None = MultiSourceDataset(
            embedding_dir, all_account_ids, training=training, seed=train_config.seed,
            max_posts=train_config.context_max_posts,
            max_comments=train_config.context_max_comments,
            max_urls=train_config.context_max_urls, url_cache=url_cache, url_data=url_data,
        )
    else:
        target_content, context_content = None, None
    sampler = LocalSubgraphSampler(
        view, max_in_neighbors=train_config.max_in_neighbors,
        max_out_neighbors=train_config.max_out_neighbors,
        max_threads=train_config.max_threads,
        max_thread_participants=train_config.max_thread_participants,
        training=training,
    )
    generator = torch.Generator().manual_seed(train_config.seed + (0 if training else 10_000))
    loader_options: dict[str, Any] = {}
    if train_config.num_workers > 0:
        # The five-fold suite creates fresh loaders after CUDA has already been
        # exercised by an earlier fold. Linux's default ``fork`` start method
        # can deadlock in that situation; ``spawn`` starts clean worker
        # interpreters and is safe across fold transitions.
        loader_options["multiprocessing_context"] = "spawn"
    return DataLoader(
        _TargetDataset(target_local), batch_size=train_config.target_batch_size,
        shuffle=training, num_workers=train_config.num_workers,
        persistent_workers=train_config.num_workers > 0,
        pin_memory=train_config.device.startswith("cuda"), generator=generator,
        worker_init_fn=_worker_seed,
        collate_fn=JointBatchCollator(
            sampler, target_content, context_content, model_config.url_max_bytes,
            use_content=train_config.use_content, account_ids=account_ids,
        ),
        **loader_options,
    )


def _run_loader(model: JointR3HFusionModel, loader: DataLoader,
                device: torch.device, config: JointTrainConfig,
                pos_weight: torch.Tensor | None,
                optimizer: torch.optim.Optimizer | None = None,
                scaler: Any | None = None
                ) -> tuple[float, list[int], list[float], list[str], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    truth: list[int] = []
    probabilities: list[float] = []
    account_ids: list[str] = []
    gate_sum = np.zeros(3, dtype=np.float64)
    present_sum = np.zeros(3, dtype=np.float64)
    topology_gate_sum = 0.0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = _move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            target_count = int(batch["target_count"])
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                output = model(
                    batch["target_content"], batch["context_content"],
                    batch["topology_features"], batch["repetition_features"],
                    batch["edge_index"], batch["edge_features"],
                    batch["thread_features"], batch["incidence_index"],
                    batch["incidence_features"],
                )
                logits = output["account_logit"][:target_count]
                baseline_logits = output["baseline_logit"][:target_count]
                labels = batch["target_labels"]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
                if config.baseline_aux_weight:
                    loss = loss + config.baseline_aux_weight * (
                        nn.functional.binary_cross_entropy_with_logits(
                            baseline_logits, labels, pos_weight=pos_weight
                        )
                    )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if config.grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
            count = len(labels)
            total_loss += float(loss.detach().cpu()) * count
            total_items += count
            truth.extend(labels.long().detach().cpu().tolist())
            probabilities.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
            account_ids.extend(batch["target_account_ids"])
            gate_sum += output["modality_gates"][:target_count].detach().float().sum(0).cpu().numpy()
            present = output["branch_present"]
            present_sum += torch.stack([
                present["repetition"][:target_count],
                present["directed"][:target_count],
                present["hypergraph"][:target_count],
            ], dim=-1).float().sum(0).cpu().numpy()
            topology_gate_sum += float(
                output["topology_gate"][:target_count].detach().float().sum().cpu()
            )
    denominator = max(1, total_items)
    diagnostics = {
        "mean_gate": dict(zip(
            ("repetition", "directed", "hypergraph"),
            (gate_sum / denominator).tolist(),
        )),
        "present_rate": dict(zip(
            ("repetition", "directed", "hypergraph"),
            (present_sum / denominator).tolist(),
        )),
        "mean_topology_gate": topology_gate_sum / denominator,
    }
    return (total_loss / denominator, truth, probabilities, account_ids,
            diagnostics)


def train_joint_model(embedding_dir: Path, url_cache: Path | None,
                      splits_path: Path, graph_arrays: Path, output_dir: Path,
                      model_config: FineAblationModelConfig,
                      train_config: JointTrainConfig) -> dict[str, Any]:
    """Train the complete parallel model without intermediate representation files."""
    model_config.validate()
    train_config.validate()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output_dir}")
    if train_config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    device = torch.device(train_config.device)

    splits = load_splits(splits_path)
    accounts = load_accounts(embedding_dir)
    graph = RawGraph.load(graph_arrays)
    split_indices = _validate_inputs(graph, splits, accounts)
    views, standardizers = _prepare_views(graph, split_indices)
    shared_urls = (
        load_url_cache(url_cache)
        if url_cache and train_config.use_content else None
    )
    if model_config.use_url and train_config.use_content:
        if shared_urls is None:
            raise ValueError("URL is enabled; provide --url-cache")
        if shared_urls[1]["split_sha256"] != split_hash(splits):
            raise ValueError("URL cache was fitted with a different split")

    loaders = {
        "train": _build_loader(
            views["train"], split_indices["train"], embedding_dir, url_cache,
            shared_urls, graph.node_ids, model_config, train_config, training=True,
        ),
        "validation": _build_loader(
            views["validation"], split_indices["validation"], embedding_dir,
            url_cache, shared_urls, graph.node_ids, model_config, train_config,
            training=False,
        ),
        "test": _build_loader(
            views["test"], split_indices["test"], embedding_dir, url_cache,
            shared_urls, graph.node_ids, model_config, train_config, training=False,
        ),
    }
    account_encoder = FineAblationRiskModel(model_config)
    joint_config = JointR3HConfig(
        content_dim=model_config.hidden_dim, topology_dim=len(TOPOLOGY_FEATURES),
        repetition_dim=len(REPETITION_FEATURES), edge_dim=graph.edge_features.shape[1],
        thread_dim=len(THREAD_FEATURES), incidence_dim=graph.incidence_features.shape[1],
        dropout=model_config.dropout, graph_dropout=train_config.graph_dropout,
        initial_gate_bias=train_config.initial_gate_bias,
        use_content=train_config.use_content,
        use_repetition=train_config.use_repetition,
        use_directed=train_config.use_directed,
        use_hypergraph=train_config.use_hypergraph,
        directed_mode=train_config.directed_mode,
        use_topology=train_config.use_topology,
        use_edge_features=train_config.use_edge_features,
    )
    model = JointR3HFusionModel(account_encoder, joint_config).to(device)
    train_labels = graph.labels[split_indices["train"]]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    pos_weight = (
        torch.tensor(negatives / max(1, positives), device=device)
        if train_config.pos_weight else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    amp_enabled = train_config.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir.mkdir(parents=True)
    metadata = {
        "model": model_config.to_dict(), "joint_model": asdict(joint_config),
        "training": train_config.to_dict(), "embedding_dir": str(embedding_dir.resolve()),
        "url_cache": str(url_cache.resolve()) if url_cache else None,
        "splits": str(splits_path.resolve()), "split_sha256": split_hash(splits),
        "graph_arrays": str(graph_arrays.resolve()),
        "protocol": "inductive sampled neighborhoods",
        "view_statistics": {name: view.statistics() for name, view in views.items()},
        "feature_standardizers": standardizers,
        "topology_features": list(TOPOLOGY_FEATURES),
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "splits.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )

    best_score, best_epoch, stale = -float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, train_config.epochs + 1):
        train_loss, _, _, _, _ = _run_loader(
            model, loaders["train"], device, train_config, pos_weight,
            optimizer, scaler,
        )
        validation_loss, val_y, val_p, _, _ = _run_loader(
            model, loaders["validation"], device, train_config, pos_weight
        )
        val_pr_auc = (
            float(average_precision_score(val_y, val_p))
            if len(set(val_y)) == 2 else float("nan")
        )
        score = val_pr_auc if np.isfinite(val_pr_auc) else -validation_loss
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss, "validation_pr_auc": val_pr_auc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "model_config": model_config.to_dict(),
                "joint_config": asdict(joint_config),
                "train_config": train_config.to_dict(), "epoch": epoch,
                "validation_pr_auc": val_pr_auc, "threshold": None,
                "threshold_details": None, "metadata": metadata,
            }, output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= train_config.patience:
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_loss, val_y, val_p, _, validation_diagnostics = _run_loader(
        model, loaders["validation"], device, train_config, pos_weight
    )
    threshold, threshold_details = select_threshold(
        val_y, val_p, train_config.threshold_strategy, train_config.target_recall
    )
    checkpoint["threshold"] = threshold
    checkpoint["threshold_details"] = threshold_details
    torch.save(checkpoint, output_dir / "best_model.pt")
    test_loss, test_y, test_p, test_ids, test_diagnostics = _run_loader(
        model, loaders["test"], device, train_config, pos_weight
    )
    result = {
        "best_epoch": best_epoch, "validation_selection_metric": "pr_auc",
        "best_validation_pr_auc": checkpoint["validation_pr_auc"],
        "validation_loss": validation_loss, "threshold_selection": threshold_details,
        "test_loss": test_loss,
        "test": classification_metrics(test_y, test_p, threshold),
        "validation_branch_diagnostics": validation_diagnostics,
        "test_branch_diagnostics": test_diagnostics,
        "protocol": "inductive sampled neighborhoods", "device": str(device),
        "history": history,
    }
    (output_dir / "test_predictions.jsonl").write_text("".join(
        json.dumps({
            "account_id": aid, "label": int(label), "probability": probability,
            "prediction": int(probability >= threshold),
        }) + "\n" for aid, label, probability in zip(test_ids, test_y, test_p)
    ), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
