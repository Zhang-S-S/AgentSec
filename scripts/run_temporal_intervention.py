#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import resource
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from comment_network import joint_fine_ablation as base
from comment_network.analysis import AnalysisConfig, build_and_analyze
from comment_network.gnn_fine_ablation import JointR3HConfig
from comment_network.joint_risk_residual_mil import (
    JointRiskResidualMILModel,
    RiskResidualMILTrainConfig,
    _run_loader,
)
from comment_network.stage2 import FeatureStandardizer, GraphView, RawGraph, induced_view
from multisource.data import load_accounts, split_hash
from multisource.evaluation import classification_metrics, select_threshold
from multisource.model_risk_residual_mil import RiskResidualMILConfig, RiskResidualMILModel
from multisource.url_features import (
    FEATURE_NAMES,
    SOURCE_TO_ID,
    URLStandardizer,
    lexical_features,
    normalize_url,
)
from agentsec.determinism import configure_determinism
from agentsec.experiment import _model_config, _training_config


TRAIN_START = "2026-05-01T00:00:00Z"
TRAIN_END = "2026-05-18T00:00:00Z"
DETECTION_END = "2026-05-25T00:00:00Z"
EVALUATION_END = "2026-06-02T00:00:00Z"
INTERVENTION_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal training, detection, and retrospective intervention simulation"
    )
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--labels-db", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "train", "detect", "intervene"),
        default="all",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_labels(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT agent_id,is_malicious FROM manual_labels "
            "WHERE is_malicious IN (0,1)"
        ).fetchall()
    finally:
        connection.close()
    labels = {str(account_id): int(label) for account_id, label in rows}
    if len(labels) != len(rows) or not labels:
        raise ValueError("Labels must be non-empty and unique")
    return labels


def read_account_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows[str(item["account_id"])] = item
    return rows


def utc_value(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def initialize_temporal_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE instances (
          record_id INTEGER PRIMARY KEY, account_id TEXT, source_type TEXT,
          shard INTEGER, row_index INTEGER, valid_mask INTEGER
        );
        CREATE INDEX by_account ON instances(account_id,source_type,record_id);
        CREATE TABLE urls_raw (
          account_id TEXT, source_type TEXT, content_id TEXT, raw_url TEXT
        );
        CREATE INDEX urls_account ON urls_raw(account_id,source_type);
        """
    )
    return connection


def temporal_view_complete(path: Path) -> bool:
    manifest = path / "temporal_manifest.json"
    if not manifest.is_file():
        return False
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return bool(value.get("complete")) and (path / "index.sqlite").is_file()


def prepare_temporal_views(
    source_embedding: Path,
    output_root: Path,
    labels: dict[str, int],
) -> dict[str, Path]:
    definitions = {
        "training": (TRAIN_START, TRAIN_END),
        "detection": (TRAIN_START, DETECTION_END),
    }
    view_paths = {
        name: output_root / "data" / f"{name}_embeddings"
        for name in definitions
    }
    pending = [name for name, path in view_paths.items() if not temporal_view_complete(path)]
    if not pending:
        return view_paths
    for name in pending:
        path = view_paths[name]
        if path.exists():
            raise RuntimeError(f"Incomplete temporal view exists: {path}")
        path.mkdir(parents=True)

    master_accounts = read_account_rows(source_embedding / "accounts.jsonl")
    connections = {
        name: initialize_temporal_index(view_paths[name] / "index.sqlite")
        for name in pending
    }
    bounds = {
        name: (utc_value(start), utc_value(end))
        for name, (start, end) in definitions.items()
    }
    instance_batches: dict[str, list[tuple[Any, ...]]] = {name: [] for name in pending}
    url_batches: dict[str, list[tuple[str, str, str, str]]] = {
        name: [] for name in pending
    }
    scanned = 0
    for metadata_path in sorted((source_embedding / "shards").glob("shard-*.jsonl")):
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                account_id = str(item["account_id"])
                if account_id not in labels:
                    continue
                source = str(item["source_type"])
                timestamp = utc_value(item.get("timestamp"))
                for name in pending:
                    start, end = bounds[name]
                    include = source.endswith("profile") or (
                        source in {"post", "comment"}
                        and timestamp is not None
                        and start <= timestamp < end
                    )
                    if not include:
                        continue
                    instance_batches[name].append((
                        int(item["record_id"]), account_id, source,
                        int(item["shard"]), int(item["row_index"]),
                        int(item["valid_mask"]),
                    ))
                    for link in item.get("urls", []):
                        raw_url = link.get("raw_url")
                        if raw_url:
                            url_batches[name].append((
                                account_id, source, str(item["content_id"]), str(raw_url)
                            ))
                scanned += 1
                if scanned % 100_000 == 0:
                    print(json.dumps({"stage": "temporal_index", "records": scanned}), flush=True)
                if any(len(batch) >= 20_000 for batch in instance_batches.values()):
                    for name in pending:
                        if instance_batches[name]:
                            connections[name].executemany(
                                "INSERT INTO instances VALUES (?,?,?,?,?,?)",
                                instance_batches[name],
                            )
                            instance_batches[name].clear()
                        if url_batches[name]:
                            connections[name].executemany(
                                "INSERT INTO urls_raw VALUES (?,?,?,?)",
                                url_batches[name],
                            )
                            url_batches[name].clear()
                        connections[name].commit()

    source_manifest = json.loads(
        (source_embedding / "embedding_manifest.json").read_text(encoding="utf-8")
    )
    for name in pending:
        connection = connections[name]
        if instance_batches[name]:
            connection.executemany(
                "INSERT INTO instances VALUES (?,?,?,?,?,?)", instance_batches[name]
            )
        if url_batches[name]:
            connection.executemany(
                "INSERT INTO urls_raw VALUES (?,?,?,?)", url_batches[name]
            )
        active = {
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT account_id FROM instances "
                "WHERE source_type IN ('post','comment') AND valid_mask=1"
            )
        }
        connection.execute(
            "CREATE TEMP TABLE active_accounts(account_id TEXT PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO active_accounts VALUES (?)", ((account_id,) for account_id in active)
        )
        connection.execute(
            "DELETE FROM instances WHERE account_id NOT IN (SELECT account_id FROM active_accounts)"
        )
        connection.execute(
            "DELETE FROM urls_raw WHERE account_id NOT IN (SELECT account_id FROM active_accounts)"
        )
        connection.commit()
        counts = dict(connection.execute(
            "SELECT source_type,COUNT(*) FROM instances GROUP BY source_type"
        ))
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()

        path = view_paths[name]
        rows = []
        for account_id in sorted(active):
            base_row = dict(master_accounts[account_id])
            base_row["label"] = labels[account_id]
            base_row["split"] = "temporal"
            rows.append(base_row)
        atomic_jsonl(path / "accounts.jsonl", rows)
        atomic_json(path / "labels.json", labels_for(active, labels))
        atomic_jsonl(
            path / "graph_labels.jsonl",
            ({"account_id": account_id, "label": labels[account_id]}
             for account_id in sorted(active)),
        )
        manifest = dict(source_manifest)
        manifest["accounts"] = len(active)
        manifest["records"] = int(sum(counts.values()))
        manifest["temporal_view"] = {
            "name": name,
            "start_inclusive": definitions[name][0],
            "end_exclusive": definitions[name][1],
            "profile_policy": "static snapshot",
            "record_counts": counts,
        }
        atomic_json(path / "embedding_manifest.json", manifest)
        os.symlink(source_embedding.resolve() / "shards", path / "shards")
        atomic_json(path / "temporal_manifest.json", {
            "complete": True,
            "name": name,
            "active_labeled_accounts": len(active),
            "positive": sum(labels[account_id] for account_id in active),
            "negative": len(active) - sum(labels[account_id] for account_id in active),
            "start_inclusive": definitions[name][0],
            "end_exclusive": definitions[name][1],
            "record_counts": counts,
        })
    return view_paths


def labels_for(account_ids: Iterable[str], labels: dict[str, int]) -> dict[str, int]:
    return {account_id: labels[account_id] for account_id in sorted(account_ids)}


def account_group(row: dict[str, Any]) -> str:
    owner = str(row.get("twitter_owner_group") or "").strip().casefold()
    return owner or f"moltbook:{row['account_id']}"


def make_train_validation_split(
    accounts: list[dict[str, Any]], seed: int
) -> dict[str, list[str]]:
    ids = np.asarray([str(row["account_id"]) for row in accounts], dtype=object)
    labels = np.asarray([int(row["label"]) for row in accounts], dtype=np.int64)
    groups = np.asarray([account_group(row) for row in accounts], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    target_rate = float(labels.mean())
    candidates = []
    placeholder = np.zeros(len(ids), dtype=np.int8)
    for train_index, validation_index in splitter.split(placeholder, labels, groups):
        score = abs(len(validation_index) / len(ids) - 0.2)
        score += abs(float(labels[validation_index].mean()) - target_rate)
        candidates.append((score, train_index, validation_index))
    _, train_index, validation_index = min(candidates, key=lambda row: row[0])
    result = {
        "train": sorted(ids[train_index].tolist()),
        "validation": sorted(ids[validation_index].tolist()),
    }
    for name, members in result.items():
        if len({labels[np.where(ids == account_id)[0][0]] for account_id in members}) != 2:
            raise ValueError(f"{name} contains only one class")
    return result


def url_rows_from_index(index_path: Path, allowed: set[str]) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    try:
        raw_rows = connection.execute(
            "SELECT account_id,source_type,content_id,raw_url FROM urls_raw"
        )
        aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
        invalid = 0
        for account_id, source, content_id, raw_url in raw_rows:
            account_id = str(account_id)
            if account_id not in allowed or source not in SOURCE_TO_ID:
                continue
            try:
                normalized = normalize_url(str(raw_url))
            except (TypeError, ValueError):
                invalid += 1
                continue
            key = (account_id, normalized.normalized_url, str(source))
            if key not in aggregated:
                aggregated[key] = {
                    "account_id": account_id,
                    "source_type": str(source),
                    "raw_url": normalized.raw_url,
                    "raw_urls": [normalized.raw_url],
                    "normalized_url": normalized.normalized_url,
                    "encoding_text": normalized.encoding_text,
                    "content_ids": [str(content_id)],
                    "count": 1,
                    "normalized": normalized,
                }
            else:
                row = aggregated[key]
                row["count"] += 1
                if str(content_id) not in row["content_ids"]:
                    row["content_ids"].append(str(content_id))
                if normalized.raw_url not in row["raw_urls"]:
                    row["raw_urls"].append(normalized.raw_url)
        rows = list(aggregated.values())
        rows.sort(key=lambda row: (
            row["account_id"], row["source_type"], row["normalized_url"]
        ))
        for row in rows:
            row["features_raw"] = lexical_features(
                row.pop("normalized"), int(row["count"])
            )
            row["source_id"] = SOURCE_TO_ID[row["source_type"]]
        if rows:
            rows[0]["_invalid_total"] = invalid
        return rows
    finally:
        connection.close()


def prepare_url_cache(
    view_dir: Path,
    output_dir: Path,
    allowed: set[str],
    fit_accounts: set[str] | None = None,
    standardizer: URLStandardizer | None = None,
) -> tuple[Path, URLStandardizer]:
    if (output_dir / "manifest.json").is_file() and (output_dir / "urls.jsonl").is_file():
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        fitted = URLStandardizer.from_dict(manifest["standardizer"])
        return output_dir, fitted
    if output_dir.exists():
        raise RuntimeError(f"Incomplete URL cache exists: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = url_rows_from_index(view_dir / "index.sqlite", allowed)
    invalid = int(rows[0].pop("_invalid_total", 0)) if rows else 0
    if standardizer is None:
        if fit_accounts is None:
            raise ValueError("fit_accounts is required when fitting URL features")
        matrix = [
            row["features_raw"] for row in rows
            if row["account_id"] in fit_accounts
        ]
        values = (
            np.stack(matrix)
            if matrix else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        )
        standardizer = URLStandardizer().fit(values)
    with (output_dir / "urls.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            features = standardizer.transform(row.pop("features_raw")[None])[0]
            row["features"] = features.tolist()
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_json(output_dir / "manifest.json", {
        "version": "moltbook-url-lexical-v1",
        "embedding_dir": str(view_dir.resolve()),
        "split_sha256": "temporal",
        "fit_split": "historical_train",
        "rows": len(rows),
        "invalid_urls": invalid,
        "source_mapping": SOURCE_TO_ID,
        "standardizer": standardizer.to_dict(),
        "temporal_index_respected": True,
    })
    return output_dir, standardizer


def build_temporal_graph(
    source_db: Path,
    view_dir: Path,
    output_dir: Path,
    end_at: str,
    seed: int,
) -> Path:
    arrays = output_dir / "graph_arrays.npz"
    if arrays.is_file() and (output_dir / "report.json").is_file():
        return arrays
    if output_dir.exists():
        raise RuntimeError(f"Incomplete graph directory exists: {output_dir}")
    build_and_analyze(AnalysisConfig(
        source_db=source_db,
        output_dir=output_dir,
        labels=view_dir / "graph_labels.jsonl",
        seed=seed,
        permutations=1,
        labeled_only=True,
        start_at=TRAIN_START,
        end_at=end_at,
        skip_statistical_analysis=True,
    ))
    return arrays


def model_arguments(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        variant="full",
        seed=args.seed,
        post_top_k=3,
        comment_top_k=5,
        neighbors=4,
        text_loss_weight=1.0,
        max_posts=128,
        max_comments=128,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=1e-4,
        device=args.device,
        num_workers=args.num_workers,
    )


def joint_config(
    model_config: RiskResidualMILConfig,
    train_config: RiskResidualMILTrainConfig,
    graph: RawGraph,
) -> JointR3HConfig:
    return JointR3HConfig(
        content_dim=model_config.hidden_dim,
        topology_dim=len(base.TOPOLOGY_FEATURES),
        repetition_dim=len(base.REPETITION_FEATURES),
        edge_dim=graph.edge_features.shape[1],
        thread_dim=len(base.THREAD_FEATURES),
        incidence_dim=graph.incidence_features.shape[1],
        dropout=model_config.dropout,
        graph_dropout=train_config.graph_dropout,
        initial_gate_bias=train_config.initial_gate_bias,
        use_content=train_config.use_content,
        use_repetition=train_config.use_repetition,
        use_directed=train_config.use_directed,
        use_hypergraph=train_config.use_hypergraph,
        directed_mode=train_config.directed_mode,
        use_topology=train_config.use_topology,
        use_edge_features=train_config.use_edge_features,
    )


def train_model(
    embedding_dir: Path,
    graph_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.is_file() and (output_dir / "results.json").is_file():
        return checkpoint_path
    if output_dir.exists():
        raise RuntimeError(f"Incomplete training directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    configure_determinism(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accounts = load_accounts(embedding_dir)
    splits = make_train_validation_split(accounts, args.seed)
    atomic_json(output_dir / "splits.json", splits)
    by_id = {str(row["account_id"]): row for row in accounts}
    atomic_json(output_dir / "split_statistics.json", {
        name: {
            "accounts": len(members),
            "positive": sum(int(by_id[account_id]["label"]) for account_id in members),
            "negative": len(members) - sum(
                int(by_id[account_id]["label"]) for account_id in members
            ),
        }
        for name, members in splits.items()
    })
    url_cache, _ = prepare_url_cache(
        embedding_dir,
        output_dir / "url_cache",
        set(by_id),
        fit_accounts=set(splits["train"]),
    )

    settings = model_arguments(args)
    model_config = _model_config(settings, embedding_dir)
    train_config = _training_config(settings)
    graph = RawGraph.load(graph_path)
    node_lookup = {str(account_id): index for index, account_id in enumerate(graph.node_ids)}
    if set(node_lookup) != set(by_id):
        raise ValueError("Training graph and embedding account sets differ")
    split_indices = {
        name: np.asarray([node_lookup[account_id] for account_id in members], dtype=np.int64)
        for name, members in splits.items()
    }
    split_indices["test"] = split_indices["validation"]
    views, standardizers = base._prepare_views(graph, split_indices)
    shared_urls = base.load_url_cache(url_cache)
    loaders = {
        "train": base._build_loader(
            views["train"], split_indices["train"], embedding_dir, url_cache,
            shared_urls, graph.node_ids, model_config, train_config, training=True,
        ),
        "validation": base._build_loader(
            views["validation"], split_indices["validation"], embedding_dir,
            url_cache, shared_urls, graph.node_ids, model_config, train_config,
            training=False,
        ),
    }
    device = torch.device(args.device)
    configuration = joint_config(model_config, train_config, graph)
    model = JointRiskResidualMILModel(
        RiskResidualMILModel(model_config), configuration
    ).to(device)
    train_labels = graph.labels[split_indices["train"]]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor(
        negatives / max(1, positives), device=device
    ) if train_config.pos_weight else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    amp_enabled = train_config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, train_config.epochs + 1):
        train_loss, _, _, _, train_diagnostics = _run_loader(
            model, loaders["train"], device, train_config, pos_weight,
            optimizer, scaler,
        )
        validation_loss, val_y, val_p, val_ids, validation_diagnostics = _run_loader(
            model, loaders["validation"], device, train_config, pos_weight,
        )
        val_pr_auc = float(average_precision_score(val_y, val_p))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_pr_auc": val_pr_auc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model_config.to_dict(),
                "joint_config": asdict(configuration),
                "train_config": train_config.to_dict(),
                "epoch": epoch,
                "validation_pr_auc": val_pr_auc,
                "threshold": None,
                "threshold_details": None,
                "feature_standardizers": standardizers,
                "time_window": {
                    "start_inclusive": TRAIN_START,
                    "end_exclusive": TRAIN_END,
                },
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= train_config.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_loss, val_y, val_p, val_ids, validation_diagnostics = _run_loader(
        model, loaders["validation"], device, train_config, pos_weight,
    )
    threshold, threshold_details = select_threshold(
        val_y, val_p, train_config.threshold_strategy, train_config.target_recall
    )
    checkpoint["threshold"] = threshold
    checkpoint["threshold_details"] = threshold_details
    torch.save(checkpoint, checkpoint_path)
    atomic_jsonl(output_dir / "validation_predictions.jsonl", (
        {
            "account_id": account_id,
            "label": int(label),
            "probability": float(probability),
            "prediction": int(probability >= threshold),
        }
        for account_id, label, probability in zip(val_ids, val_y, val_p)
    ))
    atomic_json(output_dir / "results.json", {
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_score,
        "validation_loss": validation_loss,
        "validation": classification_metrics(val_y, val_p, threshold),
        "threshold_selection": threshold_details,
        "history": history,
        "train_diagnostics": train_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "protocol": "historical train/validation only",
    })
    return checkpoint_path


def standardizer_from_dict(value: dict[str, Any]) -> FeatureStandardizer:
    return FeatureStandardizer(
        tuple(value["names"]),
        tuple(value["log1p_names"]),
        np.asarray(value["mean"], dtype=np.float32),
        np.asarray(value["scale"], dtype=np.float32),
    )


def inference_view(
    graph: RawGraph, saved: dict[str, dict[str, Any]]
) -> base.JointGraphView:
    raw = induced_view(
        graph,
        np.zeros((len(graph.node_ids), 1), dtype=np.float32),
        np.arange(len(graph.node_ids), dtype=np.int64),
    )
    standardizers = {
        name: standardizer_from_dict(value) for name, value in saved.items()
    }
    comment_column = graph.edge_feature_names.index("comment_count")
    raw_topology = base.compute_topology_features(raw, comment_column)
    transformed = GraphView(
        global_nodes=raw.global_nodes,
        labels=raw.labels,
        content=raw.content,
        repetition_features=standardizers["repetition"].transform(
            raw.repetition_features
        ),
        edge_index=raw.edge_index,
        edge_features=standardizers["edge"].transform(raw.edge_features),
        thread_features=standardizers["thread"].transform(raw.thread_features),
        incidence_index=raw.incidence_index,
        incidence_features=standardizers["incidence"].transform(
            raw.incidence_features
        ),
    )
    author_column = graph.incidence_feature_names.index("role_author")
    incidence_scaler = standardizers["incidence"]
    author_cutoff = float(
        (0.5 - incidence_scaler.mean[author_column])
        / incidence_scaler.scale[author_column]
    )
    return base.JointGraphView(
        transformed,
        standardizers["topology"].transform(raw_topology),
        role_author_cutoff=author_cutoff,
    )


def detect(
    embedding_dir: Path,
    graph_path: Path,
    checkpoint_path: Path,
    training_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    prediction_path = output_dir / "predictions.jsonl"
    if prediction_path.is_file() and (output_dir / "results.json").is_file():
        return prediction_path
    if output_dir.exists():
        raise RuntimeError(f"Incomplete detection directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    accounts = load_accounts(embedding_dir)
    allowed = {str(row["account_id"]) for row in accounts}
    train_manifest = json.loads(
        (training_dir / "url_cache" / "manifest.json").read_text(encoding="utf-8")
    )
    prepare_url_cache(
        embedding_dir,
        output_dir / "url_cache",
        allowed,
        standardizer=URLStandardizer.from_dict(train_manifest["standardizer"]),
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    model_config = RiskResidualMILConfig(**checkpoint["model_config"])
    train_config = RiskResidualMILTrainConfig(**checkpoint["train_config"])
    train_config.device = args.device
    train_config.num_workers = args.num_workers
    graph = RawGraph.load(graph_path)
    if set(graph.node_ids.tolist()) != allowed:
        raise ValueError("Detection graph and embedding account sets differ")
    view = inference_view(graph, checkpoint["feature_standardizers"])
    url_data = base.load_url_cache(output_dir / "url_cache")
    loader = base._build_loader(
        view,
        np.arange(len(graph.node_ids), dtype=np.int64),
        embedding_dir,
        output_dir / "url_cache",
        url_data,
        graph.node_ids,
        model_config,
        train_config,
        training=False,
    )
    model = JointRiskResidualMILModel(
        RiskResidualMILModel(model_config),
        JointR3HConfig(**checkpoint["joint_config"]),
    ).to(torch.device(args.device))
    model.load_state_dict(checkpoint["model_state"])
    loss, truth, probabilities, account_ids, diagnostics = _run_loader(
        model, loader, torch.device(args.device), train_config, None
    )
    threshold = float(checkpoint["threshold"])
    rows = [
        {
            "account_id": account_id,
            "label": int(label),
            "probability": float(probability),
            "prediction": int(probability >= threshold),
        }
        for account_id, label, probability in zip(account_ids, truth, probabilities)
    ]
    rows.sort(key=lambda row: (-row["probability"], row["account_id"]))
    atomic_jsonl(prediction_path, rows)
    atomic_json(output_dir / "results.json", {
        "loss": loss,
        "threshold": threshold,
        "metrics": classification_metrics(truth, probabilities, threshold),
        "diagnostics": diagnostics,
        "accounts": len(rows),
        "predicted_risky": sum(row["prediction"] for row in rows),
        "time_window": {
            "start_inclusive": TRAIN_START,
            "end_exclusive": DETECTION_END,
        },
    })
    return prediction_path


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_out_strength(path: Path) -> dict[str, float]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[str(row["account_id"])] = float(row["out_strength"])
    return result


def future_exposures(
    source_db: Path, labels: dict[str, int]
) -> tuple[dict[str, Counter[str]], dict[str, Any]]:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    connection = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    try:
        query = """
            SELECT c.author_id,c.parent_id,p.author_id,pc.author_id
            FROM comments c
            LEFT JOIN posts p ON p.id=c.post_id
            LEFT JOIN comments pc ON pc.id=c.parent_id
            WHERE julianday(c.created_at) >= julianday(?)
              AND julianday(c.created_at) < julianday(?)
        """
        scanned = 0
        for source, parent_id, post_author, parent_author in connection.execute(
            query, (DETECTION_END, EVALUATION_END)
        ):
            scanned += 1
            if source is None:
                continue
            source = str(source)
            target = (
                str(parent_author) if parent_id and parent_author
                else str(post_author) if post_author
                else None
            )
            if labels.get(source) == 1 and target is not None and labels.get(target) == 0:
                by_source[source][target] += 1
    finally:
        connection.close()
    baseline_targets = set().union(*(set(values) for values in by_source.values())) if by_source else set()
    return dict(by_source), {
        "comments_scanned": scanned,
        "risky_sources": len(by_source),
        "exposed_normal_accounts": len(baseline_targets),
        "risky_to_normal_interactions": sum(
            sum(values.values()) for values in by_source.values()
        ),
    }


def exposure_after(
    by_source: dict[str, Counter[str]], removed: set[str]
) -> tuple[int, int]:
    targets: set[str] = set()
    interactions = 0
    for source, counts in by_source.items():
        if source in removed:
            continue
        targets.update(counts)
        interactions += sum(counts.values())
    return len(targets), interactions


def budget_size(total: int, ratio: float) -> int:
    return 0 if ratio == 0 else min(total, max(1, math.ceil(total * ratio)))


def evaluate_ranking(
    name: str,
    ranking: list[str],
    labels: dict[str, int],
    by_source: dict[str, Counter[str]],
    baseline: dict[str, Any],
    budget_total: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    budget_total = len(ranking) if budget_total is None else budget_total
    for ratio in INTERVENTION_RATIOS:
        size = budget_size(budget_total, ratio)
        if size > len(ranking):
            raise ValueError("Intervention budget exceeds ranking size")
        removed = set(ranking[:size])
        exposed, interactions = exposure_after(by_source, removed)
        rows.append({
            "method": name,
            "intervention_ratio": ratio,
            "removed_accounts": size,
            "true_risky_removed": sum(labels.get(account_id) == 1 for account_id in removed),
            "normal_removed": sum(labels.get(account_id) == 0 for account_id in removed),
            "exposed_normal_accounts": exposed,
            "exposure_reduction": (
                (baseline["exposed_normal_accounts"] - exposed)
                / max(1, baseline["exposed_normal_accounts"])
            ),
            "remaining_risky_to_normal_interactions": interactions,
            "interaction_reduction": (
                (baseline["risky_to_normal_interactions"] - interactions)
                / max(1, baseline["risky_to_normal_interactions"])
            ),
        })
    return rows


def random_baseline(
    universe: list[str],
    budget_total: int,
    labels: dict[str, int],
    by_source: dict[str, Counter[str]],
    baseline: dict[str, Any],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    values = np.asarray(universe, dtype=object)
    for ratio in INTERVENTION_RATIOS:
        size = budget_size(budget_total, ratio)
        trials = []
        for _ in range(repeats):
            removed = set(rng.choice(values, size=size, replace=False).tolist()) if size else set()
            exposed, interactions = exposure_after(by_source, removed)
            trials.append((
                sum(labels.get(account_id) == 1 for account_id in removed),
                sum(labels.get(account_id) == 0 for account_id in removed),
                exposed,
                interactions,
            ))
        matrix = np.asarray(trials, dtype=np.float64)
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0, ddof=1) if repeats > 1 else np.zeros(4)
        rows.append({
            "method": "Random",
            "intervention_ratio": ratio,
            "removed_accounts": size,
            "true_risky_removed": float(means[0]),
            "true_risky_removed_std": float(stds[0]),
            "normal_removed": float(means[1]),
            "normal_removed_std": float(stds[1]),
            "exposed_normal_accounts": float(means[2]),
            "exposed_normal_accounts_std": float(stds[2]),
            "exposure_reduction": (
                (baseline["exposed_normal_accounts"] - means[2])
                / max(1, baseline["exposed_normal_accounts"])
            ),
            "exposure_reduction_std": float(
                stds[2] / max(1, baseline["exposed_normal_accounts"])
            ),
            "remaining_risky_to_normal_interactions": float(means[3]),
            "remaining_risky_to_normal_interactions_std": float(stds[3]),
            "interaction_reduction": (
                (baseline["risky_to_normal_interactions"] - means[3])
                / max(1, baseline["risky_to_normal_interactions"])
            ),
            "interaction_reduction_std": float(
                stds[3] / max(1, baseline["risky_to_normal_interactions"])
            ),
        })
    return rows


def run_intervention(
    source_db: Path,
    labels: dict[str, int],
    prediction_path: Path,
    detection_graph_dir: Path,
    output_dir: Path,
    repeats: int,
    seed: int,
) -> Path:
    summary_path = output_dir / "intervention_summary.json"
    if summary_path.is_file():
        return summary_path
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(prediction_path)
    candidates = [
        str(row["account_id"]) for row in predictions if int(row["prediction"]) == 1
    ]
    all_accounts = [str(row["account_id"]) for row in predictions]
    if not candidates:
        raise ValueError("The validation-selected threshold produced no risky candidates")
    scores = {str(row["account_id"]): float(row["probability"]) for row in predictions}
    out_strength = load_out_strength(detection_graph_dir / "nodes.csv")
    agentsec_ranking = sorted(candidates, key=lambda account_id: (-scores[account_id], account_id))
    agentsec_activity_ranking = sorted(
        candidates,
        key=lambda account_id: (-out_strength.get(account_id, 0.0), account_id),
    )
    degree_ranking = sorted(
        all_accounts,
        key=lambda account_id: (-out_strength.get(account_id, 0.0), account_id),
    )
    by_source, baseline = future_exposures(source_db, labels)
    rows = []
    rows.extend(evaluate_ranking(
        "AgentSec", agentsec_ranking, labels, by_source, baseline
    ))
    rows.extend(evaluate_ranking(
        "AgentSec + historical out-degree", agentsec_activity_ranking,
        labels, by_source, baseline
    ))
    rows.extend(evaluate_ranking(
        "Historical out-degree", degree_ranking, labels, by_source, baseline,
        budget_total=len(candidates),
    ))
    rows.extend(random_baseline(
        all_accounts, len(candidates), labels, by_source, baseline, repeats, seed
    ))
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "intervention_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_jsonl(output_dir / "risk_ranking.jsonl", (
        {
            "rank": rank,
            "account_id": account_id,
            "probability": scores[account_id],
            "label": labels.get(account_id),
            "historical_out_strength": out_strength.get(account_id, 0.0),
        }
        for rank, account_id in enumerate(agentsec_ranking, start=1)
    ))
    atomic_json(summary_path, {
        "protocol": {
            "training_window": [TRAIN_START, TRAIN_END],
            "detection_window": [TRAIN_START, DETECTION_END],
            "evaluation_window": [DETECTION_END, EVALUATION_END],
            "intervention_candidate_pool": "validation-threshold predicted risky accounts",
            "budget_definition": "fraction of the predicted-risk candidate count",
            "primary_rankings": [
                "descending AgentSec probability",
                "historical out-degree within AgentSec candidates",
            ],
            "comparison_universe": "all accounts detectable by 2026-05-24",
            "simulation_boundary": (
                "Direct outgoing comments are removed retrospectively; downstream causal "
                "behavioral changes are not modeled."
            ),
        },
        "candidate_accounts": len(candidates),
        "candidate_true_risky": sum(labels.get(account_id) == 1 for account_id in candidates),
        "candidate_normal": sum(labels.get(account_id) == 0 for account_id in candidates),
        "baseline": baseline,
        "random_repeats": repeats,
        "results": rows,
    })
    return summary_path


def prepare_all(args: argparse.Namespace, labels: dict[str, int]) -> dict[str, Path]:
    views = prepare_temporal_views(
        args.embedding_dir.resolve(), args.output_root.resolve(), labels
    )
    graphs = {
        "training": build_temporal_graph(
            args.source_db.resolve(), views["training"],
            args.output_root / "data" / "training_graph", TRAIN_END, args.seed,
        ),
        "detection": build_temporal_graph(
            args.source_db.resolve(), views["detection"],
            args.output_root / "data" / "detection_graph", DETECTION_END, args.seed,
        ),
    }
    atomic_json(args.output_root / "experiment_plan.json", {
        "seed": args.seed,
        "source_db": str(args.source_db.resolve()),
        "labels_db": str(args.labels_db.resolve()),
        "embedding_dir": str(args.embedding_dir.resolve()),
        "training_window": [TRAIN_START, TRAIN_END],
        "detection_window": [TRAIN_START, DETECTION_END],
        "evaluation_window": [DETECTION_END, EVALUATION_END],
        "intervention_ratios": list(INTERVENTION_RATIOS),
        "model": {
            "post_top_k": 3,
            "comment_top_k": 5,
            "neighbors": 4,
            "text_loss_weight": 1.0,
        },
    })
    return {**views, **{f"{name}_graph": path for name, path in graphs.items()}}


def main() -> None:
    args = parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min(65536, hard):
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    if args.random_repeats < 1:
        raise ValueError("random-repeats must be positive")
    for path in (args.source_db, args.labels_db, args.embedding_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels_db)
    artifacts = prepare_all(args, labels)
    if args.stage == "prepare":
        print(json.dumps({"status": "prepared", **{k: str(v) for k, v in artifacts.items()}}, indent=2))
        return
    training_dir = args.output_root / "training"
    checkpoint = train_model(
        artifacts["training"], artifacts["training_graph"], training_dir, args
    )
    if args.stage == "train":
        print(json.dumps({"status": "trained", "checkpoint": str(checkpoint)}, indent=2))
        return
    detection_dir = args.output_root / "detection"
    predictions = detect(
        artifacts["detection"], artifacts["detection_graph"], checkpoint,
        training_dir, detection_dir, args,
    )
    if args.stage == "detect":
        print(json.dumps({"status": "detected", "predictions": str(predictions)}, indent=2))
        return
    summary = run_intervention(
        args.source_db.resolve(), labels, predictions,
        args.output_root / "data" / "detection_graph",
        args.output_root / "intervention_policy_comparison",
        args.random_repeats, args.seed,
    )
    print(json.dumps({"status": "complete", "summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
