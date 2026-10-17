"""Cross-validation and result utilities for SARA experiments."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from comment_network.stage2 import RawGraph
from multisource.data import split_hash
from multisource.url_features import load_url_cache, prepare_url_cache


SEED = 42
FOLD_COUNT = 5

METRICS = (
    "accuracy", "precision", "recall", "f1", "mcc",
    "balanced_accuracy", "specificity", "roc_auc", "pr_auc",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _account_group(account: dict[str, Any]) -> str:
    owner = str(account.get("twitter_owner_group") or "").strip().casefold()
    return owner or f"moltbook:{account['account_id']}"


def _validate_partition(splits: dict[str, list[str]], accounts: list[dict[str, Any]]) -> None:
    expected = {str(row["account_id"]) for row in accounts}
    membership: dict[str, str] = {}
    for split_name, account_ids in splits.items():
        if not account_ids:
            raise ValueError(f"Fold has an empty {split_name} split")
        for account_id in account_ids:
            if account_id in membership:
                raise ValueError(f"Account {account_id} occurs in multiple splits")
            membership[account_id] = split_name
    if set(membership) != expected:
        raise ValueError("Fold does not cover the labeled account set exactly once")
    group_splits: dict[str, set[str]] = {}
    for row in accounts:
        group_splits.setdefault(_account_group(row), set()).add(
            membership[str(row["account_id"])]
        )
    if any(len(names) != 1 for names in group_splits.values()):
        raise ValueError("twitter_owner_group leakage was detected inside a fold")


def _split_stats(ids: list[str], by_id: dict[str, dict[str, Any]]) -> dict[str, int | float]:
    positives = sum(int(by_id[account_id]["label"]) for account_id in ids)
    return {
        "accounts": len(ids),
        "positive": positives,
        "negative": len(ids) - positives,
        "positive_rate": positives / max(1, len(ids)),
    }


def make_five_folds(accounts: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    """Return five group-isolated 70/10/20 train/validation/test partitions."""
    account_ids = np.asarray([str(row["account_id"]) for row in accounts], dtype=object)
    labels = np.asarray([int(row["label"]) for row in accounts], dtype=np.int64)
    groups = np.asarray([_account_group(row) for row in accounts], dtype=object)
    if len(set(account_ids.tolist())) != len(account_ids):
        raise ValueError("Duplicate account IDs are not allowed")
    if set(labels.tolist()) - {0, 1}:
        raise ValueError("Five-fold evaluation requires binary labels")
    if len(set(groups.tolist())) < FOLD_COUNT:
        raise ValueError("At least five independent owner groups are required")

    placeholder = np.zeros(len(accounts), dtype=np.int8)
    outer = StratifiedGroupKFold(
        n_splits=FOLD_COUNT, shuffle=True, random_state=SEED
    )
    result: list[dict[str, list[str]]] = []
    test_coverage: list[int] = []
    global_positive_rate = float(labels.mean())

    for fold_index, (development_indices, test_indices) in enumerate(
        outer.split(placeholder, labels, groups), start=1
    ):
        # One of eight inner folds is about 10% of the complete dataset because
        # the outer development set contains about 80%.
        inner_labels = labels[development_indices]
        inner_groups = groups[development_indices]
        inner_placeholder = np.zeros(len(development_indices), dtype=np.int8)
        inner = StratifiedGroupKFold(
            n_splits=8, shuffle=True, random_state=SEED + fold_index
        )
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for train_local, validation_local in inner.split(
            inner_placeholder, inner_labels, inner_groups
        ):
            validation_global = development_indices[validation_local]
            validation_rate = float(labels[validation_global].mean())
            size_error = abs(len(validation_global) / len(accounts) - 0.10)
            rate_error = abs(validation_rate - global_positive_rate)
            candidates.append((size_error + rate_error, train_local, validation_local))
        if not candidates:
            raise RuntimeError(f"Could not construct validation split for fold {fold_index}")
        _, train_local, validation_local = min(candidates, key=lambda item: item[0])
        train_indices = development_indices[train_local]
        validation_indices = development_indices[validation_local]
        splits = {
            "train": sorted(account_ids[train_indices].tolist()),
            "validation": sorted(account_ids[validation_indices].tolist()),
            "test": sorted(account_ids[test_indices].tolist()),
        }
        _validate_partition(splits, accounts)
        if any(len({int(row["label"]) for row in accounts
                    if str(row["account_id"]) in set(splits[name])}) != 2
               for name in ("train", "validation", "test")):
            raise ValueError(f"Fold {fold_index} has a split containing only one class")
        result.append(splits)
        test_coverage.extend(int(index) for index in test_indices)

    if sorted(test_coverage) != list(range(len(accounts))):
        raise RuntimeError("Outer folds do not test every account exactly once")
    return result


def load_reference_folds(reference_root: Path,
                         accounts: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    """Load the exact five folds used by the original ablation experiment."""
    folds: list[dict[str, list[str]]] = []
    test_ids: list[str] = []
    for fold_number in range(1, FOLD_COUNT + 1):
        path = reference_root / f"fold_{fold_number:02d}" / "splits.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference split: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        splits = {
            name: [str(account_id) for account_id in raw[name]]
            for name in ("train", "validation", "test")
        }
        _validate_partition(splits, accounts)
        folds.append(splits)
        test_ids.extend(splits["test"])

    expected = sorted(str(row["account_id"]) for row in accounts)
    if sorted(test_ids) != expected:
        raise ValueError(
            "Reference folds must test every labeled account exactly once"
        )
    return folds


def _fold_files(output_root: Path, folds: list[dict[str, list[str]]],
                accounts: list[dict[str, Any]]) -> list[Path]:
    by_id = {str(row["account_id"]): row for row in accounts}
    split_paths: list[Path] = []
    for fold_number, splits in enumerate(folds, start=1):
        fold_dir = output_root / "folds" / f"fold_{fold_number:02d}"
        split_path = fold_dir / "splits.json"
        stats = {name: _split_stats(ids, by_id) for name, ids in splits.items()}
        if split_path.exists():
            saved = json.loads(split_path.read_text(encoding="utf-8"))
            if saved != splits:
                raise ValueError(f"Saved fold differs from seed-{SEED} fold: {split_path}")
        else:
            _atomic_json(split_path, splits)
        _atomic_json(fold_dir / "split_statistics.json", stats)
        split_paths.append(split_path)
    return split_paths


def _url_cache_for_fold(embedding_dir: Path, fold_dir: Path,
                        splits: dict[str, list[str]]) -> Path:
    cache = fold_dir / "url_cache"
    if cache.exists():
        manifest_path, rows_path = cache / "manifest.json", cache / "urls.jsonl"
        if manifest_path.is_file() and rows_path.is_file():
            _, manifest = load_url_cache(cache)
            if manifest.get("split_sha256") == split_hash(splits):
                return cache
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = fold_dir / f"url_cache_incomplete_{stamp}"
        cache.rename(archived)
    print(json.dumps({"status": "preparing_fold_url_cache", "path": str(cache)}), flush=True)
    prepare_url_cache(embedding_dir, cache, splits)
    return cache


def _reference_url_cache(reference_splits_root: Path, fold_number: int,
                         splits: dict[str, list[str]]) -> Path | None:
    cache = reference_splits_root / f"fold_{fold_number:02d}" / "url_cache"
    if not ((cache / "manifest.json").is_file()
            and (cache / "urls.jsonl").is_file()):
        return None
    _, manifest = load_url_cache(cache)
    return cache if manifest.get("split_sha256") == split_hash(splits) else None


def _archive_incomplete_run(run_dir: Path, archive_root: Path) -> None:
    if not run_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{run_dir.parent.name}_{run_dir.name}_{stamp}"
    suffix = 1
    while target.exists():
        target = archive_root / f"{run_dir.parent.name}_{run_dir.name}_{stamp}_{suffix}"
        suffix += 1
    run_dir.rename(target)
    print(json.dumps({"status": "archived_incomplete_run", "path": str(target)}), flush=True)


def _pooled_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probability = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "specificity": float(tn / max(1, tn + fp)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "confusion_matrix": matrix.tolist(),
        "accounts": len(rows),
        "positive": int(truth.sum()),
        "negative": int(len(truth) - truth.sum()),
    }


def _validate_graph_accounts(graph_path: Path, accounts: list[dict[str, Any]]) -> None:
    graph = RawGraph.load(graph_path)
    account_labels = {str(row["account_id"]): int(row["label"]) for row in accounts}
    if set(graph.node_ids.tolist()) != set(account_labels):
        raise ValueError("graph_arrays node IDs do not exactly match embedding account IDs")
    for account_id, label in zip(graph.node_ids.tolist(), graph.labels.tolist()):
        if account_labels[str(account_id)] != int(label):
            raise ValueError(f"Label mismatch for graph account {account_id}")


