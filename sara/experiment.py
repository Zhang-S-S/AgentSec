"""Five-fold training and ablation entry point for SARA."""
from __future__ import annotations

import argparse
import csv
import json
import resource
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from comment_network.joint_pure_mil_ablation import train_joint_pure_mil_model
from comment_network.joint_risk_residual_mil import (
    RiskResidualMILTrainConfig,
    train_joint_risk_residual_mil_model,
)
from comment_network.joint_risk_residual_mil_single_init_ablation import (
    SingleInitRiskResidualMILTrainConfig,
    train_single_init_risk_residual_mil_model,
)
from multisource.data import load_accounts
from multisource.model_pure_mil_ablation import PureMILConfig
from multisource.model_risk_residual_mil import RiskResidualMILConfig
from multisource.model_risk_residual_mil_single_init_ablation import (
    SingleInitRiskResidualMILConfig,
)

from . import cross_validation as cv


VARIANTS = {
    "full": "Full SARA model",
    "without_residual_topk": "Replace Residual Top-k MIL with mean pooling",
    "pure_mil": "Replace Residual Top-k MIL with attention MIL",
    "without_graph": "Remove the directed graph contribution",
    "without_directionality": "Replace directed propagation with undirected propagation",
    "without_auxiliary": "Remove profile and URL information",
    "without_text_supervision": "Remove text-only supervision",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate SARA")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--graph-arrays", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-splits-root", type=Path)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--post-top-k", type=int, default=3)
    parser.add_argument("--comment-top-k", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-posts", type=int, default=128)
    parser.add_argument("--max-comments", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _raise_open_file_limit(target: int = 65536) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    requested = min(target, hard)
    if soft < requested:
        resource.setrlimit(resource.RLIMIT_NOFILE, (requested, hard))


def _model_config(args: argparse.Namespace, embedding_dir: Path) -> Any:
    manifest = json.loads(
        (embedding_dir / "embedding_manifest.json").read_text(encoding="utf-8")
    )
    use_auxiliary = args.variant != "without_auxiliary"
    common = dict(
        embedding_dim=int(manifest["config"]["embedding_dimension"]),
        hidden_dim=256,
        transformer_layers=2,
        transformer_heads=8,
        transformer_ffn_dim=1024,
        dropout=0.2,
        mil_pooling="normalized_lse",
        top_k=5,
        content_mode="post_comment",
        fusion="hierarchical",
        url_pooling="attention",
        use_post=True,
        use_comment=True,
        use_moltbook_profile=use_auxiliary,
        use_twitter_profile=use_auxiliary,
        use_alignment=use_auxiliary,
        use_url=use_auxiliary,
        use_mil_branch=True,
        behavior_encoder=False,
        post_top_k=args.post_top_k,
        comment_top_k=args.comment_top_k,
        residual_gate_bias=-2.0,
        instance_behavior_dim=0,
    )
    if args.variant == "pure_mil":
        return PureMILConfig(**common)
    if args.variant in {
        "without_residual_topk", "without_graph", "without_directionality"
    }:
        return SingleInitRiskResidualMILConfig(
            **common,
            use_residual_topk_mil=args.variant != "without_residual_topk",
        )
    return RiskResidualMILConfig(**common)


def _training_config(args: argparse.Namespace) -> Any:
    common = dict(
        seed=args.seed,
        target_batch_size=16,
        max_posts=args.max_posts,
        max_comments=args.max_comments,
        max_urls=64,
        context_max_posts=32,
        context_max_comments=32,
        context_max_urls=16,
        max_in_neighbors=args.neighbors,
        max_out_neighbors=args.neighbors,
        max_threads=2,
        max_thread_participants=4,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        grad_clip=5.0,
        baseline_aux_weight=0.0,
        text_loss_weight=(
            0.0 if args.variant == "without_text_supervision"
            else args.text_loss_weight
        ),
        instance_loss_weight=0.0,
        positive_top_k=5,
        use_instance_behavior=False,
        device=args.device,
        num_workers=args.num_workers,
        amp=True,
        pos_weight=True,
        threshold_strategy="max_f1",
        target_recall=0.8,
        graph_dropout=0.15,
        initial_gate_bias=-2.5,
        use_content=True,
        use_repetition=False,
        use_directed=True,
        use_hypergraph=False,
        directed_mode="both",
        use_topology=False,
        use_edge_features=True,
    )
    if args.variant in {
        "without_residual_topk", "without_graph", "without_directionality"
    }:
        return SingleInitRiskResidualMILTrainConfig(
            **common,
            use_directionality=args.variant != "without_directionality",
            disable_directed_contribution=args.variant == "without_graph",
        )
    return RiskResidualMILTrainConfig(**common)


def _trainer(variant: str) -> Callable[..., dict[str, Any]]:
    if variant == "pure_mil":
        return train_joint_pure_mil_model
    if variant in {
        "without_residual_topk", "without_graph", "without_directionality"
    }:
        return train_single_init_risk_residual_mil_model
    return train_joint_risk_residual_mil_model


def _completed(run_dir: Path) -> dict[str, Any] | None:
    required = (
        run_dir / "results.json",
        run_dir / "train_config.json",
        run_dir / "test_predictions.jsonl",
    )
    if not all(path.is_file() for path in required):
        return None
    return json.loads(required[0].read_text(encoding="utf-8"))


def _summary(output_root: Path, variant: str) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold in range(1, cv.FOLD_COUNT + 1):
        run_dir = output_root / "runs" / f"fold_{fold:02d}"
        result = _completed(run_dir)
        if result is None:
            continue
        fold_results.append({
            "fold": fold,
            "best_epoch": result["best_epoch"],
            "best_validation_pr_auc": result["best_validation_pr_auc"],
            **result["test"],
        })
        with (run_dir / "test_predictions.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["fold"] = fold
                predictions.append(row)

    metrics: dict[str, dict[str, float | None]] = {}
    for metric in cv.METRICS:
        values = np.asarray(
            [float(row[metric]) for row in fold_results], dtype=np.float64
        )
        metrics[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
        }
    predictions.sort(key=lambda row: (int(row["fold"]), str(row["account_id"])))
    if len({str(row["account_id"]) for row in predictions}) != len(predictions):
        raise RuntimeError("An account occurs in more than one test fold")
    if predictions:
        cv._atomic_jsonl(output_root / "oof_predictions.jsonl", predictions)
    summary = {
        "variant": variant,
        "description": VARIANTS[variant],
        "seed": cv.SEED,
        "completed_folds": len(fold_results),
        "fold_results": fold_results,
        "mean_std": metrics,
        "pooled_oof": (
            cv._pooled_metrics(predictions)
            if len(fold_results) == cv.FOLD_COUNT else None
        ),
    }
    cv._atomic_json(output_root / "summary.json", summary)
    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        for metric, values in metrics.items():
            writer.writerow({"metric": metric, **values})
    return summary


def main() -> None:
    args = _arguments()
    _raise_open_file_limit()
    cv.SEED = args.seed
    embedding_dir = args.embedding_dir.resolve()
    graph_arrays = args.graph_arrays.resolve()
    output_root = args.output_root.resolve()
    manifest = embedding_dir / "embedding_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not graph_arrays.is_file():
        raise FileNotFoundError(graph_arrays)

    accounts = load_accounts(embedding_dir)
    cv._validate_graph_accounts(graph_arrays, accounts)
    if args.reference_splits_root:
        reference_root = args.reference_splits_root.resolve()
        folds = cv.load_reference_folds(reference_root, accounts)
    else:
        reference_root = None
        folds = cv.make_five_folds(accounts)
    output_root.mkdir(parents=True, exist_ok=True)
    split_paths = cv._fold_files(output_root, folds, accounts)
    model_config = _model_config(args, embedding_dir)
    train_config = _training_config(args)
    plan = {
        "variant": args.variant,
        "description": VARIANTS[args.variant],
        "seed": args.seed,
        "folds": cv.FOLD_COUNT,
        "embedding_dir": str(embedding_dir),
        "graph_arrays": str(graph_arrays),
        "reference_splits_root": str(reference_root) if reference_root else None,
        "model": model_config.to_dict(),
        "training": train_config.to_dict(),
        "artifact_sha256": {
            "embedding_manifest": cv._sha256(manifest),
            "graph_arrays": cv._sha256(graph_arrays),
            "fold_splits": [cv._sha256(path) for path in split_paths],
        },
    }
    plan_path = output_root / "experiment_plan.json"
    if plan_path.is_file():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != plan:
            raise ValueError(f"Existing plan differs from this run: {plan_path}")
    else:
        cv._atomic_json(plan_path, plan)
    print(json.dumps({"status": "plan_ready", "variant": args.variant}), flush=True)
    if args.dry_run:
        return

    url_caches: list[Path] = []
    for fold_number, splits in enumerate(folds, start=1):
        cache = None
        if reference_root:
            cache = cv._reference_url_cache(reference_root, fold_number, splits)
        if cache is None:
            cache = cv._url_cache_for_fold(
                embedding_dir,
                output_root / "folds" / f"fold_{fold_number:02d}",
                splits,
            )
        url_caches.append(cache)

    trainer = _trainer(args.variant)
    _summary(output_root, args.variant)
    archive_root = output_root / "incomplete_archives"
    for fold_number in range(1, cv.FOLD_COUNT + 1):
        run_dir = output_root / "runs" / f"fold_{fold_number:02d}"
        if _completed(run_dir) is not None:
            print(json.dumps({"status": "skipped", "fold": fold_number}), flush=True)
            continue
        if run_dir.exists():
            cv._archive_incomplete_run(run_dir, archive_root)
        progress = {
            "status": "running",
            "variant": args.variant,
            "fold": fold_number,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        cv._atomic_json(output_root / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        try:
            result = trainer(
                embedding_dir=embedding_dir,
                url_cache=url_caches[fold_number - 1],
                splits_path=split_paths[fold_number - 1],
                graph_arrays=graph_arrays,
                output_dir=run_dir,
                model_config=model_config,
                train_config=train_config,
            )
        except Exception as error:
            failure = {
                **progress,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            cv._atomic_json(output_root / "progress.json", failure)
            raise
        summary = _summary(output_root, args.variant)
        finished = {
            "status": "completed_fold",
            "variant": args.variant,
            "fold": fold_number,
            "best_epoch": result["best_epoch"],
            "completed_folds": summary["completed_folds"],
        }
        cv._atomic_json(output_root / "progress.json", finished)
        print(json.dumps(finished), flush=True)

    final = _summary(output_root, args.variant)
    complete = {
        "status": "complete",
        "variant": args.variant,
        "completed_folds": final["completed_folds"],
        "summary": str((output_root / "summary.json").resolve()),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    cv._atomic_json(output_root / "progress.json", complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
