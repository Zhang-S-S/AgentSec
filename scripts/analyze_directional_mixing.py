#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon


LABEL_NAMES = {0: "normal", 1: "risky"}


def _asymmetry(risky_to_normal: float, normal_to_risky: float) -> float:
    total = risky_to_normal + normal_to_risky
    return (
        float((risky_to_normal - normal_to_risky) / total)
        if total > 0 else 0.0
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _mixing_matrix(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.float64)
    np.add.at(matrix, (source_labels, target_labels), weights)
    return matrix


def _cross_summary(matrix: np.ndarray) -> dict[str, Any]:
    risky_to_normal = float(matrix[1, 0])
    normal_to_risky = float(matrix[0, 1])
    cross_total = risky_to_normal + normal_to_risky
    return {
        "risky_to_normal": risky_to_normal,
        "normal_to_risky": normal_to_risky,
        "risky_to_normal_share": _ratio(risky_to_normal, cross_total),
        "normal_to_risky_share": _ratio(normal_to_risky, cross_total),
        "risky_to_normal_over_normal_to_risky": _ratio(
            risky_to_normal, normal_to_risky
        ),
        "directional_asymmetry": _asymmetry(
            risky_to_normal, normal_to_risky
        ),
    }


def _source_bootstrap(
    labels: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    source_labels = labels[source]
    target_labels = labels[target]
    risky_nodes = np.flatnonzero(labels == 1)
    normal_nodes = np.flatnonzero(labels == 0)
    risky_to_normal = np.bincount(
        source,
        weights=weights * ((source_labels == 1) & (target_labels == 0)),
        minlength=len(labels),
    )
    normal_to_risky = np.bincount(
        source,
        weights=weights * ((source_labels == 0) & (target_labels == 1)),
        minlength=len(labels),
    )
    rng = np.random.default_rng(seed)
    asymmetries = np.empty(iterations, dtype=np.float64)
    ratios = np.empty(iterations, dtype=np.float64)
    ratios.fill(np.nan)
    for index in range(iterations):
        sampled_risky = rng.choice(risky_nodes, len(risky_nodes), replace=True)
        sampled_normal = rng.choice(normal_nodes, len(normal_nodes), replace=True)
        rn = float(risky_to_normal[sampled_risky].sum())
        nr = float(normal_to_risky[sampled_normal].sum())
        asymmetries[index] = _asymmetry(rn, nr)
        if nr > 0:
            ratios[index] = rn / nr
    return {
        "directional_asymmetry_95ci": np.quantile(
            asymmetries, [0.025, 0.975]
        ).tolist(),
        "direction_ratio_95ci": np.nanquantile(
            ratios, [0.025, 0.975]
        ).tolist(),
    }


def _label_permutation_test(
    labels: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    edge_weights: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    source_labels = labels[source]
    target_labels = labels[target]
    observed_edge = _asymmetry(
        float(np.sum((source_labels == 1) & (target_labels == 0))),
        float(np.sum((source_labels == 0) & (target_labels == 1))),
    )
    observed_weight = _asymmetry(
        float(edge_weights[(source_labels == 1) & (target_labels == 0)].sum()),
        float(edge_weights[(source_labels == 0) & (target_labels == 1)].sum()),
    )
    rng = np.random.default_rng(seed)
    null_edge = np.empty(iterations, dtype=np.float64)
    null_weight = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        shuffled = rng.permutation(labels)
        shuffled_source = shuffled[source]
        shuffled_target = shuffled[target]
        rn = (shuffled_source == 1) & (shuffled_target == 0)
        nr = (shuffled_source == 0) & (shuffled_target == 1)
        null_edge[index] = _asymmetry(float(rn.sum()), float(nr.sum()))
        null_weight[index] = _asymmetry(
            float(edge_weights[rn].sum()), float(edge_weights[nr].sum())
        )
    return {
        "iterations": iterations,
        "edge_count_two_sided_p": float(
            (1 + np.sum(np.abs(null_edge) >= abs(observed_edge)))
            / (iterations + 1)
        ),
        "comment_weight_two_sided_p": float(
            (1 + np.sum(np.abs(null_weight) >= abs(observed_weight)))
            / (iterations + 1)
        ),
        "edge_count_null_95ci": np.quantile(
            null_edge, [0.025, 0.975]
        ).tolist(),
        "comment_weight_null_95ci": np.quantile(
            null_weight, [0.025, 0.975]
        ).tolist(),
    }


def _dyadic_test(
    labels: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    pairs: dict[tuple[int, int], list[float]] = {}
    for source_index, target_index, weight in zip(
        source.tolist(), target.tolist(), weights.tolist()
    ):
        source_label = int(labels[source_index])
        target_label = int(labels[target_index])
        if source_label == target_label:
            continue
        if source_label == 1:
            key = (source_index, target_index)
            direction = 0
        else:
            key = (target_index, source_index)
            direction = 1
        pair = pairs.setdefault(key, [0.0, 0.0])
        pair[direction] += float(weight)

    values = np.asarray(list(pairs.values()), dtype=np.float64)
    if not len(values):
        return {
            "connected_cross_label_pairs": 0,
            "risky_to_normal_only": 0,
            "normal_to_risky_only": 0,
            "reciprocal": 0,
            "exclusive_direction_binomial_p": None,
            "paired_comment_weight_wilcoxon_p": None,
        }
    rn_present = values[:, 0] > 0
    nr_present = values[:, 1] > 0
    rn_only = int(np.sum(rn_present & ~nr_present))
    nr_only = int(np.sum(~rn_present & nr_present))
    reciprocal = int(np.sum(rn_present & nr_present))
    discordant = rn_only + nr_only
    binomial_p = (
        float(binomtest(rn_only, discordant, 0.5).pvalue)
        if discordant else None
    )
    differences = values[:, 0] - values[:, 1]
    nonzero = differences != 0
    wilcoxon_p = (
        float(wilcoxon(differences[nonzero]).pvalue)
        if np.any(nonzero) else None
    )
    return {
        "connected_cross_label_pairs": int(len(values)),
        "risky_to_normal_only": rn_only,
        "normal_to_risky_only": nr_only,
        "reciprocal": reciprocal,
        "risky_to_normal_only_share_among_exclusive": _ratio(
            rn_only, discordant
        ),
        "exclusive_direction_binomial_p": binomial_p,
        "paired_comment_weight_wilcoxon_p": wilcoxon_p,
    }


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "directional_mixing.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "mixing_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["measure", "source", "target", "value", "row_share"],
        )
        writer.writeheader()
        for measure in ("unique_edges", "comment_interactions"):
            matrix = np.asarray(result["mixing_matrices"][measure], dtype=float)
            row_totals = matrix.sum(axis=1)
            for source_label in (0, 1):
                for target_label in (0, 1):
                    writer.writerow({
                        "measure": measure,
                        "source": LABEL_NAMES[source_label],
                        "target": LABEL_NAMES[target_label],
                        "value": matrix[source_label, target_label],
                        "row_share": (
                            matrix[source_label, target_label]
                            / row_totals[source_label]
                            if row_totals[source_label] else 0.0
                        ),
                    })

    edge = result["cross_direction"]["unique_edges"]
    weighted = result["cross_direction"]["comment_interactions"]
    dyad = result["dyadic_direction_test"]
    report = f"""# Directed Cross-label Interaction Analysis

## Unique directed account pairs

- Risky to Normal: {edge['risky_to_normal']:.0f}
- Normal to Risky: {edge['normal_to_risky']:.0f}
- Ratio: {edge['risky_to_normal_over_normal_to_risky']}
- Directional asymmetry: {edge['directional_asymmetry']:.6f}
- Bootstrap 95% CI: {edge['directional_asymmetry_95ci']}

## Comment-count-weighted interactions

- Risky to Normal: {weighted['risky_to_normal']:.0f}
- Normal to Risky: {weighted['normal_to_risky']:.0f}
- Ratio: {weighted['risky_to_normal_over_normal_to_risky']}
- Directional asymmetry: {weighted['directional_asymmetry']:.6f}
- Bootstrap 95% CI: {weighted['directional_asymmetry_95ci']}

## Cross-label dyads

- Risky to Normal only: {dyad['risky_to_normal_only']}
- Normal to Risky only: {dyad['normal_to_risky_only']}
- Reciprocal: {dyad['reciprocal']}
- Exact binomial p-value: {dyad['exclusive_direction_binomial_p']}
- Paired interaction-weight Wilcoxon p-value: {dyad['paired_comment_weight_wilcoxon_p']}

## Label-permutation test

- Unique-edge asymmetry p-value: {result['label_permutation_test']['edge_count_two_sided_p']}
- Comment-weight asymmetry p-value: {result['label_permutation_test']['comment_weight_two_sided_p']}
"""
    (output_dir / "directional_mixing.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze direction asymmetry between risky and normal accounts"
    )
    parser.add_argument("--graph-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.permutations < 1 or args.bootstrap < 1:
        parser.error("permutations and bootstrap must be positive")

    with np.load(args.graph_arrays, allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        edge_index = archive["edge_index"].astype(np.int64)
        edge_features = archive["edge_features"].astype(np.float64)
        edge_feature_names = [str(value) for value in archive["edge_feature_names"]]
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Both normal and risky nodes are required")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if "comment_count" not in edge_feature_names:
        raise ValueError("comment_count is missing from edge features")

    source, target = edge_index
    keep = source != target
    source, target = source[keep], target[keep]
    comment_weights = edge_features[keep, edge_feature_names.index("comment_count")]
    if np.any(comment_weights < 0):
        raise ValueError("comment_count must be non-negative")
    source_labels, target_labels = labels[source], labels[target]
    edge_matrix = _mixing_matrix(
        source_labels, target_labels, np.ones(len(source), dtype=np.float64)
    )
    comment_matrix = _mixing_matrix(
        source_labels, target_labels, comment_weights
    )
    edge_cross = _cross_summary(edge_matrix)
    edge_cross.update(_source_bootstrap(
        labels, source, target, np.ones(len(source)), args.bootstrap, args.seed
    ))
    comment_cross = _cross_summary(comment_matrix)
    comment_cross.update(_source_bootstrap(
        labels, source, target, comment_weights, args.bootstrap, args.seed + 1
    ))

    result = {
        "graph_arrays": str(args.graph_arrays.resolve()),
        "seed": args.seed,
        "nodes": {
            "total": int(len(labels)),
            "normal": int(np.sum(labels == 0)),
            "risky": int(np.sum(labels == 1)),
        },
        "self_loops_excluded": int(np.sum(~keep)),
        "directed_edges_analyzed": int(len(source)),
        "mixing_matrices": {
            "unique_edges": edge_matrix.tolist(),
            "comment_interactions": comment_matrix.tolist(),
        },
        "cross_direction": {
            "unique_edges": edge_cross,
            "comment_interactions": comment_cross,
        },
        "dyadic_direction_test": _dyadic_test(
            labels, source, target, comment_weights
        ),
        "label_permutation_test": _label_permutation_test(
            labels,
            source,
            target,
            comment_weights,
            args.permutations,
            args.seed,
        ),
    }
    _write_outputs(args.output_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
