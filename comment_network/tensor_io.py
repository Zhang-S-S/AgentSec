from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


NODE_EXCLUDED = {"account_id", "node_index", "label", "is_agent", "community_id"}
EDGE_FEATURES = [
    "comment_count", "top_level_count", "reply_count", "post_fallback_count",
    "distinct_posts", "normalized_duplicate_ratio", "max_duplicate_cluster_share",
    "duration_days", "is_self",
]
THREAD_FEATURES = [
    "comment_count", "reply_count", "self_comment_count", "unique_commenters",
    "participant_count", "max_depth", "duration_days",
]
INCIDENCE_FEATURES = [
    "role_author", "comment_count", "reply_fraction",
    "normalized_duplicate_ratio", "max_duplicate_cluster_share",
]


def _as_float(row: dict, name: str) -> float:
    return float(row.get(name, 0) or 0)


def write_graph_arrays(output_path: Path, node_rows: list[dict], edge_rows: list[dict],
                       hyperedge_rows: list[dict], incidence_rows: list[dict]) -> None:
    """Write directed-graph and thread-hypergraph tensors without a PyG dependency."""
    node_feature_names = [name for name in node_rows[0] if name not in NODE_EXCLUDED]
    thread_ids = [str(row["post_id"]) for row in hyperedge_rows]
    thread_index = {post_id: index for index, post_id in enumerate(thread_ids)}
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            node_ids=np.asarray([row["account_id"] for row in node_rows]),
            labels=np.asarray([row["label"] for row in node_rows], np.int8),
            node_features=np.asarray(
                [[_as_float(row, name) for name in node_feature_names] for row in node_rows],
                np.float32,
            ),
            node_feature_names=np.asarray(node_feature_names),
            edge_index=np.asarray(
                [[int(row["source_index"]), int(row["target_index"])] for row in edge_rows],
                np.int64,
            ).T,
            edge_features=np.asarray(
                [[_as_float(row, name) for name in EDGE_FEATURES] for row in edge_rows],
                np.float32,
            ),
            edge_feature_names=np.asarray(EDGE_FEATURES),
            thread_ids=np.asarray(thread_ids),
            thread_features=np.asarray(
                [[_as_float(row, name) for name in THREAD_FEATURES] for row in hyperedge_rows],
                np.float32,
            ),
            thread_feature_names=np.asarray(THREAD_FEATURES),
            incidence_index=np.asarray(
                [[int(row["node_index"]), thread_index[str(row["post_id"])]]
                 for row in incidence_rows],
                np.int64,
            ).T,
            incidence_features=np.asarray([
                [
                    float(row["role"] == "author"),
                    _as_float(row, "comment_count"),
                    _as_float(row, "reply_count") / max(_as_float(row, "comment_count"), 1.0),
                    _as_float(row, "normalized_duplicate_ratio"),
                    _as_float(row, "max_duplicate_cluster_share"),
                ]
                for row in incidence_rows
            ], np.float32),
            incidence_feature_names=np.asarray(INCIDENCE_FEATURES),
        )
    temporary.replace(output_path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def materialize_graph_arrays(output_dir: Path) -> Path:
    """Rebuild graph_arrays.npz from auditable CSVs in an analysis directory."""
    path = output_dir / "graph_arrays.npz"
    write_graph_arrays(
        path,
        _read_csv(output_dir / "nodes.csv"),
        _read_csv(output_dir / "edges.csv"),
        _read_csv(output_dir / "thread_hyperedges.csv"),
        _read_csv(output_dir / "incidences.csv"),
    )
    return path
