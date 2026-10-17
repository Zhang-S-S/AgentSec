from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.stats import hypergeom, mannwhitneyu

from .tensor_io import write_graph_arrays


URL = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\"'\u3000]+|"
    r"(?<![\w@])(?:[\w-]+\.)+[a-z]{2,63}(?::\d+)?(?:[/?#][^\s<>\"']*)?"
)
EMAIL = re.compile(r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
MENTION = re.compile(r"(?<!\w)@[\w.-]+", re.UNICODE)
UUID = re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class AnalysisConfig:
    source_db: Path
    output_dir: Path
    labels: Path | None = None
    seed: int = 42
    permutations: int = 500
    labeled_only: bool = False


@dataclass
class EdgeStats:
    comments: int = 0
    top_level: int = 0
    replies: int = 0
    post_fallback: int = 0
    posts: set[str] = field(default_factory=set)
    normalized_counts: Counter[bytes] = field(default_factory=Counter)
    first_at: str | None = None
    last_at: str | None = None

    def add(self, relation: str, post_id: str, normalized_hash: bytes | None,
            created_at: str | None) -> None:
        self.comments += 1
        if relation == "post_author":
            self.top_level += 1
        elif relation == "parent_author":
            self.replies += 1
        else:
            self.post_fallback += 1
        self.posts.add(post_id)
        if normalized_hash is not None:
            self.normalized_counts[normalized_hash] += 1
        if created_at:
            self.first_at = created_at if self.first_at is None else min(self.first_at, created_at)
            self.last_at = created_at if self.last_at is None else max(self.last_at, created_at)


@dataclass
class ContentStats:
    total: int = 0
    nonempty: int = 0
    raw_counts: Counter[bytes] = field(default_factory=Counter)
    normalized_counts: Counter[bytes] = field(default_factory=Counter)
    normalized_first_target: dict[bytes, str] = field(default_factory=dict)
    normalized_first_post: dict[bytes, str] = field(default_factory=dict)
    cross_target: set[bytes] = field(default_factory=set)
    cross_post: set[bytes] = field(default_factory=set)

    def add(self, content: str | None, target: str | None, post_id: str) -> bytes | None:
        self.total += 1
        if content is None or not content.strip():
            return None
        self.nonempty += 1
        raw_hash = _digest(content)
        normalized_hash = _digest(normalize_comment(content))
        self.raw_counts[raw_hash] += 1
        self.normalized_counts[normalized_hash] += 1
        if target is not None:
            previous_target = self.normalized_first_target.setdefault(normalized_hash, target)
            if previous_target != target:
                self.cross_target.add(normalized_hash)
        previous_post = self.normalized_first_post.setdefault(normalized_hash, post_id)
        if previous_post != post_id:
            self.cross_post.add(normalized_hash)
        return normalized_hash


@dataclass
class IncidenceStats:
    comments: int = 0
    replies: int = 0
    normalized_counts: Counter[bytes] = field(default_factory=Counter)

    def add(self, is_reply: bool, normalized_hash: bytes | None) -> None:
        self.comments += 1
        self.replies += int(is_reply)
        if normalized_hash is not None:
            self.normalized_counts[normalized_hash] += 1


@dataclass
class ThreadStats:
    author_id: str | None
    comments: int = 0
    replies: int = 0
    self_comments: int = 0
    max_depth: int = 0
    commenters: set[str] = field(default_factory=set)
    first_at: str | None = None
    last_at: str | None = None

    def add(self, source: str, depth: int, created_at: str | None, is_reply: bool) -> None:
        self.comments += 1
        self.replies += int(is_reply)
        self.self_comments += int(self.author_id is not None and source == self.author_id)
        self.max_depth = max(self.max_depth, depth)
        self.commenters.add(source)
        if created_at:
            self.first_at = created_at if self.first_at is None else min(self.first_at, created_at)
            self.last_at = created_at if self.last_at is None else max(self.last_at, created_at)


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).digest()


def normalize_comment(text: str) -> str:
    """Deterministic, URL-insensitive template normalization; no network access."""
    value = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    value = TAG.sub(" ", value)
    value = EMAIL.sub(" <email> ", value)
    value = URL.sub(" <url> ", value)
    value = MENTION.sub(" <user> ", value)
    value = UUID.sub(" <id> ", value)
    return SPACE.sub(" ", value).strip()


def _read_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def load_labels(path: Path | None) -> tuple[dict[str, int], dict]:
    if path is None:
        return {}, {"path": None, "labeled": 0, "positive": 0, "negative": 0}
    labels: dict[str, int] = {}
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with _read_db(path) as db:
            rows = db.execute(
                "SELECT agent_id,is_malicious FROM manual_labels WHERE is_malicious IN (0,1)"
            )
            labels = {str(account_id): int(label) for account_id, label in rows}
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    label = item.get("label", item.get("is_malicious"))
                    account_id = item.get("account_id", item.get("agent_id"))
                    if label in (0, 1) and account_id is not None:
                        labels[str(account_id)] = int(label)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("accounts", payload.get("labels", []))
        for item in rows:
            label = item.get("label", item.get("is_malicious"))
            account_id = item.get("account_id", item.get("agent_id"))
            if label in (0, 1) and account_id is not None:
                labels[str(account_id)] = int(label)
    metadata = {
        "path": str(path.resolve()), "labeled": len(labels),
        "positive": sum(labels.values()), "negative": len(labels) - sum(labels.values()),
        "sha256": _file_sha256(path),
    }
    return labels, metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration_days(first: str | None, last: str | None) -> float:
    if not first or not last:
        return 0.0
    try:
        start = datetime.fromisoformat(first.replace("Z", "+00:00"))
        end = datetime.fromisoformat(last.replace("Z", "+00:00"))
        return max(0.0, (end - start).total_seconds() / 86400.0)
    except ValueError:
        return 0.0


def _repeat_summary(stats: ContentStats | IncidenceStats | EdgeStats) -> dict[str, float]:
    counts = stats.normalized_counts
    total = sum(counts.values())
    duplicate = total - len(counts)
    repeated_clusters = sum(value > 1 for value in counts.values())
    result = {
        "normalized_unique": float(len(counts)),
        "normalized_duplicate_instances": float(duplicate),
        "normalized_duplicate_ratio": duplicate / total if total else 0.0,
        "max_duplicate_cluster_share": max(counts.values(), default=0) / total if total else 0.0,
        "repeated_clusters": float(repeated_clusters),
    }
    if isinstance(stats, ContentStats):
        raw_duplicate = stats.nonempty - len(stats.raw_counts)
        result.update({
            "raw_unique": float(len(stats.raw_counts)),
            "exact_duplicate_instances": float(raw_duplicate),
            "exact_duplicate_ratio": raw_duplicate / stats.nonempty if stats.nonempty else 0.0,
            "cross_target_repeat_fraction": (
                len(stats.cross_target) / repeated_clusters if repeated_clusters else 0.0
            ),
            "cross_post_repeat_fraction": (
                len(stats.cross_post) / repeated_clusters if repeated_clusters else 0.0
            ),
        })
    return result


def _bh_adjust(rows: list[dict], key: str = "p_value") -> None:
    valid = [(index, row[key]) for index, row in enumerate(rows) if np.isfinite(row.get(key, np.nan))]
    valid.sort(key=lambda item: item[1])
    adjusted = [1.0] * len(valid)
    running = 1.0
    for rank in range(len(valid), 0, -1):
        value = valid[rank - 1][1] * len(valid) / rank
        running = min(running, value)
        adjusted[rank - 1] = min(1.0, running)
    for (index, _), value in zip(valid, adjusted):
        rows[index]["q_value_bh"] = float(value)


def _feature_comparisons(node_rows: list[dict], features: list[str]) -> list[dict]:
    result = []
    for feature in features:
        positive = np.asarray([row[feature] for row in node_rows if row["label"] == 1], float)
        negative = np.asarray([row[feature] for row in node_rows if row["label"] == 0], float)
        if not len(positive) or not len(negative):
            continue
        try:
            test = mannwhitneyu(positive, negative, alternative="two-sided")
            u, p_value = float(test.statistic), float(test.pvalue)
            effect = 2 * u / (len(positive) * len(negative)) - 1
        except ValueError:
            p_value, effect = 1.0, 0.0
        result.append({
            "feature": feature, "positive_n": int(len(positive)), "negative_n": int(len(negative)),
            "positive_mean": float(positive.mean()), "negative_mean": float(negative.mean()),
            "positive_median": float(np.median(positive)), "negative_median": float(np.median(negative)),
            "positive_q25": float(np.quantile(positive, 0.25)),
            "positive_q75": float(np.quantile(positive, 0.75)),
            "negative_q25": float(np.quantile(negative, 0.25)),
            "negative_q75": float(np.quantile(negative, 0.75)),
            "rank_biserial": float(effect), "p_value": p_value,
        })
    _bh_adjust(result)
    return result


def activity_adjusted_feature_tests(node_rows: list[dict], features: list[str], seed: int,
                                    permutations: int) -> list[dict]:
    """Shuffle labels within activity deciles to separate behavior from sheer volume."""
    labeled_indices = np.asarray([
        index for index, row in enumerate(node_rows) if row["label"] in (0, 1)
    ])
    labels = np.asarray([node_rows[index]["label"] for index in labeled_indices], np.int8)
    strata = _degree_strata(node_rows, labeled_indices)
    count_like = {
        "post_count", "comments_authored", "out_degree", "in_degree", "out_strength",
        "in_strength", "pagerank", "core_number", "component_size",
        "authored_thread_mean_participants", "authored_thread_mean_comments",
    }
    rng = np.random.default_rng(seed + 2)
    output = []
    for feature in features:
        values = np.asarray([node_rows[index][feature] for index in labeled_indices], float)
        transformed = np.log1p(values) if feature in count_like else values
        observed = float(transformed[labels == 1].mean() - transformed[labels == 0].mean())
        null = []
        for _ in range(permutations):
            permuted = labels.copy()
            for members in strata:
                permuted[members] = rng.permutation(permuted[members])
            null.append(float(
                transformed[permuted == 1].mean() - transformed[permuted == 0].mean()
            ))
        distribution = np.asarray(null)
        center = float(distribution.mean())
        p_value = (1 + np.sum(np.abs(distribution - center) >= abs(observed - center))) / (
            permutations + 1
        )
        output.append({
            "feature": feature, "transform": "log1p" if feature in count_like else "identity",
            "observed_positive_minus_negative": observed, "null_mean": center,
            "null_ci95": [float(x) for x in np.quantile(distribution, [0.025, 0.975])],
            "permutation_p_two_sided": float(p_value),
            "observed_minus_null": observed - center,
        })
    _bh_adjust(output, "permutation_p_two_sided")
    return output


def _degree_strata(rows: list[dict], labeled_indices: np.ndarray) -> list[np.ndarray]:
    activity = np.asarray([
        math.log1p(rows[index]["comments_authored"] + rows[index]["post_count"])
        for index in labeled_indices
    ])
    cuts = np.unique(np.quantile(activity, np.linspace(0, 1, 11)))
    bins = np.digitize(activity, cuts[1:-1], right=True)
    return [np.flatnonzero(bins == value) for value in np.unique(bins)]


def _mixing_metrics(source: np.ndarray, target: np.ndarray, weights: np.ndarray,
                    labels: np.ndarray) -> dict[str, float]:
    source_labels, target_labels = labels[source], labels[target]
    total = weights.sum()
    rr = weights[(source_labels == 1) & (target_labels == 1)].sum()
    risk_out = weights[source_labels == 1].sum()
    benign_out = weights[source_labels == 0].sum()
    return {
        "risk_risk_share": float(rr / total) if total else 0.0,
        "risk_target_given_risk_source": float(rr / risk_out) if risk_out else 0.0,
        "risk_target_given_benign_source": float(
            weights[(source_labels == 0) & (target_labels == 1)].sum() / benign_out
        ) if benign_out else 0.0,
        "same_label_share": float(weights[source_labels == target_labels].sum() / total) if total else 0.0,
    }


def _permutation_mixing(node_rows: list[dict], edge_rows: list[dict], seed: int,
                        permutations: int) -> dict:
    labeled_global = np.asarray([index for index, row in enumerate(node_rows) if row["label"] in (0, 1)])
    local = {global_index: local_index for local_index, global_index in enumerate(labeled_global)}
    labels = np.asarray([node_rows[index]["label"] for index in labeled_global], dtype=np.int8)
    strata = _degree_strata(node_rows, labeled_global)
    source, target, relation_weights = [], [], {"all": [], "post": [], "reply": []}
    for row in edge_rows:
        if row["is_self"] or row["source_index"] not in local or row["target_index"] not in local:
            continue
        source.append(local[row["source_index"]]); target.append(local[row["target_index"]])
        relation_weights["all"].append(row["comment_count"])
        relation_weights["post"].append(row["top_level_count"] + row["post_fallback_count"])
        relation_weights["reply"].append(row["reply_count"])
    source_array, target_array = np.asarray(source, int), np.asarray(target, int)
    rng = np.random.default_rng(seed)
    output = {}
    for relation, raw_weights in relation_weights.items():
        weights = np.asarray(raw_weights, float)
        keep = weights > 0
        observed = _mixing_metrics(source_array[keep], target_array[keep], weights[keep], labels)
        null = {key: [] for key in observed}
        for _ in range(permutations):
            permuted = labels.copy()
            for members in strata:
                permuted[members] = rng.permutation(permuted[members])
            values = _mixing_metrics(source_array[keep], target_array[keep], weights[keep], permuted)
            for key, value in values.items():
                null[key].append(value)
        output[relation] = {}
        for key, value in observed.items():
            distribution = np.asarray(null[key], float)
            center = float(distribution.mean()) if len(distribution) else float("nan")
            two_sided = (1 + np.sum(np.abs(distribution - center) >= abs(value - center))) / (
                len(distribution) + 1
            )
            output[relation][key] = {
                "observed": value, "null_mean": center,
                "null_ci95": [float(x) for x in np.quantile(distribution, [0.025, 0.975])],
                "enrichment": value / center if center else float("nan"),
                "permutation_p_two_sided": float(two_sided),
            }
    output["labeled_edges"] = len(source)
    output["permutations"] = permutations
    output["permutation_scheme"] = "label shuffle within log1p(post+comment activity) deciles"
    return output


def _hypergraph_mixing(node_rows: list[dict], node_index: dict[str, int],
                       threads: dict[str, ThreadStats], seed: int, permutations: int) -> dict:
    labeled_global = np.asarray([index for index, row in enumerate(node_rows) if row["label"] in (0, 1)])
    local = {global_index: local_index for local_index, global_index in enumerate(labeled_global)}
    labels = np.asarray([node_rows[index]["label"] for index in labeled_global], dtype=np.int8)
    strata = _degree_strata(node_rows, labeled_global)
    row_indices, col_indices, thread_sizes = [], [], []
    for thread in threads.values():
        participants = set(thread.commenters)
        if thread.author_id:
            participants.add(thread.author_id)
        members = sorted({local[node_index[account]] for account in participants
                          if account in node_index and node_index[account] in local})
        if len(members) < 2:
            continue
        row = len(thread_sizes)
        row_indices.extend([row] * len(members)); col_indices.extend(members)
        thread_sizes.append(len(members))
    matrix = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
        shape=(len(thread_sizes), len(labels)),
    )
    total_pairs = float(sum(size * (size - 1) / 2 for size in thread_sizes))

    def statistic(current: np.ndarray) -> float:
        risk_count = np.asarray(matrix @ current).ravel()
        return float(np.sum(risk_count * (risk_count - 1) / 2) / total_pairs) if total_pairs else 0.0

    observed = statistic(labels)
    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(permutations):
        permuted = labels.copy()
        for members in strata:
            permuted[members] = rng.permutation(permuted[members])
        null.append(statistic(permuted))
    distribution = np.asarray(null, float)
    center = float(distribution.mean()) if len(distribution) else float("nan")
    p_value = (1 + np.sum(np.abs(distribution - center) >= abs(observed - center))) / (
        len(distribution) + 1
    )
    return {
        "threads_with_at_least_two_labeled_participants": len(thread_sizes),
        "labeled_co_participation_pairs_with_multiplicity": total_pairs,
        "risk_risk_pair_share": {
            "observed": observed, "null_mean": center,
            "null_ci95": [float(x) for x in np.quantile(distribution, [0.025, 0.975])],
            "enrichment": observed / center if center else float("nan"),
            "permutation_p_two_sided": float(p_value),
        },
        "permutations": permutations,
        "permutation_scheme": "label shuffle within log1p(post+comment activity) deciles",
    }


def _write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _write_sqlite(path: Path, node_rows: list[dict], edge_rows: list[dict],
                  hyperedge_rows: list[dict], incidence_rows: list[dict], metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    db = sqlite3.connect(temporary)
    try:
        for table, rows in (
            ("nodes", node_rows), ("edges", edge_rows),
            ("thread_hyperedges", hyperedge_rows), ("incidences", incidence_rows),
        ):
            if not rows:
                continue
            fields = list(rows[0])
            columns = ",".join('"' + name.replace('"', '""') + '"' for name in fields)
            db.execute(f"CREATE TABLE {table} ({columns})")
            placeholders = ",".join("?" for _ in fields)
            db.executemany(
                f"INSERT INTO {table} VALUES ({placeholders})",
                ([row.get(name) for name in fields] for row in rows),
            )
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", (
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in metadata.items()
        ))
        db.execute("CREATE UNIQUE INDEX nodes_id ON nodes(account_id)")
        db.execute("CREATE UNIQUE INDEX edges_pair ON edges(source_id,target_id)")
        db.execute("CREATE INDEX incidence_thread ON incidences(post_id)")
        db.execute("CREATE INDEX incidence_account ON incidences(account_id)")
        db.commit()
    finally:
        db.close()
    temporary.replace(path)


def build_and_analyze(config: AnalysisConfig) -> dict:
    if config.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    labels, label_metadata = load_labels(config.labels)
    if config.labeled_only and not labels:
        raise ValueError("labeled_only requires a non-empty label file")
    edges: dict[tuple[str, str], EdgeStats] = {}
    contents: defaultdict[str, ContentStats] = defaultdict(ContentStats)
    incidences: dict[tuple[str, str], IncidenceStats] = {}
    threads: dict[str, ThreadStats] = {}
    audit = Counter()

    with _read_db(config.source_db) as db:
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"Source database failed quick_check: {integrity}")
        agents = {str(row[0]) for row in db.execute("SELECT id FROM agents")}
        post_counts = {str(aid): int(count) for aid, count in db.execute(
            "SELECT author_id,COUNT(*) FROM posts WHERE author_id IS NOT NULL GROUP BY author_id"
        )}
        query = """
            SELECT c.id,c.post_id,c.parent_id,c.author_id,c.content,c.created_at,
                   COALESCE(c.depth,0),p.author_id,pc.author_id
            FROM comments c
            LEFT JOIN posts p ON p.id=c.post_id
            LEFT JOIN comments pc ON pc.id=c.parent_id
        """
        cursor = db.execute(query)
        for index, (_comment_id, post_id, parent_id, source, content, created_at, depth,
                    post_author, parent_author) in enumerate(cursor, 1):
            audit["comments_total"] += 1
            if source is None or source == "":
                audit["missing_source"] += 1
                continue
            source, post_id = str(source), str(post_id)
            if config.labeled_only and source not in labels:
                audit["excluded_unlabeled_source"] += 1
                continue
            if parent_id and parent_author:
                target, relation = str(parent_author), "parent_author"
            elif not parent_id and post_author:
                target, relation = str(post_author), "post_author"
            elif parent_id and post_author:
                target, relation = str(post_author), "post_fallback"
            else:
                target, relation = None, "unresolved"
            normalized_hash = contents[source].add(content, target, post_id)
            thread_author = str(post_author) if post_author else None
            if config.labeled_only and thread_author not in labels:
                thread_author = None
            thread = threads.setdefault(post_id, ThreadStats(
                thread_author
            ))
            thread.add(source, int(depth), created_at, bool(parent_id))
            incidence = incidences.setdefault((post_id, source), IncidenceStats())
            incidence.add(bool(parent_id), normalized_hash)
            if target is None:
                audit["unresolved_target"] += 1
            elif config.labeled_only and target not in labels:
                audit["excluded_unlabeled_target"] += 1
            else:
                audit[f"resolved_{relation}"] += 1
                edge = edges.setdefault((source, target), EdgeStats())
                edge.add(relation, post_id, normalized_hash, created_at)
                audit["self_comments"] += int(source == target)
            if index % 100000 == 0:
                print(json.dumps({"stage": "scan", "comments": index,
                                  "edges": len(edges), "threads": len(threads)}), flush=True)

    if config.labeled_only:
        all_nodes = sorted(labels)
    else:
        all_nodes = sorted(agents | set(labels) | set(contents)
                           | {node for pair in edges for node in pair})
    node_index = {account_id: index for index, account_id in enumerate(all_nodes)}
    graph = nx.DiGraph(); graph.add_nodes_from(all_nodes)
    edge_rows = []
    for (source, target), stats in sorted(edges.items()):
        repeat = _repeat_summary(stats)
        row = {
            "source_id": source, "target_id": target,
            "source_index": node_index[source], "target_index": node_index[target],
            "comment_count": stats.comments, "top_level_count": stats.top_level,
            "reply_count": stats.replies, "post_fallback_count": stats.post_fallback,
            "distinct_posts": len(stats.posts),
            "normalized_duplicate_ratio": repeat["normalized_duplicate_ratio"],
            "max_duplicate_cluster_share": repeat["max_duplicate_cluster_share"],
            "first_at": stats.first_at, "last_at": stats.last_at,
            "duration_days": _duration_days(stats.first_at, stats.last_at),
            "is_self": int(source == target),
        }
        edge_rows.append(row)
        if source != target:
            graph.add_edge(source, target, weight=stats.comments,
                           top_level=stats.top_level, reply=stats.replies)

    undirected = nx.Graph(); undirected.add_nodes_from(all_nodes)
    for source, target, data in graph.edges(data=True):
        previous = undirected.get_edge_data(source, target, {}).get("weight", 0)
        undirected.add_edge(source, target, weight=previous + data["weight"])
    active_nodes = [node for node in all_nodes if undirected.degree(node) > 0]
    active_undirected = undirected.subgraph(active_nodes).copy()
    pagerank = nx.pagerank(graph, weight="weight", max_iter=500, tol=1e-10)
    core_number = nx.core_number(undirected)
    clustering = nx.clustering(undirected)
    component_size = {}
    for component in nx.connected_components(undirected):
        for node in component:
            component_size[node] = len(component)
    # Louvain internally iterates over sets. String hashes are randomized by
    # each Python process, so a fixed algorithm seed alone is insufficient for
    # cross-process reproducibility. Map the sorted account IDs to integers
    # before community detection, then map the result back.
    deterministic_nodes = sorted(active_nodes)
    deterministic_index = {node: index for index, node in enumerate(deterministic_nodes)}
    integer_graph = nx.relabel_nodes(active_undirected, deterministic_index, copy=True)
    integer_communities = nx.community.louvain_communities(
        integer_graph, weight="weight", seed=config.seed
    ) if active_nodes else []
    communities = [
        {deterministic_nodes[index] for index in group}
        for group in integer_communities
    ]
    community_id = {node: index for index, group in enumerate(communities) for node in group}
    modularity = nx.community.modularity(active_undirected, communities, weight="weight") if communities else 0.0

    outgoing = defaultdict(list); incoming = defaultdict(list)
    self_comment_counts = defaultdict(int)
    for row in edge_rows:
        if row["is_self"]:
            self_comment_counts[row["source_id"]] += row["comment_count"]
            continue
        outgoing[row["source_id"]].append((row["target_id"], row["comment_count"]))
        incoming[row["target_id"]].append((row["source_id"], row["comment_count"]))
    authored_thread_values = defaultdict(list)
    for thread in threads.values():
        if thread.author_id:
            participants = set(thread.commenters) | {thread.author_id}
            authored_thread_values[thread.author_id].append((len(participants), thread.comments))

    node_rows = []
    for account_id in all_nodes:
        out_values, in_values = outgoing[account_id], incoming[account_id]
        out_neighbors = {target for target, _ in out_values}
        in_neighbors = {source for source, _ in in_values}
        mutual = out_neighbors & in_neighbors
        out_strength = sum(weight for _, weight in out_values)
        in_strength = sum(weight for _, weight in in_values)
        probabilities = np.asarray([weight for _, weight in out_values], float)
        probabilities = probabilities / probabilities.sum() if probabilities.size else probabilities
        hhi = float(np.sum(probabilities ** 2)) if probabilities.size else 0.0
        entropy = float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0
        entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        content = contents[account_id]
        repeat = _repeat_summary(content)
        authored = authored_thread_values[account_id]
        node_rows.append({
            "account_id": account_id, "node_index": node_index[account_id],
            "label": labels.get(account_id, -1), "is_agent": int(account_id in agents),
            "post_count": post_counts.get(account_id, 0),
            "comments_authored": content.total,
            "out_degree": len(out_neighbors), "in_degree": len(in_neighbors),
            "out_strength": out_strength, "in_strength": in_strength,
            "self_comment_count": self_comment_counts[account_id],
            "mutual_neighbors": len(mutual),
            "mutual_neighbor_ratio": len(mutual) / len(out_neighbors | in_neighbors)
            if out_neighbors or in_neighbors else 0.0,
            "target_hhi": hhi, "target_entropy": entropy,
            "pagerank": pagerank.get(account_id, 0.0),
            "core_number": core_number.get(account_id, 0),
            "clustering": clustering.get(account_id, 0.0),
            "component_size": component_size.get(account_id, 1),
            "community_id": community_id.get(account_id, -1),
            "exact_duplicate_ratio": repeat["exact_duplicate_ratio"],
            "normalized_duplicate_ratio": repeat["normalized_duplicate_ratio"],
            "max_duplicate_cluster_share": repeat["max_duplicate_cluster_share"],
            "cross_target_repeat_fraction": repeat["cross_target_repeat_fraction"],
            "cross_post_repeat_fraction": repeat["cross_post_repeat_fraction"],
            "authored_threads": len(authored),
            "authored_thread_mean_participants": float(np.mean([x[0] for x in authored])) if authored else 0.0,
            "authored_thread_mean_comments": float(np.mean([x[1] for x in authored])) if authored else 0.0,
        })

    hyperedge_rows, incidence_rows = [], []
    for post_id, thread in sorted(threads.items()):
        participants = set(thread.commenters)
        if thread.author_id:
            participants.add(thread.author_id)
        hyperedge_rows.append({
            "post_id": post_id, "author_id": thread.author_id,
            "comment_count": thread.comments, "reply_count": thread.replies,
            "self_comment_count": thread.self_comments,
            "unique_commenters": len(thread.commenters), "participant_count": len(participants),
            "max_depth": thread.max_depth, "first_at": thread.first_at, "last_at": thread.last_at,
            "duration_days": _duration_days(thread.first_at, thread.last_at),
        })
    for (post_id, account_id), stats in sorted(incidences.items()):
        repeat = _repeat_summary(stats)
        incidence_rows.append({
            "post_id": post_id, "account_id": account_id,
            "node_index": node_index[account_id], "role": "commenter",
            "comment_count": stats.comments, "reply_count": stats.replies,
            "normalized_duplicate_ratio": repeat["normalized_duplicate_ratio"],
            "max_duplicate_cluster_share": repeat["max_duplicate_cluster_share"],
        })
    for post_id, thread in sorted(threads.items()):
        if thread.author_id:
            incidence_rows.append({
                "post_id": post_id, "account_id": thread.author_id,
                "node_index": node_index[thread.author_id], "role": "author",
                "comment_count": 0, "reply_count": 0,
                "normalized_duplicate_ratio": 0.0, "max_duplicate_cluster_share": 0.0,
            })

    comparison_features = [
        "post_count", "comments_authored", "out_degree", "in_degree", "out_strength",
        "in_strength", "mutual_neighbor_ratio", "target_hhi", "target_entropy", "pagerank",
        "core_number", "clustering", "component_size", "exact_duplicate_ratio",
        "normalized_duplicate_ratio", "max_duplicate_cluster_share",
        "cross_target_repeat_fraction", "cross_post_repeat_fraction",
        "authored_thread_mean_participants", "authored_thread_mean_comments",
    ]
    comparisons = _feature_comparisons(node_rows, comparison_features) if labels else []
    adjusted_comparisons = activity_adjusted_feature_tests(
        node_rows, comparison_features, config.seed, config.permutations
    ) if labels else []
    mixing = _permutation_mixing(node_rows, edge_rows, config.seed, config.permutations) if labels else {}
    hypergraph = _hypergraph_mixing(
        node_rows, node_index, threads, config.seed, config.permutations
    ) if labels else {}

    community_rows = []
    if labels:
        # Community enrichment is defined on the active graph. Isolated labeled
        # accounts cannot belong to a detected community and must not enter its
        # reference rate.
        active_labels = {node: labels[node] for node in active_nodes if node in labels}
        total_positive, total_labeled = sum(active_labels.values()), len(active_labels)
        for identifier, group in enumerate(communities):
            group_labels = [active_labels[node] for node in group if node in active_labels]
            if not group_labels:
                continue
            positives = sum(group_labels); labeled_count = len(group_labels)
            p_value = float(hypergeom.sf(positives - 1, total_labeled, total_positive, labeled_count))
            community_rows.append({
                "community_id": identifier, "nodes": len(group), "labeled": labeled_count,
                "positive": positives, "risk_rate": positives / labeled_count,
                "global_risk_rate": total_positive / total_labeled,
                "fold_enrichment": (positives / labeled_count) / (total_positive / total_labeled),
                "p_value": p_value,
            })
        _bh_adjust(community_rows)

    node_fields = list(node_rows[0]); edge_fields = list(edge_rows[0])
    hyperedge_fields = list(hyperedge_rows[0]); incidence_fields = list(incidence_rows[0])
    _write_csv(config.output_dir / "nodes.csv", node_rows, node_fields)
    _write_csv(config.output_dir / "edges.csv", edge_rows, edge_fields)
    _write_csv(config.output_dir / "thread_hyperedges.csv", hyperedge_rows, hyperedge_fields)
    _write_csv(config.output_dir / "incidences.csv", incidence_rows, incidence_fields)
    _write_csv(config.output_dir / "feature_comparisons.csv", comparisons,
               list(comparisons[0]) if comparisons else ["feature"])
    _write_csv(config.output_dir / "community_enrichment.csv", community_rows,
               list(community_rows[0]) if community_rows else ["community_id"])

    write_graph_arrays(
        config.output_dir / "graph_arrays.npz",
        node_rows, edge_rows, hyperedge_rows, incidence_rows,
    )
    metadata = {
        "source_db": str(config.source_db.resolve()),
        "source_size": config.source_db.stat().st_size,
        "source_sha256": _file_sha256(config.source_db),
        "label_source": label_metadata, "seed": config.seed, "permutations": config.permutations,
        "node_scope": "labeled_only" if config.labeled_only else "all_source_agents",
        "edge_definition": "comment author -> parent-comment author; top-level -> post author",
        "normalization": "NFKC+casefold; HTML tags removed; URL/email/mention/UUID placeholders; whitespace collapsed",
    }
    _write_sqlite(
        config.output_dir / "comment_network.sqlite", node_rows, edge_rows,
        hyperedge_rows, incidence_rows, metadata,
    )
    report = {
        "status": "complete", "metadata": metadata,
        "audit": dict(audit),
        "network": {
            "nodes": len(node_rows), "active_nodes": len(active_nodes), "directed_edges": len(edge_rows),
            "directed_nonself_edges": graph.number_of_edges(), "weak_components": nx.number_connected_components(undirected),
            "largest_weak_component": max(component_size.values(), default=0),
            "communities_active_graph": len(communities), "louvain_modularity": float(modularity),
            "threads": len(hyperedge_rows), "incidences": len(incidence_rows),
        },
        "label_coverage": {
            **label_metadata,
            "labels_in_source_agents": sum(account in agents for account in labels),
            "labeled_active_nodes": sum(row["label"] in (0, 1) and row["out_degree"] + row["in_degree"] > 0
                                        for row in node_rows),
            "positive_active_nodes": sum(row["label"] == 1 and row["out_degree"] + row["in_degree"] > 0
                                         for row in node_rows),
        },
        "feature_comparisons": comparisons,
        "activity_adjusted_feature_tests": adjusted_comparisons,
        "edge_mixing": mixing, "thread_hypergraph_mixing": hypergraph,
        "community_enrichment": sorted(community_rows, key=lambda row: row["p_value"])[:50],
        "interpretation_boundary": (
            "Observational comment edges support association/aggregation claims only; "
            "they do not establish causal propagation or information diffusion."
        ),
    }
    (config.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
