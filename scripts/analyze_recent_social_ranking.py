#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
RECENT_START = "2026-05-18T00:00:00Z"
RECENT_END = "2026-05-25T00:00:00Z"
FUTURE_END = "2026-06-02T00:00:00Z"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--labels-db", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_labels(path):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT agent_id,is_malicious FROM manual_labels "
            "WHERE is_malicious IN (0,1)"
        ).fetchall()
    finally:
        connection.close()
    return {str(account_id): int(label) for account_id, label in rows}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def comment_rows(connection, start, end):
    query = """
        SELECT c.author_id,c.parent_id,p.author_id,pc.author_id
        FROM comments c
        LEFT JOIN posts p ON p.id=c.post_id
        LEFT JOIN comments pc ON pc.id=c.parent_id
        WHERE julianday(c.created_at) >= julianday(?)
          AND julianday(c.created_at) < julianday(?)
    """
    return connection.execute(query, (start, end))


def resolve_target(parent_id, post_author, parent_author):
    if parent_id and parent_author:
        return str(parent_author)
    if post_author:
        return str(post_author)
    return None


def recent_social_behavior(source_db, universe):
    out_targets = defaultdict(set)
    out_interactions = Counter()
    connection = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    try:
        for source, parent_id, post_author, parent_author in comment_rows(
            connection, RECENT_START, RECENT_END
        ):
            if source is None:
                continue
            source = str(source)
            target = resolve_target(parent_id, post_author, parent_author)
            if source not in universe or target is None or source == target:
                continue
            out_targets[source].add(target)
            out_interactions[source] += 1
    finally:
        connection.close()
    return {
        account_id: {
            "recent_out_degree": len(out_targets[account_id]),
            "recent_out_interactions": int(out_interactions[account_id]),
        }
        for account_id in universe
    }


def future_exposures(source_db, labels):
    by_source = defaultdict(Counter)
    connection = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    try:
        for source, parent_id, post_author, parent_author in comment_rows(
            connection, RECENT_END, FUTURE_END
        ):
            if source is None:
                continue
            source = str(source)
            target = resolve_target(parent_id, post_author, parent_author)
            if labels.get(source) == 1 and target is not None and labels.get(target) == 0:
                by_source[source][target] += 1
    finally:
        connection.close()
    targets = set().union(*(set(counts) for counts in by_source.values())) if by_source else set()
    return dict(by_source), {
        "exposed_normal_accounts": len(targets),
        "risky_to_normal_interactions": sum(sum(counts.values()) for counts in by_source.values()),
    }


def exposure_after(by_source, removed):
    targets = set()
    interactions = 0
    for source, counts in by_source.items():
        if source in removed:
            continue
        targets.update(counts)
        interactions += sum(counts.values())
    return len(targets), interactions


def budget_size(total, ratio):
    return 0 if ratio == 0 else min(total, max(1, math.ceil(total * ratio)))


def evaluate(name, ranking, labels, by_source, baseline, budget_total=None):
    rows = []
    budget_total = len(ranking) if budget_total is None else budget_total
    for ratio in RATIOS:
        size = budget_size(budget_total, ratio)
        removed = set(ranking[:size])
        exposed, interactions = exposure_after(by_source, removed)
        true_risky = sum(labels.get(account_id) == 1 for account_id in removed)
        rows.append({
            "method": name,
            "intervention_ratio": ratio,
            "removed_accounts": size,
            "true_risky_removed": true_risky,
            "normal_removed": size - true_risky,
            "selection_precision": true_risky / size if size else None,
            "exposed_normal_accounts": exposed,
            "exposure_reduction": (
                baseline["exposed_normal_accounts"] - exposed
            ) / max(1, baseline["exposed_normal_accounts"]),
            "remaining_risky_to_normal_interactions": interactions,
            "interaction_reduction": (
                baseline["risky_to_normal_interactions"] - interactions
            ) / max(1, baseline["risky_to_normal_interactions"]),
        })
    return rows


def main():
    args = parse_args()
    labels = read_labels(args.labels_db)
    predictions = read_jsonl(args.predictions)
    candidates = [
        str(row["account_id"]) for row in predictions if int(row["prediction"]) == 1
    ]
    universe = {str(row["account_id"]) for row in predictions}
    behavior = recent_social_behavior(args.source_db, universe)
    by_source, baseline = future_exposures(args.source_db, labels)

    rankings = {
        "AgentSec + recent out-degree": sorted(
            candidates,
            key=lambda account_id: (
                -behavior[account_id]["recent_out_degree"],
                -behavior[account_id]["recent_out_interactions"],
                account_id,
            ),
        ),
        "AgentSec + recent interaction volume": sorted(
            candidates,
            key=lambda account_id: (
                -behavior[account_id]["recent_out_interactions"],
                -behavior[account_id]["recent_out_degree"],
                account_id,
            ),
        ),
    }
    new_rows = []
    for name, ranking in rankings.items():
        new_rows.extend(evaluate(name, ranking, labels, by_source, baseline))
    recent_degree_all = sorted(
        universe,
        key=lambda account_id: (
            -behavior[account_id]["recent_out_degree"],
            -behavior[account_id]["recent_out_interactions"],
            account_id,
        ),
    )
    new_rows.extend(evaluate(
        "Recent out-degree (all accounts)",
        recent_degree_all,
        labels,
        by_source,
        baseline,
        budget_total=len(candidates),
    ))

    with args.base_summary.open(encoding="utf-8") as handle:
        base = json.load(handle)
    result = {
        "recent_behavior_window": [RECENT_START, RECENT_END],
        "future_evaluation_window": [RECENT_END, FUTURE_END],
        "candidate_accounts": len(candidates),
        "baseline": baseline,
        "new_results": new_rows,
        "comparison_results": base["results"] + new_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "recent_social_ranking_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in new_rows for key in row})
    with (args.output_dir / "recent_social_ranking_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(new_rows)

    ranking_rows = []
    for account_id in candidates:
        ranking_rows.append({
            "account_id": account_id,
            "label": labels.get(account_id),
            **behavior[account_id],
        })
    ranking_rows.sort(
        key=lambda row: (-row["recent_out_degree"], -row["recent_out_interactions"], row["account_id"])
    )
    with (args.output_dir / "recent_social_behavior.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_rows[0].keys())
        writer.writeheader()
        writer.writerows(ranking_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
