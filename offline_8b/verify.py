from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .common import MODEL_ID, SOURCES, deny_network, digest, file_hash, read_db, write_json
from .encode import check_shards, validate_vectors


def verify(input_dir, prepared_dir=None, samples=2, seed=42):
    import numpy as np
    deny_network()
    root = Path(input_dir)
    manifest = json.loads((root / "embedding_manifest.json").read_text())
    if not manifest["complete"] or manifest["config"]["model_name"] != MODEL_ID:
        raise ValueError("Not a complete 8B cache")
    dimension = manifest["config"]["embedding_dimension"]
    db = read_db(root / "index.sqlite")
    check_shards(root, db)
    splits = json.loads((root / "splits.json").read_text())
    membership = {}
    for split, ids in splits.items():
        for aid in ids:
            if aid in membership:
                raise ValueError(f"Account overlap: {aid}")
            membership[aid] = split
    accounts = {}
    with (root / "accounts.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            aid = item["account_id"]
            if aid in accounts or item["split"] != membership.get(aid, "unassigned"):
                raise ValueError("Duplicated account or changed split")
            accounts[aid] = item
    labels = json.loads((root / "labels.json").read_text())
    if not set(membership) <= accounts.keys() or any(accounts[a]["label"] != label for a, label in labels.items()):
        raise ValueError("Missing account or changed label")
    if prepared_dir:
        prepared = Path(prepared_dir)
        prep_manifest = json.loads((prepared / "prepare_manifest.json").read_text())
        if file_hash(prepared / "index.sqlite") != manifest["config"]["prepared_index_sha256"]:
            raise ValueError("Prepared source fingerprint mismatch")
        for filename in ("splits.json", "labels.json", "accounts.jsonl"):
            if file_hash(root / filename) != file_hash(prepared / filename):
                raise ValueError(f"Changed {filename}")
        db.execute("ATTACH DATABASE ? AS prepared", (f"file:{prepared / 'index.sqlite'}?mode=ro",))
        missing = db.execute("SELECT COUNT(*) FROM prepared.records r LEFT JOIN instances i USING(record_id) "
                             "WHERE i.record_id IS NULL OR i.account_id!=r.account_id OR i.source_type!=r.source_type").fetchone()[0]
        if missing or prep_manifest["account_count"] != len(accounts):
            raise ValueError("Missing or mismatched source instances/accounts")
    rng, seen, reservoirs = random.Random(seed), Counter(), {}
    counts, profiles = Counter(), Counter()
    total = 0
    for shard, rows in db.execute("SELECT shard,rows FROM shards ORDER BY shard"):
        if shard % 16 == 0:
            print(f"Checking shard {shard}; {total} instances checked", flush=True)
        stem = root / "shards" / f"shard-{shard:06d}"
        vectors = np.load(str(stem) + ".npy", mmap_mode="r")
        with open(str(stem) + ".jsonl", encoding="utf-8") as f:
            metadata = [json.loads(line) for line in f]
        if rows != len(metadata) or vectors.dtype != np.float32:
            raise ValueError("Shard row count or storage dtype mismatch")
        validate_vectors(vectors, [m["valid_mask"] for m in metadata], dimension)
        for index, item in enumerate(metadata):
            aid, kind = item["account_id"], item["source_type"]
            ref = db.execute("SELECT account_id,source_type,shard,row_index,valid_mask FROM instances WHERE record_id=?",
                             (item["record_id"],)).fetchone()
            if ref != (aid, kind, shard, index, item["valid_mask"]) or item["row_index"] != index or item["shard"] != shard:
                raise ValueError("Index/metadata mismatch")
            if aid not in accounts or item["split"] != accounts[aid]["split"] or kind not in SOURCES:
                raise ValueError("Wrong account/split/source")
            if item["text_hash"] != digest(item["text"]) or item["raw_text_hash"] != digest(item["raw_text"]):
                raise ValueError("Text hash mismatch")
            if item["is_empty"] != (not item["text"]) or item["valid_mask"] != int(not item["is_empty"]):
                raise ValueError("Empty text mask mismatch")
            if item["url_count"] != len(item["urls"]) or item["has_url"] != bool(item["urls"]):
                raise ValueError("URL metadata mismatch")
            limit = manifest["config"]["profile_max_length"] if kind.endswith("profile") else manifest["config"]["max_length"]
            if item["retained_token_count"] > limit or item["is_truncated"] != (item["original_token_count"] > item["retained_token_count"]):
                raise ValueError("Truncation metadata mismatch")
            if kind.endswith("profile"):
                profiles[(aid, kind)] += 1
                if item["profile_mask"] != item["valid_mask"]:
                    raise ValueError("Profile mask mismatch")
            if prepared_dir:
                original = json.loads(db.execute("SELECT payload FROM prepared.records WHERE record_id=?", (item["record_id"],)).fetchone()[0])
                if any(item.get(k) != v for k, v in original.items()):
                    raise ValueError("Raw or preprocessed source metadata changed")
            counts[kind] += 1
            total += 1
            categories = ["random"]
            if any(ord(c) > 127 for c in item["raw_text"]):
                categories.append("non_ascii_multilingual_candidates")
            if item["has_url"]:
                categories.append("url")
            if kind.endswith("profile") and item["is_empty"]:
                categories.append("empty_profile")
            if item["is_truncated"]:
                categories.append("long_truncated")
            if item.get("exact_duplicate_count", 0) or item.get("normalized_duplicate_count", 0):
                categories.append("duplicate")
            for category in categories:
                seen[category] += 1
                group = reservoirs.setdefault(category, [])
                sample = {"account_id": aid, "content_id": item["content_id"], "source_type": kind,
                          "raw_text": item["raw_text"], "model_text": item["text"],
                          "tensor_shape": [dimension], "valid_mask": item["valid_mask"],
                          "original_token_count": item["original_token_count"],
                          "retained_token_count": item["retained_token_count"]}
                if len(group) < samples:
                    group.append(sample)
                else:
                    chosen = rng.randrange(seen[category])
                    if chosen < samples:
                        group[chosen] = sample
    if total != manifest["records"] or total != db.execute("SELECT COUNT(*) FROM instances").fetchone()[0]:
        raise ValueError("Extra/missing index rows")
    if len(accounts) != manifest["accounts"] or any(profiles[(aid, kind)] != 1 for aid in accounts
                                                  for kind in ("moltbook_profile", "twitter_profile")):
        raise ValueError("Accounts/profiles are not one-to-one")
    report = {"passed": True, "accounts": len(accounts), "instances": total,
              "dimension": dimension, "source_counts": dict(counts),
              "network": "Runtime socket access denied; no URL fetching code; HF local-only",
              "sample_categories_present": dict(seen), "samples": reservoirs}
    write_json(root / "quality_report.json", report)
    for category, examples in reservoirs.items():
        for item in examples:
            print(f"\n[{category}] {item['source_type']} tensor_shape={item['tensor_shape']} mask={item['valid_mask']}")
            # Display clipping only; never used as model input. Full text in report.
            print("before:", repr(item["raw_text"][:800]))
            print("after: ", repr(item["model_text"][:800]))
            print("tokens:", item["original_token_count"], "->", item["retained_token_count"])
    from .loader import AccountEmbeddings
    loader = AccountEmbeddings(root)
    for aid in rng.sample(sorted(accounts), min(samples, len(accounts))):
        item = loader[aid]
        print("Account shapes:", aid, {k: list(v.shape) for k, v in item.items() if isinstance(v, np.ndarray)})
    loader.close()
    db.close()
    print(f"PASS: {len(accounts)} accounts, {total} instances, dimension={dimension}")
    return report


def main():
    p = argparse.ArgumentParser(description="Check all vectors, masks, source alignment, splits and samples")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--prepared-dir")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.samples < 0:
        p.error("samples must be nonnegative")
    verify(args.input_dir, args.prepared_dir, args.samples, args.seed)


if __name__ == "__main__":
    main()
