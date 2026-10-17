from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import importlib.metadata
from pathlib import Path
import sqlite3

from .common import (VERSION, OutputLock, canonical, compatible_manifest, deny_network,
                     digest, file_hash, read_db, source_fingerprint, write_json)
from .text import Cleaner, simhash


def load_split_labels(splits_path, accounts_json=None, label_db=None):
    splits = json.loads(Path(splits_path).read_text())
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits.json must contain train, validation, test")
    membership = {}
    for split, ids in splits.items():
        if not isinstance(ids, list):
            raise ValueError("Each split must be a list of account IDs")
        for aid in ids:
            aid = str(aid)
            if aid in membership:
                raise ValueError(f"Repeated/overlapping account in splits: {aid}")
            membership[aid] = split
    if accounts_json:
        rows = [(str(r["account_id"]), r["label"])
                for r in json.loads(Path(accounts_json).read_text())]
    else:
        with read_db(label_db) as db:
            rows = [(str(a), b) for a, b in db.execute(
                "SELECT agent_id,is_malicious FROM manual_labels WHERE is_malicious IN (0,1)")]
    labels = dict(rows)
    if len(labels) != len(rows) or any(v not in (0, 1) for v in labels.values()):
        raise ValueError("Labels must be unique account-level binary labels")
    missing = set(membership) - labels.keys()
    if missing:
        raise ValueError(f"Split accounts have no labels: {len(missing)}")
    return splits, membership, labels


def init_index(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS accounts (
      account_id TEXT PRIMARY KEY, split TEXT NOT NULL, label INTEGER,
      twitter_owner_group TEXT, profile_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS records (
      record_id INTEGER PRIMARY KEY, account_id TEXT NOT NULL, source_type TEXT NOT NULL,
      content_id TEXT NOT NULL, raw_hash TEXT NOT NULL, text_hash TEXT NOT NULL,
      simhash TEXT, payload TEXT NOT NULL, UNIQUE(source_type,content_id));
    CREATE INDEX IF NOT EXISTS records_account ON records(account_id,source_type,record_id);
    CREATE TABLE IF NOT EXISTS progress (stage TEXT PRIMARY KEY, cursor INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS duplicate_stats (
      account_id TEXT, source_type TEXT, payload TEXT, PRIMARY KEY(account_id,source_type));
    """)


def ensure_account(db, aid, membership, labels, profile=None):
    profile = profile or {}
    owner = str(profile.get("x_handle") or "").strip().lstrip("@").casefold()
    # Some exports store a profile URL instead of a handle; still no network.
    if "/" in owner:
        from urllib.parse import urlsplit
        owner = urlsplit(owner if "://" in owner else "https://" + owner).path.strip("/").split("/")[0]
    db.execute("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?,?)",
               (aid, membership.get(aid, "unassigned"), labels.get(aid), owner or None, canonical(profile)))


def make_record(cleaner, aid, cid, kind, timestamp, split, raw, fields, identities=(), urls=()):
    item = cleaner.clean(raw, identities=identities, external_urls=urls)
    item.update(account_id=aid, content_id=cid, source_type=kind, timestamp=timestamp,
                split=split, raw_text=raw, raw_fields=fields,
                profile_mask=item["valid_mask"] if kind.endswith("profile") else None)
    for link in item["urls"]:
        link.update(account_id=aid, content_id=cid, source_type=kind)
    return item


def insert_record(db, item):
    fingerprint = simhash(item["text"]) if item["source_type"] in {"post", "comment"} else None
    db.execute("INSERT INTO records(account_id,source_type,content_id,raw_hash,text_hash,simhash,payload) "
               "VALUES(?,?,?,?,?,?,?)", (item["account_id"], item["source_type"], item["content_id"],
               item["raw_text_hash"], item["text_hash"], str(fingerprint) if fingerprint is not None else None,
               canonical(item)))


def duplicate_statistics(rows):
    raw_counts = Counter(r[0] for r in rows)
    clean_counts = Counter(r[1] for r in rows)
    unique = {}
    for raw, clean, sh in rows:
        if sh is not None:
            unique[clean] = int(sh)
    buckets, neighbors = defaultdict(list), {}
    # Four disjoint bands guarantee a candidate for Hamming distance <= 3.
    # Counts are per account AND content type; no test/train corpus fitting.
    for key, bits in unique.items():
        candidates = set()
        bands = [(i, (bits >> (i * 16)) & 65535) for i in range(4)]
        for band in bands:
            candidates.update(buckets[band])
        for other in candidates:
            if (bits ^ unique[other]).bit_count() <= 3:
                neighbors[key] = neighbors.get(key, 0) + clean_counts[other]
                neighbors[other] = neighbors.get(other, 0) + clean_counts[key]
        for band in bands:
            buckets[band].append(key)
    return {"instances": len(rows), "exact_duplicate_instances": sum(n - 1 for n in raw_counts.values()),
            "normalized_duplicate_instances": sum(n - 1 for n in clean_counts.values()),
            "near_duplicate_pairs": sum(clean_counts[k] * v for k, v in neighbors.items()) // 2,
            "near_duplicate_definition": "distinct normalized hashes; 64-bit char-trigram SimHash Hamming <= 3",
            "raw_counts": dict(raw_counts), "normalized_counts": dict(clean_counts),
            "near_counts": neighbors}


def run(args):
    deny_network()
    root = Path(args.output_dir).resolve()
    source_path = Path(args.source_db).resolve()
    if root == source_path.parent or source_path.is_relative_to(root):
        raise ValueError("Output must be a new directory separate from the source database")
    splits, membership, labels = load_split_labels(args.splits, args.accounts_json, args.label_db)
    print("Fingerprinting read-only source database...", flush=True)
    config = {"preprocessing_version": VERSION, "source": source_fingerprint(source_path),
              "splits_sha256": file_hash(args.splits), "labels_sha256": digest(canonical(labels)),
              "url_policy": "origin-only; no domains in embedding; offline bundled PSL",
              "tldextract_version": importlib.metadata.version("tldextract"),
              "language_policy": "not detected; null metadata, never a model feature",
              "duplicates": "per-account/source raw SHA256 + clean SHA256 + trigram SimHash64 radius3"}
    with OutputLock(root):
        manifest = compatible_manifest(root, "prepare_manifest.json", config, args.resume)
        if manifest["complete"]:
            print("Compatible preprocessing already complete.")
            return
        cleaner = Cleaner()
        db = sqlite3.connect(root / "index.sqlite")
        db.execute("PRAGMA journal_mode=WAL")
        init_index(db)
        source = read_db(source_path)
        source.row_factory = sqlite3.Row
        source.execute("BEGIN")
        def cursor(stage):
            row = db.execute("SELECT cursor FROM progress WHERE stage=?", (stage,)).fetchone()
            return row[0] if row else 0
        def checkpoint(stage, last):
            db.execute("INSERT OR REPLACE INTO progress VALUES (?,?)", (stage, last))
            db.commit()
        agent_last = cursor("agents")
        for count, row in enumerate(source.execute("SELECT rowid AS _rowid,* FROM agents WHERE rowid>? ORDER BY rowid",
                                  (agent_last,)), 1):
            profile = dict(row)
            ensure_account(db, str(row["id"]), membership, labels, profile)
            agent_last = row["_rowid"]
            if count % args.page_size == 0:
                checkpoint("agents", agent_last)
        checkpoint("agents", agent_last)
        for aid in set(labels) | set(membership):
            ensure_account(db, aid, membership, labels)
        db.commit()
        # One scan per source table, avoiding unindexed per-author source queries.
        for kind, table in (("post", "posts"), ("comment", "comments")):
            last = cursor(kind)
            while True:
                rows = source.execute(f"SELECT rowid AS _rowid,* FROM {table} WHERE rowid>? ORDER BY rowid LIMIT ?",
                                      (last, args.page_size)).fetchall()
                if not rows:
                    break
                for row in rows:
                    fields = dict(row)
                    aid = str(row["author_id"]) if row["author_id"] else "__missing_author__:" + kind + ":" + str(row["id"])
                    ensure_account(db, aid, membership, labels)
                    raw = ((row["title"] or "") + "\n\n" + (row["content"] or "")) if kind == "post" else (row["content"] or "")
                    # Raw text/fields are retained exactly; only the model copy is cleaned.
                    item = make_record(cleaner, aid, str(row["id"]), kind, row["created_at"],
                                       membership.get(aid, "unassigned"), raw, fields,
                                       identities=(aid,), urls=(row["url"],) if kind == "post" else ())
                    item["author_missing"] = row["author_id"] is None or row["author_id"] == ""
                    insert_record(db, item)
                    last = row["_rowid"]
                checkpoint(kind, last)
                print(f"{kind}: scanned through source rowid {last}", flush=True)
        # Exactly two profile slots for every account, including missing authors/profiles.
        profile_fields = {
            "moltbook_profile": (("description", "Profile description"), ("location", "Location"), ("website", "Website")),
            "twitter_profile": (("x_bio", "Profile description"), ("x_location", "Location"), ("x_website", "Website")),
        }
        profile_last = cursor("profiles")
        for count, account in enumerate(db.execute("SELECT rowid,account_id,split,profile_json FROM accounts WHERE rowid>? ORDER BY rowid",
                                  (profile_last,)), 1):
            rid, aid, split, encoded = account
            profile = json.loads(encoded)
            for kind, allowed in profile_fields.items():
                fields = {key: profile[key] for key, _ in allowed if isinstance(profile.get(key), str) and profile[key].strip()}
                raw = "\n".join(f"{label}: {fields[key]}" for key, label in allowed if key in fields)
                item = make_record(cleaner, aid, aid + ":" + kind, kind, profile.get("last_updated"),
                                   split, raw, fields, identities=(aid, profile.get("x_handle")))
                # Field headings must never turn an empty/HTML-only profile into valid text.
                if not any(cleaner.clean(v)["text"] for v in fields.values()):
                    item.update(text="", text_hash=digest(""), is_empty=True, valid_mask=0, profile_mask=0)
                item["timestamp_kind"] = "profile_last_updated"
                insert_record(db, item)
            profile_last = rid
            if count % args.page_size == 0:
                checkpoint("profiles", profile_last)
                print(f"Profiles: processed through account rowid {profile_last}", flush=True)
        checkpoint("profiles", profile_last)
        source.close()
        print("Computing within-account duplicate statistics...", flush=True)
        for count, (aid,) in enumerate(db.execute("SELECT account_id FROM accounts ORDER BY account_id"), 1):
            for kind in ("post", "comment"):
                if db.execute("SELECT 1 FROM duplicate_stats WHERE account_id=? AND source_type=?", (aid, kind)).fetchone():
                    continue
                rows = db.execute("SELECT raw_hash,text_hash,simhash FROM records WHERE account_id=? AND source_type=?",
                                  (aid, kind)).fetchall()
                stats = duplicate_statistics(rows)
                db.execute("INSERT INTO duplicate_stats VALUES (?,?,?)", (aid, kind, canonical(stats)))
            db.commit()
            if count % 500 == 0:
                print(f"Duplicate statistics: {count} accounts", flush=True)
        # Check the immutable inputs again before certifying this dataset.
        if source_fingerprint(source_path) != config["source"]:
            raise RuntimeError("Source changed during preprocessing; use a stable snapshot and new output")
        write_json(root / "splits.json", splits)
        write_json(root / "labels.json", labels)
        with (root / "accounts.jsonl").open("w", encoding="utf-8") as out:
            for aid, split, label, owner in db.execute(
                    "SELECT account_id,split,label,twitter_owner_group FROM accounts ORDER BY account_id"):
                out.write(canonical(dict(account_id=aid, split=split, label=label, twitter_owner_group=owner)) + "\n")
        owners = db.execute("SELECT twitter_owner_group,COUNT(DISTINCT split),GROUP_CONCAT(DISTINCT split) "
                            "FROM accounts WHERE twitter_owner_group IS NOT NULL AND split!='unassigned' "
                            "GROUP BY twitter_owner_group HAVING COUNT(DISTINCT split)>1").fetchall()
        write_json(root / "owner_split_audit.json", {"cross_split_owners": owners,
                   "action": "Reported only; original splits unchanged. Use owner groups for a future separate group-split experiment."})
        manifest.update(complete=True, account_count=db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0],
                        record_counts=dict(db.execute("SELECT source_type,COUNT(*) FROM records GROUP BY source_type")))
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()
        manifest["index_sha256"] = file_hash(root / "index.sqlite")
        write_json(root / "prepare_manifest.json", manifest)
        print(canonical(manifest), flush=True)


def main():
    p = argparse.ArgumentParser(description="Read-only multilingual preprocessing; never access content URLs")
    p.add_argument("--source-db", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", required=True)
    labels = p.add_mutually_exclusive_group(required=True)
    labels.add_argument("--accounts-json")
    labels.add_argument("--label-db")
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.page_size < 1:
        p.error("page-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
