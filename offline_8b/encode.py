from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import zlib

from .common import (INSTRUCTION, MODEL_ID, SOURCES, VERSION, OutputLock, canonical,
                     compatible_manifest, deny_network, digest, file_hash, read_db, write_json)

def resolve_model(path, revision=None):
    path = Path(path).expanduser().absolute()
    if (path / "snapshots").is_dir():
        if revision is None and (path / "refs/main").is_file():
            revision = (path / "refs/main").read_text().strip()
        if revision:
            path = path / "snapshots" / revision
        else:
            candidates = [p for p in (path / "snapshots").iterdir() if (p / "config.json").is_file()]
            if len(candidates) != 1:
                raise ValueError("Multiple/no local snapshots; specify --model-revision COMMIT")
            path = candidates[0]
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Local model config missing: {path}")
    config = json.loads((path / "config.json").read_text())
    if (config.get("model_type") != "qwen3" or config.get("hidden_size") != 4096
            or config.get("num_hidden_layers") != 36 or config.get("vocab_size") != 151665):
        raise ValueError("Expected official Qwen3-Embedding-8B architecture (4096 dimensions, 36 layers)")
    modules_file = path / "modules.json"
    if not modules_file.is_file():
        raise ValueError("Missing modules.json; refusing SentenceTransformers fallback pooling")
    modules = json.loads(modules_file.read_text())
    types = [m["type"] for m in modules]
    allowed = ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling"]
    if types not in (allowed, allowed + ["sentence_transformers.models.Normalize"]):
        raise ValueError("Expected official Transformer -> Pooling [-> Normalize] modules")
    pooling_file = path / "1_Pooling/config.json"
    if not pooling_file.exists():
        raise ValueError("Use the full official SentenceTransformers snapshot, including 1_Pooling")
    pooling = json.loads(pooling_file.read_text())
    if (pooling.get("word_embedding_dimension") != 4096 or not pooling.get("include_prompt", True)
            or not pooling.get("pooling_mode_lasttoken")) or any(
            value for key, value in pooling.items() if key.startswith("pooling_mode_") and key != "pooling_mode_lasttoken"):
        raise ValueError("Snapshot does not use official last-token-only pooling")
    # Content digest also protects local folders without a Hub snapshot commit.
    # Hashing model weights costs one sequential read, before GPU loading.
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt"})
    fingerprint = digest(canonical({str(p.relative_to(path)): file_hash(p) for p in files}))
    resolved_revision = path.name if path.parent.name == "snapshots" else "local-sha256:" + fingerprint
    if revision and path.parent.name == "snapshots" and path.name != revision:
        raise ValueError("Requested revision differs from local snapshot")
    return path, resolved_revision, fingerprint


def tokenize_record(tokenizer, text, max_length, instruction=True):
    if not text:
        return [], {"original_token_count": 0, "retained_token_count": 0, "is_truncated": False}
    prefix = f"Instruct: {INSTRUCTION}\nQuery: " if instruction else ""
    ids = tokenizer(prefix + text, add_special_tokens=True, truncation=False)["input_ids"]
    original = len(ids)
    if instruction:
        prefix_size = len(tokenizer(prefix, add_special_tokens=True, truncation=False)["input_ids"])
        if max_length < prefix_size + 2:
            raise ValueError("max_length must leave room for instruction and content head/tail")
    else:
        prefix_size = 0
    if original > max_length:
        head = max(prefix_size + 1, (max_length + 1) // 2)
        tail = max_length - head
        ids = ids[:head] + ids[-tail:]
    return ids, {"original_token_count": original, "retained_token_count": len(ids),
                 "is_truncated": original > len(ids)}


def init_output(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS tokens (
      record_id INTEGER PRIMARY KEY, input_key TEXT, ids BLOB, original INTEGER,
      retained INTEGER, source_type TEXT);
    CREATE TABLE IF NOT EXISTS shards (
      shard INTEGER PRIMARY KEY, rows INTEGER, vector_sha TEXT, metadata_sha TEXT);
    CREATE TABLE IF NOT EXISTS instances (
      record_id INTEGER PRIMARY KEY, account_id TEXT, source_type TEXT,
      shard INTEGER, row_index INTEGER, valid_mask INTEGER);
    CREATE INDEX IF NOT EXISTS by_account ON instances(account_id,source_type,record_id);
    CREATE TABLE IF NOT EXISTS vector_cache (input_key TEXT PRIMARY KEY, shard INTEGER, row_index INTEGER);
    """)


def length_report(db):
    report = {}
    for kind in SOURCES:
        hist = list(db.execute("SELECT original,COUNT(*) FROM tokens WHERE source_type=? GROUP BY original ORDER BY original", (kind,)))
        total = sum(n for _, n in hist)
        entry = {"count": total, "histogram": dict(hist)}
        for q in (0.5, 0.9, 0.95, 0.99, 1.0):
            seen = 0
            for length, count in hist:
                seen += count
                if seen >= max(1, total * q):
                    entry[f"p{int(q * 100)}"] = length
                    break
        entry["truncated_count"] = db.execute("SELECT COUNT(*) FROM tokens WHERE source_type=? AND original>retained", (kind,)).fetchone()[0]
        report[kind] = entry
    return report


class Encoder:
    def __init__(self, model_path, tokenizer, args, dtype):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch, self.tokenizer = torch, tokenizer
        kwargs = {"torch_dtype": getattr(torch, dtype)}
        if args.flash_attention_2:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = SentenceTransformer(str(model_path), device=args.device,
                    local_files_only=True, trust_remote_code=False,
                    model_kwargs=kwargs, tokenizer_kwargs={"padding_side": "left"})
        if self.model.get_sentence_embedding_dimension() != 4096:
            raise ValueError("Model does not produce native 4096-dimensional embeddings")
        self.model.eval()
        self.model[0].auto_model.config.use_cache = False
        self.device, self.dimension, self.batch_size = args.device, args.embedding_dim, args.batch_size

    def batch(self, rows):
        from sentence_transformers.util import truncate_embeddings
        torch = self.torch
        features = self.tokenizer.pad({"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]},
                                      padding=True, return_tensors="pt")
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.inference_mode():
            vectors = self.model(features)["sentence_embedding"].float()
            if self.dimension != 4096:
                # Official ST Matryoshka dimension API, never manual array slicing.
                vectors = truncate_embeddings(vectors, self.dimension)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy()

    def encode(self, rows):
        import numpy as np
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        result = np.empty((len(rows), self.dimension), dtype=np.float32)
        offset = 0
        while offset < len(order):
            indices = order[offset:offset + self.batch_size]
            failed = False
            try:
                result[indices] = self.batch([rows[i] for i in indices])
            except self.torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise RuntimeError("OOM at batch size 1; change hardware or explicitly configure a new experiment. Length/dimension unchanged.") from None
                failed = True
            if failed:
                self.batch_size = max(1, len(indices) // 2)
                gc.collect()
                self.torch.cuda.empty_cache()
                print(f"CUDA OOM: retrying with batch_size={self.batch_size}; model/length/dimension unchanged", flush=True)
                continue
            offset += len(indices)
        return result


def validate_vectors(vectors, masks, dimension):
    import numpy as np
    if vectors.shape != (len(masks), dimension) or not np.isfinite(vectors).all():
        raise ValueError("Embedding shape mismatch or NaN/Inf")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, masks, atol=2e-4):
        raise ValueError("Expected unit vectors for valid inputs and zero vectors for empty inputs")


def check_shards(root, db):
    for shard, rows, vhash, mhash in db.execute("SELECT * FROM shards ORDER BY shard"):
        stem = Path(root) / "shards" / f"shard-{shard:06d}"
        if file_hash(str(stem) + ".npy") != vhash or file_hash(str(stem) + ".jsonl") != mhash:
            raise ValueError(f"Corrupted completed shard {shard}; restore it before resuming")


def encode_pending(prepared, root, db, encoder, dimension, shard_size):
    """Bounded shards, SQLite transaction as commit marker, content-hash reuse."""
    import numpy as np
    from collections import OrderedDict
    shard_cache = OrderedDict()
    def cached(shard):
        if shard not in shard_cache:
            shard_cache[shard] = np.load(root / "shards" / f"shard-{shard:06d}.npy", mmap_mode="r")
            if len(shard_cache) > 8:
                shard_cache.popitem(last=False)
        shard_cache.move_to_end(shard)
        return shard_cache[shard]
    db.execute("ATTACH DATABASE ? AS prepared", (f"file:{prepared / 'index.sqlite'}?mode=ro",))
    last = db.execute("SELECT account_id,source_type,record_id FROM instances "
                      "ORDER BY account_id DESC,source_type DESC,record_id DESC LIMIT 1").fetchone()
    last = last or ("", "", 0)
    while True:
        rows = db.execute("SELECT r.record_id,r.payload,t.input_key,t.ids,t.original,t.retained "
                          "FROM prepared.records r JOIN tokens t USING(record_id) "
                          "WHERE (r.account_id,r.source_type,r.record_id) > (?,?,?) "
                          "ORDER BY r.account_id,r.source_type,r.record_id LIMIT ?", (*last, shard_size)).fetchall()
        if not rows:
            break
        next_shard = db.execute("SELECT COALESCE(MAX(shard)+1,0) FROM shards").fetchone()[0]
        vectors = np.zeros((len(rows), dimension), dtype=np.float32)
        unique, source_rows, metadata = {}, {}, []
        stats_cache = {}
        for index, (rid, payload, key, packed, original, retained) in enumerate(rows):
            item = json.loads(payload)
            item.update(record_id=rid, shard=next_shard, row_index=index,
                        original_token_count=original, retained_token_count=retained,
                        is_truncated=original > retained, input_hash=key)
            pair = (item["account_id"], item["source_type"])
            if item["source_type"] in {"post", "comment"}:
                if pair not in stats_cache:
                    stats_cache[pair] = json.loads(db.execute(
                        "SELECT payload FROM prepared.duplicate_stats WHERE account_id=? AND source_type=?", pair).fetchone()[0])
                stats = stats_cache[pair]
                item["exact_duplicate_count"] = stats["raw_counts"][item["raw_text_hash"]] - 1
                item["normalized_duplicate_count"] = stats["normalized_counts"][item["text_hash"]] - 1
                item["near_duplicate_count"] = stats["near_counts"].get(item["text_hash"], 0)
            metadata.append(item)
            if not retained:
                continue
            hit = db.execute("SELECT shard,row_index FROM vector_cache WHERE input_key=?", (key,)).fetchone()
            if hit:
                vectors[index] = cached(hit[0])[hit[1]]
            else:
                unique.setdefault(key, json.loads(zlib.decompress(packed)))
                source_rows.setdefault(key, []).append(index)
        if unique:
            keys = list(unique)
            generated = encoder.encode([unique[k] for k in keys])
            for key, value in zip(keys, generated):
                vectors[source_rows[key]] = value
        validate_vectors(vectors, [m["valid_mask"] for m in metadata], dimension)
        stem = root / "shards" / f"shard-{next_shard:06d}"
        with open(str(stem) + ".npy.tmp", "wb") as f:
            np.save(f, vectors, allow_pickle=False)
        with open(str(stem) + ".jsonl.tmp", "w", encoding="utf-8") as f:
            for item in metadata:
                f.write(canonical(item) + "\n")
        os.replace(str(stem) + ".npy.tmp", str(stem) + ".npy")
        os.replace(str(stem) + ".jsonl.tmp", str(stem) + ".jsonl")
        with db:
            db.execute("INSERT INTO shards VALUES (?,?,?,?)", (next_shard, len(rows),
                       file_hash(str(stem) + ".npy"), file_hash(str(stem) + ".jsonl")))
            db.executemany("INSERT INTO instances VALUES (?,?,?,?,?,?)", [
                (m["record_id"], m["account_id"], m["source_type"], next_shard, i, m["valid_mask"])
                for i, m in enumerate(metadata)])
            db.executemany("INSERT OR IGNORE INTO vector_cache VALUES (?,?,?)",
                           [(key, next_shard, indexes[0]) for key, indexes in source_rows.items()])
        print(f"Committed shard {next_shard}: {len(rows)} instances, {len(unique)} new model inputs", flush=True)
        last = (metadata[-1]["account_id"], metadata[-1]["source_type"], metadata[-1]["record_id"])
    db.execute("DETACH DATABASE prepared")


def run(args):
    deny_network()
    prepared, root = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if prepared == root or root.is_relative_to(prepared):
        raise ValueError("Use a separate new embedding output directory")
    prep_manifest = json.loads((prepared / "prepare_manifest.json").read_text())
    if not prep_manifest["complete"] or prep_manifest["config"]["preprocessing_version"] != VERSION:
        raise ValueError("Preprocessing is incomplete/incompatible")
    if file_hash(prepared / "index.sqlite") != prep_manifest["index_sha256"]:
        raise ValueError("Preprocessed database has changed")
    print("Resolving and fingerprinting local model snapshot...", flush=True)
    model_path, revision, model_hash = resolve_model(args.model_path, args.model_revision)
    import torch
    from transformers import AutoTokenizer
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device unavailable; no silent CPU fallback")
    if args.dtype == "auto":
        if args.device.startswith("cuda"):
            with torch.cuda.device(args.device):
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = "float32"
    else:
        dtype = args.dtype
    if dtype == "bfloat16" and args.device.startswith("cuda"):
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 unsupported by requested GPU; explicitly select --dtype float16")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left", trust_remote_code=False)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "sentence-transformers", "numpy")}
    config = {"model_name": MODEL_ID, "model_revision": revision, "model_sha256": model_hash,
              "native_dimension": 4096, "embedding_dimension": args.embedding_dim,
              "dimension_method": "native" if args.embedding_dim == 4096 else "sentence_transformers.util.truncate_embeddings (native MRL)",
              "compute_dtype": dtype, "storage_dtype": "float32", "pooling": "last-token",
              "padding_side": "left", "normalization": "L2 float32", "instruction": INSTRUCTION if args.instruction else None,
              "instruction_template": "Instruct: {instruction}\nQuery: {text}" if args.instruction else "{text}",
              "max_length": args.max_length, "profile_max_length": args.profile_max_length,
              "truncation": "token-head-tail-preserve-instruction", "preprocessing_version": VERSION,
              "prepared_index_sha256": prep_manifest["index_sha256"],
              "prepare_config": prep_manifest["config"], "versions": versions,
              "flash_attention_2": args.flash_attention_2, "network_policy": "deny sockets + HF offline + local_files_only"}
    with OutputLock(root):
        manifest = compatible_manifest(root, "embedding_manifest.json", config, args.resume)
        db = sqlite3.connect(root / "index.sqlite", uri=True)
        init_output(db)
        check_shards(root, db)
        if manifest["complete"]:
            print("Compatible embedding cache already complete.")
            db.close()
            return
        source = read_db(prepared / "index.sqlite")
        cursor = db.execute("SELECT COALESCE(MAX(record_id),0) FROM tokens").fetchone()[0]
        print("Collecting full token-length distributions before model inference...", flush=True)
        count = 0
        for rid, kind, payload in source.execute("SELECT record_id,source_type,payload FROM records WHERE record_id>? ORDER BY record_id", (cursor,)):
            item = json.loads(payload)
            limit = args.profile_max_length if kind.endswith("profile") else args.max_length
            ids, info = tokenize_record(tokenizer, item["text"], limit, args.instruction)
            key = digest(canonical({"text_hash": item["text_hash"], "max_length": limit,
                                    "ids": ids, "config": digest(canonical(config))}))
            db.execute("INSERT INTO tokens VALUES (?,?,?,?,?,?)", (rid, key, zlib.compress(canonical(ids).encode()),
                       info["original_token_count"], info["retained_token_count"], kind))
            count += 1
            if count % 1000 == 0:
                db.commit()
                print(f"Tokenized {count} new instances", flush=True)
        db.commit()
        report = length_report(db)
        write_json(root / "token_lengths.json", report)
        print(canonical({kind: {k: v for k, v in values.items() if k != "histogram"}
                         for kind, values in report.items()}), flush=True)
        source.close()
        if args.stats_only:
            print("Statistics ready; no model weights loaded. Resume with the same settings to embed.")
            db.close()
            return
        (root / "shards").mkdir(exist_ok=True)
        encoder = Encoder(model_path, tokenizer, args, dtype)
        encode_pending(prepared, root, db, encoder, args.embedding_dim, args.shard_size)
        # Copy immutable account/split/label metadata for a self-contained cache.
        import shutil
        for name in ("accounts.jsonl", "splits.json", "labels.json", "owner_split_audit.json"):
            shutil.copyfile(prepared / name, root / name)
        source = read_db(prepared / "index.sqlite")
        with (root / "duplicate_stats.jsonl").open("w", encoding="utf-8") as out:
            for aid, kind, payload in source.execute("SELECT * FROM duplicate_stats ORDER BY account_id,source_type"):
                stats = json.loads(payload)
                out.write(canonical({"account_id": aid, "source_type": kind, **{
                    k: v for k, v in stats.items() if k not in {"raw_counts", "normalized_counts", "near_counts"}}}) + "\n")
        expected = source.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        actual = db.execute("SELECT COUNT(*) FROM instances").fetchone()[0]
        if actual != expected:
            raise ValueError(f"Missing instances: {actual} != {expected}")
        source.close()
        manifest.update(complete=True, records=actual, accounts=prep_manifest["account_count"],
                        final_batch_size=encoder.batch_size,
                        shards=db.execute("SELECT COUNT(*) FROM shards").fetchone()[0])
        db.close()
        write_json(root / "embedding_manifest.json", manifest)
        print("Embedding complete. Run offline_8b.verify for the full audit.", flush=True)


def main():
    p = argparse.ArgumentParser(description="Offline Qwen3-Embedding-8B, resume-safe multimodal encoding")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-revision")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--profile-max-length", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=4096)
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--instruction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--flash-attention-2", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 32 <= args.embedding_dim <= 4096:
        p.error("Native MRL embedding dimension must be 32..4096")
    if min(args.batch_size, args.shard_size) < 1 or not all(2 <= n <= 32768 for n in (args.max_length, args.profile_max_length)):
        p.error("Positive batch/shard size and token limits in 2..32768 required")
    run(args)


if __name__ == "__main__":
    main()
