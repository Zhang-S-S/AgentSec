from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
from pathlib import Path
from urllib.parse import quote

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
VERSION = "moltbook-multimodal-v1"
SOURCES = ("post", "comment", "moltbook_profile", "twitter_profile")
INSTRUCTION = (
    "Represent this multilingual social-media content for risky-account detection, "
    "preserving signals of abuse, violence, deception, manipulation, spam, "
    "low-quality interference, and malicious external links."
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint(path):
    path = Path(path).resolve()
    # An empty WAL contains no committed data and SQLite may create/remove this
    # zero-byte sidecar while opening a read-only database. Treat missing and
    # empty WAL files as the same logical source state.
    wal_path = Path(str(path) + "-wal")
    wal_sha256 = file_hash(wal_path) if wal_path.is_file() and wal_path.stat().st_size else None
    return {"sha256": file_hash(path), "wal_sha256": wal_sha256}


def read_db(path):
    return sqlite3.connect(f"file:{quote(str(Path(path).resolve()), safe='/')}?mode=ro", uri=True)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def deny_network():
    """Process-wide fail-closed network guard, including model/tokenizer libraries."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    def denied(*args, **kwargs):
        raise RuntimeError("Network access is forbidden in the offline embedding pipeline")
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.socket.sendto = denied
    socket.getaddrinfo = denied


class OutputLock:
    def __init__(self, root):
        self.root = Path(root)

    def __enter__(self):
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = (self.root / ".lock").open("a")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Another process is writing {self.root}")
        return self

    def __exit__(self, *args):
        self.file.close()


def compatible_manifest(root, name, config, resume):
    path = Path(root) / name
    if path.exists():
        old = json.loads(path.read_text())
        if not resume:
            raise ValueError(f"Output exists: {path}. Use --resume or a new directory")
        if old["config"] != config:
            raise ValueError(f"Incompatible cache configuration: {path}; use a new directory")
        return old
    other = [p for p in Path(root).iterdir() if p.name != ".lock"]
    if other:
        raise ValueError(f"Refusing nonempty unrecognized output directory: {root}")
    obj = {"config": config, "complete": False}
    write_json(path, obj)
    return obj
