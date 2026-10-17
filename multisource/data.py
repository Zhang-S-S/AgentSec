from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .url_features import encode_bytes, load_url_cache


def load_accounts(embedding_dir: Path) -> list[dict]:
    accounts = []
    with (embedding_dir / "accounts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("label") in (0, 1):
                accounts.append(item)
    if not accounts:
        raise ValueError(f"No labeled accounts in {embedding_dir / 'accounts.jsonl'}")
    return accounts


def load_splits(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"train", "validation", "test"}:
        raise ValueError("splits must contain exactly train, validation, and test")
    seen: set[str] = set()
    result = {}
    for name in ("train", "validation", "test"):
        result[name] = [str(x) for x in value[name]]
        if not result[name]:
            raise ValueError(f"Split {name!r} is empty")
        if len(result[name]) != len(set(result[name])):
            raise ValueError(f"Split {name!r} contains duplicate account IDs")
        overlap = seen.intersection(result[name])
        if overlap:
            raise ValueError(f"Account crosses splits: {next(iter(overlap))}")
        seen.update(result[name])
    return result


def split_hash(splits: dict[str, list[str]]) -> str:
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class ShardCache:
    def __init__(self, root: Path, max_open: int = 8) -> None:
        self.root, self.max_open = root, max_open
        self.values: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, shard: int) -> np.ndarray:
        value = self.values.pop(shard, None)
        if value is None:
            value = np.load(self.root / "shards" / f"shard-{shard:06d}.npy", mmap_mode="r")
        self.values[shard] = value
        while len(self.values) > self.max_open:
            self.values.popitem(last=False)
        return value


def content_presence(embedding_dir: Path) -> dict[str, tuple[bool, bool]]:
    uri = f"file:{(embedding_dir / 'index.sqlite').resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute(
            "SELECT account_id, source_type, MAX(valid_mask) FROM instances "
            "WHERE source_type IN ('post','comment') GROUP BY account_id,source_type"
        ).fetchall()
    result: dict[str, list[bool]] = {}
    for aid, kind, valid in rows:
        result.setdefault(str(aid), [False, False])[0 if kind == "post" else 1] = bool(valid)
    return {key: (value[0], value[1]) for key, value in result.items()}


def modality_presence(embedding_dir: Path) -> dict[str, dict[str, bool]]:
    uri = f"file:{(embedding_dir / 'index.sqlite').resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute(
            "SELECT account_id,source_type,MAX(valid_mask) FROM instances GROUP BY account_id,source_type"
        ).fetchall()
    result: dict[str, dict[str, bool]] = {}
    for aid, source, valid in rows:
        result.setdefault(str(aid), {})[str(source)] = bool(valid)
    return result


def eligible_split_ids(embedding_dir: Path, splits: dict[str, list[str]], content_mode: str) -> dict[str, list[str]]:
    presence = content_presence(embedding_dir)
    result = {}
    for split, ids in splits.items():
        selected = []
        for aid in ids:
            post, comment = presence.get(aid, (False, False))
            if content_mode == "post_only" and not post:
                continue
            if content_mode == "comment_only" and not comment:
                continue
            if content_mode == "post_comment" and not (post or comment):
                raise ValueError(f"Account {aid!r} has neither a valid Post nor Comment")
            selected.append(aid)
        if not selected:
            raise ValueError(f"No eligible accounts in {split} for {content_mode}")
        result[split] = selected
    return result


class MultiSourceDataset(Dataset):
    def __init__(self, embedding_dir: Path, account_ids: Sequence[str], *, training: bool,
                 seed: int, max_posts: int, max_comments: int, max_urls: int,
                 url_cache: Path | None = None,
                 url_data: tuple[dict[str, list[dict]], dict] | None = None) -> None:
        self.embedding_dir = embedding_dir
        manifest = json.loads((embedding_dir / "embedding_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError("Embedding cache is incomplete")
        self.dimension = int(manifest["config"]["embedding_dimension"])
        accounts = {str(x["account_id"]): x for x in load_accounts(embedding_dir)}
        missing = set(account_ids) - accounts.keys()
        if missing:
            raise ValueError(f"Split contains unknown account: {next(iter(missing))}")
        self.accounts = [accounts[str(aid)] for aid in account_ids]
        self.training, self.seed = training, seed
        self.limits = {"post": max_posts, "comment": max_comments}
        self.max_urls = max_urls
        self.urls, self.url_manifest = (url_data if url_data is not None else
                                        load_url_cache(url_cache) if url_cache else ({}, None))
        self.db: sqlite3.Connection | None = None
        self.shards: ShardCache | None = None

    def __len__(self) -> int:
        return len(self.accounts)

    def _resources(self) -> tuple[sqlite3.Connection, ShardCache]:
        if self.db is None:
            uri = f"file:{(self.embedding_dir / 'index.sqlite').resolve()}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True)
            self.shards = ShardCache(self.embedding_dir)
        return self.db, self.shards  # type: ignore[return-value]

    def _vectors(self, aid: str, source: str) -> tuple[np.ndarray, np.ndarray]:
        db, shards = self._resources()
        refs = db.execute(
            "SELECT shard,row_index,valid_mask FROM instances "
            "WHERE account_id=? AND source_type=? ORDER BY record_id", (aid, source)
        ).fetchall()
        if source in {"post", "comment"}:
            refs = [row for row in refs if row[2]]
            limit = self.limits[source]
            if limit > 0 and len(refs) > limit:
                if self.training:
                    refs = random.sample(refs, limit)
                else:
                    rng = random.Random(f"{self.seed}:{aid}:{source}")
                    refs = rng.sample(refs, limit)
                refs.sort(key=lambda row: (row[0], row[1]))
        if not refs:
            return np.empty((0, self.dimension), np.float32), np.empty(0, np.bool_)
        values = np.empty((len(refs), self.dimension), np.float32)
        for output_row, (shard, row, _valid) in enumerate(refs):
            values[output_row] = shards.get(int(shard))[int(row)]
        return values, np.asarray([bool(x[2]) for x in refs], dtype=np.bool_)

    def _profile(self, aid: str, source: str) -> tuple[np.ndarray, bool]:
        values, masks = self._vectors(aid, source)
        if len(values) != 1:
            raise ValueError(f"Expected exactly one {source} slot for account {aid}, got {len(values)}")
        return values[0], bool(masks[0])

    def __getitem__(self, index: int) -> dict:
        account = self.accounts[index]
        aid = str(account["account_id"])
        posts, post_mask = self._vectors(aid, "post")
        comments, comment_mask = self._vectors(aid, "comment")
        if not (post_mask.any() or comment_mask.any()):
            raise ValueError(f"Data error: account {aid!r} has neither valid Post nor Comment")
        moltbook, moltbook_mask = self._profile(aid, "moltbook_profile")
        twitter, twitter_mask = self._profile(aid, "twitter_profile")
        urls = list(self.urls.get(aid, []))
        if self.max_urls > 0 and len(urls) > self.max_urls:
            if self.training:
                urls = random.sample(urls, self.max_urls)
            else:
                rng = random.Random(f"{self.seed}:{aid}:url")
                urls = rng.sample(urls, self.max_urls)
        return {
            "account_id": aid, "label": float(account["label"]),
            "posts": posts, "post_mask": post_mask,
            "comments": comments, "comment_mask": comment_mask,
            "moltbook_profile": moltbook, "moltbook_profile_mask": moltbook_mask,
            "twitter_profile": twitter, "twitter_profile_mask": twitter_mask,
            "urls": urls,
        }


def _pad_content(rows: list[dict], key: str, mask_key: str, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max((len(row[key]) for row in rows), default=0)
    values = np.zeros((len(rows), length, dimension), dtype=np.float32)
    masks = np.zeros((len(rows), length), dtype=np.bool_)
    for index, row in enumerate(rows):
        size = len(row[key])
        values[index, :size] = row[key]
        masks[index, :size] = row[mask_key]
    return torch.from_numpy(values), torch.from_numpy(masks)


def collate_accounts(rows: list[dict], url_max_bytes: int = 256) -> dict:
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    dimension = rows[0]["moltbook_profile"].shape[0]
    posts, post_mask = _pad_content(rows, "posts", "post_mask", dimension)
    comments, comment_mask = _pad_content(rows, "comments", "comment_mask", dimension)
    url_count = max((len(row["urls"]) for row in rows), default=0)
    byte_length = max(5, max((len(encode_bytes(url["encoding_text"], url_max_bytes))
                              for row in rows for url in row["urls"]), default=0))
    byte_length = min(byte_length, url_max_bytes)
    url_bytes = np.zeros((len(rows), url_count, byte_length), dtype=np.int64)
    url_features = np.zeros((len(rows), url_count, 18), dtype=np.float32)
    url_source = np.zeros((len(rows), url_count), dtype=np.int64)
    url_mask = np.zeros((len(rows), url_count), dtype=np.bool_)
    url_metadata = []
    for row_index, row in enumerate(rows):
        url_metadata.append(row["urls"])
        for url_index, item in enumerate(row["urls"]):
            encoded = encode_bytes(item["encoding_text"], url_max_bytes)
            url_bytes[row_index, url_index, :len(encoded)] = encoded
            url_features[row_index, url_index] = item["features"]
            url_source[row_index, url_index] = item["source_id"]
            url_mask[row_index, url_index] = True
    return {
        "account_ids": [row["account_id"] for row in rows],
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
        "posts": posts, "post_mask": post_mask,
        "comments": comments, "comment_mask": comment_mask,
        "moltbook_profile": torch.from_numpy(np.stack([row["moltbook_profile"] for row in rows])),
        "moltbook_profile_mask": torch.tensor([row["moltbook_profile_mask"] for row in rows]),
        "twitter_profile": torch.from_numpy(np.stack([row["twitter_profile"] for row in rows])),
        "twitter_profile_mask": torch.tensor([row["twitter_profile_mask"] for row in rows]),
        "url_bytes": torch.from_numpy(url_bytes), "url_features": torch.from_numpy(url_features),
        "url_source": torch.from_numpy(url_source), "url_mask": torch.from_numpy(url_mask),
        "url_metadata": url_metadata,
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}
