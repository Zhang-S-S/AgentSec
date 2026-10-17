from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

from .common import SOURCES, read_db


class AccountEmbeddings:
    """Read all instances, in source order, with no pooling or instance sampling."""
    def __init__(self, root, cache_shards=4):
        self.root = Path(root)
        manifest = json.loads((self.root / "embedding_manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError("Embedding cache is incomplete")
        self.dimension = manifest["config"]["embedding_dimension"]
        self.accounts = {}
        with (self.root / "accounts.jsonl").open() as f:
            for line in f:
                item = json.loads(line)
                self.accounts[item["account_id"]] = item
        self.db = read_db(self.root / "index.sqlite")
        self.cache, self.cache_shards = OrderedDict(), cache_shards

    def close(self):
        self.db.close()
        self.cache.clear()

    def _shard(self, index):
        import numpy as np
        if index not in self.cache:
            stem = self.root / "shards" / f"shard-{index:06d}"
            with open(str(stem) + ".jsonl", encoding="utf-8") as f:
                metadata = [json.loads(line) for line in f]
            self.cache[index] = np.load(str(stem) + ".npy", mmap_mode="r"), metadata
            if len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, account_id):
        import numpy as np
        result = dict(self.accounts[account_id])
        for kind in SOURCES:
            refs = self.db.execute("SELECT shard,row_index FROM instances WHERE account_id=? AND source_type=? ORDER BY record_id",
                                   (account_id, kind)).fetchall()
            matrix = np.empty((len(refs), self.dimension), dtype=np.float32)
            metadata = []
            for index, (shard, row) in enumerate(refs):
                vectors, items = self._shard(shard)
                matrix[index] = vectors[row]
                metadata.append(items[row])
            masks = np.asarray([item["valid_mask"] for item in metadata], dtype=np.bool_)
            if kind.endswith("profile"):
                if len(refs) != 1:
                    raise ValueError(f"Expected one {kind} slot for {account_id}")
                result[kind + "_embedding"] = matrix[0]
                result[kind + "_mask"] = int(masks[0])
                result[kind + "_metadata"] = metadata[0]
            else:
                result[kind + "_embeddings"] = matrix
                result[kind + "_mask"] = masks
                result[kind + "_metadata"] = metadata
        return result
