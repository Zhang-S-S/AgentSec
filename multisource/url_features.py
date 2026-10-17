from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlsplit

import numpy as np


URL_FEATURE_VERSION = "moltbook-url-lexical-v1"
FEATURE_NAMES = (
    "url_length", "host_length", "path_length", "digit_ratio", "special_ratio",
    "subdomain_depth", "character_entropy", "is_https", "is_ip", "has_punycode",
    "has_abnormal_port", "is_shortener", "has_suspicious_keyword", "query_key_count",
    "path_depth", "log1p_count", "unicode_host", "tld_length",
)
SOURCE_TO_ID = {"post": 0, "comment": 1, "moltbook_profile": 2, "twitter_profile": 3}

# Versioned, corpus-independent dictionaries. They are never fitted on validation/test.
SHORTENER_DOMAINS_V1 = frozenset({
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly", "is.gd",
    "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy",
})
SUSPICIOUS_KEYWORDS_V1 = frozenset({
    "login", "verify", "verification", "wallet", "seed", "password", "token",
    "claim", "bonus", "airdrop", "giveaway", "free", "urgent", "download",
    "invoice", "payment", "crypto", "admin", "security", "update",
})


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _is_ip(host: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


@lru_cache(maxsize=1)
def _psl_extractor():
    import tldextract
    return tldextract.TLDExtract(
        suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
    )


@lru_cache(maxsize=65536)
def _registered_parts(host: str) -> tuple[str, str, int]:
    """Use only tldextract's installed PSL snapshot; online updates are disabled."""
    if _is_ip(host):
        return host, "", 0
    item = _psl_extractor()(host)
    registered = item.top_domain_under_public_suffix or host
    depth = len([x for x in item.subdomain.split(".") if x])
    return registered, item.suffix, depth


@dataclass(frozen=True)
class NormalizedURL:
    raw_url: str
    normalized_url: str
    encoding_text: str
    host: str
    registered_domain: str
    tld: str
    path: str
    query_keys: tuple[str, ...]
    scheme: str
    port: int | None
    unicode_host: bool
    punycode: bool
    abnormal_port: bool


def normalize_url(raw_url: str) -> NormalizedURL:
    """Pure string parsing. This function never dereferences or resolves a URL."""
    raw = str(raw_url).strip()
    if not raw:
        raise ValueError("URL is empty")
    candidate = raw if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) else "https://" + raw.lstrip("/")
    parsed = urlsplit(candidate)
    if not parsed.hostname:
        raise ValueError(f"URL has no host: {raw!r}")
    original_host = parsed.hostname
    unicode_host = any(ord(char) > 127 for char in original_host)
    punycode_before = "xn--" in original_host.casefold()
    try:
        host = original_host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as error:
        raise ValueError(f"Invalid IDNA host: {raw!r}") from error
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"Invalid port: {raw!r}") from error
    scheme = parsed.scheme.lower()
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    normalized_port = None if default_port else port
    abnormal_port = normalized_port is not None
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    netloc = display_host + (f":{normalized_port}" if normalized_port is not None else "")
    path = quote(unquote(parsed.path or "/"), safe="/%:@!$&'()*+,;=-._~")
    query_keys = tuple(sorted(key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)))
    query = "&".join(quote(key, safe="-._~") for key in query_keys)
    normalized = f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")
    # Required byte/character encoder text excludes scheme, fragment, and values.
    encoding_text = netloc + path + (("?" + "&".join(query_keys)) if query_keys else "")
    registered, tld, subdomain_depth = _registered_parts(host)
    del subdomain_depth
    return NormalizedURL(
        raw_url=raw, normalized_url=normalized, encoding_text=encoding_text,
        host=host, registered_domain=registered, tld=tld, path=path,
        query_keys=query_keys, scheme=scheme, port=normalized_port,
        unicode_host=unicode_host, punycode=punycode_before or "xn--" in host,
        abnormal_port=abnormal_port,
    )


def lexical_features(url: NormalizedURL, count: int = 1) -> np.ndarray:
    text = url.encoding_text
    length = max(1, len(text))
    _, _, depth = _registered_parts(url.host)
    lowered = text.casefold()
    keyword = any(word in lowered for word in SUSPICIOUS_KEYWORDS_V1)
    values = [
        len(url.normalized_url), len(url.host), len(url.path),
        sum(char.isdigit() for char in text) / length,
        sum(not char.isalnum() for char in text) / length,
        depth, _entropy(text), float(url.scheme == "https"), float(_is_ip(url.host)),
        float(url.punycode), float(url.abnormal_port),
        float(url.registered_domain in SHORTENER_DOMAINS_V1), float(keyword),
        len(url.query_keys), len([part for part in url.path.split("/") if part]),
        math.log1p(count), float(url.unicode_host), len(url.tld),
    ]
    return np.asarray(values, dtype=np.float32)


def encode_bytes(text: str, max_length: int) -> list[int]:
    """UTF-8 bytes +1 (zero is padding), deterministic head-tail truncation."""
    values = [byte + 1 for byte in text.encode("utf-8")]
    if len(values) > max_length:
        head = (max_length + 1) // 2
        values = values[:head] + values[-(max_length - head):]
    return values


class URLStandardizer:
    def __init__(self, mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> None:
        self.mean = mean
        self.scale = scale

    def fit(self, rows: np.ndarray) -> "URLStandardizer":
        if rows.ndim != 2 or rows.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Unexpected URL feature matrix shape")
        if len(rows) == 0:
            self.mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            self.scale = np.ones(len(FEATURE_NAMES), dtype=np.float32)
        else:
            self.mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = rows.std(axis=0, dtype=np.float64).astype(np.float32)
            self.scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        return self

    def transform(self, rows: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("URLStandardizer has not been fitted")
        return ((rows - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        if self.mean is None or self.scale is None:
            raise RuntimeError("URLStandardizer has not been fitted")
        return {"feature_names": list(FEATURE_NAMES), "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> "URLStandardizer":
        if value.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("URL feature schema mismatch")
        return cls(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))


def _split_hash(splits: dict[str, list[str]]) -> str:
    encoded = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_url_cache(embedding_dir: Path, output_dir: Path, splits: dict[str, list[str]],
                      keep_duplicates: bool = False) -> dict:
    """Scan cached metadata only and write account-indexed, epoch-independent URL features."""
    output_dir.mkdir(parents=True, exist_ok=False)
    membership = {str(aid): split for split, ids in splits.items() for aid in ids}
    aggregated: dict[tuple[str, str, str], dict] = {}
    duplicate_rows: list[dict] = []
    invalid = 0
    for metadata_path in sorted((embedding_dir / "shards").glob("shard-*.jsonl")):
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                aid, source, content_id = str(item["account_id"]), item["source_type"], str(item["content_id"])
                if aid not in membership or source not in SOURCE_TO_ID:
                    continue
                for link in item.get("urls", []):
                    raw = link.get("raw_url")
                    try:
                        normalized = normalize_url(raw)
                    except (ValueError, TypeError):
                        invalid += 1
                        continue
                    base = {
                        "account_id": aid, "source_type": source,
                        "raw_url": normalized.raw_url, "raw_urls": [normalized.raw_url],
                        "normalized_url": normalized.normalized_url,
                        "encoding_text": normalized.encoding_text, "content_ids": [content_id], "count": 1,
                        "normalized": asdict(normalized),
                    }
                    if keep_duplicates:
                        duplicate_rows.append(base)
                    else:
                        key = (aid, normalized.normalized_url, source)
                        if key in aggregated:
                            aggregated[key]["count"] += 1
                            if content_id not in aggregated[key]["content_ids"]:
                                aggregated[key]["content_ids"].append(content_id)
                            if normalized.raw_url not in aggregated[key]["raw_urls"]:
                                aggregated[key]["raw_urls"].append(normalized.raw_url)
                        else:
                            aggregated[key] = base
    rows = duplicate_rows if keep_duplicates else list(aggregated.values())
    rows.sort(key=lambda x: (x["account_id"], x["source_type"], x["normalized_url"], x["content_ids"]))
    raw_features = []
    for row in rows:
        normalized = NormalizedURL(**row.pop("normalized"))
        # Avoid double-counting when duplicate instances are explicitly retained.
        count = 1 if keep_duplicates else int(row["count"])
        features = lexical_features(normalized, count)
        row["features"] = features.tolist()
        row["source_id"] = SOURCE_TO_ID[row["source_type"]]
        raw_features.append(features)
    train_matrix = np.stack([
        feature for row, feature in zip(rows, raw_features)
        if membership[row["account_id"]] == "train"
    ]) if any(membership[row["account_id"]] == "train" for row in rows) else np.empty((0, len(FEATURE_NAMES)), np.float32)
    standardizer = URLStandardizer().fit(train_matrix)
    for row, features in zip(rows, raw_features):
        row["features"] = standardizer.transform(features[None])[0].tolist()
    with (output_dir / "urls.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "version": URL_FEATURE_VERSION,
        "embedding_dir": str(embedding_dir.resolve()),
        "split_sha256": _split_hash(splits),
        "fit_split": "train", "keep_duplicates": keep_duplicates,
        "frequency_feature_enabled": not keep_duplicates,
        "rows": len(rows), "invalid_urls": invalid,
        "source_mapping": SOURCE_TO_ID, "shortener_dictionary": "v1",
        "suspicious_keyword_dictionary": "v1", "psl_policy": "bundled-offline",
        "tldextract_version": importlib.metadata.version("tldextract"),
        "normalization": "lowercase-IDNA-host; no-fragment; sorted-query-names-without-values; default-port-removed",
        "byte_vocab": "padding=0, UTF-8 byte b maps to b+1", "standardizer": standardizer.to_dict(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_url_cache(path: Path) -> tuple[dict[str, list[dict]], dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != URL_FEATURE_VERSION:
        raise ValueError("Incompatible URL cache version")
    grouped: dict[str, list[dict]] = defaultdict(list)
    with (path / "urls.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[str(row["account_id"])].append(row)
    return dict(grouped), manifest
