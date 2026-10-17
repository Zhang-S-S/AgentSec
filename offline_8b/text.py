from __future__ import annotations

import hashlib
import html
import ipaddress
import re
import unicodedata
from html.parser import HTMLParser
from urllib.parse import urlsplit

from .common import digest

# Includes schemeless links; no URL is ever dereferenced. Keep raw matches separately.
URL = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\"'\u3000]+|(?<![\w@])(?:[\w-]+\.)+[a-z]{2,63}(?::\d+)?(?:[/?#][^\s<>\"']*)?")
EMAIL = re.compile(r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
MENTION = re.compile(r"(?<!\w)@[\w.-]+", re.UNICODE)
UUID = re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")


class TextHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links = [], []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if self.skip:
            return
        if tag in {"url", "user", "email"}:
            self.parts.append(f"<{tag.upper()}>")
        elif tag in {"p", "div", "br", "li", "section", "h1", "h2", "h3"}:
            self.parts.append("\n")
        for key, value in attrs:
            if key in {"href", "src"} and value and not value.startswith(("#", "javascript:", "data:")):
                if value.startswith("mailto:"):
                    self.parts.append(" <EMAIL> ")
                else:
                    self.links.append(value)
                    self.parts.append(" <URL> ")
            elif key == "alt" and value:
                self.parts.append(value)

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
        elif tag in {"p", "div", "li", "section"} and not self.skip:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def normalize_chars(text):
    text = unicodedata.normalize("NFKC", html.unescape(text))
    # ZWJ/ZWNJ and emoji tag sequences have real linguistic/emoji semantics.
    return "".join(c for c in text if c in "\n\t\r\u200c\u200d"
                   or 0xE0020 <= ord(c) <= 0xE007F
                   or unicodedata.category(c) not in {"Cc", "Cf", "Cs"})


class Cleaner:
    def __init__(self, domain_extractor=None):
        if domain_extractor is None:
            import tldextract
            domain_extractor = tldextract.TLDExtract(
                suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
        self.domains = domain_extractor

    def url_info(self, raw):
        candidate = normalize_chars(raw).strip()
        try:
            p = urlsplit(candidate if "://" in candidate else "https://" + candidate.lstrip("/"))
            host = (p.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
            if not host:
                raise ValueError("missing hostname")
            try:
                ipaddress.ip_address(host)
                domain = host
            except ValueError:
                extracted = self.domains(host)
                domain = extracted.top_domain_under_public_suffix or host
            netloc = f"[{host}]" if ":" in host else host
            # Origin-only canonicalization deliberately drops ALL paths as well as
            # userinfo/query/fragment: path segments can contain invite IDs/tokens.
            normalized = f"{p.scheme.lower()}://{netloc}/"
            return {"raw_url": raw, "normalized_url": normalized,
                    "main_domain": domain, "valid": True}
        except (ValueError, UnicodeError):
            return {"raw_url": raw, "normalized_url": None, "main_domain": None, "valid": False}

    def clean(self, raw, identities=(), external_urls=()):
        # Capture URL spellings before NFKC/HTML decoding for auditability.
        original_urls = [m.group() for m in URL.finditer(EMAIL.sub("<EMAIL>", raw))]
        text = normalize_chars(raw)
        parser = TextHTML()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
        text = EMAIL.sub("<EMAIL>", text)
        urls = []
        def replace(match):
            candidate = match.group()
            stripped = candidate.rstrip(".,!?;:，。！？；：)]}）】")
            suffix = candidate[len(stripped):]
            urls.append(stripped)
            return "<URL>" + suffix
        text = URL.sub(replace, text)
        urls.extend(parser.links)
        for url in external_urls:
            if url and url.strip():
                urls.append(url)
                text += "\n<URL>"
        text = MENTION.sub(lambda m: "<USER>" + ("." if m.group().endswith(".") else ""), text)
        text = UUID.sub("<USER>", text)
        for identity in identities:
            if identity:
                text = re.sub(r"(?<!\w)" + re.escape(normalize_chars(str(identity))) + r"(?!\w)",
                              "<USER>", text, flags=re.IGNORECASE)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        # Pair normalized matches with pre-cleaning spelling where possible.
        spellings = {}
        for url in original_urls:
            stripped = url.rstrip(".,!?;:，。！？；：)]}）】")
            spellings.setdefault(normalize_chars(stripped), []).append(stripped)
        records = []
        for url in urls:
            options = spellings.get(normalize_chars(url), [])
            records.append(self.url_info(options.pop(0) if options else url))
        return {"text": text, "text_hash": digest(text), "raw_text_hash": digest(raw),
                "urls": records, "has_url": bool(records), "url_count": len(records),
                "is_empty": not bool(text), "valid_mask": int(bool(text)),
                "language": None, "language_confidence": None}


def simhash(text):
    """64-bit character-trigram SimHash, preserving case and all scripts."""
    if not text:
        return None
    features = {text[i:i + 3] for i in range(max(1, len(text) - 2))}
    import numpy as np
    hashes = b"".join(hashlib.blake2b(part.encode(), digest_size=8).digest() for part in features)
    bits = np.unpackbits(np.frombuffer(hashes, dtype=np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    majority = bits.sum(axis=0) * 2 >= len(features)
    return int.from_bytes(np.packbits(majority, bitorder="little").tobytes(), "little")
