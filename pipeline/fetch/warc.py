# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Minimal WARC ingest (Phase 4).

CommonCrawl ships the open web as WARC archives. Ingesting WARC lets the acquisition loop run
at scale against an archived crawl — the practical way to demonstrate TB-scale processing
without crawling the live web. This is a dependency-free reader for the subset that matters
for pretraining: ``response`` records, from which we recover the target URI, content-type, and
HTML/text body.

Records are length-delimited (``Content-Length``), parsed over bytes for correctness and
decoded as UTF-8 (errors replaced). ``read_warc`` is gzip-aware (``.warc.gz``). Not a full
WARC 1.1 implementation — it skips request/metadata records and any record without a length.
"""

from __future__ import annotations

import gzip
from pathlib import Path

_CRLF = b"\r\n"
_HEADER_SEP = b"\r\n\r\n"


def _parse_headers(block: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in block.split(_CRLF):
        if b":" in line:
            k, _, v = line.partition(b":")
            headers[k.decode("latin-1").strip().lower()] = v.decode("latin-1").strip()
    return headers


def iter_warc_records(data: bytes):
    """Yield ``{warc_type, target_uri, content_type, content}`` for each WARC record in ``data``."""
    pos = 0
    n = len(data)
    while pos < n:
        # A record begins with a "WARC/<ver>" line; find the WARC header block.
        start = data.find(b"WARC/", pos)
        if start == -1:
            break
        sep = data.find(_HEADER_SEP, start)
        if sep == -1:
            break
        warc_headers = _parse_headers(data[start:sep])
        body_start = sep + len(_HEADER_SEP)
        try:
            length = int(warc_headers.get("content-length", "0"))
        except ValueError:
            length = 0
        block = data[body_start : body_start + length]
        pos = body_start + length

        record = {
            "warc_type": warc_headers.get("warc-type", ""),
            "target_uri": warc_headers.get("warc-target-uri", ""),
            "content_type": "",
            "content": "",
        }
        if record["warc_type"] == "response":
            http_sep = block.find(_HEADER_SEP)
            if http_sep != -1:
                http_headers = _parse_headers(block[:http_sep])
                payload = block[http_sep + len(_HEADER_SEP) :]
                record["content_type"] = http_headers.get("content-type", "")
            else:
                payload = block
            record["content"] = payload.decode("utf-8", errors="replace")
        else:
            record["content"] = block.decode("utf-8", errors="replace")
        yield record


def read_warc(path: str | Path):
    """Read a WARC file (gzip-aware by extension) and iterate its records."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            data = fh.read()
    else:
        data = path.read_bytes()
    return iter_warc_records(data)


def records_to_documents(records, *, html_only: bool = True):
    """Convert ``response`` WARC records into pipeline document dicts.

    ``html_only`` keeps only text/html responses (the common pretraining filter). Each doc gets
    ``url``, ``mime``, and ``content`` — ready for the clean → dedup → score pipeline.
    """
    for rec in records:
        if rec.get("warc_type") != "response":
            continue
        ctype = rec.get("content_type", "")
        if html_only and "html" not in ctype.lower():
            continue
        uri = rec.get("target_uri")
        if not uri:
            continue
        yield {"url": uri, "mime": ctype or "text/html", "content": rec.get("content", "")}


__all__ = ["iter_warc_records", "read_warc", "records_to_documents"]
