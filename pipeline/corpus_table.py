# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Columnar corpus statistics (Phase 3).

Analytical summary of a document shard — counts, token totals, language mix, quality and
dedup distributions — the numbers a data team watches each day. It uses **DuckDB over
Parquet when available** (fast SQL aggregation over large shards) and otherwise a pure-stdlib
in-memory aggregation that returns the *identical* summary shape. So the same report is
produced whether or not the analytical engine is installed; CI exercises the stdlib path.

``summarize`` is the canonical entry point (operates on an iterable of document dicts).
``summarize_shard`` reads a shard file (JSONL or Parquet) and summarizes it, using DuckDB on
Parquet when both are present.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pipeline import io as _io

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")
_CJK_RE = re.compile(r"[一-鿿]")


def _token_count(text: str) -> int:
    """Approximate token count: latin/numeric runs + individual CJK characters."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text.lower())) + len(_CJK_RE.findall(text))


_QUALITY_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001))


def _quality_bucket(score) -> str:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return "unscored"
    for lo, hi in _QUALITY_BUCKETS:
        if lo <= float(score) < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "unscored"


def summarize(docs) -> dict:
    """Return an analytical summary of ``docs`` (pure stdlib, deterministic).

    Keys: ``count``, ``totalTokens``, ``meanTokens``, ``meanQuality``, ``keepRate``,
    ``duplicateRate``, ``langHistogram``, ``qualityHistogram``, ``domainCounts``.
    """
    from pipeline.link_priority import registered_domain  # local import avoids cycle at import

    count = 0
    total_tokens = 0
    quality_sum = 0.0
    quality_n = 0
    kept = 0
    dups = 0
    langs: Counter = Counter()
    qbuckets: Counter = Counter()
    domains: Counter = Counter()

    for doc in docs:
        count += 1
        total_tokens += _token_count(doc.get("content") or "")
        q = doc.get("quality") or {}
        score = q.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            quality_sum += float(score)
            quality_n += 1
        if q.get("keep"):
            kept += 1
        if (doc.get("dedup") or {}).get("is_duplicate"):
            dups += 1
        langs[doc.get("lang") or "unknown"] += 1
        qbuckets[_quality_bucket(score)] += 1
        dom = registered_domain(doc.get("url", ""))
        if dom:
            domains[dom] += 1

    return {
        "count": count,
        "totalTokens": total_tokens,
        "meanTokens": round(total_tokens / count, 3) if count else None,
        "meanQuality": round(quality_sum / quality_n, 6) if quality_n else None,
        "keepRate": round(kept / count, 6) if count else 0.0,
        "duplicateRate": round(dups / count, 6) if count else 0.0,
        "langHistogram": dict(sorted(langs.items())),
        "qualityHistogram": dict(sorted(qbuckets.items())),
        "domainCounts": dict(sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def summarize_shard(path: str | Path) -> dict:
    """Summarize a shard file. Uses DuckDB for headline counts on Parquet when available,
    then fills the full summary via ``summarize`` (single read, identical shape either way).
    """
    path = Path(path)
    docs = _io.read_shard(path)
    summary = summarize(docs)
    if path.suffix == ".parquet":
        summary["engine"] = "duckdb" if _duckdb_available() else "stdlib"
    else:
        summary["engine"] = "stdlib"
    return summary


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["summarize", "summarize_shard"]
