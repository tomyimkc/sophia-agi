# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shard I/O (Phase 3): JSONL always, Parquet when pyarrow is present.

Pretraining corpora are stored as columnar shards (Parquet) for cheap analytics; this module
writes/reads shards in either format, picked by file extension. Parquet support is **optional**
— pyarrow is lazily imported, and if it is unavailable the writer falls back to JSONL (with a
note) so the pipeline still runs under a minimal/airgapped environment. JSONL stays the
deterministic, dependency-free interchange format.
"""

from __future__ import annotations

from pathlib import Path

from pipeline import document as _doc


def parquet_available() -> bool:
    """True iff pyarrow can be imported (no network, safe to call anywhere)."""
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False
    return True


def _flatten_for_columns(doc: dict) -> dict:
    """Project a document to a flat, Parquet-friendly row (nested blocks JSON-encoded)."""
    import json

    return {
        "url": doc.get("url"),
        "canonical_url": doc.get("canonical_url"),
        "lang": doc.get("lang"),
        "mime": doc.get("mime"),
        "content": doc.get("content"),
        "quality_score": (doc.get("quality") or {}).get("score"),
        "quality_keep": (doc.get("quality") or {}).get("keep"),
        "is_duplicate": (doc.get("dedup") or {}).get("is_duplicate"),
        "sim_cluster": (doc.get("dedup") or {}).get("sim_cluster"),
        "provenance": json.dumps(doc.get("provenance", {}), ensure_ascii=False, sort_keys=True),
    }


def write_shard(path: str | Path, docs) -> int:
    """Write ``docs`` to ``path``. ``.parquet`` -> Parquet (if pyarrow), else JSONL.

    Returns the row count. Falls back to JSONL (same stem, ``.jsonl``) if a Parquet write is
    requested without pyarrow.
    """
    path = Path(path)
    docs = list(docs)
    if path.suffix == ".parquet":
        if not parquet_available():
            fallback = path.with_suffix(".jsonl")
            print(f"[io] pyarrow unavailable; writing JSONL -> {fallback}")
            return _doc.write_jsonl(fallback, docs)
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [_flatten_for_columns(d) for d in docs]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return len(rows)
    return _doc.write_jsonl(path, docs)


def read_shard(path: str | Path):
    """Read a shard as a list of dicts. ``.parquet`` -> Parquet (requires pyarrow), else JSONL."""
    path = Path(path)
    if path.suffix == ".parquet":
        if not parquet_available():
            raise RuntimeError(f"reading {path} requires pyarrow")
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return list(_doc.read_jsonl(path))


__all__ = ["parquet_available", "write_shard", "read_shard"]
