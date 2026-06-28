# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sharded corpus writer + catalog (Phase 5).

Batches processed documents into fixed-size shards, writes each to an ``ObjectStore`` (local
filesystem by default; S3/MinIO via the same Protocol), stamps each shard with a
``pipeline.manifest`` entry, and assembles a **catalog** — the index a training job reads to
discover shards. This is the data-lake write path: PB-scale corpora are just many catalogued
shards.
"""

from __future__ import annotations

import json

from pipeline import manifest as _man


def write_sharded(
    docs,
    store,
    *,
    prefix: str = "corpus",
    shard_size: int = 1000,
    pre_dedup_count: int | None = None,
) -> dict:
    """Write ``docs`` to ``store`` as ``prefix/part-NNNNN.jsonl`` shards + a catalog.

    Returns the catalog dict: ``{prefix, shardCount, totalRows, shards: [manifest, ...]}``.
    Each shard also gets a sibling ``.manifest.json`` object. The catalog is stored at
    ``prefix/_catalog.json``.
    """
    docs = list(docs)
    shards: list[dict] = []
    total = 0
    for idx in range(0, len(docs), shard_size):
        batch = docs[idx : idx + shard_size]
        part = f"{prefix}/part-{idx // shard_size:05d}.jsonl"
        store.put_shard(part, batch)
        m = _man.build_manifest(batch, shard_path=part)
        store.put(part.replace(".jsonl", ".manifest.json"),
                  json.dumps(m, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
        shards.append(m)
        total += len(batch)

    catalog = {
        "prefix": prefix,
        "shardCount": len(shards),
        "totalRows": total,
        "preDedupCount": pre_dedup_count,
        "shards": shards,
    }
    store.put(f"{prefix}/_catalog.json",
              json.dumps(catalog, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    return catalog


__all__ = ["write_sharded"]
