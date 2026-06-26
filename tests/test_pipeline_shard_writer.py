#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.shard_writer (Phase 5).

Verifies that documents are split into fixed-size shards, each shard + sibling manifest is
written to the object store, a catalog is emitted with correct totals, and the shards
round-trip back. Stdlib-only. Offline.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.shard_writer import write_sharded  # noqa: E402
from pipeline.store.objectstore import LocalObjectStore  # noqa: E402


def _docs(n):
    return [{"url": f"https://a.com/{i}", "content": f"doc number {i} " * 5,
             "quality": {"score": 0.8, "keep": True}} for i in range(n)]


def test_sharding_splits_and_catalogs():
    with tempfile.TemporaryDirectory() as td:
        store = LocalObjectStore(td)
        catalog = write_sharded(_docs(25), store, prefix="corpus", shard_size=10)
        assert catalog["shardCount"] == 3  # 10 + 10 + 5
        assert catalog["totalRows"] == 25
        # Each shard + manifest present, plus the catalog.
        listing = store.list("corpus/")
        assert "corpus/part-00000.jsonl" in listing
        assert "corpus/part-00000.manifest.json" in listing
        assert "corpus/_catalog.json" in listing


def test_shard_roundtrip_and_catalog_object():
    with tempfile.TemporaryDirectory() as td:
        store = LocalObjectStore(td)
        write_sharded(_docs(15), store, prefix="c", shard_size=10)
        first = store.get_shard("c/part-00000.jsonl")
        assert len(first) == 10
        second = store.get_shard("c/part-00001.jsonl")
        assert len(second) == 5
        catalog = json.loads(store.get("c/_catalog.json"))
        assert sum(s["rowCount"] for s in catalog["shards"]) == 15


def test_empty_input():
    with tempfile.TemporaryDirectory() as td:
        store = LocalObjectStore(td)
        catalog = write_sharded([], store, prefix="c", shard_size=10)
        assert catalog["shardCount"] == 0
        assert catalog["totalRows"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.shard_writer tests passed")
