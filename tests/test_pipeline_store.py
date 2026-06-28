#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.store (Phase 5) — object store, KV seen-set, work queue.

Verifies LocalObjectStore put/get/exists/list + shard round-trip + root-escape guard;
SqliteSeenSet/MemorySeenSet add-returns-newness, persistence, and dedup semantics; and
FileQueue FIFO ordering, durability across reopen (restart), and compaction. Stdlib-only
(sqlite3 is in the stdlib). Offline, no deps.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.store.kv import MemorySeenSet, SqliteSeenSet  # noqa: E402
from pipeline.store.objectstore import LocalObjectStore, ObjectStore  # noqa: E402
from pipeline.store.queue import FileQueue, WorkQueue  # noqa: E402


# ---------------------------- object store --------------------------------- #

def test_object_store_put_get_list():
    with tempfile.TemporaryDirectory() as td:
        store = LocalObjectStore(td)
        assert isinstance(store, ObjectStore)  # satisfies the Protocol
        store.put("shards/a.bin", b"hello")
        assert store.exists("shards/a.bin")
        assert store.get("shards/a.bin") == b"hello"
        store.put("shards/b.bin", b"world")
        assert store.list("shards/") == ["shards/a.bin", "shards/b.bin"]


def test_object_store_shard_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        store = LocalObjectStore(td)
        docs = [{"url": "u1", "content": "c1"}, {"url": "u2", "content": "c2"}]
        assert store.put_shard("part-0001.jsonl", docs) == 2
        assert store.get_shard("part-0001.jsonl") == docs


def test_object_store_rejects_escape():
    with tempfile.TemporaryDirectory() as td:
        store = LocalObjectStore(td)
        try:
            store.put("../escape.bin", b"x")
        except ValueError:
            return
        raise AssertionError("expected ValueError on root escape")


# ------------------------------ seen-set ----------------------------------- #

def test_memory_seen_set_newness():
    s = MemorySeenSet()
    assert s.add("http://a") is True
    assert s.add("http://a") is False  # already seen
    assert s.contains("http://a") is True
    assert len(s) == 1


def test_sqlite_seen_set_persists():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "seen.db"
        s1 = SqliteSeenSet(p)
        assert s1.add("k1") is True
        assert s1.add("k1") is False
        s1.close()
        # Reopen -> still remembers (durable across "restart").
        s2 = SqliteSeenSet(p)
        assert s2.contains("k1") is True
        assert s2.add("k1") is False
        assert len(s2) == 1
        s2.close()


# ------------------------------- queue ------------------------------------- #

def test_file_queue_fifo():
    with tempfile.TemporaryDirectory() as td:
        q = FileQueue(Path(td) / "q.jsonl")
        assert isinstance(q, WorkQueue)
        q.push({"u": 1})
        q.push({"u": 2})
        assert len(q) == 2
        assert q.pop() == {"u": 1}
        assert q.pop() == {"u": 2}
        assert q.pop() is None
        assert len(q) == 0


def test_file_queue_durable_across_reopen():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "q.jsonl"
        q1 = FileQueue(path)
        q1.push({"u": 1})
        q1.push({"u": 2})
        assert q1.pop() == {"u": 1}
        # Simulate restart: a fresh FileQueue on the same path resumes after the cursor.
        q2 = FileQueue(path)
        assert len(q2) == 1
        assert q2.pop() == {"u": 2}


def test_file_queue_skips_poison_line():
    # A torn write / corrupt line must not make pop() loop forever.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "q.jsonl"
        q = FileQueue(path)
        q.push({"u": 1})
        # Append a corrupt line directly (simulating a crash mid-write).
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
        q.push({"u": 2})
        assert q.pop() == {"u": 1}
        assert q.pop() == {"u": 2}  # poison line skipped, not re-popped forever
        assert q.pop() is None


def test_file_queue_compact():
    with tempfile.TemporaryDirectory() as td:
        q = FileQueue(Path(td) / "q.jsonl")
        q.push({"u": 1})
        q.push({"u": 2})
        q.pop()
        q.compact()
        assert len(q) == 1
        assert q.pop() == {"u": 2}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.store tests passed")
