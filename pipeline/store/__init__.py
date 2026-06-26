# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data infrastructure adapters (Phase 5): object store, KV seen-set, work queue.

The JD's 数据基建 — the storage/coordination layer the acquisition + cleaning pipeline runs
on. Each adapter is defined as a small Protocol with a **local default** (filesystem object
store, SQLite/in-memory KV, file-backed queue) so the whole pipeline is testable and runnable
offline, plus a documented seam to swap in production backends (S3/MinIO, RocksDB/Redis,
Redis Streams/NATS) without touching pipeline code.

This keeps Phase 5 honest: the *interfaces* and a working local implementation ship and are
tested here; the cloud backends are adapters you wire up when running on real infra (see
``docs/06-Roadmap/data-engineering-plan.md`` and the scale runbook).
"""

from __future__ import annotations

from pipeline.store.kv import MemorySeenSet, SeenSet, SqliteSeenSet
from pipeline.store.objectstore import LocalObjectStore, ObjectStore
from pipeline.store.queue import FileQueue, WorkQueue
from pipeline.store.redis_backends import RedisListQueue, RedisSeenSet
from pipeline.store.s3 import S3ObjectStore

__all__ = [
    # interfaces
    "ObjectStore", "SeenSet", "WorkQueue",
    # local defaults
    "LocalObjectStore", "SqliteSeenSet", "MemorySeenSet", "FileQueue",
    # cloud backends
    "S3ObjectStore", "RedisSeenSet", "RedisListQueue",
]
