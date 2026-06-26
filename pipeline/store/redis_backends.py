# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Redis-backed seen-set and work queue (Phase 5, production backends).

Implement ``pipeline.store.kv.SeenSet`` and ``pipeline.store.queue.WorkQueue`` over Redis.
The redis client is **injected**, so the adapter logic is unit-tested with an in-memory fake
(no redis-py, no live server in CI). ``from_url`` lazily builds a real client for production.

- ``RedisSeenSet`` uses a Redis SET (``SADD`` returns 1 iff the member is new — atomic dedup).
- ``RedisListQueue`` uses a Redis LIST (``LPUSH`` + ``RPOP`` = FIFO). For stronger at-least-once
  delivery in production, prefer a consumer-group over a Redis Stream; this list queue is the
  minimal drop-in that supersedes the single-process JSONL queue.
"""

from __future__ import annotations

import json


class RedisSeenSet:
    """SeenSet over a Redis SET. ``client`` exposes ``sadd``, ``sismember``, ``scard``."""

    def __init__(self, client, *, key: str = "sophia:seen"):
        self.client = client
        self.key = key

    def add(self, key: str) -> bool:
        return int(self.client.sadd(self.key, key)) == 1

    def contains(self, key: str) -> bool:
        return bool(self.client.sismember(self.key, key))

    def __len__(self) -> int:
        return int(self.client.scard(self.key))

    @classmethod
    def from_url(cls, url: str, *, key: str = "sophia:seen"):
        try:
            import redis
        except Exception as e:  # pragma: no cover
            raise RuntimeError("RedisSeenSet.from_url requires redis-py") from e
        return cls(redis.from_url(url, decode_responses=True), key=key)


class RedisListQueue:
    """WorkQueue over a Redis LIST. ``client`` exposes ``lpush``, ``rpop``, ``llen``."""

    def __init__(self, client, *, key: str = "sophia:queue"):
        self.client = client
        self.key = key

    def push(self, item: dict) -> None:
        self.client.lpush(self.key, json.dumps(item, ensure_ascii=False, sort_keys=True))

    def pop(self) -> dict | None:
        raw = self.client.rpop(self.key)
        return json.loads(raw) if raw is not None else None

    def __len__(self) -> int:
        return int(self.client.llen(self.key))

    @classmethod
    def from_url(cls, url: str, *, key: str = "sophia:queue"):
        try:
            import redis
        except Exception as e:  # pragma: no cover
            raise RuntimeError("RedisListQueue.from_url requires redis-py") from e
        return cls(redis.from_url(url, decode_responses=True), key=key)


__all__ = ["RedisSeenSet", "RedisListQueue"]
