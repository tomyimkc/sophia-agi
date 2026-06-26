#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the cloud store adapters (S3 + Redis) via injected in-memory fakes.

boto3/redis are not installed in CI, but the adapters take an injected client — so a fake
client backed by a dict/set/list exercises all the adapter logic (key mapping, prefix
stripping, listing, atomic add, FIFO) and confirms each adapter satisfies the same Protocol as
its local counterpart. No network, no cloud SDK. Offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.store.kv import SeenSet  # noqa: E402
from pipeline.store.objectstore import ObjectStore  # noqa: E402
from pipeline.store.queue import WorkQueue  # noqa: E402
from pipeline.store.redis_backends import RedisListQueue, RedisSeenSet  # noqa: E402
from pipeline.store.s3 import S3ObjectStore  # noqa: E402


class FakeS3:
    def __init__(self):
        self.store: dict = {}

    def put_object(self, Bucket, Key, Body):
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": self.store[(Bucket, Key)]}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise KeyError(Key)
        return {}

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for (b, k) in self.store if b == Bucket and k.startswith(Prefix)]}


class FakeRedis:
    def __init__(self):
        self.sets: dict = {}
        self.lists: dict = {}

    def sadd(self, key, member):
        s = self.sets.setdefault(key, set())
        if member in s:
            return 0
        s.add(member)
        return 1

    def sismember(self, key, member):
        return member in self.sets.get(key, set())

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def lpush(self, key, val):
        self.lists.setdefault(key, []).insert(0, val)

    def rpop(self, key):
        lst = self.lists.get(key, [])
        return lst.pop() if lst else None

    def llen(self, key):
        return len(self.lists.get(key, []))


# --------------------------------- S3 -------------------------------------- #

def test_s3_object_store_conforms_and_roundtrips():
    store = S3ObjectStore("bucket", FakeS3(), prefix="lake")
    assert isinstance(store, ObjectStore)
    store.put("a/x.bin", b"hello")
    assert store.exists("a/x.bin")
    assert not store.exists("a/missing")
    assert store.get("a/x.bin") == b"hello"
    store.put("a/y.bin", b"world")
    assert store.list("a/") == ["a/x.bin", "a/y.bin"]  # prefix stripped back to logical keys


def test_s3_shard_roundtrip():
    store = S3ObjectStore("bucket", FakeS3())
    docs = [{"url": "u1"}, {"url": "u2"}]
    assert store.put_shard("part-0.jsonl", docs) == 2
    assert store.get_shard("part-0.jsonl") == docs


# ------------------------------- Redis ------------------------------------- #

def test_redis_seen_set_conforms_and_dedups():
    s = RedisSeenSet(FakeRedis())
    assert isinstance(s, SeenSet)
    assert s.add("http://a") is True
    assert s.add("http://a") is False
    assert s.contains("http://a") is True
    assert len(s) == 1


def test_redis_list_queue_fifo():
    q = RedisListQueue(FakeRedis())
    assert isinstance(q, WorkQueue)
    q.push({"u": 1})
    q.push({"u": 2})
    assert len(q) == 2
    assert q.pop() == {"u": 1}
    assert q.pop() == {"u": 2}
    assert q.pop() is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all cloud-adapter tests passed")
