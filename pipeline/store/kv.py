# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Key/value seen-set adapter (Phase 5).

At scale, the crawler needs a persistent "have I seen this URL / this content fingerprint?"
set that doesn't fit in RAM — the JD's 键值数据库 role. ``SeenSet`` is the interface;
``SqliteSeenSet`` is a durable local default (sqlite, single file), ``MemorySeenSet`` a
volatile one for tests. A production RocksDB/Redis adapter implements the same three methods.

``add`` returns whether the key was *newly* added, so dedup ("admit only if unseen") is a
single atomic call.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SeenSet(Protocol):
    def add(self, key: str) -> bool: pass
    def contains(self, key: str) -> bool: pass
    def __len__(self) -> int: pass


class MemorySeenSet:
    """In-memory seen-set (volatile; for tests and small runs)."""

    def __init__(self):
        self._s: set[str] = set()

    def add(self, key: str) -> bool:
        if key in self._s:
            return False
        self._s.add(key)
        return True

    def contains(self, key: str) -> bool:
        return key in self._s

    def __len__(self) -> int:
        return len(self._s)


class SqliteSeenSet:
    """Durable seen-set backed by sqlite (one file). Safe across process restarts."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY)")
        self._conn.commit()

    def add(self, key: str) -> bool:
        cur = self._conn.execute("INSERT OR IGNORE INTO seen(k) VALUES (?)", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def contains(self, key: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM seen WHERE k = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


__all__ = ["SeenSet", "MemorySeenSet", "SqliteSeenSet"]
