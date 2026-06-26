# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Work queue adapter (Phase 5).

The pipeline decouples stages (fetch → clean → dedup → score → shard) through a queue — the
JD's 消息队列 role. ``WorkQueue`` is the interface; ``FileQueue`` is a durable local default
(append-only JSONL + a persisted cursor, survives restarts, at-least-once). A production
adapter implements the same methods over Redis Streams / NATS / Kafka without changing callers.

VISION.md flags the current single-process JSONL queue as a known limitation; this gives the
pipeline a clean seam to grow past it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkQueue(Protocol):
    def push(self, item: dict) -> None: ...
    def pop(self) -> dict | None: ...
    def __len__(self) -> int: ...


class FileQueue:
    """Durable FIFO queue: append-only JSONL log + a cursor file of consumed-line count.

    Restart-safe (the cursor persists how many items were consumed). ``pop`` advances the
    cursor only after the item is read, so a crash between read and process re-delivers the
    item (at-least-once) rather than dropping it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cursor_path = self.path.with_suffix(self.path.suffix + ".cursor")
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        if not self._cursor_path.exists():
            self._cursor_path.write_text("0", encoding="utf-8")

    def _consumed(self) -> int:
        try:
            return int(self._cursor_path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            return 0

    def _set_consumed(self, n: int) -> None:
        self._cursor_path.write_text(str(n), encoding="utf-8")

    def _lines(self) -> list[str]:
        return [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def push(self, item: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    def pop(self) -> dict | None:
        lines = self._lines()
        consumed = self._consumed()
        if consumed >= len(lines):
            return None
        item = json.loads(lines[consumed])
        self._set_consumed(consumed + 1)
        return item

    def __len__(self) -> int:
        return max(0, len(self._lines()) - self._consumed())

    def compact(self) -> None:
        """Drop already-consumed lines to bound the log size; resets the cursor to 0."""
        remaining = self._lines()[self._consumed() :]
        self.path.write_text("".join(ln + "\n" for ln in remaining), encoding="utf-8")
        self._set_consumed(0)


__all__ = ["WorkQueue", "FileQueue"]
