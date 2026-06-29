# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Object store adapter (Phase 5).

The data lake stores corpus shards as objects. ``ObjectStore`` is the minimal interface the
pipeline needs; ``LocalObjectStore`` implements it on the filesystem so everything runs
offline. A production S3/MinIO adapter implements the same four methods over boto3 — pipeline
code (shard writers, catalog) depends only on the Protocol, never the backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...


class LocalObjectStore:
    """Filesystem-backed object store rooted at ``root`` (keys are relative paths)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are POSIX-style; guard against escaping the root.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes object store root: {key!r}")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return out

    # ---- shard conveniences (JSONL bytes) ---- #

    def put_shard(self, key: str, docs) -> int:
        """Serialize ``docs`` as JSONL and store under ``key``; returns the row count."""
        lines = [json.dumps(d, ensure_ascii=False, sort_keys=True) for d in docs]
        self.put(key, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        return len(lines)

    def get_shard(self, key: str) -> list[dict]:
        """Load a JSONL shard stored under ``key``."""
        text = self.get(key).decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]


__all__ = ["ObjectStore", "LocalObjectStore"]
