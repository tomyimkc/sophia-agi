"""Durable, idempotent task queue — so the founder can hand work to Sophia and walk
away. Backed by append-only JSONL (in-memory when path is None). At-least-once with
idempotency: enqueuing the same ``idempotency_key`` returns the same ``task_id`` and
never double-creates. ``next_task`` leases the oldest pending task; ``complete_task``
marks it done. Survives restarts; safe to retry everything.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sophia_contract.errors import ContractError
from sophia_contract.stores import _append_jsonl, _read_jsonl


def _task_id(key: str) -> str:
    return f"task_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}"


class TaskQueue:
    def __init__(self, path: "Path | None" = None, *, clock=None):
        self.path = path
        self.clock = clock or (lambda: "")
        self._tasks: dict = {}      # task_id -> latest state
        self._by_key: dict = {}     # idempotency_key -> task_id
        self._order: list = []      # task_ids in enqueue order
        for row in _read_jsonl(path):
            self._apply(row)

    def _apply(self, row: dict) -> None:
        tid = row["task_id"]
        if tid not in self._tasks:
            self._order.append(tid)
        self._tasks[tid] = {**self._tasks.get(tid, {}), **row}
        if row.get("idempotency_key"):
            self._by_key[row["idempotency_key"]] = tid

    def enqueue(self, request: dict) -> dict:
        """Idempotently add a task. request: {idempotency_key, kind, payload?}."""
        key = request.get("idempotency_key")
        kind = request.get("kind")
        if not isinstance(key, str) or not key.strip():
            raise ContractError("BAD_REQUEST", "idempotency_key is required")
        if not isinstance(kind, str) or not kind.strip():
            raise ContractError("BAD_REQUEST", "kind is required")
        existing = self._by_key.get(key)
        if existing:
            return self.status(existing)
        row = {"task_id": _task_id(key), "idempotency_key": key, "kind": kind,
               "payload": request.get("payload", {}), "state": "pending", "at": self.clock()}
        self._apply(row)
        _append_jsonl(self.path, row)
        return self.status(row["task_id"])

    def next_task(self, *, lease_by: str = "worker") -> "dict | None":
        """Lease the oldest pending task (marks it 'leased'). None if queue empty."""
        for tid in self._order:
            if self._tasks[tid]["state"] == "pending":
                upd = {"task_id": tid, "state": "leased", "leased_by": lease_by, "at": self.clock()}
                self._apply(upd)
                _append_jsonl(self.path, upd)
                return self.status(tid)
        return None

    def complete(self, task_id: str, *, result=None, state: str = "done") -> dict:
        if task_id not in self._tasks:
            raise ContractError("BAD_REQUEST", f"unknown task_id {task_id!r}")
        if state not in ("done", "failed"):
            raise ContractError("BAD_REQUEST", "state must be 'done' or 'failed'")
        upd = {"task_id": task_id, "state": state, "result": result, "at": self.clock()}
        self._apply(upd)
        _append_jsonl(self.path, upd)
        return self.status(task_id)

    def status(self, task_id: str) -> dict:
        t = self._tasks.get(task_id)
        if not t:
            raise ContractError("BAD_REQUEST", f"unknown task_id {task_id!r}")
        return {k: v for k, v in t.items()}

    def pending_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t["state"] == "pending")
