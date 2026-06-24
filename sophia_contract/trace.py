# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Structured tracing in a Langfuse-compatible shape.

Each service call emits one trace event with the fields a Langfuse ingestor expects
(``id, name, startTime, endTime, input, output, metadata, level``), appended to
``traces.jsonl`` (in-memory when path is None). This gives the solo founder an audit
trail and a drop-in feed for Langfuse without a network dependency at runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sophia_contract.stores import _append_jsonl, _read_jsonl


def _trace_id(seed: str) -> str:
    return f"trace_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


class Tracer:
    def __init__(self, path: "Path | None" = None, *, clock=None, enabled: bool = True):
        self.path = path
        self.clock = clock or (lambda: "")
        self.enabled = enabled
        self._events: list = list(_read_jsonl(path))

    def span(self, name: str, *, input, output, level: str = "DEFAULT",
             metadata: "dict | None" = None) -> "dict | None":
        """Record one Langfuse-style span. ``level`` ∈ {DEFAULT, WARNING, ERROR}."""
        if not self.enabled:
            return None
        ts = self.clock()
        seed = f"{name}:{ts}:{json.dumps(input, sort_keys=True, default=str)}"
        event = {
            "id": _trace_id(seed),
            "name": name,
            "startTime": ts,
            "endTime": ts,
            "input": input,
            "output": output,
            "level": level,
            "metadata": metadata or {},
        }
        self._events.append(event)
        _append_jsonl(self.path, event)
        return event

    def events(self) -> "list[dict]":
        return list(self._events)
