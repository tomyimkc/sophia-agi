"""Append-only audit log for declassification (#3).

Every declassification (and every denied attempt) is hash-chained: each entry's
hash covers its content AND the previous entry's hash. ``verify()`` alone catches
**in-place edits, reordering, and front/middle deletion**.

Honest limitation (inherent to any unanchored hash chain): ``verify()`` on its own
CANNOT detect **tail truncation** (dropping the most recent — e.g. incriminating —
entries) or a **wholesale rebuild/forged append** by a writer with log access,
because nothing commits to the length or head. To close that, persist the
``count`` + ``head()`` out-of-band (an external anchor) and pass them to
``verify(expected_count=…, expected_head=…)`` — which then catches truncation and
rebuild too. This makes "bounded, logged declassification" auditable *given an
external anchor*; the log is an in-memory model (persistence/runtime wiring is next).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

_GENESIS = "GENESIS"


def _hash(index: int, prev: str, action: str, detail: dict) -> str:
    payload = json.dumps(
        {"i": index, "prev": prev, "action": action, "detail": detail},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    index: int
    prev_hash: str
    action: str
    detail: dict
    hash: str


@dataclass
class AuditLog:
    entries: list = field(default_factory=list)

    def append(self, action: str, detail: dict) -> AuditEntry:
        prev = self.entries[-1].hash if self.entries else _GENESIS
        idx = len(self.entries)
        entry = AuditEntry(idx, prev, action, dict(detail), _hash(idx, prev, action, detail))
        self.entries.append(entry)
        return entry

    def head(self) -> str:
        """The current head hash — persist this (with ``count``) as an external
        anchor to detect tail truncation / rebuild."""
        return self.entries[-1].hash if self.entries else _GENESIS

    @property
    def count(self) -> int:
        return len(self.entries)

    def verify(self, *, expected_count: "int | None" = None, expected_head: "str | None" = None) -> bool:
        """Chain integrity (catches edits, reorders, front/middle deletion). Pass
        ``expected_count`` and/or ``expected_head`` from an out-of-band anchor to
        ALSO catch tail truncation and forged append/rebuild."""
        prev = _GENESIS
        for i, e in enumerate(self.entries):
            if e.index != i or e.prev_hash != prev or e.hash != _hash(i, prev, e.action, e.detail):
                return False
            prev = e.hash
        if expected_count is not None and len(self.entries) != expected_count:
            return False
        if expected_head is not None and self.head() != expected_head:
            return False
        return True

    def records(self, action: "str | None" = None) -> list:
        return [e for e in self.entries if action is None or e.action == action]
