"""Tamper-evident append-only audit log for declassification (#3).

Every declassification (and every denied attempt) is recorded in a hash-chained
log: each entry's hash covers its content AND the previous entry's hash, so editing
or deleting any past entry breaks the chain — ``verify()`` then returns False. This
is what makes "bounded, logged declassification" auditable rather than a promise.
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

    def verify(self) -> bool:
        """True iff the chain is intact (no entry altered, removed, or reordered)."""
        prev = _GENESIS
        for i, e in enumerate(self.entries):
            if e.index != i or e.prev_hash != prev or e.hash != _hash(i, prev, e.action, e.detail):
                return False
            prev = e.hash
        return True

    def records(self, action: "str | None" = None) -> list:
        return [e for e in self.entries if action is None or e.action == action]
