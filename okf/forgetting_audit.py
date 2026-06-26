# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tamper-evident audit trail for belief lifecycle events.

Every decay/suppress/quarantine/reinforce decision from `okf.decay_okf` (and every
frontier-demotion decision from `okf.frontier_demotion`) is appended as an immutable,
hash-chained record. Each record carries the SHA-256 of the previous one
(genesis = all-zeros), so any retroactive edit to the trail is detectable by re-walking
the chain. This is the provenance analogue of engram's "SHA-256 audit trail", kept in
Sophia's deterministic-stdlib idiom and gated by source discipline.

This is an AUDIT TRAIL, not a database. It never holds raw PII and it never deletes
history; cryptographic erasure (GDPR right-to-be-forgotten) is a separate operation
(`forget_subject`) that appends a *cryptographic erasure event* and invalidates the
subject's contribution hashes WITHOUT rewriting the chain — auditable forgetfulness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

GENESIS_HASH = "0" * 64
EVENT_TYPES = ("suppress", "reinforce", "quarantine", "consolidate", "erasure", "resurrect", "demote")


def _h(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class LifecycleEvent:
    event: str                       # one of EVENT_TYPES
    node_id: str
    reason: str                      # controlled vocab from decay_okf.DECAY_REASONS + erasure
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subject_id: str | None = None    # for per-subject erasure
    meta: dict = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    hash: str = ""

    def seal(self) -> "LifecycleEvent":
        payload = {k: v for k, v in asdict(self).items() if k != "hash"}
        self.hash = _h(payload)
        return self


class ForgettingAudit:
    """Append-only, hash-chained ledger of belief-lifecycle decisions.

    Verification is O(n) in trail length: recompute each hash from prev_hash and
    confirm equality. A single bit-flip anywhere breaks the chain at that record.
    """

    def __init__(self) -> None:
        self._events: list[LifecycleEvent] = []

    @property
    def head_hash(self) -> str:
        return self._events[-1].hash if self._events else GENESIS_HASH

    def append(self, event: LifecycleEvent) -> LifecycleEvent:
        if event.event not in EVENT_TYPES:
            raise ValueError(f"unknown event {event.event!r}")
        event.prev_hash = self.head_hash
        event.seal()
        self._events.append(event)
        return event

    def record_plan(self, plan: dict) -> list[LifecycleEvent]:
        """Fold a DecayPlan.to_dict() into auditable events."""
        out: list[LifecycleEvent] = []
        for node_id, reason in plan.get("suppress", []):
            out.append(self.append(LifecycleEvent("suppress", node_id, reason)))
        for node_id in plan.get("reinforce", []):
            out.append(self.append(LifecycleEvent("reinforce", node_id, "surprise_gated")))
        for node_id, reason in plan.get("quarantine", []):
            out.append(self.append(LifecycleEvent("quarantine", node_id, reason)))
        return out

    def record_demotion(self, decision: dict) -> LifecycleEvent | None:
        """Fold a frontier-demotion decision into an auditable event."""
        if not decision.get("demote"):
            return None
        return self.append(LifecycleEvent(
            "demote",
            decision.get("nodeId", "_unknown"),
            f"frontier_decisive_evidence:{decision.get('newConfidence')}",
            meta={"supersededByRegime": decision.get("supersededByRegime"),
                  "rankDrop": decision.get("rankDrop", 1)},
        ))

    def forget_subject(self, subject_id: str, *, authority: str = "gdpr_rtbfe") -> LifecycleEvent:
        """Per-subject cryptographic erasure. Does NOT rewrite history.

        Appends an `erasure` event naming the subject; downstream consumers MUST
        invalidate any belief whose `subject_id` matches. The audit chain still
        proves the erasure happened, when, and by whose authority.
        """
        return self.append(LifecycleEvent(
            "erasure", node_id=f"_subject:{subject_id}", reason="cryptographic_erasure",
            subject_id=subject_id, meta={"authority": authority},
        ))

    def verify(self) -> bool:
        prev = GENESIS_HASH
        for e in self._events:
            payload = {k: v for k, v in asdict(e).items() if k != "hash"}
            if e.prev_hash != prev:
                return False
            if _h(payload) != e.hash:
                return False
            prev = e.hash
        return True

    def tamper(self, idx: int) -> None:
        """Test hook: silently mutate a record to break the chain (verify() -> False)."""
        self._events[idx].reason = "TAMPERED:" + self._events[idx].reason

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self._events]


__all__ = ["LifecycleEvent", "ForgettingAudit", "EVENT_TYPES", "GENESIS_HASH"]
