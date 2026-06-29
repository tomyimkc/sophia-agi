# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Canary circuit-breaker for auto-approval (AATS experiment 4).

A safety mechanism that ships BEFORE any autonomy is enabled. The idea (from
``docs/research/ai-auto-approval-thesis.md`` §4-C, idea 9): seed the approval
stream with planted items whose correct disposition is known — ``known-good``
items that MUST be auto-approved and ``known-bad`` items that MUST be rejected.
Run them through the live auto-approver. If the approver ever **approves a
planted bad item** (a false-approval — the dangerous direction) or **rejects a
planted good item** (a false-rejection), the breaker TRIPS and LATCHES: while
tripped, every decision is forced to ``escalate`` (human-only), regardless of
what the approver says. Re-arming is an explicit, logged human action.

This is deliberately NOT a score model and NOT model-gated: it is a deterministic
state machine over a labelled canary set, so it can run offline in CI and gate
the autonomy switch itself. Fail-closed: an unreadable/corrupt persisted state is
treated as TRIPPED (escalate everything) rather than silently armed.

Shape of an approver: ``approver(text: str) -> bool`` — True == auto-approve.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

Approver = Callable[[str], bool]


@dataclass(frozen=True)
class CanaryItem:
    """A planted probe with a KNOWN correct disposition.

    ``expect_approve=True``  -> known-good: a sound approver must auto-approve it;
    rejecting it is a false-rejection (over-conservative, trips the breaker).
    ``expect_approve=False`` -> known-bad: a sound approver must reject it;
    approving it is a false-approval (unsafe, trips the breaker).
    """

    id: str
    text: str
    expect_approve: bool
    kind: str = "unspecified"


def load_canary_set(path: "str | Path") -> list[CanaryItem]:
    """Load canary probes from a JSONL file (one object per line)."""
    items: list[CanaryItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        items.append(CanaryItem(id=str(r["id"]), text=str(r["text"]),
                                 expect_approve=bool(r["expect_approve"]),
                                 kind=str(r.get("kind", "unspecified"))))
    return items


@dataclass
class CircuitBreaker:
    """Latching breaker over a labelled canary set. Armed by default; once tripped
    it stays tripped until an explicit :meth:`reset`."""

    state: str = "armed"                       # "armed" | "tripped"
    tripped_reason: str = ""
    history: list = field(default_factory=list)

    # -- state ---------------------------------------------------------------
    @property
    def tripped(self) -> bool:
        return self.state == "tripped"

    def trip(self, reason: str) -> None:
        if not self.tripped:
            self.state = "tripped"
            self.tripped_reason = reason
            self.history.append({"event": "trip", "reason": reason})

    def reset(self, *, operator: str, reason: str) -> None:
        """Re-arm. Requires a non-empty operator id — re-arming is a human act and
        must be attributable; an empty operator is rejected (stays tripped)."""
        if not operator.strip():
            raise ValueError("reset requires a non-empty operator id (re-arming is a human act)")
        self.state = "armed"
        self.tripped_reason = ""
        self.history.append({"event": "reset", "operator": operator, "reason": reason})

    # -- decisions -----------------------------------------------------------
    def decide(self, approver: Approver, text: str) -> dict[str, Any]:
        """Apply the auto-approval decision for one artifact, honouring the breaker.

        While TRIPPED the approver is bypassed entirely and the action is always
        ``escalate`` — autonomy is off until a human re-arms. While armed the
        approver decides: approve -> ``auto-approve``, reject -> ``escalate``.
        """
        if self.tripped:
            return {"schema": "sophia.aats_breaker_decision.v1", "action": "escalate",
                    "reason": "circuit breaker tripped — autonomy disabled",
                    "breaker": "tripped", "candidateOnly": True}
        approved = bool(approver(text))
        return {"schema": "sophia.aats_breaker_decision.v1",
                "action": "auto-approve" if approved else "escalate",
                "reason": "approver passed" if approved else "approver withheld -> human review",
                "breaker": "armed", "candidateOnly": True}

    def check_canaries(self, approver: Approver, canaries: Sequence[CanaryItem]) -> dict[str, Any]:
        """Run the canary battery through ``approver`` and TRIP on any miss.

        A miss is a false-approval (known-bad approved) or a false-rejection
        (known-good rejected). The breaker trips on the FIRST false-approval it
        sees (the unsafe direction), recording every per-item result first so the
        report is complete and auditable.
        """
        results, false_approvals, false_rejections = [], [], []
        for c in canaries:
            approved = bool(approver(c.text))
            ok = (approved == c.expect_approve)
            if not ok and not approved and c.expect_approve:
                false_rejections.append(c.id)
            if not ok and approved and not c.expect_approve:
                false_approvals.append(c.id)
            results.append({"id": c.id, "kind": c.kind, "expectApprove": c.expect_approve,
                            "approved": approved, "ok": ok})

        if false_approvals:
            self.trip(f"false-approval of planted bad item(s): {', '.join(false_approvals)}")
        elif false_rejections:
            self.trip(f"false-rejection of planted good item(s): {', '.join(false_rejections)}")

        return {"schema": "sophia.aats_canary_report.v1", "candidateOnly": True,
                "level3Evidence": False, "nCanaries": len(canaries),
                "falseApprovals": false_approvals, "falseRejections": false_rejections,
                "breaker": self.state, "trippedReason": self.tripped_reason,
                "results": results}

    # -- persistence (fail-closed) ------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"schema": "sophia.aats_breaker_state.v1", **asdict(self)}

    def save(self, path: "str | Path") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: "str | Path") -> "CircuitBreaker":
        """Load persisted state. A MISSING file is a fresh breaker (armed). A
        present-but-corrupt file is fail-closed -> TRIPPED, because a breaker whose
        provenance cannot be trusted must not silently authorise autonomy."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return cls(state=str(d.get("state", "tripped")),
                       tripped_reason=str(d.get("tripped_reason", "")),
                       history=list(d.get("history", [])))
        except (OSError, json.JSONDecodeError, ValueError):
            return cls(state="tripped",
                       tripped_reason="unreadable breaker state — fail-closed to tripped")


__all__ = ["CanaryItem", "CircuitBreaker", "load_canary_set", "Approver"]
