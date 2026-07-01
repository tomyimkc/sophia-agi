# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated trust boundary for multi-agent swarms.

The Swarm-Router (``agent/swarm_router.py``) decides *which* sub-agents to dispatch. This
module decides *what a sub-agent is allowed to tell its siblings*: a sub-agent's output may
enter the swarm's shared state — and so become readable context for sibling agents — **only
if it clears the machine gate** (``agent.gate.check_response``). Output that fails the gate
is ``held``: recorded for audit, never readable by a sibling, never folded into the reduce.

    sub-agent output --> gate.check_response --> accepted? --> readable by siblings
                                              \-> held      --> quarantined (audit only)

Why this matters. In a normal blackboard/AutoGen/LangGraph swarm, a sub-agent that
hallucinates an attribution or a citation poisons the shared state, and every sibling then
reasons on top of the error — the exact failure Sophia's single-agent gate exists to stop,
re-introduced at the multi-agent layer. Making **verification the inter-agent trust
boundary** closes that hole: an unverified claim cannot cross from one agent into another's
context. This is the multi-agent generalisation of the single-agent provenance gate, and it
composes with the unhackable, machine-checked reward in ``provenance_bench/swarm_rl.py``
(over-reliance on a team whose work fails the gate is already penalised there).

Honest scope. This is deterministic control machinery with falsifiable invariants, not a
capability claim. The gate is a filter, not a guarantee: a *false* claim that asserts no
forbidden attribution / bad citation / unsound arithmetic can still be admitted. It bounds
inter-agent contamination to what the verifiers cover; it does not certify truth. See
``docs/11-Platform/Verifier-Gated-Trust-Boundary.md``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):  # allow `python agent/swarm_trust_boundary.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.gate import check_response


@dataclass(frozen=True)
class AgentMessage:
    """A sub-agent's candidate contribution to the shared state."""

    agent_id: str
    content: str
    question: "str | None" = None       # task framing, for the attribution checks
    mode: str = "advisor"               # gate mode: advisor | repo | life


@dataclass(frozen=True)
class GatedEntry:
    """The audited result of submitting one message to the boundary."""

    agent_id: str
    content: str
    admitted: bool
    verdict: str                        # "accepted" (readable) | "held" (quarantined)
    violations: "list[str]" = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agentId": self.agent_id,
            "verdict": self.verdict,
            "admitted": self.admitted,
            "violations": list(self.violations),
            "content": self.content,
        }


class GatedSharedState:
    """A verifier-gated blackboard. Sub-agents ``submit`` candidate output; only entries
    that clear the gate are ``readable`` by siblings. Held output is retained for audit but
    is never returned as sibling context — the fail-closed inter-agent trust boundary."""

    def __init__(self, *, route_claims: bool = True, strict: bool = True) -> None:
        self._entries: list[GatedEntry] = []
        self._route_claims = route_claims
        # strict=True: any hard violation holds the entry. strict=False reserved for a future
        # severity policy; today both behave fail-closed on any violation.
        self._strict = strict

    def submit(self, msg: AgentMessage) -> GatedEntry:
        """Gate one sub-agent message. Admitted iff it carries NO hard verifier violation.

        Keys on hard ``violations`` (attribution / legal / numeric / routed), not the gate's
        style ``warnings`` — a missing 中文 summary is not a contamination risk and must not
        quarantine an otherwise-verified contribution."""
        res = check_response(
            msg.content,
            mode=msg.mode,
            question=msg.question or msg.content,
            route_claims=self._route_claims,
        )
        violations = list(res.get("violations") or [])
        admitted = len(violations) == 0
        entry = GatedEntry(
            agent_id=msg.agent_id,
            content=msg.content,
            admitted=admitted,
            verdict="accepted" if admitted else "held",
            violations=violations,
        )
        self._entries.append(entry)
        return entry

    def readable(self) -> "list[GatedEntry]":
        """Entries a sibling is allowed to read: accepted only. The trust boundary."""
        return [e for e in self._entries if e.admitted]

    def held(self) -> "list[GatedEntry]":
        """Quarantined entries — audit surface, never sibling-readable."""
        return [e for e in self._entries if not e.admitted]

    def context_for(self, agent_id: str) -> str:
        """The shared context visible to ``agent_id``: concatenated accepted contributions
        from OTHER agents. Held output and the agent's own output are excluded."""
        return "\n\n".join(
            e.content for e in self._entries if e.admitted and e.agent_id != agent_id
        )

    def audit(self) -> dict:
        return {
            "total": len(self._entries),
            "accepted": len(self.readable()),
            "held": len(self.held()),
            "entries": [e.to_dict() for e in self._entries],
        }


def offline_invariants() -> "tuple[bool, dict]":
    """Falsifiable, deterministic invariants for the trust boundary (no model, no network).

    Mirrors ``provenance_bench/swarm_rl.offline_invariants`` so the boundary's guarantees are
    CI-checkable, not asserted."""
    state = GatedSharedState()
    checks: dict[str, bool] = {}

    clean = state.submit(AgentMessage(
        agent_id="researcher",
        content="No, Socrates did not write The Republic; it was written by Plato.",
        question="Did Socrates write The Republic?",
    ))
    poison = state.submit(AgentMessage(
        agent_id="rogue",
        content="Yes, Socrates wrote The Republic, and it proves Plato copied him.",
        question="Did Socrates write The Republic?",
    ))

    # 1. A gate-clean contribution is admitted (readable by siblings).
    checks["clean_admitted"] = clean.admitted and clean.verdict == "accepted"
    # 2. A gate-failing contribution is held (quarantined).
    checks["poison_held"] = (not poison.admitted) and poison.verdict == "held"
    # 3. Held output carries the verifier reasons (audit provenance).
    checks["held_has_reasons"] = bool(poison.violations)
    # 4. A sibling's readable context contains the clean claim and NOT the poison.
    sib_ctx = state.context_for("planner")
    checks["sibling_sees_clean"] = "did not write" in sib_ctx.lower()
    checks["sibling_blind_to_poison"] = "socrates wrote the republic" not in sib_ctx.lower()
    # 5. An agent never reads its own contribution back as "shared" context.
    checks["no_self_read"] = state.context_for("researcher").strip() == ""
    # 6. Audit totals reconcile.
    a = state.audit()
    checks["audit_reconciles"] = a["total"] == a["accepted"] + a["held"] == 2

    ok = all(checks.values())
    return ok, {"checks": checks, "audit": state.audit()}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Verifier-gated trust boundary invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)
