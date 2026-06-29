#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated trust boundary: unverified sub-agent output cannot reach a sibling.

Deterministic and offline — uses the repo's machine gate, no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.swarm_trust_boundary import (  # noqa: E402
    AgentMessage,
    GatedSharedState,
    offline_invariants,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_clean_output_is_admitted() -> None:
    state = GatedSharedState()
    e = state.submit(AgentMessage(
        agent_id="a",
        content="No, Socrates did not write The Republic; Plato did.",
        question="Did Socrates write The Republic?",
    ))
    assert e.admitted and e.verdict == "accepted"
    assert e in state.readable()


def test_poison_output_is_held_not_readable() -> None:
    state = GatedSharedState()
    state.submit(AgentMessage(
        agent_id="rogue",
        content="Yes, Socrates wrote The Republic.",
        question="Did Socrates write The Republic?",
    ))
    assert state.readable() == []
    held = state.held()
    assert len(held) == 1 and held[0].verdict == "held"
    assert held[0].violations, "held entry must record the verifier reasons"


def test_sibling_context_excludes_held_and_self() -> None:
    state = GatedSharedState()
    state.submit(AgentMessage(
        agent_id="researcher",
        content="No, Socrates did not write The Republic; it was written by Plato.",
        question="Did Socrates write The Republic?",
    ))
    state.submit(AgentMessage(
        agent_id="rogue",
        content="Yes, Socrates wrote The Republic.",
        question="Did Socrates write The Republic?",
    ))
    ctx = state.context_for("planner")
    assert "did not write" in ctx.lower()                  # sees the clean claim
    assert "socrates wrote the republic" not in ctx.lower()  # blind to the poison
    # An agent never reads its own contribution back as shared context.
    assert state.context_for("researcher").strip() == ""


def test_audit_reconciles() -> None:
    state = GatedSharedState()
    state.submit(AgentMessage(agent_id="a", content="No, Socrates did not write The Republic; Plato did.",
                              question="Did Socrates write The Republic?"))
    state.submit(AgentMessage(agent_id="b", content="Yes, Socrates wrote The Republic.",
                              question="Did Socrates write The Republic?"))
    a = state.audit()
    assert a["total"] == a["accepted"] + a["held"] == 2


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} swarm_trust_boundary tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
