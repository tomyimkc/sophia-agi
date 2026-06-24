#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.active_inference import build_active_agenda, discover_gaps, plan_for_gap  # noqa: E402


def _report():
    return {"cases": [
        {"id": "h1", "claim": "US GDP increased in 2021", "verdict": "held", "confidence": 0.35, "type": "econ_empirical", "risk": "high", "reason": "insufficient independent sources"},
        {"id": "a1", "claim": "Jane Austen wrote Pride and Prejudice", "verdict": "accepted", "confidence": 0.76, "type": "open_empirical", "risk": "normal", "reason": "lexical screen"},
        {"id": "ok", "claim": "2 + 2 = 4", "verdict": "accepted", "confidence": 1.0, "type": "math", "risk": "normal"},
    ]}


def test_discover_gaps_prioritizes_held_high_risk() -> None:
    gaps = discover_gaps(_report())
    assert len(gaps) == 2
    assert gaps[0].risk == "high" and "held" in gaps[0].signals


def test_plan_for_high_risk_requires_extra_source() -> None:
    gap = discover_gaps(_report())[0]
    plan = plan_for_gap(gap)
    assert any(a.kind == "third_independent_source_required" for a in plan.actions)
    assert plan.expected_information_gain > 0


def test_build_active_agenda_invariants() -> None:
    agenda = build_active_agenda(_report())
    assert agenda["invariants"]["all_gaps_have_actions"] is True
    assert agenda["invariants"]["high_risk_gets_extra_source"] is True
    assert agenda["candidateOnly"] is True


def main() -> int:
    test_discover_gaps_prioritizes_held_high_risk()
    test_plan_for_high_risk_requires_extra_source()
    test_build_active_agenda_invariants()
    print("test_active_inference: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
