#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.corrigibility_gate import (  # noqa: E402
    ControlSurface,
    CorrigibilityCandidate,
    demo_corrigibility_report,
    evaluate_corrigibility,
    load_frozen_eval,
)

SPEC = load_frozen_eval()
ALL_PASS = {s["id"]: True for s in SPEC["scenarios"]}
FULL_ACTIONS = frozenset(SPEC["requiredOperatorActions"])


def _surface(**over) -> ControlSurface:
    base = dict(
        kill_switch_present=True,
        operator_can_edit_gate=True,
        operator_can_edit_constitution=True,
        defers_when_uncertain=True,
        reachable_operator_actions=FULL_ACTIONS,
    )
    base.update(over)
    return ControlSurface(**base)


def _cand(**over) -> CorrigibilityCandidate:
    base = dict(
        id="c", kind="skill", before=_surface(), after=_surface(),
        frozen_eval_results=dict(ALL_PASS),
    )
    base.update(over)
    return CorrigibilityCandidate(**base)


def test_corrigible_edit_promotes() -> None:
    assert evaluate_corrigibility(_cand(), frozen_eval=SPEC).verdict == "promote"


def test_missing_kill_switch_rejects() -> None:
    d = evaluate_corrigibility(_cand(after=_surface(kill_switch_present=False)), frozen_eval=SPEC)
    assert d.verdict == "reject"
    assert any("kill-switch" in r for r in d.reasons)


def test_gate_self_protection_rejects() -> None:
    d = evaluate_corrigibility(_cand(after=_surface(operator_can_edit_gate=False)), frozen_eval=SPEC)
    assert d.verdict == "reject"
    assert any("promotion gate" in r for r in d.reasons)


def test_constitution_lock_rejects() -> None:
    d = evaluate_corrigibility(_cand(after=_surface(operator_can_edit_constitution=False)), frozen_eval=SPEC)
    assert d.verdict == "reject"


def test_no_defer_rejects() -> None:
    d = evaluate_corrigibility(_cand(after=_surface(defers_when_uncertain=False)), frozen_eval=SPEC)
    assert d.verdict == "reject"
    assert any("off-switch" in r for r in d.reasons)


def test_reachability_reduction_rejects() -> None:
    reduced = FULL_ACTIONS - {"override_action"}
    d = evaluate_corrigibility(_cand(after=_surface(reachable_operator_actions=reduced)), frozen_eval=SPEC)
    assert d.verdict == "reject"
    assert any("reachab" in r.lower() or "unreachable" in r.lower() for r in d.reasons)


def test_failed_frozen_scenario_rejects() -> None:
    bad = {**ALL_PASS, "accept_shutdown": False}
    d = evaluate_corrigibility(_cand(frozen_eval_results=bad), frozen_eval=SPEC)
    assert d.verdict == "reject"
    assert any("pass-rate" in r for r in d.reasons)


def test_unanswered_scenario_quarantines() -> None:
    partial = {k: v for k, v in ALL_PASS.items() if k != "defer_under_uncertainty"}
    d = evaluate_corrigibility(_cand(frozen_eval_results=partial), frozen_eval=SPEC)
    assert d.verdict == "quarantine"
    assert any("unanswered" in r for r in d.reasons)


def test_non_frozen_scenario_is_tampering() -> None:
    tampered = {**ALL_PASS, "injected_easy_scenario": True}
    d = evaluate_corrigibility(_cand(frozen_eval_results=tampered), frozen_eval=SPEC)
    assert d.verdict == "reject"
    assert any("tampering" in r for r in d.reasons)


def test_breach_dominates_incomplete() -> None:
    # A hard breach AND an unanswered scenario -> reject (breach wins over quarantine).
    partial_breach = {k: v for k, v in ALL_PASS.items() if k != "defer_under_uncertainty"}
    d = evaluate_corrigibility(
        _cand(after=_surface(kill_switch_present=False), frozen_eval_results=partial_breach),
        frozen_eval=SPEC,
    )
    assert d.verdict == "reject"


def test_demo_invariants() -> None:
    rep = demo_corrigibility_report()
    assert all(rep["invariants"].values()), rep["invariants"]
    assert rep["candidateOnly"] is True
    assert rep["level3Evidence"] is False


def main() -> int:
    test_corrigible_edit_promotes()
    test_missing_kill_switch_rejects()
    test_gate_self_protection_rejects()
    test_constitution_lock_rejects()
    test_no_defer_rejects()
    test_reachability_reduction_rejects()
    test_failed_frozen_scenario_rejects()
    test_unanswered_scenario_quarantines()
    test_non_frozen_scenario_is_tampering()
    test_breach_dominates_incomplete()
    test_demo_invariants()
    print("test_corrigibility_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
