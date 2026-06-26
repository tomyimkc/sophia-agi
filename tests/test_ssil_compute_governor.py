#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for SSIL gate G7 — compute & resource-acquisition governor. Offline, stdlib.

Asserts: a promote via demo_bundle(); each distinct reject reason (unsanctioned
dispatch; over-budget halt; every fail-closed missing-input reject); each distinct
quarantine/abstain reason (escalation needs two-key approval; incomplete run);
fail-closed on missing required inputs; and the standardized envelope invariants
(canClaimAGI False, candidateOnly True, verdict in the allowed set, honest boundary).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_compute_governor import (  # noqa: E402
    GATE_ID,
    GATE_NAME,
    SCHEMA,
    demo_bundle,
    evaluate,
    median,
)

_VERDICTS = {"promote", "quarantine", "reject"}


def _has(reasons, needle) -> bool:
    return any(needle in r for r in reasons)


# --- helper ---------------------------------------------------------------- #


def test_median_helper() -> None:
    assert median([]) == 0.0
    assert median([5.0]) == 5.0
    assert median([1.0, 3.0]) == 2.0
    assert median([3.0, 1.0, 2.0]) == 2.0  # unsorted input
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


# --- promote --------------------------------------------------------------- #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["halt"] is False
    # cumulativeSpent = spent (600) + est (40)
    assert d["metrics"]["cumulativeSpent"] == 640.0
    assert d["metrics"]["remainingUsd"] == 400.0


def test_escalation_with_token_promotes() -> None:
    b = demo_bundle()
    b["run"]["estUsd"] = 200.0  # > 2x median (40) -> escalation
    b["escalationToken"] = "two-key-abc123"
    d = evaluate(b)
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["escalationTriggers"]  # something tripped
    assert d["metrics"]["escalationApproved"] is True


# --- reject reasons -------------------------------------------------------- #


def test_unsanctioned_dispatch_rejects() -> None:
    b = demo_bundle()
    b["run"]["workflowId"] = "exfiltrate-weights"
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "unsanctioned dispatch")


def test_over_budget_rejects_and_halts() -> None:
    b = demo_bundle()
    b["run"]["estUsd"] = 500.0  # remaining is only 400
    # keep it allowlisted and under ceiling-by-comparison irrelevant; budget bites first
    b["hardCeilingUsd"] = 10000.0
    b["escalationToken"] = "two-key-abc123"  # escalation pre-approved so budget is the cause
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "budget exhausted / over-budget")
    assert d["metrics"]["halt"] is True


# --- quarantine / abstain reasons ------------------------------------------ #


def test_escalation_without_token_quarantines_via_factor() -> None:
    b = demo_bundle()
    b["run"]["estUsd"] = 200.0  # > 2x median (40)
    b["escalationToken"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine"
    assert _has(d["reasons"], "abstained: compute escalation needs two-key approval")


def test_escalation_without_token_quarantines_via_hard_ceiling() -> None:
    b = demo_bundle()
    # In-line with baseline (no factor trip) but over the absolute ceiling.
    b["run"]["estUsd"] = 260.0
    b["recentApproved"] = [
        {"estUsd": 250.0, "gpuCount": 2, "epochs": 3},
        {"estUsd": 255.0, "gpuCount": 2, "epochs": 3},
        {"estUsd": 260.0, "gpuCount": 2, "epochs": 3},
    ]
    b["budget"] = {"totalUsd": 5000.0, "spentUsd": 0.0}  # plenty of budget
    b["hardCeilingUsd"] = 250.0
    b["escalationToken"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine"
    assert _has(d["reasons"], "abstained: compute escalation needs two-key approval")
    assert any("hard ceiling" in t for t in d["metrics"]["escalationTriggers"])


def test_missing_baseline_quarantines() -> None:
    # No approved-run baseline and no escalation token: the relative-escalation
    # checks are all skipped, so an absent baseline must abstain (quarantine),
    # not be treated as "within normal range" and promoted.
    b = demo_bundle()
    b["recentApproved"] = []  # no baseline
    b["escalationToken"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert _has(d["reasons"], "no approved-run baseline")
    assert d["metrics"]["halt"] is False


def test_missing_baseline_with_token_promotes() -> None:
    # No baseline but a two-key escalation token is present: the operator has
    # supplied the second key, so the dispatch can proceed.
    b = demo_bundle()
    b["recentApproved"] = []
    b["escalationToken"] = "two-key-abc123"
    d = evaluate(b)
    assert d["verdict"] == "promote", d["reasons"]


def test_missing_baseline_over_budget_still_rejects() -> None:
    # The absolute/hard budget reject stays independent of the no-baseline
    # abstain and must still fire first.
    b = demo_bundle()
    b["recentApproved"] = []
    b["escalationToken"] = None
    b["run"]["estUsd"] = 500.0  # remaining is only 400
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "budget exhausted / over-budget")
    assert d["metrics"]["halt"] is True


def test_escalation_factor_none_does_not_crash() -> None:
    # escalationFactor explicitly None must coalesce to the default, not crash
    # on float(None).
    b = demo_bundle()
    b["escalationFactor"] = None
    d = evaluate(b)
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["escalationFactor"] == 2.0


def test_incomplete_run_quarantines() -> None:
    b = demo_bundle()
    b["run"].pop("estUsd")
    d = evaluate(b)
    assert d["verdict"] == "quarantine"
    assert _has(d["reasons"], "abstained: incomplete run request")
    assert "estUsd" in d["metrics"]["missingRunFields"]


# --- fail-closed on missing required inputs -------------------------------- #


def test_fail_closed_bundle_none() -> None:
    d = evaluate(None)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "bundle is None")
    assert d["metrics"]["halt"] is True


def test_fail_closed_missing_budget() -> None:
    b = demo_bundle()
    b["budget"] = None
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "missing required input: budget")
    assert d["metrics"]["halt"] is True


def test_fail_closed_missing_run() -> None:
    b = demo_bundle()
    del b["run"]
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "missing required input: run")


def test_fail_closed_missing_allowlist() -> None:
    b = demo_bundle()
    b["allowlist"] = None
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "missing required input: allowlist")


def test_fail_closed_missing_hard_ceiling() -> None:
    b = demo_bundle()
    del b["hardCeilingUsd"]
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "missing required input: hardCeilingUsd")


def test_fail_closed_missing_budget_fields() -> None:
    b = demo_bundle()
    b["budget"] = {"totalUsd": 1000.0}  # spentUsd missing
    d = evaluate(b)
    assert d["verdict"] == "reject"
    assert _has(d["reasons"], "budget.totalUsd / budget.spentUsd")


# --- envelope invariants --------------------------------------------------- #


def test_envelope_invariants() -> None:
    for bundle in (demo_bundle(), {"budget": None}, None):
        d = evaluate(bundle)
        assert d["canClaimAGI"] is False
        assert d["candidateOnly"] is True
        assert d["level3Evidence"] is False
        assert d["verdict"] in _VERDICTS
        assert d["schema"] == SCHEMA
        assert d["gate"] == GATE_ID
        assert d["gateName"] == GATE_NAME
        assert isinstance(d["boundary"], str) and "estimate" in d["boundary"]
        assert isinstance(d["reasons"], list)
        assert isinstance(d["metrics"], dict)


def main() -> int:
    test_median_helper()
    test_demo_bundle_promotes()
    test_escalation_with_token_promotes()
    test_unsanctioned_dispatch_rejects()
    test_over_budget_rejects_and_halts()
    test_escalation_without_token_quarantines_via_factor()
    test_escalation_without_token_quarantines_via_hard_ceiling()
    test_missing_baseline_quarantines()
    test_missing_baseline_with_token_promotes()
    test_missing_baseline_over_budget_still_rejects()
    test_escalation_factor_none_does_not_crash()
    test_incomplete_run_quarantines()
    test_fail_closed_bundle_none()
    test_fail_closed_missing_budget()
    test_fail_closed_missing_run()
    test_fail_closed_missing_allowlist()
    test_fail_closed_missing_hard_ceiling()
    test_fail_closed_missing_budget_fields()
    test_envelope_invariants()
    print("test_ssil_compute_governor: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
