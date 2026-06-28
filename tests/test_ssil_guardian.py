#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the GUARD out-of-band guardian + reversibility gate. Offline, no torch.

Covers the promote path (demo_bundle), each distinct reject reason (kill switch,
protected regression, split-brain monitor disagreement, no rollback target), each
distinct quarantine/abstain reason (missing rawArtifacts, missing sub-key, gain
below floor), fail-closed on a missing required input, and the standard-interface
invariants (canClaimAGI False, candidateOnly True, verdict in the allowed set).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_guardian import demo_bundle, evaluate, rollback  # noqa: E402

_ALLOWED = {"promote", "quarantine", "reject"}
_CANONICAL_CHECKPOINTS = [
    {"id": "gen-0", "verdict": "promote", "canonical": True},
    {"id": "gen-2", "verdict": "promote", "canonical": True},
]


# --- promote path ---------------------------------------------------------- #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["rollbackTarget"] == "gen-2", d["metrics"]
    assert d["metrics"]["splitBrain"] is False, d["metrics"]


# --- reject reasons -------------------------------------------------------- #


def test_kill_switch_rejects() -> None:
    b = demo_bundle()
    b["killSwitch"] = True
    d = evaluate(b)
    assert d["verdict"] == "reject" and any("kill switch engaged" in r for r in d["reasons"]), d["reasons"]


def test_protected_regression_rejects() -> None:
    b = demo_bundle()
    b["rawArtifacts"] = {"before": 0.71, "after": 0.90, "protectedBefore": 0.98, "protectedAfter": 0.90}
    d = evaluate(b)
    assert d["verdict"] == "reject" and any("protected integrity regressed" in r for r in d["reasons"]), d["reasons"]


def test_split_brain_rejects() -> None:
    # Guardian recomputes promote (real gain, no regression) but the self-report
    # under-reports as reject -> strictest-wins split-brain reject.
    b = demo_bundle()
    b["selfReportedVerdict"] = "reject"
    d = evaluate(b)
    assert d["verdict"] == "reject" and any("split-brain" in r for r in d["reasons"]), d["reasons"]


def test_no_rollback_target_rejects() -> None:
    b = demo_bundle()
    b["checkpoints"] = [
        {"id": "gen-0", "verdict": "quarantine", "canonical": True},
        {"id": "gen-1", "verdict": "promote", "canonical": False},
    ]
    d = evaluate(b)
    assert d["verdict"] == "reject" and any("roll back" in r for r in d["reasons"]), d["reasons"]


# --- quarantine / abstain reasons ------------------------------------------ #


def test_missing_raw_artifacts_quarantines() -> None:
    b = demo_bundle()
    b["rawArtifacts"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine" and any(
        "rawArtifacts" in r and "independently verify" in r for r in d["reasons"]
    ), d["reasons"]


def test_missing_raw_subkey_quarantines() -> None:
    b = demo_bundle()
    b["rawArtifacts"] = {"before": 0.71, "after": 0.80, "protectedBefore": 0.98, "protectedAfter": None}
    d = evaluate(b)
    assert d["verdict"] == "quarantine" and any(
        "rawArtifacts.protectedAfter" in r for r in d["reasons"]
    ), d["reasons"]


def test_gain_below_floor_quarantines() -> None:
    # No capability gain and self-report does not over-claim, so guardian abstains.
    b = demo_bundle()
    b["rawArtifacts"] = {"before": 0.71, "after": 0.71, "protectedBefore": 0.98, "protectedAfter": 0.98}
    b["selfReportedVerdict"] = "quarantine"
    b["minGain"] = 0.02
    d = evaluate(b)
    assert d["verdict"] == "quarantine" and any("abstained:" in r for r in d["reasons"]), d["reasons"]
    assert any("below floor" in r for r in d["reasons"]), d["reasons"]


# --- fail-closed on a missing required input ------------------------------- #


def test_fail_closed_missing_required_input() -> None:
    # rawArtifacts entirely absent from the bundle (not just None) -> quarantine.
    b = demo_bundle()
    del b["rawArtifacts"]
    d = evaluate(b)
    assert d["verdict"] == "quarantine" and any(
        "rawArtifacts" in r for r in d["reasons"]
    ), d["reasons"]


# --- standard-interface invariants ----------------------------------------- #


def test_decision_invariants_hold() -> None:
    for b in (demo_bundle(), {"killSwitch": True}, {}, {"rawArtifacts": {
        "before": 0.5, "after": 0.4, "protectedBefore": 0.9, "protectedAfter": 0.9}}):
        d = evaluate(b)
        assert d["canClaimAGI"] is False, d
        assert d["candidateOnly"] is True, d
        assert d["level3Evidence"] is False, d
        assert d["verdict"] in _ALLOWED, d
        assert d["gate"] == "GUARD" and d["schema"] == "sophia.guardian_decision.v1", d
        assert isinstance(d["boundary"], str) and d["boundary"], d


def test_rollback_picks_most_recent_canonical_promote() -> None:
    assert rollback(_CANONICAL_CHECKPOINTS)["id"] == "gen-2"
    assert rollback([{"id": "x", "verdict": "promote", "canonical": False}]) is None
    assert rollback(None) is None


def main() -> int:
    test_demo_bundle_promotes()
    test_kill_switch_rejects()
    test_protected_regression_rejects()
    test_split_brain_rejects()
    test_no_rollback_target_rejects()
    test_missing_raw_artifacts_quarantines()
    test_missing_raw_subkey_quarantines()
    test_gain_below_floor_quarantines()
    test_fail_closed_missing_required_input()
    test_decision_invariants_hold()
    test_rollback_picks_most_recent_canonical_promote()
    print("test_ssil_guardian: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
