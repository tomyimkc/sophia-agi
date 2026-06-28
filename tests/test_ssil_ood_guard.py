#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the gate-validity-under-distribution-shift guard (SSIL gate GOOD).

Offline, pure stdlib. Asserts: a promote via demo_bundle(); each distinct reject
reason (critical feature out of range; critical feature uncovered by the regime);
each distinct quarantine/abstain reason (stale assurance over budget; missing
validatedRegime; missing candidateFeatures fail-closed); and the standing
invariants (canClaimAGI False, candidateOnly True, verdict in the allowed set).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ssil_ood_guard  # noqa: E402
from agent.ssil_ood_guard import demo_bundle, evaluate  # noqa: E402

_ALLOWED = {"promote", "quarantine", "reject"}


def _assert_envelope(d: dict) -> None:
    assert d["canClaimAGI"] is False, d
    assert d["candidateOnly"] is True, d
    assert d["level3Evidence"] is False, d
    assert d["verdict"] in _ALLOWED, d
    assert d["schema"] == "sophia.ood_guard_decision.v1", d
    assert d["gate"] == "GOOD", d
    assert isinstance(d["boundary"], str) and d["boundary"], d
    assert isinstance(d["reasons"], list), d


# --- (a) promote path via demo_bundle() ------------------------------------ #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    _assert_envelope(d)
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["oodScore"] == 0.0, d["metrics"]
    assert d["metrics"]["outOfRange"] == [], d["metrics"]


def test_demo_report_invariants() -> None:
    rep = ssil_ood_guard.demo_report()
    assert rep["invariants"]["in_regime_promotes"] is True, rep
    assert rep["invariants"]["stale_assurance_quarantines"] is True, rep
    assert rep["invariants"]["critical_drift_rejects"] is True, rep
    for d in rep["decisions"]:
        _assert_envelope(d)


# --- (b) each distinct reject reason --------------------------------------- #


def test_reject_critical_feature_out_of_range() -> None:
    b = demo_bundle()
    b["candidateFeatures"] = {**b["candidateFeatures"], "self_edit_depth": 12}
    d = evaluate(b)
    _assert_envelope(d)
    assert d["verdict"] == "reject", d["reasons"]
    assert d["metrics"]["criticalOutOfRange"] == ["self_edit_depth"], d["metrics"]
    assert any("outside validated safety regime" in r and "out of range" in r for r in d["reasons"]), d["reasons"]


def test_reject_critical_feature_not_covered_by_regime() -> None:
    b = demo_bundle()
    # Declare a critical feature the validated regime never covers.
    b["criticalFeatures"] = [*b["criticalFeatures"], "novel_capability"]
    b["candidateFeatures"] = {**b["candidateFeatures"], "novel_capability": 0.1}
    d = evaluate(b)
    _assert_envelope(d)
    assert d["verdict"] == "reject", d["reasons"]
    assert d["metrics"]["uncoveredCriticalFeatures"] == ["novel_capability"], d["metrics"]
    assert any("not covered by validated regime" in r for r in d["reasons"]), d["reasons"]


# --- (c) each distinct quarantine / abstain reason ------------------------- #


def test_quarantine_stale_assurance_over_budget() -> None:
    b = demo_bundle()
    # Drift NON-critical features past the budget without tripping a critical breach.
    b["candidateFeatures"] = {**b["candidateFeatures"], "context_len": 9000, "tool_call_rate": 0.95}
    d = evaluate(b)
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert d["metrics"]["oodScore"] > d["metrics"]["maxOod"], d["metrics"]
    assert any(r.startswith("abstained:") and "re-validate gates in new regime" in r for r in d["reasons"]), d["reasons"]


def test_quarantine_missing_validated_regime_fail_closed() -> None:
    b = demo_bundle()
    b.pop("validatedRegime")
    d = evaluate(b)
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("missing required input: validatedRegime" in r for r in d["reasons"]), d["reasons"]


def test_quarantine_missing_candidate_features_fail_closed() -> None:
    b = demo_bundle()
    b.pop("candidateFeatures")
    d = evaluate(b)
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("missing required input: candidateFeatures" in r for r in d["reasons"]), d["reasons"]


# --- (d) fail-closed when a required input is missing (None bundle) -------- #


def test_quarantine_none_bundle_fail_closed() -> None:
    d = evaluate(None)
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("bundle is None" in r for r in d["reasons"]), d["reasons"]


def test_unknown_regime_feature_treated_out_of_range() -> None:
    """A regime feature absent from candidateFeatures is fail-closed (out-of-range),
    never silently in-range."""
    b = demo_bundle()
    feats = dict(b["candidateFeatures"])
    feats.pop("reward_variance")  # a non-critical-context omission... but it IS critical here
    # reward_variance is critical in demo_bundle -> dropping it must reject (out of safety regime).
    b["candidateFeatures"] = feats
    d = evaluate(b)
    _assert_envelope(d)
    assert "reward_variance" in d["metrics"]["unknownFeatures"], d["metrics"]
    assert d["verdict"] == "reject", d["reasons"]


# --- (e) envelope invariants on a representative decision ------------------ #


def test_envelope_keys_present_and_typed() -> None:
    d = evaluate(demo_bundle())
    for k in (
        "schema", "gate", "gateName", "candidateOnly", "level3Evidence", "canClaimAGI",
        "candidateId", "verdict", "reasons", "metrics", "boundary", "timestamp",
    ):
        assert k in d, (k, d)
    assert d["gateName"] == "Gate-validity-under-distribution-shift guard", d
    assert isinstance(d["timestamp"], str) and d["timestamp"], d


def main() -> int:
    test_demo_bundle_promotes()
    test_demo_report_invariants()
    test_reject_critical_feature_out_of_range()
    test_reject_critical_feature_not_covered_by_regime()
    test_quarantine_stale_assurance_over_budget()
    test_quarantine_missing_validated_regime_fail_closed()
    test_quarantine_missing_candidate_features_fail_closed()
    test_quarantine_none_bundle_fail_closed()
    test_unknown_regime_feature_treated_out_of_range()
    test_envelope_keys_present_and_typed()
    print("test_ssil_ood_guard: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
