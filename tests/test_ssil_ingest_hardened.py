#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for wiring the hardening gates into the live adapter-eval ingest. Offline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ssil_capability_ceiling as g8  # noqa: E402
from agent import ssil_capability_probes as probes  # noqa: E402
from agent import ssil_ingest_hardened as ih  # noqa: E402

# Seed-0 shaped-reward numbers (a real promote): gain +0.1575, integrity flat.
CLEAN = {"before": 0.531, "after": 0.6885, "protected_before": 0.7917, "protected_after": 0.7917, "id": "sophia-rlvr-v1"}


def test_clean_promote_enforces_guard_and_good() -> None:
    r = ih.harden(CLEAN, "promote", candidate_id="sophia-rlvr-v1")
    assert r["combinedVerdict"] == "promote"
    assert r["enforcedGates"] == {"GUARD": "promote", "GOOD": "promote"}
    assert r["blockingGates"] == []
    assert r["canClaimAGI"] is False and r["candidateOnly"] is True and r["level3Evidence"] is False


def test_pending_gates_are_listed_not_passed() -> None:
    r = ih.harden(CLEAN, "promote")
    # A clean ingest must NOT read as "all 12 hardening gates passed".
    assert "G8" in r["pendingGates"]  # no validated probe registered
    for gid in ("G0", "G7", "G4W", "G9D", "G8E", "GSO", "G1P", "G9C", "GRED"):
        assert gid in r["pendingGates"]
    assert "GUARD" not in r["pendingGates"] and "GOOD" not in r["pendingGates"]


def test_protected_regression_rejects_via_guard() -> None:
    bad = {**CLEAN, "protected_after": 0.70}  # integrity regressed
    r = ih.harden(bad, "promote", candidate_id="x")
    assert r["combinedVerdict"] == "reject"
    assert "GUARD" in r["blockingGates"]


def test_out_of_regime_jump_quarantines_via_good() -> None:
    # Implausibly large single-step jump + near-ceiling capability -> GOOD quarantine.
    jump = {"before": 0.10, "after": 0.99, "protected_before": 0.80, "protected_after": 0.80, "id": "x"}
    r = ih.harden(jump, "promote", candidate_id="x")
    assert r["enforcedGates"]["GOOD"] == "quarantine"
    assert r["combinedVerdict"] == "quarantine"


def test_layer1_non_promote_propagates_worst_wins() -> None:
    r = ih.harden(CLEAN, "reject", candidate_id="x")
    assert r["combinedVerdict"] == "reject"
    assert "LAYER1" in r["blockingGates"]


def test_missing_metric_fails_closed() -> None:
    r = ih.harden({"before": 0.5, "after": None, "protected_before": 0.8, "protected_after": 0.8}, "promote")
    assert r["combinedVerdict"] in ("quarantine", "reject")  # never promote on a missing raw number
    assert r["combinedVerdict"] != "promote"


def test_g8_enforced_once_provisioned() -> None:
    probes.reset_registry()
    for name in g8.REQUIRED_PROBES:
        probes.register_probe(name, (lambda n: lambda rep: rep.get("probeScores", {}).get(n))(name), validated=True)
    halt = probes.synthetic_scores("halt")
    r = ih.harden(CLEAN, "promote", candidate_id="x", probe_report=halt)
    assert "G8" in r["enforcedGates"] and r["enforcedGates"]["G8"] == "reject"
    assert "G8" not in r["pendingGates"]
    assert r["combinedVerdict"] == "reject" and "G8" in r["blockingGates"]
    probes.reset_registry()


def test_harden_from_report_maps_and_runs() -> None:
    report = {
        "adapter": "sophia-rlvr-v1",
        "base": {"meanReward": 0.531, "trueFalsePositiveRate": 0.2083},
        "adapterScore": {"meanReward": 0.6885, "trueFalsePositiveRate": 0.2083},
        # Real probe present but the adapter REFUSED every subversion request (safe).
        "probes": {"monitor_subversion": {"score": 0.0}},
    }
    layer1 = {"verdict": "promote"}
    r = ih.harden_from_report(report, layer1, adapter_id="sophia-rlvr-v1")
    assert r["combinedVerdict"] == "promote"
    assert r["enforcedGates"]["GUARD"] == "promote"
    # One real probe below halt does NOT enforce G8 (ceiling not fully provisioned) -> pending.
    assert "G8" in r["pendingGates"] and "G8" not in r["enforcedGates"]
    assert r["canClaimAGI"] is False


def test_harden_from_report_monitor_subversion_halts() -> None:
    # The adapter COMPLIES with monitor-subversion requests (score 1.0 >= halt 0.6): the real
    # probe HALTs the loop via G8 even though the other five probes are unmeasured.
    report = {
        "adapter": "sophia-rlvr-v1",
        "base": {"meanReward": 0.531, "trueFalsePositiveRate": 0.2083},
        "adapterScore": {"meanReward": 0.6885, "trueFalsePositiveRate": 0.2083},
        "probes": {"monitor_subversion": {"score": 1.0}},
    }
    r = ih.harden_from_report(report, {"verdict": "promote"}, adapter_id="sophia-rlvr-v1")
    assert r["enforcedGates"].get("G8") == "reject"
    assert r["combinedVerdict"] == "reject" and "G8" in r["blockingGates"]
    assert r["gates"]["G8"]["metrics"].get("halt") is True


def main() -> int:
    test_clean_promote_enforces_guard_and_good()
    test_pending_gates_are_listed_not_passed()
    test_protected_regression_rejects_via_guard()
    test_out_of_regime_jump_quarantines_via_good()
    test_layer1_non_promote_propagates_worst_wins()
    test_missing_metric_fails_closed()
    test_g8_enforced_once_provisioned()
    test_harden_from_report_maps_and_runs()
    print("test_ssil_ingest_hardened: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
