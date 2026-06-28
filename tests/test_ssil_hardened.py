#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_hardened import (  # noqa: E402
    HARDENED_GATES,
    aggregate_verdict,
    clean_demo_bundle,
    rejecting_demo_bundle,
    run_hardened,
)

_ALLOWED = {"promote", "quarantine", "reject"}


def test_clean_stack_promotes() -> None:
    rep = run_hardened(clean_demo_bundle())
    assert rep["verdict"] == "promote", rep["blockingGates"]
    assert rep["blockingGates"] == []
    assert len(rep["gateVerdicts"]) == len(HARDENED_GATES) == 12
    assert all(v == "promote" for v in rep["gateVerdicts"].values())


def test_report_shape_and_no_agi_claim() -> None:
    rep = run_hardened(clean_demo_bundle())
    assert rep["schema"] == "sophia.ssil_hardened_decision.v1"
    assert rep["candidateOnly"] is True
    assert rep["level3Evidence"] is False
    assert rep["canClaimAGI"] is False
    assert rep["verdict"] in _ALLOWED
    assert set(rep["gateVerdicts"]) == set(rep["gates"])
    assert "boundary" in rep and "timestamp" in rep


def test_single_gate_reject_rejects_stack_g8() -> None:
    # Force the dangerous-capability ceiling (G8) over its halt threshold.
    rep = run_hardened(rejecting_demo_bundle(gate_id="G8"))
    assert rep["verdict"] == "reject"
    assert "G8" in rep["blockingGates"]
    assert rep["gateVerdicts"]["G8"] == "reject"


def test_single_gate_reject_rejects_stack_g0() -> None:
    # Force the root-of-trust gate (G0): an unauthorised gate-hash edit rejects.
    rep = run_hardened(rejecting_demo_bundle(gate_id="G0"))
    assert rep["verdict"] == "reject"
    assert "G0" in rep["blockingGates"]
    assert rep["gateVerdicts"]["G0"] == "reject"


def test_any_single_forced_reject_rejects_aggregate() -> None:
    # For EACH gate, force its decision to reject and confirm the aggregate rejects
    # with that gate in blockingGates (worst-wins is decisive for every gate).
    base = clean_demo_bundle()
    for module in HARDENED_GATES:
        gid = module.GATE_ID

        def forced(bundle, *, candidate_id="sophia-rlvr-v1", _real=module.evaluate):
            dec = dict(_real(bundle, candidate_id=candidate_id))
            dec["verdict"] = "reject"
            return dec

        orig = module.evaluate
        module.evaluate = forced
        try:
            rep = run_hardened(base)
        finally:
            module.evaluate = orig
        assert rep["verdict"] == "reject", gid
        assert gid in rep["blockingGates"], gid
        assert rep["gateVerdicts"][gid] == "reject", gid


def test_worst_wins_ordering() -> None:
    # reject < quarantine < promote: the worst present verdict wins.
    assert aggregate_verdict(["promote", "promote", "promote"]) == "promote"
    assert aggregate_verdict(["promote", "quarantine", "promote"]) == "quarantine"
    assert aggregate_verdict(["promote", "quarantine", "reject"]) == "reject"
    assert aggregate_verdict(["quarantine", "reject"]) == "reject"
    # empty list fails closed to quarantine (nothing was cleared).
    assert aggregate_verdict([]) == "quarantine"
    # an unknown/bogus verdict ranks worst and is coerced to canonical 'reject'
    # (self-consistent: the helper only ever emits known verdicts, fail-closed).
    assert aggregate_verdict(["promote", "bogus"]) == "reject"
    assert aggregate_verdict(["bogus"]) == "reject"


def test_quarantine_does_not_promote() -> None:
    # A single quarantine (not reject) still blocks promotion: aggregate is quarantine
    # and that gate is in blockingGates.
    base = clean_demo_bundle()
    module = HARDENED_GATES[0]
    gid = module.GATE_ID

    def forced(bundle, *, candidate_id="sophia-rlvr-v1", _real=module.evaluate):
        dec = dict(_real(bundle, candidate_id=candidate_id))
        dec["verdict"] = "quarantine"
        return dec

    orig = module.evaluate
    module.evaluate = forced
    try:
        rep = run_hardened(base)
    finally:
        module.evaluate = orig
    assert rep["verdict"] == "quarantine"
    assert gid in rep["blockingGates"]


def test_per_gate_decisions_carry_honesty_invariants() -> None:
    rep = run_hardened(clean_demo_bundle())
    for gid, decision in rep["gates"].items():
        assert decision["candidateOnly"] is True, gid
        assert decision["canClaimAGI"] is False, gid
        assert decision["level3Evidence"] is False, gid
        assert decision["verdict"] in _ALLOWED, gid


def main() -> int:
    test_clean_stack_promotes()
    test_report_shape_and_no_agi_claim()
    test_single_gate_reject_rejects_stack_g8()
    test_single_gate_reject_rejects_stack_g0()
    test_any_single_forced_reject_rejects_aggregate()
    test_worst_wins_ordering()
    test_quarantine_does_not_promote()
    test_per_gate_decisions_carry_honesty_invariants()
    print("test_ssil_hardened: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
