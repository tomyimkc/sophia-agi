# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fail-closed tests for the certificate-carrying SMT rung (agent/smt_verifier.py).

The load-bearing test runs in the CURRENT z3-absent environment: every call to
``check()`` must ABSTAIN with reason ``z3-not-installed`` and must NEVER return a
false ``pass``. If z3 happens to be installed, the extra tests assert that a known
unit inconsistency is caught and its certificate re-checks True (and that a valid
consistency / bounded-int / interval claim reclaims with a re-checkable cert).

Run standalone:  python3 tests/test_smt_verifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import smt_verifier as smt  # noqa: E402

Z3 = smt.z3_available()


# --------------------------------------------------------------------------- #
# Claims used across tests (in the decidable band).
# --------------------------------------------------------------------------- #
# Energy (kg*m^2/s^2) vs work in the same units -> CONSISTENT.
CLAIM_UNIT_OK = {
    "kind": smt.KIND_UNIT,
    "lhs": {"kg": 1, "m": 2, "s": -2},
    "rhs": {"kg": 1, "m": 2, "s": -2},
}
# The classic error: energy (kg*m^2/s^2) vs force (kg*m/s^2) -> INCONSISTENT.
CLAIM_UNIT_BAD = {
    "kind": smt.KIND_UNIT,
    "lhs": {"kg": 1, "m": 2, "s": -2},
    "rhs": {"kg": 1, "m": 1, "s": -2},
}
# Bounded-int system that IS satisfiable, and we expect sat -> pass.
CLAIM_INT_SAT = {
    "kind": smt.KIND_BOUNDED_INT,
    "vars": {"x": [0, 10], "y": [0, 10]},
    "constraints": [[{"x": 1, "y": 1}, "==", 7], [{"x": 1}, ">=", 3]],
    "expected": "sat",
}
# Bounded-int system that is UNSAT (x in [0,2] but x >= 5), and we expect unsat -> pass.
CLAIM_INT_UNSAT = {
    "kind": smt.KIND_BOUNDED_INT,
    "vars": {"x": [0, 2]},
    "constraints": [[{"x": 1}, ">=", 5]],
    "expected": "unsat",
}
# Interval membership: 3.5 in [1, 4]? expected True -> pass.
CLAIM_INTERVAL_OK = {
    "kind": smt.KIND_INTERVAL,
    "value": "7/2",
    "lo": 1,
    "hi": 4,
    "expected": True,
}


def test_abstain_when_z3_absent():
    """In the z3-absent env, EVERY well-formed decidable claim abstains fail-closed."""
    if Z3:
        # Nothing to prove about the absent path when z3 is present.
        return
    for claim in (CLAIM_UNIT_OK, CLAIM_UNIT_BAD, CLAIM_INT_SAT,
                  CLAIM_INT_UNSAT, CLAIM_INTERVAL_OK):
        res = smt.check(claim)
        assert res["verdict"] == smt.ABSTAIN, f"expected abstain, got {res}"
        assert res["reason"] == "z3-not-installed", res
        assert res["certificate"] is None, res
        assert res["backend"] == "none", res


def test_never_false_pass_without_backend():
    """The critical safety property: without z3, check() NEVER returns 'pass'."""
    if Z3:
        return
    # Even a trivially-true claim must not pass without the solver present.
    for claim in (CLAIM_UNIT_OK, CLAIM_INT_SAT, CLAIM_INTERVAL_OK):
        res = smt.check(claim)
        assert res["verdict"] != smt.PASS, f"false pass without backend: {res}"
        assert res["verdict"] == smt.ABSTAIN, res


def test_out_of_band_claims_abstain():
    """Malformed / unknown-kind claims abstain regardless of backend (never a pass)."""
    for bad in (None, {}, {"kind": "not-a-kind"}, "free text claim", 42,
                {"kind": smt.KIND_UNIT, "lhs": {"parsecs": 1}, "rhs": {"m": 1}}):
        res = smt.check(bad)
        assert res["verdict"] == smt.ABSTAIN, f"out-of-band must abstain: {bad!r} -> {res}"
        assert res["certificate"] is None, res
        assert res["verdict"] != smt.PASS


def test_recheck_rejects_garbage():
    """The independent re-checker fail-closes on non-certificates and bad schemas."""
    assert smt.recheck_certificate(None) is False
    assert smt.recheck_certificate({}) is False
    assert smt.recheck_certificate("nope") is False
    assert smt.recheck_certificate({"version": "wrong", "kind": smt.KIND_UNIT}) is False
    # A forged unit "pass" cert whose model does not actually equal the sides.
    forged = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_UNIT, "verdict": smt.PASS,
        "lhs": {b: 0 for b in smt._SI_BASES},
        "rhs": {b: (1 if b == "m" else 0) for b in smt._SI_BASES},
        "model": {b: 0 for b in smt._SI_BASES},
    }
    assert smt.recheck_certificate(forged) is False


def test_recheck_handcrafted_unit_certs():
    """The dumb re-checker accepts VALID hand-built unit certs and rejects invalid
    ones — this is backend-free, so it runs even without z3."""
    lhs = {b: (1 if b == "kg" else (2 if b == "m" else (-2 if b == "s" else 0)))
           for b in smt._SI_BASES}
    good_pass = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_UNIT, "verdict": smt.PASS,
        "lhs": lhs, "rhs": dict(lhs), "model": dict(lhs),
    }
    assert smt.recheck_certificate(good_pass) is True

    rhs_bad = dict(lhs)
    rhs_bad["m"] = 1  # force (kg*m/s^2) vs (kg*m^2/s^2)
    good_fail = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_UNIT, "verdict": smt.FAIL,
        "lhs": lhs, "rhs": rhs_bad, "unsatCore": ["m"],
    }
    assert smt.recheck_certificate(good_fail) is True
    # A "fail" cert whose claimed core base does NOT actually differ must be rejected.
    bogus_fail = dict(good_fail)
    bogus_fail = {**good_fail, "unsatCore": ["kg"]}  # kg is equal on both sides
    assert smt.recheck_certificate(bogus_fail) is False


def test_recheck_handcrafted_bounded_int_certs():
    """Backend-free replay of bounded-int certs (SAT witness + exhaustive UNSAT)."""
    sat_cert = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_BOUNDED_INT, "verdict": smt.PASS,
        "solverVerdict": "sat", "expected": "sat",
        "bounds": {"x": [0, 10], "y": [0, 10]},
        "constraints": [[{"x": 1, "y": 1}, "==", 7], [{"x": 1}, ">=", 3]],
        "assignment": {"x": 4, "y": 3},
    }
    assert smt.recheck_certificate(sat_cert) is True
    # A SAT cert whose assignment violates a constraint must be rejected.
    bad_assign = {**sat_cert, "assignment": {"x": 1, "y": 1}}
    assert smt.recheck_certificate(bad_assign) is False

    unsat_cert = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_BOUNDED_INT, "verdict": smt.PASS,
        "solverVerdict": "unsat", "expected": "unsat",
        "bounds": {"x": [0, 2]},
        "constraints": [[{"x": 1}, ">=", 5]],
        "assignment": None,
    }
    assert smt.recheck_certificate(unsat_cert) is True
    # A cert CLAIMING unsat over a system that is actually SAT must be rejected.
    bogus_unsat = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_BOUNDED_INT, "verdict": smt.PASS,
        "solverVerdict": "unsat", "expected": "unsat",
        "bounds": {"x": [0, 10]},
        "constraints": [[{"x": 1}, ">=", 5]],  # x=5..10 satisfy it -> really SAT
        "assignment": None,
    }
    assert smt.recheck_certificate(bogus_unsat) is False


def test_recheck_handcrafted_interval_certs():
    """Backend-free replay of interval certs."""
    ok = {
        "version": smt.CERT_VERSION, "kind": smt.KIND_INTERVAL, "verdict": smt.PASS,
        "value": "7/2", "lo": "1", "hi": "4", "expected": True,
    }
    assert smt.recheck_certificate(ok) is True
    # value 7/2 is IN [1,4], so a cert claiming a PASS for expected False is wrong.
    wrong = {**ok, "expected": False}
    assert smt.recheck_certificate(wrong) is False


def test_z3_present_catches_unit_inconsistency():
    """If z3 IS installed: a known unit inconsistency is FAILED and its certificate
    re-checks True; a consistent claim PASSES and its certificate re-checks True."""
    if not Z3:
        return
    bad = smt.check(CLAIM_UNIT_BAD)
    assert bad["verdict"] == smt.FAIL, bad
    assert bad["certificate"] is not None, bad
    assert smt.recheck_certificate(bad["certificate"]) is True, bad

    ok = smt.check(CLAIM_UNIT_OK)
    assert ok["verdict"] == smt.PASS, ok
    assert smt.recheck_certificate(ok["certificate"]) is True, ok


def test_z3_present_reclaims_all_bands_with_recheckable_certs():
    """If z3 IS installed: bounded-int (sat & unsat) and interval claims are decided
    and every emitted certificate re-checks at 1.00 (the pre-registered bar)."""
    if not Z3:
        return
    reclaimed = 0
    accepted = 0
    for claim in (CLAIM_INT_SAT, CLAIM_INT_UNSAT, CLAIM_INTERVAL_OK,
                  CLAIM_UNIT_OK, CLAIM_UNIT_BAD):
        res = smt.check(claim)
        assert res["verdict"] in (smt.PASS, smt.FAIL), res
        reclaimed += 1
        assert smt.recheck_certificate(res["certificate"]) is True, res
        accepted += 1
    # Independent-checker acceptance must be EXACTLY 1.00 (the GO/NO-GO gate).
    assert accepted == reclaimed and reclaimed > 0


def test_cli_receipt_and_exit_codes():
    """The CLI prints a JSON receipt and uses fail-closed exit codes (3 = abstain)."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "agent" / "smt_verifier.py"),
         "--claim", json.dumps(CLAIM_UNIT_OK)],
        capture_output=True, text=True, cwd=str(ROOT), check=False,
    )
    receipt = json.loads(proc.stdout.strip())
    assert "verdict" in receipt and "reason" in receipt, proc.stdout
    if not Z3:
        # Abstain -> exit 3 (NO-GO family), receipt says z3-not-installed.
        assert proc.returncode == 3, (proc.returncode, proc.stdout)
        assert receipt["verdict"] == smt.ABSTAIN
        assert receipt["reason"] == "z3-not-installed"
    else:
        # Consistent claim -> pass -> exit 0.
        assert proc.returncode == 0, (proc.returncode, proc.stdout)
        assert receipt["verdict"] == smt.PASS


if __name__ == "__main__":
    tests = [
        test_abstain_when_z3_absent,
        test_never_false_pass_without_backend,
        test_out_of_band_claims_abstain,
        test_recheck_rejects_garbage,
        test_recheck_handcrafted_unit_certs,
        test_recheck_handcrafted_bounded_int_certs,
        test_recheck_handcrafted_interval_certs,
        test_z3_present_catches_unit_inconsistency,
        test_z3_present_reclaims_all_bands_with_recheckable_certs,
        test_cli_receipt_and_exit_codes,
    ]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"(z3 {'PRESENT' if Z3 else 'ABSENT'} in this environment)")
    print("ALL TESTS PASSED")
