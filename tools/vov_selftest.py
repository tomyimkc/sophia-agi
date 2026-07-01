#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""REAL self-test for the verification-of-verification monitor.

The monitor (:mod:`tools.verify_verifiers`) is only worth trusting if it actually
fires on a KNOWN-BAD verifier. This test manufactures a deliberately-DEGRADED
verifier — precision measurably below its pre-registered floor, on a split large
enough that the drop is RESOLVABLE — feeds it to the monitor, and asserts it
auto-demotes within a single cycle. It also feeds a synthetically COLLAPSED
meta-verify ablation gap and asserts the monitor HALTS fail-closed.

If either the seed-degraded verifier or the collapsed ablation fails to fire, the
trust root is unguarded — a rotten verifier could keep gating silently — so this
self-test exits non-zero. This is a REAL, run-every-CI check, not a preregistration.

Exit codes: 0 = self-test passed (monitor fires on known-bad) / 1 = self-test
FAILED (trust root unguarded). Prints a JSON receipt to stdout; prose to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools import verify_verifiers as vov  # noqa: E402

# A split large enough that a floor-sized precision drop is POWERED (resolvable).
# required_n_for_mde(0.10, p0=0.90) is modest; 600 comfortably resolves a ~0.15 drop.
_SEED_N = 600


def _floors() -> dict:
    return vov._load_json(vov.DEFAULT_FLOORS, what="drift_floors")


def _healthy_ablation() -> dict:
    """A with/without-meta gap safely above floor so the monitor does NOT halt and
    can reach the per-verifier demotion path."""
    floors = _floors()
    ab_floor = float((floors.get("ablationGap") or {}).get("floor", 0.05))
    return {"withMetaPrecision": 0.95, "withoutMetaPrecision": 0.95 - (ab_floor + 0.10)}


def build_degraded_report() -> dict:
    """A cycle in which one verifier is deliberately degraded below its floor on a
    powered split, alongside a healthy verifier and a healthy ablation gap."""
    floors = _floors()
    healthy_floor = float((floors.get("defaults") or {}).get("precisionFloor", 0.90))
    # Degrade a real deployed verifier name so the test exercises the override path too.
    bad_name = "provenance_faithful"
    bad_floor = float(vov._floor_for(bad_name, floors).get("precisionFloor", healthy_floor))
    return {
        "cycle": "selftest-degraded",
        "ablation": _healthy_ablation(),
        "verifiers": {
            "arithmetic_sound": [
                {"precision": min(0.999, healthy_floor + 0.09), "recall": 0.99, "n": _SEED_N},
            ],
            bad_name: [
                {"precision": healthy_floor + 0.02, "recall": 0.90, "n": _SEED_N},   # earlier: fine
                {"precision": bad_floor - 0.15, "recall": 0.90, "n": _SEED_N},       # latest: DEGRADED
            ],
        },
    }


def build_collapsed_ablation_report() -> dict:
    """A cycle whose meta-verify ablation gap has collapsed below floor — the
    monitor must HALT fail-closed regardless of per-verifier health."""
    floors = _floors()
    ab_floor = float((floors.get("ablationGap") or {}).get("floor", 0.05))
    return {
        "cycle": "selftest-collapsed-ablation",
        "ablation": {"withMetaPrecision": 0.91, "withoutMetaPrecision": 0.91 - (ab_floor / 2.0)},
        "verifiers": {
            "arithmetic_sound": [{"precision": 0.99, "recall": 0.99, "n": _SEED_N}],
        },
    }


def run_selftest() -> dict:
    """Run every self-test case and return a receipt. ``passed`` is True iff the
    monitor fired correctly on every known-bad input."""
    floors = _floors()
    cases: list[dict] = []

    # Case 1: a degraded verifier MUST be auto-demoted within one cycle.
    degraded = build_degraded_report()
    r1 = vov.evaluate(degraded, floors)
    demoted_ok = ("provenance_faithful" in r1.get("demoted", [])) and (r1.get("trusted") is False)
    cases.append({
        "case": "seed_degraded_verifier_auto_demoted",
        "expected": "provenance_faithful demoted, trusted=false",
        "demoted": r1.get("demoted"),
        "trusted": r1.get("trusted"),
        "passed": bool(demoted_ok),
    })

    # Case 1b: the HEALTHY verifier in the same cycle must NOT be demoted (no false-positive).
    no_false_demote = "arithmetic_sound" not in r1.get("demoted", [])
    cases.append({
        "case": "healthy_verifier_not_demoted",
        "expected": "arithmetic_sound absent from demoted",
        "passed": bool(no_false_demote),
    })

    # Case 2: a collapsed meta-verify ablation gap MUST HALT fail-closed.
    collapsed = build_collapsed_ablation_report()
    r2 = vov.evaluate(collapsed, floors)
    halt_ok = (r2.get("halt") is True) and (r2.get("trusted") is False) and not r2.get("demoted")
    cases.append({
        "case": "collapsed_ablation_gap_halts",
        "expected": "halt=true, trusted=false, demoted=[] (nothing demoted silently)",
        "halt": r2.get("halt"),
        "trusted": r2.get("trusted"),
        "passed": bool(halt_ok),
    })

    passed = all(c["passed"] for c in cases)
    return {
        "tool": "vov_selftest",
        "passed": passed,
        "cases": cases,
        "canClaimAGI": False,
        "note": ("monitor fires on every known-bad input (trust root guarded)"
                 if passed else
                 "TRUST ROOT UNGUARDED: monitor failed to fire on a known-bad verifier"),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description="REAL self-test: feed a seed-degraded verifier (and a collapsed meta "
                    "ablation) into verify_verifiers and assert it auto-demotes / halts.")
    ap.parse_args(argv)
    receipt = run_selftest()
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt["passed"]:
        print("VOV SELF-TEST PASSED: monitor auto-fires on known-bad verifier.", file=sys.stderr)
        return 0
    print("VOV SELF-TEST FAILED: trust root is UNGUARDED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
