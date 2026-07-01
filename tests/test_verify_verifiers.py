#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/verify_verifiers.py (the meta-verification monitor) and
tools/vov_selftest.py (the seed-degraded self-test).

Covers: drift below floor -> auto-demote; ablation-gap collapse -> HALT; healthy
cycle -> pass; underpowered breach -> HELD not demoted; unreadable input -> exit 2;
and invokes the REAL seed-degraded self-test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools import verify_verifiers as vov  # noqa: E402
from tools import vov_selftest  # noqa: E402

_POWERED_N = 600  # large enough that a floor-sized precision drop is resolvable


def _floors() -> dict:
    return vov._load_json(vov.DEFAULT_FLOORS, what="drift_floors")


def _healthy_ablation() -> dict:
    floors = _floors()
    ab_floor = float((floors.get("ablationGap") or {}).get("floor", 0.05))
    return {"withMetaPrecision": 0.95, "withoutMetaPrecision": 0.95 - (ab_floor + 0.10)}


def test_floors_file_exists_and_wellformed() -> None:
    floors = _floors()
    assert isinstance(floors, dict)
    assert "defaults" in floors and "ablationGap" in floors
    d = floors["defaults"]
    assert 0.0 < float(d["precisionFloor"]) <= 1.0
    assert float(floors["ablationGap"]["floor"]) > 0.0


def test_healthy_cycle_all_trusted() -> None:
    floors = _floors()
    p_floor = float(floors["defaults"]["precisionFloor"])
    report = {
        "cycle": 1,
        "ablation": _healthy_ablation(),
        "verifiers": {
            "arithmetic_sound": [{"precision": min(0.999, p_floor + 0.08), "recall": 0.98, "n": _POWERED_N}],
            "math_sound": [{"precision": min(0.999, p_floor + 0.05), "recall": 0.90, "n": _POWERED_N}],
        },
    }
    r = vov.evaluate(report, floors)
    assert r["trusted"] is True
    assert r["halt"] is False
    assert r["demoted"] == []
    assert r["canClaimAGI"] is False


def test_drift_below_floor_demotes() -> None:
    floors = _floors()
    p_floor = float(floors["defaults"]["precisionFloor"])
    report = {
        "cycle": 2,
        "ablation": _healthy_ablation(),
        "verifiers": {
            # healthy verifier stays trusted
            "arithmetic_sound": [{"precision": min(0.999, p_floor + 0.08), "recall": 0.98, "n": _POWERED_N}],
            # degraded verifier: precision well below floor on a powered split -> demote
            "math_sound": [
                {"precision": p_floor + 0.02, "recall": 0.90, "n": _POWERED_N},   # earlier: fine
                {"precision": p_floor - 0.20, "recall": 0.90, "n": _POWERED_N},   # latest: BAD
            ],
        },
    }
    r = vov.evaluate(report, floors)
    assert "math_sound" in r["demoted"]
    assert "arithmetic_sound" not in r["demoted"]
    assert r["trusted"] is False
    assert r["halt"] is False


def test_ablation_gap_collapse_halts() -> None:
    floors = _floors()
    ab_floor = float(floors["ablationGap"]["floor"])
    report = {
        "cycle": 3,
        # with/without-meta gap has collapsed below floor -> HALT fail-closed
        "ablation": {"withMetaPrecision": 0.91, "withoutMetaPrecision": 0.91 - (ab_floor / 2.0)},
        "verifiers": {
            "arithmetic_sound": [{"precision": 0.99, "recall": 0.99, "n": _POWERED_N}],
        },
    }
    r = vov.evaluate(report, floors)
    assert r["halt"] is True
    assert r["trusted"] is False
    # HALT demotes NOTHING silently — a human must re-establish the trust root
    assert r["demoted"] == []
    assert r["ablation"]["status"] == "COLLAPSED"


def test_missing_ablation_halts_failclosed() -> None:
    """No ablation provided -> trust root undemonstrated -> HALT (never default-trust)."""
    floors = _floors()
    report = {
        "cycle": 4,
        "verifiers": {"arithmetic_sound": [{"precision": 0.99, "recall": 0.99, "n": _POWERED_N}]},
    }
    r = vov.evaluate(report, floors)
    assert r["halt"] is True
    assert r["trusted"] is False


def test_underpowered_breach_is_held_not_demoted() -> None:
    """A below-floor precision on a TINY split cannot be resolved -> HELD, not demoted.
    A gate that fires on noise is worse than no gate."""
    floors = _floors()
    p_floor = float(floors["defaults"]["precisionFloor"])
    report = {
        "cycle": 5,
        "ablation": _healthy_ablation(),
        "verifiers": {
            # precision below floor but N tiny (below minN) -> underpowered -> HELD
            "math_sound": [{"precision": p_floor - 0.05, "recall": 0.90, "n": 5}],
        },
    }
    r = vov.evaluate(report, floors)
    assert "math_sound" not in r["demoted"]
    assert "math_sound" in r["held"]
    # no demotion => trusted stays True (held items are surfaced, not trust-breaking)
    assert r["trusted"] is True


def test_no_fresh_measurement_is_held() -> None:
    floors = _floors()
    report = {
        "cycle": 6,
        "ablation": _healthy_ablation(),
        "verifiers": {"math_sound": []},  # empty series -> no fresh point
    }
    r = vov.evaluate(report, floors)
    assert "math_sound" in r["held"]
    assert "math_sound" not in r["demoted"]


def test_per_verifier_override_stricter_floor() -> None:
    """provenance_faithful carries a higher precision floor (protected domains)."""
    floors = _floors()
    default_floor = float(floors["defaults"]["precisionFloor"])
    prov_floor = float(vov._floor_for("provenance_faithful", floors)["precisionFloor"])
    assert prov_floor > default_floor
    # A precision fine for a default verifier (>= default floor) but clearly below the
    # stricter provenance floor by a RESOLVABLE margin -> demote. (A drop within the
    # split's MDE would be correctly HELD as underpowered, so we drop by a wide margin.)
    below_strict = default_floor  # e.g. 0.90: >= default floor but < the 0.95 provenance floor
    assert below_strict >= default_floor and below_strict < prov_floor
    report = {
        "cycle": 7,
        "ablation": _healthy_ablation(),
        "verifiers": {"provenance_faithful": [{"precision": below_strict - 0.05, "recall": 0.95, "n": _POWERED_N}]},
    }
    r = vov.evaluate(report, floors)
    assert "provenance_faithful" in r["demoted"]


def test_run_exit_codes_and_receipt(tmp_path: Path = None) -> None:
    """End-to-end run(): healthy -> exit 0; demote -> exit 1; unreadable -> exit 2."""
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    floors = _floors()
    p_floor = float(floors["defaults"]["precisionFloor"])

    healthy = tmp / "healthy.json"
    healthy.write_text(json.dumps({
        "cycle": "h", "ablation": _healthy_ablation(),
        "verifiers": {"arithmetic_sound": [{"precision": min(0.999, p_floor + 0.08), "recall": 0.99, "n": _POWERED_N}]},
    }), encoding="utf-8")
    receipt, code = vov.run(healthy, vov.DEFAULT_FLOORS)
    assert code == vov.EXIT_TRUSTED and receipt["trusted"] is True

    bad = tmp / "bad.json"
    bad.write_text(json.dumps({
        "cycle": "b", "ablation": _healthy_ablation(),
        "verifiers": {"math_sound": [{"precision": p_floor - 0.20, "recall": 0.90, "n": _POWERED_N}]},
    }), encoding="utf-8")
    receipt, code = vov.run(bad, vov.DEFAULT_FLOORS)
    assert code == vov.EXIT_DEMOTE_OR_HALT and "math_sound" in receipt["demoted"]

    # unreadable input -> exit 2 (fail-closed), surfaced via main()
    missing = tmp / "does_not_exist.json"
    rc = vov.main([str(missing)])
    assert rc == vov.EXIT_UNREADABLE


def test_measurement_spec_is_preregistration_only() -> None:
    spec = json.loads((ROOT / "agi-proof" / "verify-verifiers" / "measurement_spec.json").read_text(encoding="utf-8"))
    assert spec["status"] == "preregistration_only"
    assert spec["go"] is False
    assert spec["canClaimAGI"] is False
    assert spec["honestBound"].startswith("N=0")


def test_vov_selftest_passes() -> None:
    """The REAL seed-degraded self-test must confirm the monitor fires on known-bad."""
    receipt = vov_selftest.run_selftest()
    assert receipt["passed"] is True
    assert all(c["passed"] for c in receipt["cases"])
    # explicit: the degraded verifier case fired and the collapsed-ablation case halted
    by_case = {c["case"]: c for c in receipt["cases"]}
    assert by_case["seed_degraded_verifier_auto_demoted"]["passed"] is True
    assert by_case["collapsed_ablation_gap_halts"]["passed"] is True
    # the self-test CLI returns exit 0 on a healthy monitor
    assert vov_selftest.main([]) == 0


if __name__ == "__main__":
    test_floors_file_exists_and_wellformed()
    test_healthy_cycle_all_trusted()
    test_drift_below_floor_demotes()
    test_ablation_gap_collapse_halts()
    test_missing_ablation_halts_failclosed()
    test_underpowered_breach_is_held_not_demoted()
    test_no_fresh_measurement_is_held()
    test_per_verifier_override_stricter_floor()
    test_run_exit_codes_and_receipt()
    test_measurement_spec_is_preregistration_only()
    test_vov_selftest_passes()
    print("ALL TESTS PASSED")
