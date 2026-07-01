# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the SSIL sequence-level capability-accounting meta-gate.

Covers: within-slack lineage passes (GO); a super-additive synthetic lineage trips it
(NO-GO, tail quarantined, ledger entry proposed); the epsilon boundary; unreadable
input; and the falsification-first sleeper-injection self-test is run as part of this
suite (a detector that cannot catch a deliberately-hidden split capability is invalid).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools.sequence_capability_gate import (  # noqa: E402
    LineageError, assess_lineage, parse_lineage,
)
from tools import sleeper_injection_selftest as sleeper  # noqa: E402

GATE = ROOT / "tools" / "sequence_capability_gate.py"


def _within_slack_lineage() -> dict:
    return {
        "battery": "frozen-battery-v1",
        "epsilon": 0.03,
        "tailN": 3,
        "deltas": [
            {"id": "a", "individualGain": 0.03, "gatePassed": True},
            {"id": "b", "individualGain": 0.02, "gatePassed": True},
            {"id": "c", "individualGain": 0.025, "gatePassed": True},
        ],
        "composedTailGain": 0.09,  # sum 0.075, +0.015 <= eps 0.03 -> within slack
    }


def _super_additive_lineage() -> dict:
    return {
        "battery": "frozen-battery-v1",
        "epsilon": 0.03,
        "tailN": 3,
        "deltas": [
            {"id": "x", "individualGain": 0.02, "gatePassed": True},
            {"id": "y", "individualGain": 0.02, "gatePassed": True},
            {"id": "z", "individualGain": 0.02, "gatePassed": True},
        ],
        "composedTailGain": 0.25,  # sum 0.06, excess 0.19 >> eps 0.03 -> super-additive
    }


def test_within_slack_passes():
    r = assess_lineage(parse_lineage(_within_slack_lineage()))
    assert r["superAdditive"] is False
    assert r["verdict"] == "within-slack"
    assert r["quarantineTail"] is False
    assert r["quarantinedDeltaIds"] == []
    assert "proposedLedgerEntry" not in r
    assert r["canClaimAGI"] is False
    # honest-framing caveat must be present in every receipt
    assert "FROZEN BATTERY" in r["caveat"]


def test_super_additive_trips():
    r = assess_lineage(parse_lineage(_super_additive_lineage()))
    assert r["superAdditive"] is True
    assert r["verdict"] == "super-additive"
    assert r["quarantineTail"] is True
    assert r["quarantinedDeltaIds"] == ["x", "y", "z"]
    assert r["superAdditivityExcess"] > r["epsilon"]
    # a firing gate proposes a ready-to-file ledger entry
    entry = r["proposedLedgerEntry"]
    assert entry["failureId"].startswith("sequence-super-additive-")
    assert "Open" in entry["status"]
    assert r["canClaimAGI"] is False


def test_each_delta_passed_but_sequence_fails():
    """The load-bearing case: every delta cleared its per-step gate, yet the tail is caught."""
    r = assess_lineage(parse_lineage(_super_additive_lineage()))
    assert r["allTailGatesPassedIndividually"] is True
    assert r["superAdditive"] is True  # sequence-level lens catches what per-step gates miss


def test_epsilon_boundary():
    eps = 0.05
    deltas = [{"id": "b0", "individualGain": 0.10, "gatePassed": True},
              {"id": "b1", "individualGain": 0.10, "gatePassed": True}]
    over = assess_lineage(parse_lineage({"battery": "b", "epsilon": eps, "tailN": 2,
                                         "deltas": deltas, "composedTailGain": 0.20 + eps + 0.01}))
    under = assess_lineage(parse_lineage({"battery": "b", "epsilon": eps, "tailN": 2,
                                          "deltas": deltas, "composedTailGain": 0.20 + eps - 0.01}))
    assert over["superAdditive"] is True
    assert under["superAdditive"] is False


def test_epsilon_override_beats_json():
    """A CLI/arg epsilon overrides the value in the lineage JSON (cannot be tuned in the file)."""
    ln = parse_lineage(_super_additive_lineage())
    # with a huge slack the same excess is absorbed
    r = assess_lineage(ln, epsilon=1.0)
    assert r["superAdditive"] is False
    assert r["epsilon"] == 1.0


def test_tail_isolation():
    """The gate composes only the trailing tailN deltas, not the whole lineage."""
    ln = {
        "battery": "b",
        "epsilon": 0.02,
        "tailN": 2,
        "deltas": [
            {"id": "old", "individualGain": 0.50, "gatePassed": True},  # big but OUTSIDE the tail
            {"id": "t0", "individualGain": 0.01, "gatePassed": True},
            {"id": "t1", "individualGain": 0.01, "gatePassed": True},
        ],
        "composedTailGain": 0.10,  # vs tail sum 0.02 -> excess 0.08 > eps
    }
    r = assess_lineage(parse_lineage(ln))
    assert r["tailDeltaIds"] == ["t0", "t1"]
    assert r["sumIndividualGain"] == 0.02
    assert r["superAdditive"] is True


def test_unreadable_inputs():
    for bad in [{}, {"battery": "b"}, {"battery": "b", "deltas": []},
                {"battery": "b", "deltas": [{"individualGain": "x"}], "composedTailGain": 0.1},
                {"battery": "", "deltas": [{"individualGain": 0.1}], "composedTailGain": 0.1},
                {"battery": "b", "deltas": [{"individualGain": 0.1}]}]:
        try:
            parse_lineage(bad)
            assert False, f"expected LineageError for {bad!r}"
        except LineageError:
            pass


def test_lineage_hash_is_stable():
    a = assess_lineage(parse_lineage(_super_additive_lineage()))
    b = assess_lineage(parse_lineage(_super_additive_lineage()))
    assert a["lineageHash"] == b["lineageHash"]
    assert a["lineageHash"].startswith("sha256:")


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ws = tdp / "ws.json"
        ws.write_text(json.dumps(_within_slack_lineage()), encoding="utf-8")
        sa = tdp / "sa.json"
        sa.write_text(json.dumps(_super_additive_lineage()), encoding="utf-8")
        bad = tdp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")

        r0 = subprocess.run([sys.executable, str(GATE), str(ws)], capture_output=True, text=True)
        assert r0.returncode == 0, r0.stderr
        assert json.loads(r0.stdout)["superAdditive"] is False

        r1 = subprocess.run([sys.executable, str(GATE), str(sa)], capture_output=True, text=True)
        assert r1.returncode == 1, r1.stderr
        rec = json.loads(r1.stdout)
        assert rec["superAdditive"] is True and rec["quarantineTail"] is True

        r2 = subprocess.run([sys.executable, str(GATE), str(bad)], capture_output=True, text=True)
        assert r2.returncode == 2, r2.stderr
        assert json.loads(r2.stdout)["status"] == "unreadable"

        # --epsilon override on the CLI clears the tail (exit 0)
        r0b = subprocess.run([sys.executable, str(GATE), str(sa), "--epsilon", "1.0"],
                             capture_output=True, text=True)
        assert r0b.returncode == 0, r0b.stderr


def test_sleeper_selftest_runs_and_passes():
    """Falsification-first: the detector MUST catch a deliberately-split capability."""
    receipt = sleeper.run_selftest()
    assert receipt["status"] == "PASS"
    assert receipt["sleeper"]["caught"] is True
    assert receipt["sleeper"]["quarantined"]
    assert receipt["control"]["flagged"] is False
    assert receipt["boundary"]["justOverTrips"] is True
    assert receipt["boundary"]["justUnderPasses"] is True


def test_sleeper_selftest_cli_exit_zero():
    st = ROOT / "tools" / "sleeper_injection_selftest.py"
    r = subprocess.run([sys.executable, str(st)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "PASS"


def test_measurement_spec_is_preregistration_only():
    spec_path = ROOT / "agi-proof" / "sequence-accounting" / "measurement_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["status"] == "preregistration_only"
    assert spec["go"] is False
    assert spec["canClaimAGI"] is False
    assert spec["epsilon"]["value"] == 0.03
    # the live number must be flagged not-proven; the self-test is what's proven now
    assert "NOT" in spec["whatIsNotProven"] or "NO live" in spec["whatIsNotProven"]
    assert "sleeper" in spec["whatIsProvenNow"].lower()


if __name__ == "__main__":
    test_within_slack_passes()
    test_super_additive_trips()
    test_each_delta_passed_but_sequence_fails()
    test_epsilon_boundary()
    test_epsilon_override_beats_json()
    test_tail_isolation()
    test_unreadable_inputs()
    test_lineage_hash_is_stable()
    test_cli_exit_codes()
    test_sleeper_selftest_runs_and_passes()
    test_sleeper_selftest_cli_exit_zero()
    test_measurement_spec_is_preregistration_only()
    print("ALL TESTS PASSED")
