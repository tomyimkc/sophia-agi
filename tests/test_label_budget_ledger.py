# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/label_budget_ledger.py — spend, exhaustion demotes gate to advisory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import label_budget_ledger as g  # noqa: E402


def _fresh_ledger() -> dict:
    return {
        "oracles": {
            "human_gold": {"labelsAvailable": 50, "labelsSpent": 0, "gates": ["calib_gate"]},
            "metered_judge": {"labelsAvailable": 100, "labelsSpent": 0, "gates": ["probe_gate"]},
            "heldout_gold": {"labelsAvailable": 10, "labelsSpent": 0, "gates": ["transfer_gate"]},
        }
    }


def test_no_demotions_when_supply_remains():
    r = g.status(_fresh_ledger())
    assert r["anyDemotions"] is False
    assert r["gatesDemoted"] == []
    assert set(r["gatesActive"]) == {"calib_gate", "probe_gate", "transfer_gate"}


def test_spend_reduces_remaining():
    led = _fresh_ledger()
    res = g.spend(led, "metered_judge", 40)
    assert res["totalSpent"] == 40
    assert led["oracles"]["metered_judge"]["labelsSpent"] == 40
    r = g.status(led)
    assert r["perOracle"]["metered_judge"]["remaining"] == 60
    assert r["anyDemotions"] is False


def test_exhaustion_demotes_gate_to_advisory():
    led = _fresh_ledger()
    g.spend(led, "heldout_gold", 10)  # exactly exhausts
    r = g.status(led)
    assert r["perOracle"]["heldout_gold"]["exhausted"] is True
    demoted_gates = {d["gate"] for d in r["gatesDemoted"]}
    assert "transfer_gate" in demoted_gates
    d = next(d for d in r["gatesDemoted"] if d["gate"] == "transfer_gate")
    assert d["status"] == "advisory"
    assert "heldout_gold" in d["exhaustedOracles"]
    # other gates stay active
    assert "probe_gate" in r["gatesActive"]


def test_overspend_clamped_and_reported():
    led = _fresh_ledger()
    res = g.spend(led, "heldout_gold", 25)  # only 10 available
    assert res["totalSpent"] == 10
    assert res["overRequested"] == 15
    assert led["oracles"]["heldout_gold"]["labelsSpent"] == 10
    r = g.status(led)
    assert r["perOracle"]["heldout_gold"]["exhausted"] is True


def test_unknown_oracle_raises():
    led = _fresh_ledger()
    try:
        g.spend(led, "no_such_oracle", 1)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_gate_demoted_if_any_dependency_exhausted():
    # A gate depending on two oracles; exhaust one -> gate demoted.
    led = {
        "oracles": {
            "o1": {"labelsAvailable": 5, "labelsSpent": 0, "gates": ["multi_gate"]},
            "o2": {"labelsAvailable": 5, "labelsSpent": 5, "gates": ["multi_gate"]},
        }
    }
    r = g.status(led)
    d = next(d for d in r["gatesDemoted"] if d["gate"] == "multi_gate")
    assert d["exhaustedOracles"] == ["o2"]
    assert set(d["dependsOn"]) == {"o1", "o2"}


def test_cli_always_exit_zero_and_dry_run():
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    ledger_path = tmp / "ledger.json"
    ledger_path.write_text(json.dumps(_fresh_ledger()), encoding="utf-8")

    script = str(ROOT / "tools" / "label_budget_ledger.py")

    # status only
    p = subprocess.run([sys.executable, script, "--ledger", str(ledger_path)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["anyDemotions"] is False

    # dry-run spend that exhausts -> demotion in receipt, file UNCHANGED
    d = subprocess.run([sys.executable, script, "--ledger", str(ledger_path),
                        "--spend", "heldout_gold:10"], capture_output=True, text=True)
    assert d.returncode == 0, d.stderr
    rec = json.loads(d.stdout)
    assert rec["anyDemotions"] is True
    assert rec["written"] is False
    # file not modified in dry-run
    assert json.loads(ledger_path.read_text())["oracles"]["heldout_gold"]["labelsSpent"] == 0

    # write spend persists
    w = subprocess.run([sys.executable, script, "--ledger", str(ledger_path),
                        "--spend", "heldout_gold:10", "--write"], capture_output=True, text=True)
    assert w.returncode == 0, w.stderr
    assert json.loads(w.stdout)["written"] is True
    assert json.loads(ledger_path.read_text())["oracles"]["heldout_gold"]["labelsSpent"] == 10


def test_seeded_config_loads():
    cfg = g.load_ledger(g.DEFAULT_LEDGER)
    assert "oracles" in cfg
    assert cfg["canClaimAGI"] is False
    # every gate listed under an oracle should be reachable via status()
    r = g.status(cfg)
    assert isinstance(r["perOracle"], dict) and len(r["perOracle"]) >= 1


if __name__ == "__main__":
    test_no_demotions_when_supply_remains()
    test_spend_reduces_remaining()
    test_exhaustion_demotes_gate_to_advisory()
    test_overspend_clamped_and_reported()
    test_unknown_oracle_raises()
    test_gate_demoted_if_any_dependency_exhausted()
    test_cli_always_exit_zero_and_dry_run()
    test_seeded_config_loads()
    print("ALL TESTS PASSED")
