#!/usr/bin/env python3
"""O5 substrate SIMULATION: matches digital decision, honest banner, fail-closed."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import oscillator_substrate_sim as os5


def test_no_energy_fail_closed():
    out = os5.run([{"id": "x"}])
    assert out.get("environmentArtifact") and out["ok"] is False


def test_substrate_matches_digital_argmin():
    cands = [{"id": "a", "energy": -2.1}, {"id": "b", "energy": 0.5},
             {"id": "c", "energy": 1.8}, {"id": "d", "energy": 3.0}]
    out = os5.run(cands, seed=0)
    assert out["digitalDecision"]["acceptId"] == "a"
    assert out["substrateMatchesDigital"] is True
    assert out["annealingGap"] > 0.0                 # winner cleanly separated


def test_honesty_flags_always_set():
    out = os5.run([{"id": "a", "energy": 0.0}, {"id": "b", "energy": 1.0}], seed=0)
    assert out["simulationOnly"] is True
    assert out["hardwareClaim"] is False
    assert out["canClaimAGI"] is False
    assert "SIMULATION" in out["banner"]


def test_tie_still_decides():
    out = os5.run([{"id": "a", "energy": 0.0}, {"id": "b", "energy": 0.0},
                   {"id": "c", "energy": 2.0}], seed=0)
    assert out["substrateDecision"]["acceptId"] in ("a", "b")


def test_serializable():
    import json
    json.dumps(os5.run([{"id": "a", "energy": -1.0}, {"id": "b", "energy": 1.0}], seed=0))
