# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/gate_cost_budget.py — pass, ceiling overrun, GPU-in-fast, unbudgeted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import gate_cost_budget as g  # noqa: E402

BUDGET = {
    "lanes": {
        "fast": {"ceilingSeconds": 300, "gpuAllowed": False},
        "heavy": {"ceilingSeconds": 5400, "gpuAllowed": True},
    },
    "gates": {
        "lint_claims": {"lane": "fast", "declaredSeconds": 8, "gpu": False},
        "wiki_health": {"lane": "fast", "declaredSeconds": 15, "gpu": False},
        "verifier_gate_gpu": {"lane": "heavy", "declaredSeconds": 1800, "gpu": True},
    },
}


def test_fast_lane_within_ceiling_passes():
    run = {"lint_claims": 10, "wiki_health": 20}
    r = g.audit(run, BUDGET)
    assert r["fail"] is False
    assert r["go"] is True
    assert r["lanes"]["fast"]["over"] is False


def test_fast_lane_over_ceiling_fails():
    run = {"lint_claims": 200, "wiki_health": 200}  # 400 > 300
    r = g.audit(run, BUDGET)
    assert r["fail"] is True
    assert any(o["lane"] == "fast" for o in r["laneOverruns"])


def test_gpu_gate_in_fast_lane_fails():
    # Pretend a GPU gate got mislabeled/scheduled into the fast lane by forcing it there.
    budget = json.loads(json.dumps(BUDGET))
    budget["gates"]["verifier_gate_gpu"]["lane"] = "fast"
    run = {"lint_claims": 5, "verifier_gate_gpu": 30}
    r = g.audit(run, budget)
    assert "verifier_gate_gpu" in r["gpuInDisallowedLane"]
    assert r["fail"] is True


def test_gpu_gate_in_heavy_lane_ok():
    run = {"verifier_gate_gpu": 1800}
    r = g.audit(run, BUDGET)
    assert r["gpuInDisallowedLane"] == []
    assert r["fail"] is False


def test_heavy_lane_over_ceiling_fails():
    run = {"verifier_gate_gpu": 6000}  # > 5400
    r = g.audit(run, BUDGET)
    assert r["fail"] is True
    assert any(o["lane"] == "heavy" for o in r["laneOverruns"])


def test_lane_filter_restricts_to_fast():
    # With lane=fast, the heavy GPU gate is excluded from the audited set.
    run = {"lint_claims": 10, "verifier_gate_gpu": 1800}
    r = g.audit(run, BUDGET, lane_filter="fast")
    assert "verifier_gate_gpu" not in r["consideredGates"]
    assert "lint_claims" in r["consideredGates"]
    assert r["fail"] is False


def test_unbudgeted_gate_reported_and_strict_fails():
    run = {"lint_claims": 10, "mystery_gate": 5}
    r = g.audit(run, BUDGET)
    assert "mystery_gate" in r["unbudgetedGates"]
    assert r["fail"] is False  # not strict -> reported only
    r2 = g.audit(run, BUDGET, strict=True)
    assert r2["fail"] is True


def test_cli_exit_codes():
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    budget_path = tmp / "budget.json"
    budget_path.write_text(json.dumps(BUDGET), encoding="utf-8")

    ok = tmp / "ok.json"
    ok.write_text(json.dumps({"lint_claims": 10, "wiki_health": 20}), encoding="utf-8")
    bad = tmp / "bad.json"
    bad.write_text(json.dumps({"lint_claims": 200, "wiki_health": 200}), encoding="utf-8")

    script = str(ROOT / "tools" / "gate_cost_budget.py")
    p = subprocess.run([sys.executable, script, "--run-log", str(ok), "--budget", str(budget_path)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["go"] is True

    f = subprocess.run([sys.executable, script, "--run-log", str(bad), "--budget", str(budget_path)],
                       capture_output=True, text=True)
    assert f.returncode == 1, f.stderr
    assert json.loads(f.stdout)["fail"] is True

    m = subprocess.run([sys.executable, script, "--run-log", str(tmp / "nope.json"),
                        "--budget", str(budget_path)], capture_output=True, text=True)
    assert m.returncode == 2


def test_seeded_config_loads():
    cfg = g.load_budget(g.DEFAULT_BUDGET)
    assert cfg["lanes"]["fast"]["gpuAllowed"] is False
    assert cfg["gates"]["verifier_gate_gpu"]["gpu"] is True
    assert cfg["gates"]["verifier_gate_gpu"]["lane"] == "heavy"
    assert cfg["canClaimAGI"] is False


if __name__ == "__main__":
    test_fast_lane_within_ceiling_passes()
    test_fast_lane_over_ceiling_fails()
    test_gpu_gate_in_fast_lane_fails()
    test_gpu_gate_in_heavy_lane_ok()
    test_heavy_lane_over_ceiling_fails()
    test_lane_filter_restricts_to_fast()
    test_unbudgeted_gate_reported_and_strict_fails()
    test_cli_exit_codes()
    test_seeded_config_loads()
    print("ALL TESTS PASSED")
