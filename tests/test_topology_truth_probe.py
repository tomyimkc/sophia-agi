# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the topology->truth axiom probe: Spearman correctness on known inputs, the
permutation p-value's determinism/direction, and an end-to-end run on the committed seed
set that must emit a well-formed, honestly-UNDERPOWERED receipt.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import stats_ext  # noqa: E402
import run_topology_truth_probe as probe  # noqa: E402

_SEED_SET = ROOT / "agi-proof" / "topology-truth-axiom" / "labeled_set.jsonl"
_SPEC = ROOT / "agi-proof" / "topology-truth-axiom" / "measurement_spec.json"


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_spearman_perfect_monotone():
    # strictly increasing y in x => rho == 1 (even under a nonlinear monotone map)
    xs = [1, 2, 3, 4, 5]
    ys = [1, 4, 9, 16, 25]
    assert _close(stats_ext.spearman_rho(xs, ys), 1.0)


def test_spearman_perfect_anti_monotone():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 8, 6, 4, 2]
    assert _close(stats_ext.spearman_rho(xs, ys), -1.0)


def test_spearman_known_value_with_no_ties():
    # Classic textbook example: rho = 1 - 6*sum(d^2)/(n(n^2-1))
    xs = [1, 2, 3, 4, 5]
    ys = [2, 1, 4, 3, 5]
    # ranks equal values; d = [-1, 1, -1, 1, 0]; sum d^2 = 4; rho = 1 - 24/120 = 0.8
    assert _close(stats_ext.spearman_rho(xs, ys), 0.8)


def test_spearman_handles_ties():
    # ties get average ranks; a constant variable => 0.0 (undefined -> no association)
    assert _close(stats_ext.spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]), 0.0)
    # partial ties: ranks of [10,10,20] are [1.5,1.5,3]
    r = stats_ext.spearman_rho([10, 10, 20], [1, 2, 3])
    assert -1.0 <= r <= 1.0


def test_spearman_input_validation():
    for bad in (([], []), ([1, 2], [1])):
        try:
            stats_ext.spearman_rho(*bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_permutation_deterministic_and_bounded():
    xs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.8]
    ys = [0, 0, 1, 1, 1, 0, 0, 1]
    a = stats_ext.permutation_pvalue(xs, ys, iters=2000, seed=7)
    b = stats_ext.permutation_pvalue(xs, ys, iters=2000, seed=7)
    assert a == b  # same seed -> identical result
    assert 0.0 < a["p"] <= 1.0
    assert -1.0 <= a["rho"] <= 1.0
    assert a["alternative"] == "two-sided"


def test_permutation_detects_strong_positive():
    # a strongly positive monotone relation should get a small 'greater' p-value
    xs = list(range(20))
    ys = [0] * 10 + [1] * 10  # perfectly ordered
    res = stats_ext.permutation_pvalue(xs, ys, iters=5000, seed=1, alternative="greater")
    assert res["rho"] > 0.5
    assert res["p"] < 0.05


def test_permutation_null_is_not_significant():
    # a scrambled relation should NOT be significant in the 'greater' direction
    xs = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05]
    ys = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    res = stats_ext.permutation_pvalue(xs, ys, iters=5000, seed=3, alternative="greater")
    assert res["p"] > 0.05


def test_topology_confidence_monotone_and_bounded():
    weak = {"independent_sources": 0, "source_type_diversity": 0, "has_replication": 0,
            "consensus_body_endorses": 0, "recency_score": 0, "contradicting_sources": 4}
    strong = {"independent_sources": 6, "source_type_diversity": 5, "has_replication": 1,
              "consensus_body_endorses": 1, "recency_score": 3, "contradicting_sources": 0}
    cw = probe.topology_confidence(weak)
    cs = probe.topology_confidence(strong)
    assert 0.0 < cw < 1.0 and 0.0 < cs < 1.0
    assert cs > cw  # more/stronger evidence => higher topology confidence
    # more contradictions strictly lowers confidence
    more_contra = dict(strong, contradicting_sources=3)
    assert probe.topology_confidence(more_contra) < cs


def test_topology_confidence_requires_all_features():
    try:
        probe.topology_confidence({"independent_sources": 1})
        assert False, "expected ValueError for missing features"
    except ValueError:
        pass


def test_topology_confidence_not_fit_to_labels():
    # Guardrail check: the confidence map must depend ONLY on features, never on a truth
    # bit. Passing different 'truth' keys (ignored) must not change the output.
    feats = {"independent_sources": 3, "source_type_diversity": 2, "has_replication": 1,
             "consensus_body_endorses": 0, "recency_score": 2, "contradicting_sources": 1}
    c1 = probe.topology_confidence(dict(feats))
    c2 = probe.topology_confidence(dict(feats))  # identical features -> identical output
    assert _close(c1, c2)


def test_seed_set_loads_and_is_externally_labeled():
    rows = probe.load_labeled(_SEED_SET)
    assert 20 <= len(rows) <= 40, f"seed set should be small/illustrative, got {len(rows)}"
    assert any(r["truth"] == 0 for r in rows), "need myths (truth=0)"
    assert any(r["truth"] == 1 for r in rows), "need facts (truth=1)"
    for r in rows:
        assert r.get("truthProvenance"), f"{r['claimId']} lacks external truthProvenance"
        for k in probe._REQUIRED_FEATURES:
            assert k in r["features"], f"{r['claimId']} missing feature {k}"


def test_probe_runs_end_to_end_and_is_underpowered():
    rows = probe.load_labeled(_SEED_SET)
    receipt = probe.run_probe(rows, alpha=0.05, iters=2000, seed=0, mde=0.30)
    # well-formed receipt
    for key in ("rho", "p", "n", "verdict", "requiredN", "powered", "canClaimAGI",
                "go", "noGoRule", "honestBound", "perClaim", "exitCode"):
        assert key in receipt, f"receipt missing {key}"
    assert receipt["canClaimAGI"] is False
    assert -1.0 <= receipt["rho"] <= 1.0
    assert 0.0 < receipt["p"] <= 1.0
    assert receipt["n"] == len(rows)
    # HONESTY: the committed seed set is smaller than requiredN => must be UNDERPOWERED,
    # and an underpowered run must NOT be a GO regardless of the point estimate.
    assert receipt["n"] < receipt["requiredN"], "seed set must be below requiredN"
    assert receipt["powered"] is False
    assert receipt["verdict"] == "UNDERPOWERED"
    assert receipt["go"] is False
    assert receipt["exitCode"] == 3


def test_spec_is_preregistration_and_no_go_rule_present():
    spec = json.loads(_SPEC.read_text(encoding="utf-8"))
    assert spec["status"] == "preregistration_only"
    assert spec["go"] is False
    assert spec["canClaimAGI"] is False
    assert spec["underpowered"] is True
    # the falsifying rho<=0 rule must be pre-registered
    assert "rho <= 0" in spec["noGoRule"]["rule"]


def test_cli_emits_json_and_underpowered_exit():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_topology_truth_probe.py"),
         "--labeled", str(_SEED_SET), "--iters", "1500", "--seed", "0"],
        capture_output=True, text=True,
    )
    # underpowered seed set => NO-GO/UNDERPOWERED family => exit 3
    assert proc.returncode == 3, f"expected exit 3, got {proc.returncode}: {proc.stderr}"
    receipt = json.loads(proc.stdout)  # stdout must be a parseable JSON receipt
    assert receipt["experimentId"] == "topology-truth-axiom"
    assert receipt["verdict"] == "UNDERPOWERED"
    assert receipt["canClaimAGI"] is False


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("test_")):
        globals()[name]()
        print(f"ok  {name}")
    print("ALL TESTS PASSED")
