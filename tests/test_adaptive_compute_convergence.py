#!/usr/bin/env python3
"""O4 adaptive compute: early-stop on easy, run-to-cap+abstain on hard, no quality loss."""
from __future__ import annotations
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import adaptive_compute_convergence as ac


def _recs():
    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
             "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho"]
    recs = []
    for i in range(15):
        recs.append({"samples": [f"the answer is {i}"] * 8, "correct": True})     # easy
    for _ in range(15):
        recs.append({"samples": [" ".join(rng.sample(words, 3)) for _ in range(8)],
                     "correct": False})                                            # hard/disjoint
    return recs


def test_too_few_samples_fail_closed():
    out = ac.run([{"samples": ["one"], "correct": True}])
    assert out.get("environmentArtifact") and out["ok"] is False


def test_easy_stops_early():
    out = ac.run(_recs(), k_max=8, r_confident=0.9, n_boot=1000, seed=0)
    easy = [p for p in out["records"] if p["correct"]]
    assert all(p["converged"] for p in easy)
    assert sum(p["kUsed"] for p in easy) / len(easy) < 4     # stops well before k_max=8


def test_hard_runs_to_cap_and_abstains():
    out = ac.run(_recs(), k_max=8, r_confident=0.9, n_boot=1000, seed=0)
    hard = [p for p in out["records"] if not p["correct"]]
    # at least some hard cases never converge -> abstain signal at k_max
    assert any(p["reason"] == "k_max_no_convergence" for p in hard)
    assert out["cost"]["nonConvergedAbstainRate"] > 0.0


def test_saves_samples_without_quality_loss():
    out = ac.run(_recs(), k_max=8, r_confident=0.9, n_boot=1000, seed=0)
    assert out["cost"]["samplesSavedFrac"] > 0.0
    assert out["quality"]["noQualityLoss"] is True


def test_serializable_and_flagged():
    import json
    out = ac.run(_recs(), k_max=8, n_boot=500, seed=0)
    json.dumps(out)
    assert out["candidateOnly"] is True
