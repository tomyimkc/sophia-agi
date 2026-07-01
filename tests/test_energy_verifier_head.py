#!/usr/bin/env python3
"""O2 energy verifier: energy ordering, Best-of-N, held-out audit, fail-closed."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import energy_verifier_head as ev


def _mixed(domains=("math", "geo", "bio"), per=5):
    recs = []
    for d in domains:
        for i in range(per):
            recs.append({"answer": f"verified {d} claim {i}, confirmed",
                         "evidence": f"https://doi.org/10.1/{d}{i} supported",
                         "accepted": True, "domain": d, "correct": True, "query": f"{d}-q{i}"})
            recs.append({"answer": f"trust me {d} {i}, no citation, proven agi",
                         "evidence": "no source", "accepted": False, "domain": d,
                         "correct": False, "query": f"{d}-q{i}"})
    return recs


def test_no_answer_fail_closed():
    out = ev.run([{"evidence": "e", "accepted": True}])
    assert out.get("environmentArtifact") and out["ok"] is False


def test_degenerate_labels_fail_closed():
    out = ev.run([{"answer": "a", "evidence": "e", "accepted": True},
                  {"answer": "b", "evidence": "e", "accepted": True}])
    assert out.get("environmentArtifact") and "degenerate" in out["reason"]


def test_energy_orders_accepted_below_rejected():
    from agent.activation_probes import train_centroid_probe
    recs = _mixed()
    probe = ev._train_energy(recs)
    acc = [ev.energy_of(probe, r) for r in recs if r["accepted"]]
    rej = [ev.energy_of(probe, r) for r in recs if not r["accepted"]]
    assert sum(acc) / len(acc) < sum(rej) / len(rej)   # accepted pairs => lower energy


def test_bestofn_and_heldout_present():
    out = ev.run(_mixed(), seed=0)
    assert out["bestOfN"]["selectionAccuracy"] == 1.0   # min-energy picks the correct candidate
    assert out["heldOutDomain"] is not None
    assert "goodhartGap" in out["heldOutDomain"]
    assert out["hiddenStateFeaturizerReady"] is False   # the real energy-head seam is a stub


def test_serializable():
    import json
    json.dumps(ev.run(_mixed(), seed=0))
