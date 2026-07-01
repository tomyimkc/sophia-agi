#!/usr/bin/env python3
"""O1 consensus gate: fail-closed paths, honest verdicts, real-win detection."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import consensus_gate_oscillator as cg


def test_empty_is_fail_closed():
    out = cg.run([])
    assert out.get("environmentArtifact") and out["ok"] is False


def test_degenerate_labels_fail_closed():
    recs = [{"samples": ["a", "a"], "correct": True}, {"samples": ["b", "b"], "correct": True}]
    out = cg.run(recs)
    assert out.get("environmentArtifact") and "degenerate" in out["reason"]


def test_paraphrase_win_detected():
    # correct answers are PARAPHRASES (split the majority count but couple under Kuramoto);
    # wrong answers are divergent guesses. Consensus should beat the majority-agreement gate.
    recs = []
    for _ in range(25):
        recs.append({"samples": ["Paris is the capital", "The capital is Paris",
                                  "It is Paris", "Paris, the capital city"], "correct": True})
    for i in range(25):
        recs.append({"samples": [f"guess alpha {i}", f"wild beta {i}",
                                 f"random gamma {i}", f"noise delta {i}"], "correct": False})
    out = cg.run(recs, n_boot=2000, seed=0)
    assert out["verdict"] == "consensus_beats_baseline"
    assert out["aurcDelta"]["consensusWins"] is True
    assert out["aurcDelta"]["ci95"][0] > 0.0          # CI excludes 0


def test_no_fabricated_win_when_signals_equal():
    # identical samples both ways -> both gates equal -> must NOT claim a win
    recs = [{"samples": ["x", "x", "x"], "correct": i % 2 == 0} for i in range(20)]
    out = cg.run(recs, n_boot=1000, seed=0)
    assert out["verdict"] in ("inconclusive", "no_improvement")
    assert out["aurcDelta"]["consensusWins"] is False


def test_output_is_serializable_and_flagged():
    import json
    recs = [{"samples": ["a", "a", "b"], "correct": True},
            {"samples": ["c", "d", "e"], "correct": False}]
    out = cg.run(recs, n_boot=500, seed=0)
    json.dumps(out)                                    # must serialize
    assert out["candidateOnly"] is True and out["hashEmbedSeam"] is True
