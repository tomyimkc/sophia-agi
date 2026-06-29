#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verification-Gated Recurrent Depth (VGRD) — the Phase-2 coupling, pure Python.

The novel thesis of this whole line of work: a recurrent-depth model's *extra loops* are
latent reasoning steps, so the trajectory of its per-loop predictions is itself a confidence
signal — and that signal can drive Sophia's fail-closed provenance gate. Where RESULTS.md
validated **self-consistency across samples** as the one working confidence signal, VGRD adds
**latent self-consistency across DEPTH**: an answerable query settles to a stable prediction
as loops accrue; an unanswerable one keeps drifting. Abstaining on the drifters is exactly
Sophia's "abstain instead of fabricate".

This module is the policy + measurement machinery, dependency-free so it runs in the same
torch-free CI lane as ``recurrent_depth.py`` and composes with the real Sophia gate via a
pluggable ``verify_fn`` (in production: ``record_claim → verify_claim`` → accepted/abstain).

    decision = vgrd_decide(trajectory, verify_fn=sophia_gate)   # accept | abstain | block

The decision is **fail-closed**: low depth-confidence → ``abstain``; a provenance check that
rejects the answer → ``block``; only a stable, verified answer → ``accept``.

What is measured here (and what is NOT). We validate the POLICY + METRIC end-to-end on a
synthetic mix of convergent ("answerable") and oscillating ("unanswerable") per-loop
trajectories: abstaining on low-confidence cases lifts selective accuracy and drives
fabrication on the unanswerable set to ~0 — the same shape as the SimpleQA selective-
prediction result, on a controlled substrate where ground truth is known. This is a
methodology study of the coupling, NOT a capability claim: the *signal-quality on real data*
needs the trained RDT checkpoint (Phase 1.2) scored under the no-overclaim gate (Phase 3).

    python -m pretraining.architecture.vgrd --quick
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Load-bearing phrase asserted by offline_invariants() AND the test (house rule).
SCOPE_KEY = "methodology study of the coupling"


# ---------------------------------------------------------------------------
# Confidence: latent self-consistency across depth
# ---------------------------------------------------------------------------

def _argmax(vec: "list[float]") -> int:
    return max(range(len(vec)), key=lambda i: vec[i])


def loop_predictions(trajectory: "list[list[float]]") -> "list[int]":
    """Per-loop argmax prediction from a trajectory of per-loop logit vectors."""
    return [_argmax(v) for v in trajectory]


def depth_confidence(trajectory: "list[list[float]]", window: int | None = None) -> "dict":
    """Latent self-consistency across the final ``window`` loops.

    Returns the modal late prediction, the agreement fraction (= confidence in [0,1]), and the
    settle step (the last loop at which the prediction changed; earlier = more confident). A
    query whose latent prediction has converged scores near 1.0; a drifting/oscillating one
    scores low."""
    preds = loop_predictions(trajectory)
    T = len(preds)
    if T == 0:
        return {"answer": None, "confidence": 0.0, "settle_step": None, "preds": []}
    w = window if window is not None else max(1, T // 2)
    tail = preds[-w:]
    # modal prediction over the tail
    counts: dict[int, int] = {}
    for p in tail:
        counts[p] = counts.get(p, 0) + 1
    answer = max(counts, key=lambda k: counts[k])
    confidence = counts[answer] / len(tail)
    # settle step: last index where the prediction differs from the final prediction
    final = preds[-1]
    settle = 0
    for i, p in enumerate(preds):
        if p != final:
            settle = i + 1
    return {"answer": answer, "confidence": round(confidence, 4),
            "settle_step": settle, "preds": preds}


# ---------------------------------------------------------------------------
# The fail-closed accept / abstain / block decision
# ---------------------------------------------------------------------------

def vgrd_decide(trajectory: "list[list[float]]", *, verify_fn=None,
                accept_threshold: float = 0.8, window: int | None = None) -> "dict":
    """Couple depth-confidence to a provenance check, fail-closed.

    - ``verify_fn(answer) -> bool`` is the provenance gate (e.g. Sophia ``verify_claim`` →
      ``accepted``). If it returns False the answer is **blocked** regardless of confidence.
    - depth-confidence below ``accept_threshold`` → **abstain** (the model has not settled).
    - a settled answer that passes verification → **accept**.

    Returns {verdict, answer, confidence, settle_step}. ``verdict`` ∈ {accept, abstain, block}.
    """
    dc = depth_confidence(trajectory, window=window)
    answer, conf = dc["answer"], dc["confidence"]
    if answer is None:
        return {"verdict": "abstain", "answer": None, "confidence": 0.0,
                "settle_step": None, "reason": "empty trajectory"}
    if conf < accept_threshold:
        return {"verdict": "abstain", "answer": answer, "confidence": conf,
                "settle_step": dc["settle_step"], "reason": "latent prediction not settled"}
    if verify_fn is not None and not verify_fn(answer):
        return {"verdict": "block", "answer": answer, "confidence": conf,
                "settle_step": dc["settle_step"], "reason": "provenance check rejected the answer"}
    return {"verdict": "accept", "answer": answer, "confidence": conf,
            "settle_step": dc["settle_step"], "reason": "settled and verified"}


# ---------------------------------------------------------------------------
# Selective-prediction harness (same shape as the SimpleQA result in RESULTS.md)
# ---------------------------------------------------------------------------

def selective_prediction(cases: "list[dict]", *, coverages=(1.0, 0.8, 0.5, 0.2),
                         window: int | None = None) -> "dict":
    """Rank cases by depth-confidence; at each coverage answer the top fraction and abstain on
    the rest. Reports selective accuracy (correct / answered) and fabrication rate
    (answered-but-unanswerable / unanswerable) per coverage.

    Each case: {"trajectory": [...], "target": int_or_None}. ``target=None`` marks a genuinely
    UNANSWERABLE case — answering it at all is a fabrication."""
    scored = []
    for c in cases:
        dc = depth_confidence(c["trajectory"], window=window)
        scored.append({"conf": dc["confidence"], "answer": dc["answer"],
                       "target": c.get("target")})
    n = len(scored)
    n_unans = sum(1 for s in scored if s["target"] is None)
    order = sorted(range(n), key=lambda i: scored[i]["conf"], reverse=True)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        answered = [scored[i] for i in order[:k]]
        correct = sum(1 for s in answered
                      if s["target"] is not None and s["answer"] == s["target"])
        fabricated = sum(1 for s in answered if s["target"] is None)
        sel_acc = correct / len(answered) if answered else 0.0
        rows.append({"coverage": cov, "answered": len(answered),
                     "selective_accuracy": round(sel_acc, 4),
                     "fabrications": fabricated,
                     "fabrication_rate_on_unanswerable": round(
                         fabricated / n_unans, 4) if n_unans else 0.0})
    return {"n": n, "n_unanswerable": n_unans, "rows": rows}


# ---------------------------------------------------------------------------
# Synthetic substrate: convergent (answerable) vs oscillating (unanswerable)
# ---------------------------------------------------------------------------

def _logits_for(pred: int, vocab: int, sharp: float = 6.0) -> "list[float]":
    v = [0.0] * vocab
    v[pred] = sharp
    return v


def make_case(answerable: bool, vocab: int, loops: int, *, seed: int) -> "dict":
    """An answerable case's per-loop prediction drifts early then SETTLES on its target; an
    unanswerable case keeps OSCILLATING between symbols and never settles."""
    rng = random.Random(seed)
    if answerable:
        target = rng.randrange(vocab)
        settle = rng.randint(1, max(1, loops // 2))
        traj = []
        for t in range(loops):
            pred = target if t >= settle else rng.randrange(vocab)
            traj.append(_logits_for(pred, vocab))
        return {"trajectory": traj, "target": target}
    # unanswerable: oscillate between two distractors for the whole trajectory
    a, b = rng.sample(range(vocab), 2)
    traj = [_logits_for(a if t % 2 == 0 else b, vocab) for t in range(loops)]
    return {"trajectory": traj, "target": None}


def _make_mix(n_ans: int, n_unans: int, vocab: int, loops: int, seed: int) -> "list[dict]":
    cases = []
    for i in range(n_ans):
        cases.append(make_case(True, vocab, loops, seed=seed + i))
    for i in range(n_unans):
        cases.append(make_case(False, vocab, loops, seed=seed + 10_000 + i))
    random.Random(seed).shuffle(cases)
    return cases


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def run_study(*, quick: bool = False, seed: int = 0) -> "dict":
    vocab, loops = (8, 8) if quick else (12, 12)
    n_ans, n_unans = (20, 20) if quick else (60, 60)
    cases = _make_mix(n_ans, n_unans, vocab, loops, seed)

    sp = selective_prediction(cases)

    # Decision breakdown at the default threshold, including a verify_fn that rejects a
    # specific (otherwise-confident) answer to exercise the BLOCK path (fail-closed).
    blocked_answer = cases[0]["target"] if cases[0]["target"] is not None else 0
    verify_fn = lambda ans: ans != blocked_answer  # noqa: E731 — reject one answer
    verdicts = {"accept": 0, "abstain": 0, "block": 0}
    fabrications = 0
    correct_accepts = 0
    for c in cases:
        d = vgrd_decide(c["trajectory"], verify_fn=verify_fn)
        verdicts[d["verdict"]] += 1
        if d["verdict"] == "accept":
            if c["target"] is None:
                fabrications += 1
            elif d["answer"] == c["target"]:
                correct_accepts += 1

    full = sp["rows"][0]["selective_accuracy"]
    low = sp["rows"][-1]["selective_accuracy"]
    return {
        "study": "Verification-Gated Recurrent Depth (VGRD) — depth-confidence → Sophia gate",
        "config": {"vocab": vocab, "loops": loops, "n_answerable": n_ans,
                   "n_unanswerable": n_unans, "seed": seed, "quick": quick},
        "selective_prediction": sp,
        "selective_accuracy_lift_low_minus_full": round(low - full, 4),
        "decision_breakdown": verdicts,
        "accepted_fabrications_on_unanswerable": fabrications,
        "block_path_exercised": verdicts["block"] > 0,
        "abstains_on_unanswerable": verdicts["abstain"] >= n_unans - verdicts["block"],
        "interpretation": (
            "Depth-confidence (latent self-consistency across loops) ranks answerable cases "
            "above unanswerable ones, so abstaining on the low-confidence tail lifts selective "
            "accuracy and drives accepted fabrications on the unanswerable set to ~0 — the "
            "Sophia property (abstain instead of fabricate), now sourced from the recurrence's "
            "own loop trajectory and gated fail-closed through verify_fn."
        ),
        "honest_scope": (
            "A " + SCOPE_KEY + ": the VGRD policy + selective-prediction metric validated on a "
            "controlled convergent/oscillating substrate where ground truth is known. NOT a "
            "capability claim. The signal-quality of depth-confidence on REAL data needs the "
            "trained RDT checkpoint (Phase 1.2) scored by >=2 judge families with CIs excluding "
            "zero (Phase 3), exactly as the SimpleQA selective-prediction result was."
        ),
    }


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Decision logic: stable+verified → accept; unsettled → abstain; verify-fail → block.
    vocab = 6
    stable = [_logits_for(3, vocab) for _ in range(8)]
    osc = [_logits_for(1 if t % 2 else 4, vocab) for t in range(8)]
    checks["accept_on_stable"] = vgrd_decide(stable)["verdict"] == "accept"
    checks["abstain_on_oscillating"] = vgrd_decide(osc)["verdict"] == "abstain"
    checks["block_on_verify_fail"] = (
        vgrd_decide(stable, verify_fn=lambda a: False)["verdict"] == "block")
    checks["empty_trajectory_abstains"] = vgrd_decide([])["verdict"] == "abstain"

    rep = run_study(quick=True, seed=0)
    sp = rep["selective_prediction"]

    # 2. Abstaining lifts selective accuracy (low coverage >= full coverage).
    detail["sel_acc_full"] = sp["rows"][0]["selective_accuracy"]
    detail["sel_acc_low"] = sp["rows"][-1]["selective_accuracy"]
    checks["abstention_lifts_selective_accuracy"] = (
        sp["rows"][-1]["selective_accuracy"] >= sp["rows"][0]["selective_accuracy"])

    # 3. At the lowest coverage, zero fabrications on the unanswerable set.
    checks["no_fabrication_at_low_coverage"] = (
        sp["rows"][-1]["fabrication_rate_on_unanswerable"] == 0.0)

    # 4. The default-threshold policy accepts ~no unanswerable case (fail-closed).
    detail["accepted_fabrications"] = rep["accepted_fabrications_on_unanswerable"]
    checks["policy_does_not_fabricate"] = rep["accepted_fabrications_on_unanswerable"] == 0

    # 5. The block path is actually exercised (verify_fn rejected a confident answer).
    checks["block_path_exercised"] = bool(rep["block_path_exercised"])

    # 6. Determinism.
    checks["deterministic"] = (run_study(quick=True, seed=0) == rep)

    # 7. Honest scope present with the load-bearing phrase.
    checks["scope_present"] = SCOPE_KEY.lower() in rep["honest_scope"].lower()

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = run_study(quick=args.quick, seed=args.seed)
    out = args.out or (HERE / ("vgrd-quick-latest.json" if args.quick else "vgrd-latest.json"))
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print("== VGRD selective prediction ==")
    for r in rep["selective_prediction"]["rows"]:
        print(f"  coverage={r['coverage']:>4}  sel_acc={r['selective_accuracy']:.3f}  "
              f"fab_rate_unanswerable={r['fabrication_rate_on_unanswerable']:.3f}")
    print(f"verdicts: {rep['decision_breakdown']}  "
          f"accepted_fabrications={rep['accepted_fabrications_on_unanswerable']}")


if __name__ == "__main__":
    main()
