# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrated abstention for VQA — risk-coverage over the trap suite (workstream C).

The roadmap's calibration pillar: measure whether a VLM *knows when it can't see
the answer*. This module reuses the existing text-side calibration layer
(``agent.calibration`` — ECE, risk-coverage, selective risk, AURC) verbatim and
drives it from visual-trap outcomes, so the multimodal eval inherits the same
honest yardstick rather than reimplementing it.

A *confidence answer function* is ``(trap) -> (answer_text, confidence in [0,1])``.
``run_with_confidence`` judges each answer (correct == affirmed the verifier gold)
and pairs correctness with the stated confidence; ``calibration_report`` then asks
the falsifiable question: *if you answer only above a confidence threshold, does
the error rate fall?* Selective risk < base risk at <100% coverage is the
"knows what it doesn't know" claim — measured, not asserted.
"""

from __future__ import annotations

import random

from agent import calibration as cal  # reuse the text-side calibration metrics
from multimodal_bench import judge as judge_mod
from multimodal_bench import runner


def run_with_confidence(traps: list, conf_answer_fn, judge_fn=None) -> list:
    """Score each trap and pair correctness with the model's stated confidence."""
    judge_fn = judge_fn or judge_mod.lexical_judge
    records = []
    for t in traps:
        answer, conf = conf_answer_fn(t)
        v = judge_fn(answer, t)
        records.append({
            "id": t["id"],
            "confidence": float(conf),
            "correct": bool(v.affirmed_gold),
            "abstained": bool(v.abstained),
        })
    return records


def calibration_report(records: list, *, coverages=(0.5, 0.8, 1.0), n_bins: int = 10) -> dict:
    """ECE + base risk + AURC + selective risk at several coverages + the curve."""
    confs = [r["confidence"] for r in records]
    correct = [r["correct"] for r in records]
    return {
        "n": len(records),
        "ece": cal.expected_calibration_error(confs, correct, n_bins=n_bins),
        "baseRisk": cal.base_risk(correct),
        "aurc": cal.area_under_risk_coverage(confs, correct),
        "selectiveRisk": {str(c): cal.selective_risk(confs, correct, c) for c in coverages},
        "riskCoverageCurve": cal.risk_coverage_curve(confs, correct),
        "abstentionRate": round(sum(1 for r in records if r["abstained"]) / len(records), 4) if records else 0.0,
    }


# --- synthetic confidence sources (offline demo / test fixtures) ----------- #


def make_synthetic_confidence_fn(*, error_rate: float, calibrated: bool, seed: int = 0):
    """A stand-in VLM whose confidence is (or isn't) honest.

    With probability ``1 - error_rate`` it answers correctly (gold), else it falls
    for the trap distractor. If ``calibrated``, confidence tracks correctness —
    high (~0.95) when right, low (~0.1) when wrong — so it matches the per-bin
    accuracy (low ECE) AND lets thresholding cut risk (low AURC). If not, it is
    uniformly overconfident (~0.97) — so its stated confidence far exceeds its
    accuracy (high ECE) and thresholding cannot separate right from wrong (flat
    risk, high AURC). Deterministic given ``seed``.
    """
    rng = random.Random(seed)

    def fn(trap: dict):
        will_be_correct = rng.random() >= error_rate
        jitter = rng.uniform(-0.04, 0.04)
        if not calibrated:
            conf = 0.97 + jitter  # confidently right or wrong, indistinguishably
        elif will_be_correct:
            conf = 0.95 + jitter
        else:
            conf = 0.10 + jitter
        answer = _gold_text(trap) if will_be_correct else _trap_text(trap)
        return answer, min(max(conf, 0.0), 1.0)

    return fn


def _gold_text(trap: dict) -> str:
    atype, gold = trap["answer_type"], str(trap["gold_answer"])
    if atype == "yesno":
        return "Yes." if gold.lower() == "yes" else "No, that is not in the image."
    if atype == "count":
        return f"There are {gold}."
    return "There is no legible text." if gold.lower() == "none" else f"It says {gold}."


def _trap_text(trap: dict) -> str:
    atype, trapped = trap["answer_type"], str(trap["trap_answer"])
    if atype == "yesno":
        return "Yes." if trapped.lower() == "yes" else "No."
    if atype == "count":
        return f"There are {trapped}."
    return f"It says {trapped}."


def demo(seed: int = 0) -> dict:
    """Offline A/B: a calibrated model vs an overconfident one on the full suite."""
    traps = runner.load_all_traps()
    calibrated = run_with_confidence(traps, make_synthetic_confidence_fn(error_rate=0.3, calibrated=True, seed=seed))
    overconf = run_with_confidence(traps, make_synthetic_confidence_fn(error_rate=0.3, calibrated=False, seed=seed))
    return {"calibrated": calibration_report(calibrated), "overconfident": calibration_report(overconf)}
