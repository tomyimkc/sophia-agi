# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Standardized concept-discipline metrics + bootstrap CI for the testbench.

Both the inference-uplift A/B harness (``ontology_improvement``) and the RLVR
eval report on the SAME metric set, computed here so the numbers are comparable:

  - conceptMergeViolationRate — fraction of items where the answer ASSERTS an
    unscoped cross-tradition identity (the failure the gate exists to prevent);
  - abstention precision / recall vs the oracle (should-abstain == not answerable);
  - groundedCorrectRate — correct verdict on answerable, non-abstaining items;
  - confidentWrongRate — answerable items answered confidently AND wrong;
  - overAbstainRate — answerable items the policy abstained on (the AlphaAlign
    failure mode the testbench must catch);
  - ECE / Brier (if per-item confidence is supplied);
  - a risk-coverage curve.

A "record" is a normalized dict:
  {answerable: bool, abstained: bool, violation: bool, correct: bool,
   confidence: float|None}

Honesty rules baked in: report the false-positive (over-abstain) cost separately,
never collapse to one aggregate, and use a bootstrap CI for any A/B delta (a point
estimate is not a result). See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import random
from typing import Any, Callable


def summarize(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    answerable = [r for r in records if r.get("answerable", True)]
    ill_posed = [r for r in records if not r.get("answerable", True)]

    violations = sum(1 for r in records if r.get("violation"))
    confident_wrong = sum(
        1 for r in answerable if not r.get("abstained") and not r.get("correct")
    )
    grounded_correct = sum(1 for r in answerable if not r.get("abstained") and r.get("correct"))
    over_abstain = sum(1 for r in answerable if r.get("abstained"))

    # abstention confusion vs oracle (should-abstain == ill-posed).
    ab_tp = sum(1 for r in ill_posed if r.get("abstained"))
    ab_fn = sum(1 for r in ill_posed if not r.get("abstained"))
    ab_fp = sum(1 for r in answerable if r.get("abstained"))
    ab_precision = ab_tp / (ab_tp + ab_fp) if (ab_tp + ab_fp) else None
    ab_recall = ab_tp / (ab_tp + ab_fn) if (ab_tp + ab_fn) else None

    confs = [(r.get("confidence"), bool(r.get("correct"))) for r in records if r.get("confidence") is not None]
    out = {
        "n": n,
        "nAnswerable": len(answerable),
        "nIllPosed": len(ill_posed),
        "conceptMergeViolationRate": violations / n,
        "groundedCorrectRate": grounded_correct / len(answerable) if answerable else None,
        "confidentWrongRate": confident_wrong / len(answerable) if answerable else None,
        "overAbstainRate": over_abstain / len(answerable) if answerable else None,
        "abstentionPrecision": ab_precision,
        "abstentionRecall": ab_recall,
        "ece": expected_calibration_error(confs) if confs else None,
        "brier": brier_score(confs) if confs else None,
        "riskCoverage": risk_coverage_curve(records),
    }
    return out


def expected_calibration_error(conf_correct: list[tuple[float, bool]], *, bins: int = 10) -> float:
    """ECE over (confidence, correct) pairs."""
    if not conf_correct:
        return 0.0
    n = len(conf_correct)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in conf_correct if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        total += (len(bucket) / n) * abs(avg_conf - acc)
    return total


def brier_score(conf_correct: list[tuple[float, bool]]) -> float:
    if not conf_correct:
        return 0.0
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in conf_correct) / len(conf_correct)


def risk_coverage_curve(records: list[dict], *, points: int = 5) -> list[dict]:
    """Risk (error rate) vs coverage as the abstention threshold varies.

    Items with a confidence are ranked; at each coverage level we 'answer' the top
    fraction and compute the error among answered answerable items. Items without a
    confidence are treated as fully covered (answered). Coarse (``points`` levels)."""
    answerable = [r for r in records if r.get("answerable", True)]
    if not answerable:
        return []
    with_conf = [r for r in answerable if r.get("confidence") is not None]
    ranked = sorted(with_conf, key=lambda r: r["confidence"], reverse=True)
    no_conf = [r for r in answerable if r.get("confidence") is None]
    curve: list[dict] = []
    for i in range(1, points + 1):
        frac = i / points
        k = int(round(frac * len(ranked)))
        answered = ranked[:k] + no_conf
        if not answered:
            curve.append({"coverage": round(frac, 3), "risk": None, "answered": 0})
            continue
        errors = sum(1 for r in answered if not r.get("correct"))
        curve.append({
            "coverage": round(len(answered) / len(answerable), 3),
            "risk": round(errors / len(answered), 4),
            "answered": len(answered),
        })
    return curve


def bootstrap_delta(
    arm_a: list[float],
    arm_b: list[float],
    *,
    reducer: Callable[[list[float]], float] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired/unpaired bootstrap of ``reduce(arm_b) - reduce(arm_a)`` (default
    reducer = mean). Returns the point delta, the (1-alpha) CI, and whether the CI
    excludes 0 — the no-overclaim gate's evidence bar."""
    reduce = reducer or (lambda xs: sum(xs) / len(xs) if xs else 0.0)
    if not arm_a or not arm_b:
        return {"delta": None, "ciLow": None, "ciHigh": None, "excludesZero": False, "nBoot": 0}
    rng = random.Random(seed)
    point = reduce(arm_b) - reduce(arm_a)
    deltas: list[float] = []
    for _ in range(n_boot):
        sa = [arm_a[rng.randrange(len(arm_a))] for _ in arm_a]
        sb = [arm_b[rng.randrange(len(arm_b))] for _ in arm_b]
        deltas.append(reduce(sb) - reduce(sa))
    deltas.sort()
    lo = deltas[int((alpha / 2) * n_boot)]
    hi = deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {
        "delta": round(point, 4),
        "ciLow": round(lo, 4),
        "ciHigh": round(hi, 4),
        "excludesZero": (lo > 0 and hi > 0) or (lo < 0 and hi < 0),
        "nBoot": n_boot,
    }


def compare_arms(baseline: list[dict], treatment: list[dict], *, seed: int = 0) -> dict[str, Any]:
    """Summaries of both arms + bootstrap deltas on the headline metrics.

    A positive uplift is: violation rate DOWN, grounded-correct UP, confident-wrong
    DOWN, with the over-abstain rate NOT up. Each delta carries a CI + excludesZero."""
    base = summarize(baseline)
    treat = summarize(treatment)

    def col(records, key):
        # per-item 0/1 series for bootstrapping a rate
        if key == "violation":
            return [1.0 if r.get("violation") else 0.0 for r in records]
        if key == "grounded_correct":
            return [1.0 if (r.get("answerable", True) and not r.get("abstained") and r.get("correct")) else 0.0
                    for r in records if r.get("answerable", True)]
        if key == "confident_wrong":
            return [1.0 if (not r.get("abstained") and not r.get("correct")) else 0.0
                    for r in records if r.get("answerable", True)]
        if key == "over_abstain":
            return [1.0 if r.get("abstained") else 0.0 for r in records if r.get("answerable", True)]
        return []

    deltas = {
        "conceptMergeViolationRate": bootstrap_delta(col(baseline, "violation"), col(treatment, "violation"), seed=seed),
        "groundedCorrectRate": bootstrap_delta(col(baseline, "grounded_correct"), col(treatment, "grounded_correct"), seed=seed),
        "confidentWrongRate": bootstrap_delta(col(baseline, "confident_wrong"), col(treatment, "confident_wrong"), seed=seed),
        "overAbstainRate": bootstrap_delta(col(baseline, "over_abstain"), col(treatment, "over_abstain"), seed=seed),
    }
    return {
        "schema": "sophia.concept_metrics_compare.v1", "candidateOnly": True,
        "level3Evidence": False, "canClaimAGI": False,
        "baseline": base, "treatment": treat, "deltas": deltas,
    }


__all__ = [
    "summarize", "expected_calibration_error", "brier_score", "risk_coverage_curve",
    "bootstrap_delta", "compare_arms",
]
