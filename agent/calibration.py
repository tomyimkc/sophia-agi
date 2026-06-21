"""Competence where no verifier exists — calibrated abstention.

When :mod:`agent.verifier_synthesis` abstains (no admitted check), the task is
unverifiable: there is no machine oracle to gate the answer. The honest behaviour
is not silence and not blind confidence — it is to answer with a *calibrated*
confidence and let a threshold decide whether to commit or defer. This module
measures whether that confidence is honest:

  - ECE (expected calibration error): binned, do stated confidences match
    observed accuracy? Low ECE ⇒ "70% sure" really means right ~70% of the time.
  - risk–coverage / selective risk: if you answer only above a confidence
    threshold, the error among answered should fall. Selective risk < base risk
    at <100% coverage is the falsifiable "knows what it doesn't know" claim.

Confidence sources are pluggable; a label-free deterministic one ships:
self-consistency (agreement across sampled answers). The metrics take confidences
and correctness from an EXTERNAL oracle — calibration is judged against ground
truth, never against the model's own say-so.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence


def _check(confidences: Sequence[float], correct: Sequence[bool]) -> int:
    n = len(confidences)
    if n != len(correct):
        raise ValueError("confidences and correct must be the same length")
    return n


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10,
) -> float:
    """Binned ECE: sum_b (|bin|/N) * |accuracy_b - confidence_b|. 0 = perfect."""
    n = _check(confidences, correct)
    if n == 0:
        return 0.0
    bins: dict[int, list] = {}
    for c, ok in zip(confidences, correct):
        c = min(max(float(c), 0.0), 1.0)
        b = min(n_bins - 1, int(c * n_bins))
        bins.setdefault(b, []).append((c, 1.0 if ok else 0.0))
    ece = 0.0
    for items in bins.values():
        conf = sum(c for c, _ in items) / len(items)
        acc = sum(o for _, o in items) / len(items)
        ece += (len(items) / n) * abs(acc - conf)
    return round(ece, 4)


def base_risk(correct: Sequence[bool]) -> float:
    """Error rate if you answer everything (100% coverage)."""
    n = len(correct)
    return round(sum(1 for ok in correct if not ok) / n, 4) if n else 0.0


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool],
) -> list:
    """Sort by confidence (desc); for each coverage prefix return its error rate.
    A well-calibrated scorer makes risk rise monotonically with coverage."""
    n = _check(confidences, correct)
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    out = []
    errs = 0
    for k, i in enumerate(order, start=1):
        if not correct[i]:
            errs += 1
        out.append({
            "coverage": round(k / n, 4),
            "risk": round(errs / k, 4),
            "threshold": round(float(confidences[i]), 4),
        })
    return out


def selective_risk(
    confidences: Sequence[float], correct: Sequence[bool], coverage: float,
) -> float:
    """Error rate among the most-confident ``coverage`` fraction of answers."""
    n = _check(confidences, correct)
    if n == 0:
        return 0.0
    k = max(1, min(n, math.ceil(coverage * n)))
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)[:k]
    errs = sum(1 for i in order if not correct[i])
    return round(errs / k, 4)


def area_under_risk_coverage(
    confidences: Sequence[float], correct: Sequence[bool],
) -> float:
    """AURC: mean selective risk over all coverages (lower is better)."""
    curve = risk_coverage_curve(confidences, correct)
    if not curve:
        return 0.0
    return round(sum(p["risk"] for p in curve) / len(curve), 4)


def self_consistency(samples: Sequence[Any]) -> tuple:
    """Label-free confidence: majority answer + fraction agreeing.

    Returns ``(answer, confidence)``. With no samples, ``(None, 0.0)``. This needs
    no ground truth at inference time, so it is a usable confidence source for the
    unverifiable case; :func:`expected_calibration_error` then audits it offline.
    """
    if not samples:
        return (None, 0.0)
    counts = Counter(str(s) for s in samples)
    answer, c = counts.most_common(1)[0]
    return (answer, round(c / len(samples), 4))


def calibration_report(
    confidences: Sequence[float], correct: Sequence[bool], *,
    coverage: float = 0.5, n_bins: int = 10,
) -> dict:
    """One-shot summary: ECE, AURC, and base vs selective risk at ``coverage``.
    ``selectiveBeatsBase`` is the falsifiable "knows what it doesn't know" flag."""
    n = _check(confidences, correct)
    br = base_risk(correct)
    sr = selective_risk(confidences, correct, coverage)
    return {
        "n": n,
        "ece": expected_calibration_error(confidences, correct, n_bins=n_bins),
        "aurc": area_under_risk_coverage(confidences, correct),
        "baseRisk": br,
        "coverage": coverage,
        "selectiveRisk": sr,
        "selectiveBeatsBase": sr < br,
    }
