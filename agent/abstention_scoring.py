# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Abstention-aware scoring — reward "I don't know", penalise confident-wrong.

Motivation (Kalai et al., *Why Language Models Hallucinate*, arXiv:2509.04664):
the dominant benchmark scoring is binary — ``+1`` correct, ``0`` for *both* a wrong
answer and an abstention. Under that rubric a confident guess weakly dominates "I
don't know" (it can only gain points), so optimal test-taking behaviour is to always
guess — exactly the incentive that produces hallucination. A fail-closed system that
abstains is *penalised* by binary scoring even though it is behaving more honestly.

This module scores a run under an **asymmetric** rubric that prices the three outcomes
distinctly::

    answered & correct  -> +1
    answered & wrong    -> -lambda      (the confident-wrong penalty)
    abstained           ->  0

``lambda = 0`` recovers the binary rubric (guessing is free); as ``lambda`` rises,
abstaining on a coin-flip becomes the rational choice. :func:`lambda_sweep` reports the
score under a grid of ``lambda`` and the **break-even** ``lambda*`` at which a system's
abstentions start to pay off versus always-guessing — the operating point Kalai argues
benchmarks should adopt.

Deterministic, pure standard library, no model, no network. Consumes the canonical
outcome-record / decision shape used across the conformal + graded paths:

    {"correct": bool, "action"|"verdict": "answer"|"hedge"|"abstain", ...}

A record is "answered" when its action is ``answer`` (``hedge`` is treated as answered
but flagged separately, since a hedge still commits content); ``abstain`` is the only
non-answer. The scorer never mutates the records and always reports both the legacy
binary score and the abstention-aware score (distinct metrics, per the no-overclaim
discipline).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

#: Default penalty grid for the sweep — 0 (binary) through strongly penalising.
DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)

_ANSWER_ACTIONS = ("answer",)
_HEDGE_ACTIONS = ("hedge",)
_ABSTAIN_ACTIONS = ("abstain", "abstained", "abstained_unverified")


def _action_of(rec: dict[str, Any]) -> str:
    a = rec.get("action") or rec.get("verdict") or ""
    a = str(a).strip().lower()
    if a in _ABSTAIN_ACTIONS:
        return "abstain"
    if a in _HEDGE_ACTIONS:
        return "hedge"
    if a in _ANSWER_ACTIONS:
        return "answer"
    # Unknown/empty action: treat as an answer only if it is not an abstention; we
    # fail closed by counting it as answered (so it can be penalised), never as a
    # free abstention.
    return "answer"


def classify(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Bucket records into answered-correct / answered-wrong / abstained counts.

    A ``hedge`` is counted as answered (it commits content) and folded into the
    correct/wrong tally by its ``correct`` flag, but also tracked separately.
    """
    n = ac = aw = ab = hedged = 0
    for rec in records:
        n += 1
        action = _action_of(rec)
        if action == "abstain":
            ab += 1
            continue
        if action == "hedge":
            hedged += 1
        if bool(rec.get("correct", False)):
            ac += 1
        else:
            aw += 1
    return {
        "n": n,
        "answeredCorrect": ac,
        "answeredWrong": aw,
        "abstained": ab,
        "hedged": hedged,
        "answered": ac + aw,
    }


def score(records: Sequence[dict[str, Any]], *, lam: float = 1.0) -> dict[str, Any]:
    """Score a run under the asymmetric rubric at penalty ``lam``.

    Returns the abstention-aware total/mean, the legacy binary total/mean (``lam=0``,
    abstention = wrong = 0), and the bucket counts. ``maxScore`` is ``n`` (all correct),
    so ``meanScore`` is in ``[-lam, 1]``.
    """
    if lam < 0:
        raise ValueError(f"lambda must be >= 0, got {lam}")
    c = classify(records)
    n = c["n"] or 1
    aware_total = c["answeredCorrect"] - lam * c["answeredWrong"]
    binary_total = c["answeredCorrect"]  # abstain and wrong both score 0
    return {
        "schema": "sophia.abstention_score.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "lambda": float(lam),
        "counts": c,
        "awareTotal": round(aware_total, 4),
        "awareMean": round(aware_total / n, 4),
        "binaryTotal": round(float(binary_total), 4),
        "binaryMean": round(binary_total / n, 4),
        "abstentionRate": round(c["abstained"] / n, 4),
        "selectiveAccuracy": (
            round(c["answeredCorrect"] / c["answered"], 4) if c["answered"] else None
        ),
    }


def always_answer_total(records: Sequence[dict[str, Any]], *, lam: float) -> float:
    """Score of the *always-guess* policy: every abstention is forced to an answer.

    The honest comparison Kalai makes — a system that never abstains scores every item
    as answered. We do not know whether a *forced* answer on an abstained item would be
    correct, so we price it at the empirical base accuracy of the items it *did* answer
    (a neutral, non-cherry-picked estimate). Reported as a baseline, clearly approximate.
    """
    c = classify(records)
    answered = c["answered"]
    base_acc = (c["answeredCorrect"] / answered) if answered else 0.0
    forced = c["abstained"]
    # forced items: base_acc fraction correct (+1), rest wrong (-lam)
    forced_total = forced * (base_acc * 1.0 + (1.0 - base_acc) * (-lam))
    return round((c["answeredCorrect"] - lam * c["answeredWrong"]) + forced_total, 4)


def lambda_sweep(
    records: Sequence[dict[str, Any]], *, lambdas: Sequence[float] = DEFAULT_LAMBDAS
) -> dict[str, Any]:
    """Score across a grid of ``lambda`` and find the break-even penalty.

    ``breakEvenLambda`` is the smallest grid ``lambda`` at which the actual policy
    (which abstains) scores **>=** the always-answer baseline — i.e. the penalty above
    which fail-closed abstention is the rational strategy. ``None`` if it never wins on
    the grid (e.g. a perfectly accurate run that should always answer).
    """
    points = []
    break_even = None
    for lam in lambdas:
        s = score(records, lam=lam)
        baseline = always_answer_total(records, lam=lam)
        wins = s["awareTotal"] >= baseline
        points.append({
            "lambda": float(lam),
            "awareTotal": s["awareTotal"],
            "awareMean": s["awareMean"],
            "alwaysAnswerTotal": baseline,
            "abstentionWins": wins,
        })
        if wins and break_even is None and lam > 0:
            break_even = float(lam)
    return {
        "schema": "sophia.abstention_lambda_sweep.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "n": classify(records)["n"],
        "curve": points,
        "breakEvenLambda": break_even,
        "honestBound": (
            "Methodology, not a capability claim: shows that under a confident-wrong "
            "penalty lambda >= breakEvenLambda, fail-closed abstention scores at least "
            "as well as always-guessing. The always-answer baseline prices forced "
            "answers at the run's own base accuracy (approximate). Requires real "
            "{correct, action} labels from a model run to be a result, not a demo."
        ),
    }


__all__ = [
    "DEFAULT_LAMBDAS",
    "classify",
    "score",
    "always_answer_total",
    "lambda_sweep",
]
