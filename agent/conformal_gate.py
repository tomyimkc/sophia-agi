# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Conformal abstention gate for the Conscience Kernel.

This converts arbitrary nonconformity scores into answer/abstain decisions with a
simple split-conformal threshold. The implementation is deterministic/offline and
uses held-out calibration rows, not model self-report, as the source of trust.
"""
from __future__ import annotations

import json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConformalPolicy:
    alpha: float
    threshold: float
    n_calibration: int
    target_coverage: float
    risk_bucket: str = "normal"

    def decide(self, nonconformity: float) -> dict[str, Any]:
        score = float(nonconformity)
        answer = score <= self.threshold
        return {
            "schema": "sophia.conformal_decision.v1",
            "verdict": "answer" if answer else "abstain",
            "nonconformity": round(score, 4),
            "threshold": round(self.threshold, 4),
            "riskBucket": self.risk_bucket,
            "coverageGuarantee": round(1 - self.alpha, 4),
            "candidateOnly": True,
            "level3Evidence": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"schema": "sophia.conformal_policy.v1", "candidateOnly": True, "level3Evidence": False}


def quantile_threshold(scores: list[float], *, alpha: float) -> float:
    if not scores:
        return 0.0
    xs = sorted(float(s) for s in scores)
    n = len(xs)
    # split conformal finite-sample quantile: ceil((n+1)*(1-alpha))/n
    k = min(n, max(1, math.ceil((n + 1) * (1 - alpha))))
    return round(xs[k - 1], 4)


def fit_conformal_policy(rows: list[dict[str, Any]], *, alpha: float = 0.1, risk_bucket: str = "normal") -> ConformalPolicy:
    bucket = [r for r in rows if r.get("risk", "normal") == risk_bucket]
    if not bucket:
        bucket = list(rows)
    # Correct calibration rows define acceptable nonconformity. Incorrect rows are
    # evaluated in the report but do not set the coverage threshold.
    correct_scores = [float(r["nonconformity"]) for r in bucket if bool(r.get("correct", False))]
    return ConformalPolicy(alpha=alpha, threshold=quantile_threshold(correct_scores, alpha=alpha), n_calibration=len(correct_scores), target_coverage=round(1 - alpha, 4), risk_bucket=risk_bucket)


def evaluate_policy(policy: ConformalPolicy, rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket = [r for r in rows if r.get("risk", "normal") == policy.risk_bucket] or list(rows)
    decisions = []
    answered = correct_answered = false_answered = 0
    for r in bucket:
        d = policy.decide(float(r["nonconformity"]))
        ok = bool(r.get("correct", False))
        if d["verdict"] == "answer":
            answered += 1
            correct_answered += int(ok)
            false_answered += int(not ok)
        decisions.append({"id": r.get("id"), "correct": ok, **d})
    n = len(bucket)
    return {
        "schema": "sophia.conformal_eval.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "policy": policy.to_dict(),
        "n": n,
        "metrics": {
            "coverage": round(answered / n, 4) if n else 0.0,
            "selectiveAccuracy": round(correct_answered / answered, 4) if answered else 0.0,
            "falseAnswerRate": round(false_answered / answered, 4) if answered else 0.0,
        },
        "decisions": decisions,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_conformal_report(rows_path: str | Path, out: str | Path, *, alpha: float = 0.1) -> dict[str, Any]:
    rows = load_jsonl(rows_path)
    reports = {risk: evaluate_policy(fit_conformal_policy(rows, alpha=alpha, risk_bucket=risk), rows) for risk in sorted({r.get("risk", "normal") for r in rows})}
    report = {"schema": "sophia.conformal_conscience_report.v1", "candidateOnly": True, "level3Evidence": False, "alpha": alpha, "byRisk": reports, "ok": all(v["metrics"]["falseAnswerRate"] <= 0.05 for v in reports.values())}
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report


def conformal_risk_control(
    rows: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
    risk_bucket: str = "normal",
    loss_bound: float = 1.0,
) -> dict[str, Any]:
    """Distribution-free finite-sample bound on the DEPLOYED false-answer rate (CRC).

    Split-conformal **risk** control (Angelopoulos, Bates, Fisch, Lei, Schuster 2022,
    arXiv:2208.02814). For the monotone loss ``L_i(tau) = 1[answered_i(tau) and not
    correct_i]`` — non-decreasing in the answer threshold ``tau`` (``answer iff
    nonconformity <= tau``) — pick the LARGEST ``tau`` such that

        (n * Rhat(tau) + B) / (n + 1) <= alpha          (B = loss_bound = 1)

    where ``Rhat(tau) = mean_i L_i(tau)`` over ALL n calibration rows. The CRC theorem
    then gives, on a fresh exchangeable point, ``E[1(answered and wrong)] <= alpha``.

    This bounds the **marginal** false-answer rate over ALL queries (an abstention
    contributes 0) — the quantity an operator deploys against — distinct from the
    *conditional* ``falseAnswerRate = false/answered`` that :func:`evaluate_policy` only
    REPORTS (never calibrates as a bound). Returns ``feasible=False`` when ``n`` is too
    small (``B/(n+1) > alpha``): even abstaining on everything cannot certify the bound —
    the honest "not enough calibration data" signal (e.g. ``alpha=0.05`` needs ``n>=19``).

    Deterministic/offline; ``candidateOnly``. Distribution-free but assumes calibration/
    test **exchangeability**; a per-``risk_bucket`` fit is the group-conditional
    refinement when risk strata are not exchangeable.
    """
    bucket = [r for r in rows if r.get("risk", "normal") == risk_bucket] or list(rows)
    n = len(bucket)
    B = float(loss_bound)
    base = {
        "schema": "sophia.conformal_risk_control.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "riskBucket": risk_bucket,
        "alpha": alpha,
        "n": n,
        "guarantee": "E[answered AND wrong] <= alpha (finite-sample, distribution-free; assumes exchangeability)",
    }
    if n == 0:
        return {**base, "feasible": False, "threshold": None, "reason": "no rows"}

    def rhat(tau: float) -> float:
        return sum(
            1.0 for r in bucket
            if float(r["nonconformity"]) <= tau and not bool(r.get("correct", False))
        ) / n

    abstain_all_bound = B / (n + 1)  # Rhat(-inf) == 0, so bound = B/(n+1)
    # The loss is non-decreasing in tau, so (n*Rhat(tau)+B)/(n+1) is non-decreasing:
    # scan candidate thresholds ascending and keep the LARGEST tau whose bound still holds.
    best_tau: float | None = None
    best_rhat = 0.0
    for tau in sorted({float(r["nonconformity"]) for r in bucket}):
        r = rhat(tau)
        if (n * r + B) / (n + 1) <= alpha:
            best_tau, best_rhat = tau, r
        else:
            break  # monotone: no larger tau can satisfy it either
    if best_tau is None:
        if abstain_all_bound <= alpha:  # certified only by answering nothing
            return {**base, "feasible": True, "threshold": float("-inf"),
                    "empiricalRisk": 0.0, "crcBound": round(abstain_all_bound, 6),
                    "answered": 0, "coverage": 0.0,
                    "note": "certified only by abstaining on all rows at this alpha"}
        return {**base, "feasible": False, "threshold": None, "empiricalRisk": None,
                "crcBound": round(abstain_all_bound, 6),
                "reason": f"n={n} too small: B/(n+1)={abstain_all_bound:.4f} > alpha={alpha}"}
    answered = sum(1 for r in bucket if float(r["nonconformity"]) <= best_tau)
    return {
        **base, "feasible": True,
        "threshold": round(best_tau, 6),
        "empiricalRisk": round(best_rhat, 6),
        "crcBound": round((n * best_rhat + B) / (n + 1), 6),
        "answered": answered,
        "coverage": round(answered / n, 4),
    }


def crc_validity_check(
    rows: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
    risk_bucket: str = "normal",
    n_splits: int = 200,
    seed: int = 0,
    holdout: float = 0.3,
) -> dict[str, Any]:
    """Empirically demonstrate the CRC guarantee (deterministic given ``seed``).

    Over ``n_splits`` random calib/test splits, fit the CRC threshold on the calib half
    and measure the realized MARGINAL false-answer rate (answered-and-wrong over all test
    rows) on the held-out half. The CRC theorem gives ``E[loss] <= alpha``, so the MEAN
    realized rate should be ``<= alpha`` (individual splits may exceed it; the mean must
    not). This validates the *machinery*, not a Sophia capability. ``candidateOnly``.
    """
    bucket = [r for r in rows if r.get("risk", "normal") == risk_bucket] or list(rows)
    rng = random.Random(seed)
    realized: list[float] = []
    for _ in range(n_splits):
        idx = list(range(len(bucket)))
        rng.shuffle(idx)
        cut = max(1, int(len(bucket) * (1.0 - holdout)))
        calib = [bucket[i] for i in idx[:cut]]
        test = [bucket[i] for i in idx[cut:]]
        if not test:
            continue
        rc = conformal_risk_control(calib, alpha=alpha, risk_bucket=risk_bucket)
        if not rc.get("feasible"):
            continue
        tau = rc["threshold"]
        loss = sum(
            1 for r in test
            if float(r["nonconformity"]) <= tau and not bool(r.get("correct", False))
        ) / len(test)
        realized.append(loss)
    mean_realized = sum(realized) / len(realized) if realized else None
    frac_within = sum(1 for x in realized if x <= alpha) / len(realized) if realized else None
    return {
        "schema": "sophia.conformal_risk_control_validity.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "alpha": alpha,
        "riskBucket": risk_bucket,
        "nSplits": n_splits,
        "feasibleSplits": len(realized),
        "seed": seed,
        "holdout": holdout,
        "meanRealizedFalseAnswerRate": round(mean_realized, 4) if mean_realized is not None else None,
        "fracSplitsWithinAlpha": round(frac_within, 4) if frac_within is not None else None,
        "valid": bool(mean_realized is not None and mean_realized <= alpha),
    }


__all__ = ["ConformalPolicy", "quantile_threshold", "fit_conformal_policy", "evaluate_policy", "load_jsonl", "write_conformal_report", "conformal_risk_control", "crc_validity_check"]
