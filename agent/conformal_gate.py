# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Conformal abstention gate for the Conscience Kernel.

This converts arbitrary nonconformity scores into answer/abstain decisions with a
simple split-conformal threshold. The implementation is deterministic/offline and
uses held-out calibration rows, not model self-report, as the source of trust.
"""
from __future__ import annotations

import json, math
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


__all__ = ["ConformalPolicy", "quantile_threshold", "fit_conformal_policy", "evaluate_policy", "load_jsonl", "write_conformal_report"]
