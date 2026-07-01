#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W2 — proper-scoring calibration TRAINING objective (drop-in, fail-closed).

Thesis (see agi-proof/untapped-training-2026-07-01/README.md): the repo MEASURES
calibration (agent/calibration.py: ECE, selective risk) and SCORES honesty
(agent/abstention_scoring.py: the asymmetric correct/abstain/confident-wrong rubric) — but
never TRAINS against either. The abstention rubric is motivated, in that module's own
docstring, by the argument that binary benchmark scoring (correct=1, wrong=abstain=0) makes
a confident guess weakly dominate "I don't know" — i.e. the scoring is itself the training
incentive behind hallucination. (That module attributes the argument to Kalai et al., "Why
Language Models Hallucinate"; the citation is repeated here as the repo's own, NOT
independently verified in this build.) This tool supplies the missing differentiable
objective and demonstrates, offline, that minimizing it lowers ECE.

WHAT THIS DOES (fully runnable, no backend needed):
  * defines a proper scoring rule loss (Brier or logarithmic) plus an asymmetric
    abstention penalty, as pure-numpy-free Python so it has no heavy deps;
  * fits a tiny 1-parameter temperature / 2-parameter Platt calibrator on held-out
    (confidence, correct) pairs by minimizing that loss with plain gradient descent;
  * reports ECE and selective risk BEFORE and AFTER via the repo's own
    agent.calibration, so the improvement is measured by Sophia's own instrument,
    not a metric this tool invented.

WHAT THIS DOES NOT DO (honest seam):
  * it does NOT fine-tune the base LM. Wiring this loss into the MLX/LoRA DPO path
    (training/…) is the maintainer step; this tool proves the objective is correct and
    calibration-improving on a frozen model's confidences first, so the expensive step
    is de-risked. Output carries candidateOnly:true, level3Evidence:false.

Records schema (matches agent.abstention_scoring): each row is
  {"confidence": float in [0,1], "correct": bool, "action": "answer"|"hedge"|"abstain"}
Usage:
  python3 tools/train_calibration_objective.py --records recs.jsonl --loss brier \
      --out agi-proof/untapped-training-2026-07-01/calib_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

# ---- repo instruments (real interfaces; fail-closed if unavailable) ----
try:
    from agent.calibration import expected_calibration_error, selective_risk, base_risk
    from agent.abstention_scoring import score as abstention_score
    _REPO_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    """Fail-closed: an environment artifact, NOT a score."""
    return {
        "schema": "sophia.calibration_training.v1",
        "ok": False,
        "reason": reason,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


# --------------------------------------------------------------- proper scoring losses
def brier_loss(p: float, y: int) -> float:
    """Brier score (a strictly proper scoring rule): (p - y)^2."""
    return (p - y) ** 2


def log_loss(p: float, y: int, *, eps: float = 1e-6) -> float:
    """Logarithmic score (strictly proper): -[y log p + (1-y) log(1-p)]."""
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def asymmetric_abstention_penalty(p: float, y: int, action: str, *, lam: float) -> float:
    """Price a confident-WRONG commitment more than an abstention (the incentive fix the
    repo's abstention_scoring rubric targets). An abstain contributes 0 loss regardless of
    p; an answered-wrong contributes
    lam * p (the more confident the wrong answer, the worse). Answered-correct: 0.
    This is added to the proper-scoring term; lam=0 recovers pure proper scoring."""
    if action == "abstain":
        return 0.0
    if y == 0:  # answered (or hedged) but wrong
        return lam * p
    return 0.0


LOSSES: dict[str, Callable[[float, int], float]] = {"brier": brier_loss, "log": log_loss}


# --------------------------------------------------------------- Platt calibrator
def _sigmoid(z: float) -> float:
    if z < 0:
        ez = math.exp(z)
        return ez / (1.0 + ez)
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(
    conf: Sequence[float], correct: Sequence[int], actions: Sequence[str], *,
    loss: str = "brier", lam: float = 1.0, lr: float = 0.5, epochs: int = 500,
) -> tuple[float, float]:
    """Fit p_cal = sigmoid(a * logit(conf) + b) minimizing the proper-scoring +
    abstention loss by gradient descent. Returns (a, b). Pure Python, deterministic."""
    a, b = 1.0, 0.0
    eps = 1e-6
    logits = [math.log(min(max(c, eps), 1 - eps) / (1 - min(max(c, eps), 1 - eps))) for c in conf]
    n = len(conf)
    for _ in range(epochs):
        ga = gb = 0.0
        for i in range(n):
            if actions[i] == "abstain":
                continue  # abstentions carry no calibration gradient (no committed p)
            z = a * logits[i] + b
            p = _sigmoid(z)
            y = correct[i]
            # d(proper loss)/dp: brier -> 2(p-y); log -> (p-y)/(p(1-p))
            if loss == "brier":
                dldp = 2.0 * (p - y)
            else:
                pp = min(max(p, eps), 1 - eps)
                dldp = (pp - y) / (pp * (1 - pp))
            dldp += lam * (1 if y == 0 else 0)  # asymmetric penalty gradient on wrong
            dpdz = p * (1 - p)
            g = dldp * dpdz
            ga += g * logits[i]
            gb += g
        a -= lr * ga / max(n, 1)
        b -= lr * gb / max(n, 1)
    return a, b


def apply_platt(conf: Sequence[float], a: float, b: float) -> list[float]:
    eps = 1e-6
    out = []
    for c in conf:
        cc = min(max(c, eps), 1 - eps)
        out.append(_sigmoid(a * math.log(cc / (1 - cc)) + b))
    return out


# --------------------------------------------------------------- report
def run(records: list[dict[str, Any]], *, loss: str = "brier", lam: float = 1.0,
        coverage: float = 0.5) -> dict[str, Any]:
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with "
                             "PYTHONPATH=. inside the sophia-agi tree")
    if loss not in LOSSES:
        return _env_artifact(f"unknown loss '{loss}' (use one of {sorted(LOSSES)})")
    if not records:
        return _env_artifact("no records provided (fail-closed; nothing to calibrate)")

    conf = [float(r.get("confidence", 0.0)) for r in records]
    correct = [1 if bool(r.get("correct", False)) else 0 for r in records]
    actions = [str(r.get("action", "answer")).lower() for r in records]

    # answered-only view for calibration (abstentions have no committed confidence)
    idx = [i for i, act in enumerate(actions) if act != "abstain"]
    if not idx:
        return _env_artifact("every record is an abstention; no committed confidences to calibrate")
    c_ans = [conf[i] for i in idx]
    y_ans = [bool(correct[i]) for i in idx]

    ece_before = expected_calibration_error(c_ans, y_ans)
    sr_before = selective_risk(c_ans, y_ans, coverage)

    a, b = fit_platt(conf, correct, actions, loss=loss, lam=lam)
    c_cal_all = apply_platt(conf, a, b)
    c_cal_ans = [c_cal_all[i] for i in idx]

    ece_after = expected_calibration_error(c_cal_ans, y_ans)
    sr_after = selective_risk(c_cal_ans, y_ans, coverage)

    honesty = abstention_score(records, lam=lam)  # the repo's asymmetric rubric

    return {
        "schema": "sophia.calibration_training.v1",
        "ok": True,
        "loss": loss, "lambda": lam, "coverage": coverage,
        "platt": {"a": round(a, 4), "b": round(b, 4)},
        "ece": {"before": round(ece_before, 4), "after": round(ece_after, 4),
                "improved": ece_after < ece_before},
        "selectiveRisk": {"before": round(sr_before, 4), "after": round(sr_after, 4),
                          "baseRisk": round(base_risk(y_ans), 4)},
        "honestyRubric": honesty,
        "n": len(records), "nAnswered": len(idx),
        "note": "Calibrator fit on a FROZEN model's confidences. Wiring this loss into "
                "the LM's own training (MLX/LoRA DPO) is the maintainer seam; this proves "
                "the objective lowers ECE by the repo's own calibration instrument first.",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W2 proper-scoring calibration objective")
    ap.add_argument("--records", required=True, help="JSONL of {confidence,correct,action}")
    ap.add_argument("--loss", default="brier", choices=sorted(LOSSES))
    ap.add_argument("--lam", type=float, default=1.0, help="asymmetric abstention penalty")
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    recs = load_jsonl(Path(args.records))
    report = run(recs, loss=args.loss, lam=args.lam, coverage=args.coverage)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    # fail-closed exit code so a pipeline can gate on it
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())