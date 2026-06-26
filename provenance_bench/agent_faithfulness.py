# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Agent-faithfulness benchmark scorer — deterministic, no-judge, reproducible.

Scores ``agent.trajectory_eval.evaluate_trajectory`` against a hand-labelled pack
of agent trajectories. Unlike the legal/medical faithfulness benchmarks (which are
*model-judged* and therefore gated on >=2 judge families + kappa + CIs), the
trajectory evaluator's default support judge is **lexical and deterministic** — so
there is no judge variance, no multi-family gate, and the run is bit-for-bit
reproducible in CI. The honesty risk that remains is **label provenance**: the
gold labels are first-party (see the pack's ``status``), so a high score means the
evaluator agrees with the intended semantics, not that an independent oracle
confirmed it.

Metrics reported:

  * ``verdictAccuracy`` (+ Wilson 95% CI) — 3-way accept/abstain/blocked match.
  * detection precision / recall / F1 on the "should-not-certify" class (any gold
    verdict other than ``accept``) — the number an "Agent Data Evaluation" function
    actually cares about: how reliably the evaluator refuses to bless a bad run.
  * ``localizationAccuracy`` — over cases that name a culprit step, how often
    ``firstUnfaithfulStep`` matches.
  * per-category breakdown and a per-case table (for the public report / audit).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.fact_check_eval import wilson_interval
from agent.trajectory_eval import evaluate_trajectory

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "benchmark" / "agent_faithfulness.json"


def load_pack(path: "Path | str" = DEFAULT_PACK) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_case(case: dict) -> dict:
    """Run the evaluator on one case and compare to its gold labels."""
    result = evaluate_trajectory(case.get("trajectory") or [])
    predicted = result["verdict"]
    gold = case["expectVerdict"]
    gold_step = case.get("expectFirstUnfaithfulStep")
    predicted_step = result.get("firstUnfaithfulStep")
    has_culprit = gold_step is not None
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "goldVerdict": gold,
        "predictedVerdict": predicted,
        "verdictCorrect": predicted == gold,
        "goldFirstUnfaithfulStep": gold_step,
        "predictedFirstUnfaithfulStep": predicted_step,
        "localizationApplicable": has_culprit,
        "localizationCorrect": (predicted_step == gold_step) if has_culprit else None,
        "faithfulnessScore": result.get("faithfulnessScore"),
    }


def _detection(rows: list[dict]) -> dict:
    """Precision/recall/F1 on the positive class = 'should not certify' (gold verdict
    is not ``accept``; predicted verdict is not ``accept``)."""
    tp = fp = fn = tn = 0
    for r in rows:
        gold_pos = r["goldVerdict"] != "accept"
        pred_pos = r["predictedVerdict"] != "accept"
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else (0.0 if (precision is not None and recall is not None) else None)
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def _per_category(rows: list[dict]) -> dict:
    cats: dict[str, dict] = {}
    for r in rows:
        c = cats.setdefault(r["category"], {"n": 0, "verdictCorrect": 0})
        c["n"] += 1
        c["verdictCorrect"] += int(r["verdictCorrect"])
    for c in cats.values():
        c["verdictAccuracy"] = round(c["verdictCorrect"] / c["n"], 4) if c["n"] else None
    return cats


def score_pack(pack: "dict | None" = None) -> dict:
    """Score a whole pack and return the public-report record (JSON-serialisable)."""
    pack = pack or load_pack()
    cases = pack.get("cases", [])
    rows = [score_case(c) for c in cases]

    n = len(rows)
    verdict_correct = sum(1 for r in rows if r["verdictCorrect"])
    verdict_acc = round(verdict_correct / n, 4) if n else None
    ci = wilson_interval(verdict_correct, n) if n else None

    loc_rows = [r for r in rows if r["localizationApplicable"]]
    loc_correct = sum(1 for r in loc_rows if r["localizationCorrect"])
    loc_acc = round(loc_correct / len(loc_rows), 4) if loc_rows else None

    return {
        "benchmark": "agent_faithfulness",
        "schema": "sophia.agent_faithfulness.report.v1",
        "scoring": "deterministic (lexical support judge); no LLM judge; reproducible",
        "labelProvenance": "first-party / seed (see pack status) — NOT independently validated",
        "n": n,
        "verdictAccuracy": verdict_acc,
        "verdictAccuracyCI95": [ci["low"], ci["high"]] if ci else None,
        "verdictCorrect": verdict_correct,
        "detection": _detection(rows),
        "localizationAccuracy": loc_acc,
        "localizationN": len(loc_rows),
        "byCategory": _per_category(rows),
        "cases": rows,
        "candidateOnly": True,
    }


__all__ = ["DEFAULT_PACK", "load_pack", "score_case", "score_pack"]
