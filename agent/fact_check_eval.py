# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evaluation metrics for the out-of-wiki fact-check gate."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from agent.fact_check_gate import decision_to_dict, fact_check_text
from selfextend.calibration_metrics import brier_score, expected_calibration_error


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def run_fact_check_eval(
    rows: list[dict[str, Any]],
    *,
    retriever=None,
    entailment=None,
    doi_resolver=None,
    url_resolver=None,
    judges=None,
    live_backend: bool = False,
    target_fabrication_rate: float = 0.01,
) -> dict[str, Any]:
    per_case = []
    for row in rows:
        decision = fact_check_text(
            row["claim"],
            retriever=retriever,
            entailment=entailment,
            doi_resolver=doi_resolver,
            url_resolver=url_resolver,
            judges=judges,
        )
        d = decision_to_dict(decision)
        correctness = verdict_correct(d["verdict"], row["label"])
        conf = max([c.get("confidence", 0.0) for c in d.get("claims", [])] or [0.0])
        per_case.append({
            "id": row.get("id"),
            "claim": row["claim"],
            "label": row["label"],
            "type": row.get("type"),
            "risk": row.get("risk", "normal"),
            "verdict": d["verdict"],
            "confidence": round(float(conf), 4),
            "correct": correctness,
            "reason": d["reason"],
            "claims": d.get("claims", []),
        })
    return summarize_cases(per_case, live_backend=live_backend, target_fabrication_rate=target_fabrication_rate)


def verdict_correct(verdict: str, label: str) -> bool:
    """Task-level correctness for gate measurement.

    True claims should be accepted. False claims are safe if rejected OR held
    (reject is stronger; hold still prevents fabrication). Unknowable claims
    should be held, not accepted as fact or rejected as false.
    """
    if label == "true":
        return verdict == "accepted"
    if label == "false":
        return verdict in {"rejected", "held"}
    if label == "unknowable":
        return verdict == "held"
    raise ValueError(f"unknown label: {label}")


def summarize_cases(
    per_case: list[dict[str, Any]],
    *,
    live_backend: bool,
    target_fabrication_rate: float = 0.01,
) -> dict[str, Any]:
    labels = {lab: [c for c in per_case if c["label"] == lab] for lab in ("true", "false", "unknowable")}
    false_or_unknown = labels["false"] + labels["unknowable"]
    answerable = labels["true"] + labels["false"]
    accepted_false = [c for c in false_or_unknown if c["verdict"] == "accepted"]
    over_abstained_true = [c for c in labels["true"] if c["verdict"] == "held"]
    false_rejected_true = [c for c in labels["true"] if c["verdict"] == "rejected"]
    resolved_correct = [c for c in answerable if (c["label"] == "true" and c["verdict"] == "accepted") or (c["label"] == "false" and c["verdict"] == "rejected")]
    # Calibration is computed on resolved truth-valued decisions only. A held
    # claim is a safe abstention, not a probability assertion that the claim is
    # true or false; including holds would make correct abstention look
    # artificially miscalibrated.
    resolved_for_calibration = [c for c in per_case if c["verdict"] in {"accepted", "rejected"}]
    pairs = [(float(c["confidence"]), bool(c["correct"])) for c in resolved_for_calibration]
    return {
        "schema": "sophia.fact_check.live_eval.v1",
        "candidateOnly": not live_backend,
        "level3Evidence": False,
        "liveBackendUsed": live_backend,
        "n": len(per_case),
        "labelCounts": {k: len(v) for k, v in labels.items()},
        "metrics": {
            "fabricationRate": _rate(len(accepted_false), len(false_or_unknown)),
            "overAbstentionRate": _rate(len(over_abstained_true), len(labels["true"])),
            "falseRejectRateOnTrue": _rate(len(false_rejected_true), len(labels["true"])),
            "correctAbstentionRateOnUnknowable": _rate(sum(1 for c in labels["unknowable"] if c["verdict"] == "held"), len(labels["unknowable"])),
            "resolvedAnswerableAccuracy": _rate(len(resolved_correct), len(answerable)),
            "overallDecisionAccuracy": _rate(sum(1 for c in per_case if c["correct"]), len(per_case)),
            "calibrationNResolved": len(pairs),
            "ece": expected_calibration_error(pairs),
            "brier": brier_score(pairs),
        },
        "confidenceIntervals": {
            "fabricationRateWilson95": wilson_interval(len(accepted_false), len(false_or_unknown)),
            "overAbstentionRateWilson95": wilson_interval(len(over_abstained_true), len(labels["true"])),
        },
        "derivedFloors": derive_acceptance_floors(per_case, target_fabrication_rate=target_fabrication_rate),
        "riskCoverage": risk_coverage(per_case),
        "cases": per_case,
    }


def derive_acceptance_floors(per_case: list[dict[str, Any]], *, target_fabrication_rate: float = 0.01) -> dict[str, Any]:
    """Derive confidence floors from measured false-accept risk, not guesswork.

    For each risk bucket, choose the lowest acceptance confidence threshold whose
    accepted false/unknowable rate is <= target. If no accepted cases exist, keep
    the current conservative default rather than pretending calibration happened.
    """
    out: dict[str, Any] = {"targetFabricationRate": target_fabrication_rate, "byRisk": {}}
    for risk in sorted({c.get("risk", "normal") for c in per_case} or {"normal"}):
        cases = [c for c in per_case if c.get("risk", "normal") == risk]
        accepted = [c for c in cases if c["verdict"] == "accepted"]
        if not accepted:
            out["byRisk"][risk] = {"floor": 0.82 if risk == "high" else 0.70, "nAccepted": 0, "note": "no accepted cases; retain default"}
            continue
        thresholds = sorted({round(float(c["confidence"]), 4) for c in accepted})
        best = None
        for th in thresholds:
            surfaced = [c for c in accepted if float(c["confidence"]) >= th]
            bad = [c for c in surfaced if c["label"] in {"false", "unknowable"}]
            fab = _rate(len(bad), len(surfaced))
            true_cov = _rate(sum(1 for c in surfaced if c["label"] == "true"), sum(1 for c in cases if c["label"] == "true"))
            if fab <= target_fabrication_rate and (best is None or true_cov > best["trueAcceptCoverage"] or (true_cov == best["trueAcceptCoverage"] and th < best["floor"])):
                best = {"floor": th, "measuredFabricationRate": fab, "trueAcceptCoverage": true_cov, "nAcceptedAtFloor": len(surfaced)}
        out["byRisk"][risk] = best or {"floor": 1.01, "measuredFabricationRate": 1.0, "trueAcceptCoverage": 0.0, "nAcceptedAtFloor": 0, "note": "no threshold met target"}
    return out


def risk_coverage(per_case: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(per_case, key=lambda c: float(c["confidence"]), reverse=True)
    out = []
    errs = 0
    for i, c in enumerate(ordered, 1):
        if not c["correct"]:
            errs += 1
        out.append({"coverage": round(i / len(ordered), 4), "risk": round(errs / i, 4), "threshold": c["confidence"]})
    return out


def wilson_interval(k: int, n: int, *, z: float = 1.96) -> dict[str, float | int]:
    if n <= 0:
        return {"k": k, "n": n, "low": 0.0, "high": 0.0}
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return {"k": k, "n": n, "low": round(max(0.0, centre - margin), 4), "high": round(min(1.0, centre + margin), 4)}


def write_report(report: dict[str, Any], out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rate(k: int, n: int) -> float:
    return round(k / n, 4) if n else 0.0


__all__ = ["load_jsonl", "run_fact_check_eval", "summarize_cases", "derive_acceptance_floors", "wilson_interval", "write_report"]
