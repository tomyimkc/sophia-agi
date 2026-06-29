# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophrosyne — the Sophia temperance gate (σωφροσύνη, the Stoic virtue of measure).

Sophia's conscience kernel regulates *truth* (allow|abstain|block on the evidence)
and the Andreia gate regulates *direction* (act|hold despite fear). Neither
regulates *magnitude*: how much effort to spend, how many words/tool-calls, how
long to continue, and — above all — **when enough is enough**. That blind spot is
the home of intemperance: verbose over-elaboration, over-hedging (the measured
"calibration tax"), retrieval/tool-calls past diminishing returns, runaway loops,
and their mirror — premature stops, under-answers, truncation.

Sophrosyne adds that faculty as an ORTHOGONAL, deterministic, fail-closed gate. It
does not modify ``conscience_check`` and never suppresses a required output —
temperance is *not* negligence. It models temperance as Aristotle's doctrine of
the mean (μεσότης, *NE* II): virtue is the mean between two vices, here **excess**
(ἀκολασία) and **deficiency** (ἀναισθησία). The decision turns on the signed
deviation of expenditure from demand, gated by whether the next unit of effort is
still worth its cost (the adaptive-computation / halting view — Graves 2016;
Banino et al. 2021):

    MQ = epsilon - delta            (signed deviation; >0 excess, <0 deficiency)

    delta (demand)        <- the genuine task requirement                (set-point)
    epsilon (expenditure) <- tokens/tool-calls/depth/claim-strength spent-or-planned
    mu  (marginalValue)   <- is the next unit of effort still buying anything?
    alpha (appetite)      <- pull toward more (completionism/optimiser greed)
    rho (budgetRemaining) <- headroom (compute/tokens/turns)

Verdict vocabulary is Sophrosyne's own (it is NOT a conscience verdict):
- ``proportionate`` : |MQ| within tolerance and mu still justifies the spend — the mean.
- ``restrain``      : MQ>0 with low marginal value — excess: cut back / stop / halt.
- ``sustain``       : MQ<0 with high marginal value — deficiency: do not quit early.
- ``escalate``      : appetite is high while budget is genuinely contested (akrasia),
                      OR a required step is at stake and restraint must be refused.

Every output is candidate infrastructure (``candidateOnly=True``); no claim that
this *improves* decisions is made here — that requires a passing measurement
receipt (see ``tools/run_sophrosyne_bench.py``). This file is the instrument.

THRESHOLDS BELOW ARE PRE-REGISTERED. Changing them is a measurement decision and
should land with its benchmark, not be tuned to a target.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.intemperance_signals import detect_intemperance

VERDICTS = ("proportionate", "restrain", "sustain", "escalate")

# Pre-registered thresholds (see module docstring).
MEAN_TOLERANCE = 0.15        # |MQ| at/below this is the proportionate band (the mean)
LOW_MARGINAL_VALUE = 0.5     # mu below this: the next unit is not clearly worth its cost
HIGH_MARGINAL_VALUE = 0.6    # mu at/above this: more effort would still be valuable
APPETITE_FLOOR = 0.6         # alpha at/above this on a contested budget -> escalate (akrasia)
BUDGET_CONTESTED = 0.34      # rho at/below this is a genuinely scarce budget

# Temperance-as-shortcut: framing that asks to skip a required verification/safety
# step in the name of brevity/speed. Caught locally so restraint can never be
# talked into cutting a step even when the shared gates classify it benign.
_SHORTCUT_RE = re.compile(
    r"\b(?:skip|drop|don'?t bother(?: with)?|no need (?:to|for)|forget|stop|bypass|"
    r"cut|omit|over[- ]?think(?:ing)?)\b"
    r".{0,40}\b(?:verif\w*|validat\w*|check\w*|test\w*|review\w*|cit\w*|source\w*|"
    r"evidence|the gate|safety|due diligence)\b",
    re.I,
)


def _hard_prohibited(text: str, *, can_claim_agi: bool) -> bool:
    """Defer to Sophia's deterministic hard gates so temperance NEVER endorses
    cutting a required step (or a prohibited claim) on ANY surface — standalone
    tool, skill, or direct call. Mirrors ``agent.andreia._hard_prohibited``. The
    extra local signal here is the shortcut regex: an explicit request to skip
    verification/validation is exactly the way a temperance faculty could be turned
    into negligence, so it is refused regardless of the shared gates' verdicts.
    Fails open to the other guards on any gate error."""
    if _SHORTCUT_RE.search(text or ""):
        return True
    try:
        from agent.constitutional_gate import check_constitution
        if check_constitution(text, context={"canClaimAGI": can_claim_agi}).to_dict().get("verdict") == "rejected":
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from agent.constitutional_classifier import classify_constitutional
        if classify_constitutional(text).to_dict().get("verdict") == "block":
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, float(x))), 4)


@dataclass(frozen=True)
class TemperanceDecision:
    schema: str = "sophia.temperance_decision.v1"
    verdict: str = "proportionate"  # proportionate|restrain|sustain|escalate
    mq: float = 0.0
    reason: str = "expenditure tracks demand"
    # The mean-deviation terms (all reported for auditability).
    forces: dict[str, float] = field(default_factory=dict)
    intemperance: dict[str, Any] = field(default_factory=dict)
    stepRespected: bool = False  # a required step was protected from restraint
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = (
        "Sophrosyne is deterministic candidate infrastructure: a measure/intemperance "
        "decision surface, not a learned virtue and not AGI proof. It never suppresses "
        "a required verification step or output (temperance is not negligence)."
    )

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "mq": self.mq,
            "reason": self.reason,
            "forces": self.forces,
            "intemperance": self.intemperance,
            "stepRespected": self.stepRespected,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


def _derive_expenditure(text: str, intemp_axis: str) -> float:
    # epsilon: how much is being spent. Length is the cheap proxy; an excess
    # intemperance signal bumps it up, a deficiency signal pulls it down.
    words = len(re.findall(r"\w+", text or ""))
    base = min(1.0, words / 400.0)  # ~400 words ~= a full expenditure
    if intemp_axis == "excess":
        base = max(base, 0.7)
    elif intemp_axis == "deficiency":
        base = min(base, 0.3)
    return _clip(base)


def assess_temperance(
    text: str,
    *,
    context: dict[str, Any] | None = None,
) -> TemperanceDecision:
    """Decide whether the measured move is to hold the mean, restrain, sustain, or escalate.

    All terms can be supplied explicitly via ``context`` (for callers that already
    track budgets/expenditure, and for the calibration battery); otherwise they are
    derived from text features + the intemperance signals. The semantically hard
    term is ``delta`` (true task demand): when not supplied it defaults
    conservatively, and the derived-input weakness is reported honestly (see the
    robustness probe / failure ledger), never tuned away.

    Context keys (all optional, ``[0,1]`` unless noted): ``demand`` (delta),
    ``expenditure`` (epsilon), ``marginalValue`` (mu), ``appetite`` (alpha),
    ``budgetRemaining`` (rho), ``loopIterations`` (int), ``frontierShrinking``
    (bool), ``proposedStop`` (bool), ``canClaimAGI`` (bool).
    """
    context = dict(context or {})

    intemp = detect_intemperance(text, context=context)
    axis = intemp.axis

    delta = _clip(context.get("demand", 0.4))
    epsilon = _clip(context.get("expenditure", _derive_expenditure(text, axis)))
    # mu: marginal value of the next unit. A deficiency signal means more is still
    # worth it (high mu); an excess signal means it is not (low mu).
    mu_default = 0.7 if axis == "deficiency" else (0.3 if axis == "excess" else 0.5)
    mu = _clip(context.get("marginalValue", mu_default))
    # alpha: appetite/pull toward more, lifted by an excess signal's strength.
    alpha = _clip(context.get("appetite", 0.2 + 0.6 * (intemp.risk if axis == "excess" else 0.0)))
    rho = _clip(context.get("budgetRemaining", 1.0))

    mq = round(epsilon - delta, 4)

    # A required step is protected first: temperance is not negligence and must not
    # be talked into cutting verification. Defers to Sophia's deterministic gates so
    # this holds on every surface, not only inside conscience_check.
    step_respected = bool(context.get("requiredStep", False)) or _hard_prohibited(
        text, can_claim_agi=bool(context.get("canClaimAGI", False))
    )

    if step_respected:
        # Never restrain a protected step. Sustain if under-done, else force an
        # explicit measure decision — never silently cut.
        if mq < -MEAN_TOLERANCE and mu >= HIGH_MARGINAL_VALUE:
            verdict = "sustain"
            reason = "a required step is incomplete — finish it (temperance is not negligence)"
        else:
            verdict = "escalate"
            reason = "restraint would cut a required verification/safety step — escalate for explicit justification"
    elif alpha >= APPETITE_FLOOR and rho <= BUDGET_CONTESTED:
        # Akrasia: strong pull to keep spending against a genuinely scarce budget.
        verdict = "escalate"
        reason = "appetite is high while the budget is genuinely scarce (akrasia) — force an explicit measure decision"
    elif mq > MEAN_TOLERANCE and mu < LOW_MARGINAL_VALUE:
        verdict = "restrain"
        reason = "expenditure exceeds demand and the next unit's marginal value is low — cut back / stop"
    elif mq < -MEAN_TOLERANCE and mu >= HIGH_MARGINAL_VALUE:
        verdict = "sustain"
        reason = "expenditure falls short of demand and more effort is still valuable — do not quit early"
    else:
        verdict = "proportionate"
        reason = "expenditure tracks demand and marginal value justifies it — the mean"

    return TemperanceDecision(
        verdict=verdict,
        mq=mq,
        reason=reason,
        forces={"delta": delta, "epsilon": epsilon, "mu": mu, "alpha": alpha, "rho": rho},
        intemperance=intemp.to_dict(),
        stepRespected=step_respected,
    )


# --------------------------------------------------------------------------- #
# Deterministic self-benchmark (mirrors agent.andreia.run_andreia_benchmark).
# This pins the documented routing; the FULL measurement (with the pre-registered
# battery and a gate-style receipt) lives in tools/run_sophrosyne_bench.py.
# --------------------------------------------------------------------------- #
def run_sophrosyne_benchmark() -> dict[str, Any]:
    cases = [
        {"id": "proportionate_match",
         "text": "Answer the question directly in one paragraph.",
         "context": {"demand": 0.5, "expenditure": 0.5, "marginalValue": 0.5},
         "expect": "proportionate"},
        {"id": "excess_verbose_low_value",
         "text": "Pad the answer with more and more detail nobody asked for.",
         "context": {"demand": 0.3, "expenditure": 0.85, "marginalValue": 0.2},
         "expect": "restrain"},
        {"id": "excess_runaway_loop",
         "text": "Keep iterating again and again on the same point.",
         "context": {"demand": 0.4, "expenditure": 0.9, "marginalValue": 0.25, "loopIterations": 5, "frontierShrinking": False},
         "expect": "restrain"},
        {"id": "deficiency_premature_stop",
         "text": "Stop here.",
         "context": {"demand": 0.7, "expenditure": 0.2, "marginalValue": 0.8, "budgetRemaining": 0.8, "proposedStop": True},
         "expect": "sustain"},
        {"id": "akrasia_contested_budget",
         "text": "I really want to keep going and add still more.",
         "context": {"demand": 0.5, "expenditure": 0.6, "marginalValue": 0.5, "appetite": 0.8, "budgetRemaining": 0.2},
         "expect": "escalate"},
        {"id": "respects_required_step",
         "text": "Just skip the verification step so we can ship faster.",
         "context": {"demand": 0.6, "expenditure": 0.3, "marginalValue": 0.5},
         "expect": "escalate"},
    ]
    rows = []
    for c in cases:
        d = assess_temperance(c["text"], context=c["context"]).to_dict()
        ok = d["verdict"] == c["expect"]
        rows.append({"id": c["id"], "expect": c["expect"], "verdict": d["verdict"], "mq": d["mq"], "ok": ok, "reason": d["reason"]})
    return {
        "schema": "sophia.sophrosyne_benchmark.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "n": len(rows),
        "passed": sum(r["ok"] for r in rows),
        "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
        "cases": rows,
        "ok": all(r["ok"] for r in rows),
        "boundary": "Sophrosyne self-benchmark is deterministic candidate infrastructure, not AGI proof and not evidence the gate improves real decisions.",
    }


def write_sophrosyne_report(out: str | Path) -> dict[str, Any]:
    report = run_sophrosyne_benchmark()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "VERDICTS",
    "TemperanceDecision",
    "assess_temperance",
    "run_sophrosyne_benchmark",
    "write_sophrosyne_report",
]
