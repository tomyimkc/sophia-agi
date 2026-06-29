# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Intemperance signals — the dual of ``agent.cowardice_signals`` on the *magnitude* axis.

Where ``deception_signals`` catches *claiming beyond the evidence* and
``cowardice_signals`` catches *holding back despite the evidence*, this module
catches the failure neither sees: spending the **wrong amount** of effort. It has
two opposite faces, after Aristotle's two vices that flank every virtue (*NE* II):

- **excess (ἀκολασία)** — verbosity, hedge-stacking past the calibrated set-point,
  retrieval/tool-calls past diminishing returns, a loop that will not halt.
- **deficiency (ἀναισθησία)** — premature stop, under-answering, truncation, a
  lazy abstention with the budget unspent and more value still on the table.

It is a deterministic, offline, black-box heuristic — text features plus a small
context dict (expenditure, demand, marginal value, budget). It decides nothing on
its own; it feeds the ``alpha`` (appetite) / ``mu`` (marginal-value) terms and the
routing in :mod:`agent.sophrosyne`.

Design invariant (mirrors cowardice_signals): a signal NEVER forces a substantive
action. The worst an *excess* signal can do is recommend trimming/stopping; the
worst a *deficiency* signal can do is recommend continuing. It can never suppress a
required output (that guard lives in :mod:`agent.sophrosyne`). Genuine efficiency
(a short answer to a simple question, an early stop with nothing left to gain) is
not deficiency and is left alone.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

# A semantic backend's intemperance-likelihood at/above this fires the seam signal.
SEMANTIC_THRESHOLD = 0.6

# Hedges: a few are calibration; a stack of them is over-hedging (the measured
# "calibration tax" — intemperance of caution). Counted, not just matched.
_HEDGE_RE = re.compile(
    r"\b(?:i think|i believe|perhaps|maybe|possibly|it seems|arguably|"
    r"to some extent|in some sense|it could be argued|one might say|"
    r"as far as i can tell|if i'm not mistaken|i would say|sort of|kind of)\b",
    re.I,
)

# Filler / padding that signals elaboration past the point (excess prose).
_FILLER_RE = re.compile(
    r"\b(?:as (?:previously|already) (?:mentioned|noted|stated)|"
    r"it (?:is|'s) (?:important|worth) (?:to note|noting|mentioning)|"
    r"needless to say|at the end of the day|in order to|due to the fact that|"
    r"for all intents and purposes|it goes without saying)\b",
    re.I,
)

# Truncation / deficiency markers: the work was cut short.
_TRUNCATION_RE = re.compile(
    r"(?:\.\.\.$|\[truncated\]|\[\.\.\.\]|left as an exercise|"
    r"\bTODO\b|\bTBD\b|and so on\b|the rest is (?:trivial|similar))",
    re.I,
)


@dataclass(frozen=True)
class IntemperanceSignal:
    id: str
    axis: str  # excess|deficiency
    severity: str  # high|moderate
    reason: str


@dataclass(frozen=True)
class IntemperanceDecision:
    schema: str = "sophia.intemperance_signals.v1"
    # measure_clear | excess_risk | excess | deficiency_risk | deficiency
    verdict: str = "measure_clear"
    axis: str = "none"  # none|excess|deficiency
    risk: float = 0.0
    signals: tuple[IntemperanceSignal, ...] = ()
    reason: str = "expenditure appears proportionate to demand"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "axis": self.axis,
            "risk": self.risk,
            "signals": [s.__dict__ for s in self.signals],
            "reason": self.reason,
        }


def _f(context: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(context.get(key, default) or 0.0)))
    except (TypeError, ValueError):
        return default


def _max_ngram_repeat(text: str, n: int = 3) -> int:
    """Largest repeat count of any n-word shingle — a cheap verbosity/loop proxy."""
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < n:
        return 0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) if grams else 0


def detect_intemperance(
    text: str,
    *,
    context: dict[str, Any] | None = None,
    semantic_backend: Callable[[str], float] | None = None,
) -> IntemperanceDecision:
    """Flag expenditure that strays from the mean — too much or too little.

    Context keys (all optional, all ``[0,1]`` unless noted):
    - ``demand``: how much the task genuinely requires (the set-point).
    - ``expenditure``: how much is being / about to be spent.
    - ``marginalValue``: is the *next* unit of effort still buying anything?
    - ``budgetRemaining``: headroom left (compute / tokens / turns).
    - ``loopIterations`` (int): rounds spent so far on this sub-goal.
    - ``frontierShrinking`` (bool): is the open work actually decreasing? A loop
      that spends rounds without a shrinking frontier is runaway excess.
    - ``proposedStop`` (bool): ``True`` when the proposed move is to stop / abstain.

    ``semantic_backend`` is the seam for the paraphrase-brittleness fix (see the
    failure ledger), a callable ``text -> intemperance_likelihood in [0,1]``; default
    ``None`` = deterministic features only (zero behaviour change). As with the
    Andreia cowardice seam, a purely offline lexical backend is expected to be
    insufficient — a real model backend is required to actually close the gap.
    """
    context = context or {}
    t = text or ""
    demand = _f(context, "demand", 0.4)
    marginal = _f(context, "marginalValue", 0.5)
    budget = _f(context, "budgetRemaining", 1.0)
    proposed_stop = bool(context.get("proposedStop", False))
    try:
        loop_iters = max(0, int(context.get("loopIterations", 0) or 0))
    except (TypeError, ValueError):
        loop_iters = 0
    frontier_shrinking = bool(context.get("frontierShrinking", True))
    signals: list[IntemperanceSignal] = []

    # ---- excess ----------------------------------------------------------- #
    if _max_ngram_repeat(t) >= 3:
        signals.append(IntemperanceSignal(
            "self_repetition", "excess", "high",
            "the same phrasing repeats — padding past the point (verbosity)",
        ))
    if len(_HEDGE_RE.findall(t)) >= 3:
        signals.append(IntemperanceSignal(
            "hedge_stacking", "excess", "moderate",
            "hedges stack past the calibration set-point (over-hedging / calibration tax)",
        ))
    if len(_FILLER_RE.findall(t)) >= 2:
        signals.append(IntemperanceSignal(
            "filler_elaboration", "excess", "moderate",
            "filler/padding phrases elaborate beyond what the task demands",
        ))
    # A loop that has spent several rounds without the frontier shrinking, while
    # the marginal value of the next round is low, is runaway expenditure.
    if loop_iters >= 3 and not frontier_shrinking and marginal < 0.5:
        signals.append(IntemperanceSignal(
            "runaway_loop", "excess", "high",
            "rounds are being spent without the open work shrinking and with low marginal value",
        ))
    if semantic_backend is not None and not proposed_stop:
        try:
            score = float(semantic_backend(t))
        except Exception:  # noqa: BLE001 — a faulty backend must never break detection
            score = 0.0
        if score >= SEMANTIC_THRESHOLD:
            signals.append(IntemperanceSignal(
                "semantic_excess", "excess", "high",
                f"semantic backend flags over-expenditure the features did not enumerate (score {round(score, 3)})",
            ))

    # ---- deficiency ------------------------------------------------------- #
    if _TRUNCATION_RE.search(t):
        signals.append(IntemperanceSignal(
            "truncation", "deficiency", "high",
            "work is cut short (truncation / 'left as an exercise' / TODO) before completion",
        ))
    # Stopping while there is budget AND the next unit would still be valuable on a
    # demanding task is a premature stop (deficiency), not efficiency.
    if proposed_stop and budget >= 0.4 and marginal >= 0.6 and demand >= 0.5:
        signals.append(IntemperanceSignal(
            "premature_stop", "deficiency", "high",
            "proposes stopping while budget remains and the next unit of effort is still valuable",
        ))

    if not signals:
        return IntemperanceDecision()

    excess = [s for s in signals if s.axis == "excess"]
    deficiency = [s for s in signals if s.axis == "deficiency"]
    # The dominant axis is whichever side carries more weight (high=0.4, mod=0.25).
    def _w(group: list[IntemperanceSignal]) -> float:
        return sum(0.4 if s.severity == "high" else 0.25 for s in group)

    we, wd = _w(excess), _w(deficiency)
    axis = "excess" if we >= wd else "deficiency"
    group = excess if axis == "excess" else deficiency
    risk = min(1.0, _w(group))
    has_high = any(s.severity == "high" for s in group)
    verdict = axis if has_high else f"{axis}_risk"
    return IntemperanceDecision(
        verdict=verdict,
        axis=axis,
        risk=round(risk, 4),
        signals=tuple(signals),
        reason=(
            "expenditure exceeds what the task demands (excess)"
            if axis == "excess"
            else "expenditure falls short of what the task demands (deficiency)"
        ),
    )


__all__ = ["IntemperanceSignal", "IntemperanceDecision", "detect_intemperance"]
