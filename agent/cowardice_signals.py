# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cowardice signals — the dual of ``agent.deception_signals``.

``deception_signals`` catches a draft that *acts/claims more than the evidence
supports* (recklessness/overclaim). This module catches the opposite failure:
a draft that *holds back despite the evidence supporting action* — "cowardice
disguised as prudence" (Holiday, *Courage Is Calling*).

It is a deterministic, offline, black-box heuristic — text features plus a small
context dict (confidence, harm-of-silence, social vs epistemic cost). It does not
decide anything on its own; it feeds the ``phi`` (inhibition) term and the
escalate routing in :mod:`agent.andreia`. White-box probes can be added later
without changing the contract.

Design invariant: a high cowardice signal NEVER forces an action. The worst it
can do is force an *explicit justification* (escalate) so a fear-driven retreat
cannot masquerade as wisdom. Genuine epistemic prudence (high risk, low
confidence) is not cowardice and is left alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

# A semantic backend's cowardice-likelihood at/above this fires the seam signal.
SEMANTIC_THRESHOLD = 0.6

# "Cowardice can look respectable" (Holiday): excuses that disguise fear as
# realism. Matched only when paired with high confidence / high harm-of-silence,
# so genuine caution is never mislabelled.
_RESPECTABLE_EXCUSE_RE = re.compile(
    r"\b(?:now is not the (?:right )?time|not the right time|let someone else|"
    r"someone else will (?:handle|deal)|keep(?:ing)? my head down|stay(?:ing)? (?:neutral|out of it)|"
    r"not my place|not my problem|wait and see|avoid(?:ing)? (?:drama|conflict)|"
    r"play(?:ing)? it safe|don'?t (?:want to )?(?:get involved|rock the boat)|"
    r"too risky to say|best not to|why stick my neck out)\b",
    re.I,
)

# Sycophancy drift: abandoning a held position under social pressure to please.
_SYCOPHANCY_RE = re.compile(
    r"\b(?:you'?re (?:probably )?right,? (?:so )?i'?ll change|i'?ll (?:just )?go along|"
    r"to avoid (?:upsetting|conflict)|i'?ll say what (?:you|they) want|"
    r"i'?ll back down|whatever (?:you|they) prefer)\b",
    re.I,
)


@dataclass(frozen=True)
class CowardiceSignal:
    id: str
    severity: str  # high|moderate
    reason: str


@dataclass(frozen=True)
class CowardiceDecision:
    schema: str = "sophia.cowardice_signals.v1"
    verdict: str = "courageous_path_clear"  # courageous_path_clear|cowardice_risk|cowardice
    risk: float = 0.0
    signals: tuple[CowardiceSignal, ...] = ()
    reason: str = "no fear-driven-retreat signal detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "risk": self.risk,
            "signals": [s.__dict__ for s in self.signals],
            "reason": self.reason,
        }


def _f(context: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(context.get(key, default) or 0.0)))
    except (TypeError, ValueError):
        return default


def detect_cowardice(
    text: str,
    *,
    context: dict[str, Any] | None = None,
    semantic_backend: Callable[[str], float] | None = None,
) -> CowardiceDecision:
    """Flag a retreat that looks driven by fear/social cost rather than by risk.

    Context keys (all optional, all ``[0,1]`` unless noted):
    - ``confidence``: how well-supported the would-be action is.
    - ``harmOfSilence``: ``psi`` — cost of staying quiet (complicity pressure).
    - ``socialCost``: reputational/comfort cost of acting (the fear term).
    - ``epistemicRisk``: genuine risk of being wrong (the prudent reason to hold).
    - ``proposedHold``: ``True`` when the proposed move is to stay silent/abstain.
      (Deliberately NOT the metacognition fact-gate route: epistemic caution about
      *whether a fact is verified* is prudence, not cowardice — only an intent to
      withhold a well-supported, high-stakes point counts here.)

    ``semantic_backend`` is the seam for the documented fix to the regex's paraphrase
    brittleness (failure-ledger: andreia-derived-signal-routing-weak-on-raw-text). It is
    a callable ``text -> cowardice_likelihood in [0,1]``; when supplied and it scores
    ``>= SEMANTIC_THRESHOLD`` on an otherwise-actionable case, a ``semantic_cowardice``
    signal fires (catching paraphrases the regex never enumerated). Default ``None`` =
    deterministic regex only (zero behaviour change). NOTE: an OFFLINE lexical-hashing
    backend is measured INSUFFICIENT (it cannot separate cowardly paraphrases from
    courageous text — see tools/run_andreia_robustness.py), so a real semantic model /
    NLI / LLM-judge backend is required to actually close the gap.
    """
    context = context or {}
    t = text or ""
    confidence = _f(context, "confidence")
    harm_of_silence = _f(context, "harmOfSilence")
    social_cost = _f(context, "socialCost")
    epistemic_risk = _f(context, "epistemicRisk")
    proposed_hold = bool(context.get("proposedHold", False))
    signals: list[CowardiceSignal] = []

    if _RESPECTABLE_EXCUSE_RE.search(t) and (confidence >= 0.6 or harm_of_silence >= 0.5):
        signals.append(CowardiceSignal(
            "respectable_excuse", "high",
            "fear is dressed as realism ('not the right time', 'let someone else') while the case for acting is clear",
        ))
    if semantic_backend is not None and (confidence >= 0.6 or harm_of_silence >= 0.5):
        try:
            score = float(semantic_backend(t))
        except Exception:  # noqa: BLE001 — a faulty backend must never break detection
            score = 0.0
        if score >= SEMANTIC_THRESHOLD:
            signals.append(CowardiceSignal(
                "semantic_cowardice", "high",
                f"semantic backend flags a fear-driven retreat the regex did not enumerate (score {round(score, 3)})",
            ))
    if proposed_hold and confidence >= 0.75 and harm_of_silence >= 0.5:
        signals.append(CowardiceSignal(
            "confidence_silence_mismatch", "high",
            "holds back despite high confidence and a high cost of silence (decision-action gap)",
        ))
    # Fear, not prudence: the inhibitor is social/reputational cost, not real risk.
    if social_cost >= 0.5 and social_cost > epistemic_risk + 0.15 and harm_of_silence >= 0.4:
        signals.append(CowardiceSignal(
            "social_cost_dominates", "moderate",
            "the dominant reason to stay silent is reputational/comfort cost, not epistemic risk",
        ))
    if _SYCOPHANCY_RE.search(t):
        signals.append(CowardiceSignal(
            "sycophancy_drift", "high",
            "abandons a held position under social pressure to please rather than for new evidence",
        ))

    if not signals:
        return CowardiceDecision()
    risk = min(1.0, sum(0.4 if s.severity == "high" else 0.25 for s in signals))
    verdict = "cowardice" if any(s.severity == "high" for s in signals) else "cowardice_risk"
    return CowardiceDecision(
        verdict=verdict,
        risk=round(risk, 4),
        signals=tuple(signals),
        reason="retreat appears fear-driven rather than risk-driven",
    )


__all__ = ["CowardiceSignal", "CowardiceDecision", "detect_cowardice"]
