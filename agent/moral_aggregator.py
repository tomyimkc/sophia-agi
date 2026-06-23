"""Bounded moral-uncertainty aggregation (the Sophia moral parliament).

Hard prohibitions live in constitutional/deontic gates. This module handles gray
zones by bounded Borda-style aggregation across simple ethical perspectives,
avoiding raw expected-value fanaticism.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

THEORIES = ("deontological", "consequentialist", "virtue", "contractualist", "care", "epistemic_humility")


@dataclass(frozen=True)
class MoralVote:
    theory: str
    score: int  # -2..2 bounded
    uncertainty: float
    reason: str


@dataclass(frozen=True)
class MoralDecision:
    schema: str = "sophia.moral_parliament.v1"
    verdict: str = "permit"  # permit|revise|escalate
    aggregate: float = 0.0
    variance: float = 0.0
    votes: tuple[MoralVote, ...] = ()
    reason: str = "bounded moral aggregate"

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "verdict": self.verdict, "aggregate": self.aggregate, "variance": self.variance, "votes": [v.__dict__ for v in self.votes], "reason": self.reason}


def _bounded(x: int) -> int:
    return max(-2, min(2, int(x)))


def vote_theory(theory: str, text: str, *, context: dict[str, Any] | None = None) -> MoralVote:
    context = context or {}
    t = text or ""
    low = t.lower()
    harmful = bool(re.search(r"\b(?:harm|deceive|bypass|exploit|unsafe|tamper|hide|manipulate)\b", low))
    helpful = bool(re.search(r"\b(?:help|clarify|verify|cite|reduce risk|protect|honest|corrigible)\b", low))
    uncertain = float(context.get("moralUncertainty", 0.25 if harmful and helpful else 0.15))
    if theory == "deontological":
        score = -2 if harmful else (1 if helpful else 0)
        reason = "rule/duty view prioritizes prohibitions"
    elif theory == "consequentialist":
        score = -1 if harmful else (2 if helpful else 0)
        reason = "outcome view estimates net welfare/risk"
    elif theory == "virtue":
        score = -1 if harmful else (1 if re.search(r"\b(?:honest|humble|careful|courage|wisdom)\b", low) else 0)
        reason = "virtue view favors honesty, humility, practical wisdom"
    elif theory == "contractualist":
        score = -2 if re.search(r"\b(?:deceive|hide|bypass|without consent)\b", low) else (1 if helpful else 0)
        reason = "contractualist view rejects non-consensual deception"
    elif theory == "care":
        score = -1 if harmful else (1 if re.search(r"\b(?:protect|care|support|reduce harm)\b", low) else 0)
        reason = "care view prioritizes relationship and vulnerability"
    else:  # epistemic_humility
        score = 2 if re.search(r"\b(?:verify|cite|abstain|uncertain|clarify)\b", low) else (-2 if re.search(r"\b(?:pretend|unverified|overclaim)\b", low) else 0)
        reason = "epistemic humility view rewards verification and uncertainty"
    return MoralVote(theory, _bounded(score), max(0.0, min(1.0, uncertain)), reason)


def moral_parliament(text: str, *, context: dict[str, Any] | None = None, theories: tuple[str, ...] = THEORIES) -> MoralDecision:
    votes = tuple(vote_theory(th, text, context=context) for th in theories)
    scores = [v.score for v in votes]
    aggregate = round(sum(scores) / (2 * len(scores)), 4) if scores else 0.0  # normalize -1..1
    variance = round(pstdev(scores) / 2, 4) if len(scores) > 1 else 0.0
    if variance >= 0.75 or any(v.uncertainty >= 0.7 for v in votes):
        verdict = "escalate"
        reason = "high moral disagreement/uncertainty"
    elif aggregate >= 0.2:
        verdict = "permit"
        reason = "bounded aggregate supports action"
    else:
        verdict = "revise"
        reason = "bounded aggregate does not support action as written"
    return MoralDecision(verdict=verdict, aggregate=aggregate, variance=variance, votes=votes, reason=reason)


__all__ = ["MoralVote", "MoralDecision", "vote_theory", "moral_parliament", "THEORIES"]
