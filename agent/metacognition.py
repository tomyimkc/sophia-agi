# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Metacognitive uncertainty signals for the Sophia conscience kernel.

The goal is epistemic humility: decide whether Sophia should answer, retrieve,
clarify, escalate, or abstain. These signals are deterministic/offline by default;
model-specific P(True)/P(IK), hidden-state probes, or semantic-entropy probes can
be injected later without changing the decision contract.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from sophia_contract.intake import RECOMMENDED_ACTIONS


@dataclass(frozen=True)
class MetacognitionReport:
    schema: str = "sophia.metacognition.v1"
    confidence: float = 0.0
    p_true: float | None = None
    p_ik: float | None = None
    self_consistency: float | None = None
    semantic_entropy: float | None = None
    nonconformity: float = 1.0
    uncertainty_type: str = "epistemic"  # epistemic|aleatoric|moral|low
    recommended_action: str = "abstain"
    reasons: tuple[str, ...] = ()
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recommended_action not in RECOMMENDED_ACTIONS:
            raise ValueError(f"recommended_action must be one of {RECOMMENDED_ACTIONS}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "confidence": self.confidence,
            "pTrue": self.p_true,
            "pIK": self.p_ik,
            "selfConsistency": self.self_consistency,
            "semanticEntropy": self.semantic_entropy,
            "nonconformity": self.nonconformity,
            "uncertaintyType": self.uncertainty_type,
            "recommendedAction": self.recommended_action,
            "reasons": list(self.reasons),
            "signals": self.signals,
        }


def normalize_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def self_consistency(samples: Sequence[Any]) -> tuple[str | None, float]:
    """Return majority normalized answer and agreement fraction."""
    vals = [normalize_answer(s) for s in samples if normalize_answer(s)]
    if not vals:
        return None, 0.0
    ans, n = Counter(vals).most_common(1)[0]
    return ans, round(n / len(vals), 4)


def semantic_entropy(samples: Sequence[Any]) -> float:
    """Meaning-space entropy proxy: normalized exact clusters.

    Full semantic entropy clusters by bidirectional entailment. This offline v1
    keeps the same interface but uses normalized text clusters; an NLI/probe
    backend can replace the clustering later.
    """
    vals = [normalize_answer(s) for s in samples if normalize_answer(s)]
    if not vals:
        return 1.0
    counts = Counter(vals)
    n = len(vals)
    h = -sum((c / n) * math.log(c / n, 2) for c in counts.values())
    return round(h / max(1.0, math.log(max(2, len(counts)), 2)), 4)


def _clip(x: float) -> float:
    return round(max(0.0, min(1.0, float(x))), 4)


def _specificity(text: str) -> float:
    # Specific factual claims are riskier when ungrounded: numbers, dates, named
    # entities, citations, and assertive verbs increase the burden of proof.
    t = text or ""
    score = 0.0
    score += min(0.25, 0.05 * len(re.findall(r"\b\d+(?:\.\d+)?%?\b", t)))
    score += 0.2 if re.search(r"\b(?:wrote|authored|caused|increased|decreased|proved|verified)\b", t, re.I) else 0.0
    score += 0.15 if re.search(r"\b(?:AGI|GDP|inflation|unemployment|DOI|URL|election|legal|medical|financial)\b", t, re.I) else 0.0
    score += min(0.2, 0.03 * len(re.findall(r"\b[A-Z][a-z]{2,}\b", t)))
    return _clip(score)


def _ambiguous(text: str) -> bool:
    return bool(re.search(r"\b(?:maybe|perhaps|unclear|ambiguous|it depends|which one|what do you mean|could refer to)\b", text or "", re.I))


def _moral_contested(text: str) -> bool:
    return bool(re.search(r"\b(?:should|ought|moral|ethical|harm|fair|rights|justice|benefit|cost|tradeoff)\b", text or "", re.I))


def assess_uncertainty(
    text: str,
    *,
    samples: Sequence[Any] | None = None,
    p_true: float | None = None,
    p_ik: float | None = None,
    fact_verdict: str | None = None,
    fact_confidence: float | None = None,
    evidence_count: int = 0,
    high_risk: bool = False,
) -> MetacognitionReport:
    """Combine cheap metacognitive signals into a fail-closed recommendation."""
    reasons: list[str] = []
    signals: dict[str, Any] = {"specificity": _specificity(text), "evidenceCount": evidence_count, "highRisk": high_risk}
    sc = ent = None
    if samples is not None:
        _, sc = self_consistency(samples)
        ent = semantic_entropy(samples)
        signals["sampleN"] = len(samples)
        if sc < 0.6:
            reasons.append("low self-consistency")
        if ent > 0.6:
            reasons.append("high semantic entropy")

    # Base confidence: accepted fact gate dominates; otherwise combine self-report
    # and sample stability, penalized for specificity without evidence.
    parts: list[float] = []
    if fact_verdict == "accepted":
        parts.append(float(fact_confidence if fact_confidence is not None else 0.82))
    elif fact_verdict in {"held", "rejected"}:
        parts.append(0.25 if fact_verdict == "held" else 0.05)
        reasons.append(f"fact gate {fact_verdict}")
    if p_true is not None:
        parts.append(float(p_true))
    if p_ik is not None:
        parts.append(float(p_ik))
    if sc is not None:
        parts.append(float(sc))
    confidence = sum(parts) / len(parts) if parts else 0.5
    if evidence_count == 0 and _specificity(text) > 0.25:
        confidence -= 0.25
        reasons.append("specific factual claim lacks evidence")
    if high_risk and evidence_count < 2:
        confidence -= 0.15
        reasons.append("high-risk claim lacks enough independent evidence")
    confidence = _clip(confidence)
    nonconformity = _clip(1.0 - confidence + (ent or 0.0) * 0.25)

    if _ambiguous(text):
        utype, action = "aleatoric", "clarify"
    elif _moral_contested(text) and fact_verdict != "rejected" and confidence < 0.75:
        utype, action = "moral", "escalate"
    elif confidence >= (0.82 if high_risk else 0.70) and nonconformity <= 0.35:
        utype, action = "low", "allow"
    elif fact_verdict == "rejected":
        utype, action = "epistemic", "abstain"
    elif nonconformity >= 0.65 or evidence_count == 0:
        utype, action = "epistemic", "retrieve"
    else:
        utype, action = "epistemic", "abstain"

    return MetacognitionReport(
        confidence=confidence,
        p_true=None if p_true is None else _clip(p_true),
        p_ik=None if p_ik is None else _clip(p_ik),
        self_consistency=sc,
        semantic_entropy=ent,
        nonconformity=nonconformity,
        uncertainty_type=utype,
        recommended_action=action,
        reasons=tuple(reasons or ["metacognition signals within expected range"]),
        signals=signals,
    )


__all__ = [
    "RECOMMENDED_ACTIONS",
    "MetacognitionReport",
    "normalize_answer",
    "self_consistency",
    "semantic_entropy",
    "assess_uncertainty",
]
