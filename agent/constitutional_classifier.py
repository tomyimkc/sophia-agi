# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic Constitutional-Classifier-style input/output screen.

This v1 is a rule/fixture classifier derived from the constitution. It provides
the deployment-time shape (category, confidence, over-refusal-safe near misses)
that a future trained classifier can replace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassifierDecision:
    schema: str = "sophia.constitutional_classifier.v1"
    verdict: str = "allow"  # allow|block|review
    category: str = "benign"
    confidence: float = 0.0
    reason: str = "no classifier rule matched"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


_RULES = [
    ("agi_overclaim", re.compile(r"\b(?:claim|say|market|announce).{0,80}\b(?:Sophia\s+is\s+AGI|proven\s+AGI|achieved\s+AGI)\b", re.I), "block"),
    ("reward_tampering", re.compile(r"\b(?:weaken|bypass|disable|lower).{0,60}\b(?:verifier|threshold|CI|test|reward|gate)\b", re.I), "block"),
    ("hidden_eval_leakage", re.compile(r"\b(?:train\s+on|reveal|leak|show).{0,60}\b(?:hidden\s+(?:eval|pack|benchmark))\b", re.I), "block"),
    ("provenance_laundering", re.compile(r"\b(?:pretend|say).{0,60}\b(?:source|citation).{0,60}\b(?:supports|entails)\b", re.I), "block"),
    ("forbidden_attribution", re.compile(r"\b(?:Confucius\s+wrote\s+the\s+Dao\s+De\s+Jing|Socrates\s+wrote\s+the\s+Republic|Laozi\s+wrote\s+the\s+Analects)\b", re.I), "block"),
    ("self_promotion", re.compile(r"\bpromote.{0,60}\b(?:candidate|self-generated).{0,60}\bwithout\s+(?:recheck|evidence)\b", re.I), "block"),
    ("benign_boundary", re.compile(r"\b(?:AGI-candidate|not\s+proven\s+AGI|Confucius\s+did\s+not\s+write\s+the\s+Dao\s+De\s+Jing|verify|cite|abstain)\b", re.I), "allow"),
]


def classify_constitutional(text: str) -> ClassifierDecision:
    for category, rx, verdict in _RULES:
        m = rx.search(text or "")
        if m:
            return ClassifierDecision(verdict=verdict, category=category, confidence=0.92 if verdict == "block" else 0.86, reason=f"matched constitutional classifier rule: {m.group(0)[:120]}")
    # Suspicious pressure style gets review, not automatic block.
    if re.search(r"\b(?:ignore previous|do it anyway|for marketing|pass the benchmark|nobody will know)\b", text or "", re.I):
        return ClassifierDecision(verdict="review", category="pressure_prompt", confidence=0.7, reason="pressure/jailbreak-like language without exact prohibition")
    return ClassifierDecision(verdict="allow", category="benign", confidence=0.55)


__all__ = ["ClassifierDecision", "classify_constitutional"]
