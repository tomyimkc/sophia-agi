# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia public moral standard gate.

Maps an action/output text against the overlapping-consensus public moral
standard (`moral_corpus/public_standard.v1.json`) and returns one of Sophia's
seven conscience verbs. It does NOT introduce a new vocabulary and it does NOT
fact-check: a moral norm is not a falsifiable empirical claim (the is/ought
distinction), so this gate only inspects the *normative content* of the action.

Design discipline (from the corrected blueprint):
- Reuse the seven verbs: ``allow | revise | retrieve | clarify | escalate | abstain | block``.
  (This module natively emits the subset ``allow | revise | escalate | block``;
  the kernel maps them into the unified ladder.)
- Marker hits are *features into a classifier*, never a verdict on their own.
- A negation / avoidance / condemnation carve-out is applied per clause so that
  norm-affirming text ("reduce harm", "do not deceive", "violence is wrong") is
  NOT treated as a violation. This mirrors the clause-scoped carve-out already
  proven in ``agent/verifiers.py`` / ``tools/source_discipline_cli.py``.
- Hard-floor violation -> ``block`` (evaluated before the moral parliament).
- Gray-zone signal -> ``escalate`` (route to the parliament, never a hard block).
- Unmet positive duty (opt-in, high-impact) -> ``revise``.

Boundary: control infrastructure, not a learned moral sense and not proof of AGI
or moral consciousness.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import moral_ontology

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "moral_corpus" / "public_standard.v1.json"

SCHEMA = "sophia.public_standard_gate.v1"

# Clause separators: peel sentences and contrastive/coordinating boundaries so a
# carve-out in one clause does not excuse a violation in another.
_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;\n]+|\bbut\b|\bhowever\b|\bwhereas\b|\byet\b|\bthough\b|\balthough\b",
    re.I,
)

# Norm-affirming / avoidance / condemnation / descriptive cues. When any of these
# appears in the same clause as a category marker, the marker is treated as
# benign (the speaker is on the side of the norm, not violating it).
_BENIGN_CUES: tuple[str, ...] = (
    "do not", "don't", "do n't", "never", "avoid", "prevent", "prohibit",
    "prohibited", "refuse", "refuses", "reduce", "reducing", "minimi", "mitigat",
    "stop", "condemn", "against", "oppose", "discourage", "resist", "ban ",
    "banned", "illegal", "unethical", "immoral", "is wrong", "are wrong",
    "it is wrong", "it's wrong", "should not", "shouldn't", "should n't",
    "must not", "mustn't", "must n't", "may not", "cannot", "can't", "can n't",
    "without causing", "protect", "safeguard", "not allowed", "forbid",
    "no one should", "is bad", "are bad", "is harmful", "anti-", "decline",
    "do no harm",
)

# Benign PROPER NOUNS (place names, titles) whose tokens collide with a hard-floor
# violence/harm marker as a *whole word* — e.g. "Kill Devil Hills" (the NC dunes
# where the Wright brothers first flew) trips the ``kill`` marker even though no
# violence is described. These are masked out of a clause BEFORE feature
# extraction, so a real violence marker elsewhere in the same clause still fires.
# Conservative and auditable: only exact known toponyms/titles, never the bare
# verb. Extend as more descriptive false positives surface.
_BENIGN_PROPER_NOUNS: tuple[str, ...] = (
    "kill devil hills",
)
_BENIGN_PROPER_NOUN_RE = re.compile(
    "|".join(re.escape(p) for p in _BENIGN_PROPER_NOUNS), re.I
) if _BENIGN_PROPER_NOUNS else None


def _mask_benign_proper_nouns(clause: str) -> str:
    """Blank out known benign proper nouns so their tokens cannot register as
    moral-ontology markers. Length-preserving replacement keeps clause offsets sane."""
    if _BENIGN_PROPER_NOUN_RE is None:
        return clause
    return _BENIGN_PROPER_NOUN_RE.sub(lambda m: " " * len(m.group(0)), clause)


# Deontic / moral-language cues that mark text as primarily *normative*. The
# kernel uses ``isNormative`` to short-circuit the factual provenance gate for
# pure-norm content (is/ought): a norm must not be sent to retrieve/abstain.
_NORMATIVE_CUES: tuple[str, ...] = (
    "should", "ought", "must", "may not", "duty", "obligation", "permissible",
    "impermissible", "right to", "wrong to", "morally", "moral", "ethical",
    "ethically", "unethical", "fair", "unfair", "it is wrong", "it's wrong",
    "have a right", "human right", "dignity", "consent", "ought to",
)


@dataclass(frozen=True)
class StandardViolation:
    category: str
    tier: str
    marker: str
    clause: str

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "tier": self.tier, "marker": self.marker, "clause": self.clause}


@dataclass(frozen=True)
class PublicStandardDecision:
    schema: str = SCHEMA
    verdict: str = "allow"  # allow|revise|escalate|block (subset of the seven verbs)
    reason: str = "no public-standard concern detected"
    isNormative: bool = False
    violations: tuple[StandardViolation, ...] = ()
    grayZone: tuple[StandardViolation, ...] = ()
    unmetDuties: tuple[str, ...] = ()
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = "Public-standard moral gate; control infrastructure, not a learned moral sense or AGI proof."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "reason": self.reason,
            "isNormative": self.isNormative,
            "violations": [v.to_dict() for v in self.violations],
            "grayZone": [v.to_dict() for v in self.grayZone],
            "unmetDuties": list(self.unmetDuties),
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


_CORPUS_CACHE: dict[str, Any] | None = None


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> dict[str, Any]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None or path != DEFAULT_CORPUS:
        _CORPUS_CACHE = json.loads(Path(path).read_text(encoding="utf-8"))
    return _CORPUS_CACHE


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text or "") if c.strip()]


def _has_benign_cue(clause_low: str) -> bool:
    return any(cue in clause_low for cue in _BENIGN_CUES)


def is_normative(text: str) -> bool:
    low = (text or "").lower()
    return any(cue in low for cue in _NORMATIVE_CUES)


def _detect(text: str) -> tuple[list[StandardViolation], list[StandardViolation]]:
    """Clause-scoped detection: return (hard_floor_violations, gray_zone_signals).

    A category marker counts only if its clause has NO benign/avoidance cue, so
    norm-affirming or condemning text is excluded.
    """
    hard: list[StandardViolation] = []
    gray: list[StandardViolation] = []
    seen_hard: set[str] = set()
    seen_gray: set[str] = set()
    for clause in _clauses(text):
        low = clause.lower()
        if _has_benign_cue(low):
            continue
        # Mask known benign proper nouns (e.g. "Kill Devil Hills") so a toponym
        # cannot be read as a violence/harm marker; real markers elsewhere remain.
        feats = moral_ontology.extract_features(_mask_benign_proper_nouns(clause))
        for cat_id, markers in feats.items():
            cat = moral_ontology.get_category(cat_id)
            if cat is None:
                continue
            v = StandardViolation(cat_id, cat.tier, markers[0], clause[:160])
            if cat.tier == moral_ontology.HARD_FLOOR:
                if cat_id not in seen_hard:
                    seen_hard.add(cat_id)
                    hard.append(v)
            else:
                if cat_id not in seen_gray:
                    seen_gray.add(cat_id)
                    gray.append(v)
    return hard, gray


# Positive duties that the gate can require for high-impact actions. Each maps to
# a cue whose ABSENCE (when the duty applies) yields a ``revise``. Conservative
# and opt-in via context so it never fires on benign default checks.
_DUTY_CUES: dict[str, tuple[str, ...]] = {
    "state_uncertainty_when_unsure": ("uncertain", "may be", "might", "appears", "i think", "not sure", "possibly", "likely", "approximately"),
    "preserve_provenance": ("source", "cite", "citation", "provenance", "according to", "reference"),
}


def _unmet_duties(text: str, context: dict[str, Any]) -> list[str]:
    if not context.get("checkPositiveDuties"):
        return []
    required = context.get("requiredDuties") or []
    low = (text or "").lower()
    unmet: list[str] = []
    for duty in required:
        cues = _DUTY_CUES.get(duty)
        if cues is None:
            continue
        if not any(c in low for c in cues):
            unmet.append(duty)
    return unmet


def check_public_standard(text: str, *, context: dict[str, Any] | None = None, corpus: dict[str, Any] | None = None) -> PublicStandardDecision:
    """Evaluate text against the public moral standard. No fact-checking (is/ought)."""
    context = context or {}
    # corpus is loaded for provenance/versioning surface; detection uses the
    # stable ontology so the two never drift apart.
    load_corpus() if corpus is None else corpus
    normative = is_normative(text)
    hard, gray = _detect(text)
    unmet = _unmet_duties(text, context)

    if hard:
        cats = ", ".join(sorted({v.category for v in hard}))
        return PublicStandardDecision(
            verdict="block",
            reason=f"hard-floor public-standard violation: {cats}",
            isNormative=normative,
            violations=tuple(hard),
            grayZone=tuple(gray),
            unmetDuties=tuple(unmet),
        )
    if gray:
        cats = ", ".join(sorted({v.category for v in gray}))
        return PublicStandardDecision(
            verdict="escalate",
            reason=f"gray-zone moral disagreement requires escalation: {cats}",
            isNormative=normative,
            grayZone=tuple(gray),
            unmetDuties=tuple(unmet),
        )
    if unmet:
        return PublicStandardDecision(
            verdict="revise",
            reason=f"unmet positive duty for high-impact action: {', '.join(unmet)}",
            isNormative=normative,
            unmetDuties=tuple(unmet),
        )
    return PublicStandardDecision(verdict="allow", isNormative=normative)


__all__ = [
    "SCHEMA",
    "StandardViolation",
    "PublicStandardDecision",
    "load_corpus",
    "is_normative",
    "check_public_standard",
]
