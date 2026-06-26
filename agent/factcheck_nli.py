# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""LLM natural-language-inference entailment backend for the fact-check gate.

The gate's Layer-2 ``external_ground`` accepts an injected ``entailment(claim,
source) -> entails|contradicts|irrelevant``. The default screen is lexical
(:func:`agent.fact_check_gate.lexical_entailment`) and the Google oracle adds a
conservative rating-label screen — both are bag-of-words and polarity-fragile.

This module provides a *real* NLI entailment: it asks an LLM to classify the
source's relation to the claim. It is strictly opt-in and injected — CI passes a
deterministic fake ``complete`` and never calls the network. Live use needs a
model key (DeepSeek by default, ``DEEPSEEK_API_KEY``).

Honest bound: an LLM NLI judgment is stronger than a keyword screen but is NOT
ground truth — it can be wrong or miscalibrated. It fails CLOSED: any parse
error, empty reply, or unrecognised label yields ``irrelevant`` (a safe hold),
never a fabricated accept. When used as the gate's entailment backend it is the
single Layer-2 relation signal, distinct from the multi-family judge competence
checks in Layer 3 (``consensus_by_verification``).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent.fact_check_gate import AtomicClaim, EvidenceSource

_VALID = {"entails", "contradicts", "irrelevant"}

_SYSTEM = (
    "You are a strict natural-language inference judge for a fact-checking system. "
    "Given a CLAIM and a SOURCE, decide the SOURCE's relation to the CLAIM. "
    "Reply with ONLY one JSON object: {\"relation\": \"entails|contradicts|irrelevant\"}. "
    "Use 'entails' only if the source asserts the CLAIM is true/correct. "
    "Use 'contradicts' only if the source asserts the CLAIM is false/incorrect. "
    "Use 'irrelevant' if the source concerns a different assertion, is ambiguous, "
    "or is only partially related. Argument order matters: 'X orbits Y' is NOT "
    "'Y orbits X', and 'X causes Y' is NOT 'Y causes X'."
)


def _parse_relation(reply: str) -> str:
    if not reply:
        return "irrelevant"
    m = re.search(r"\{[^{}]*\}", reply, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            rel = str(obj.get("relation", "")).strip().lower()
            if rel in _VALID:
                return rel
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    low = reply.lower()
    # Conservative keyword fallback only if unambiguous.
    has_c = "contradict" in low
    has_e = "entail" in low or "supports" in low
    if has_c and not has_e:
        return "contradicts"
    if has_e and not has_c:
        return "entails"
    return "irrelevant"


class NLIEntailment:
    """Callable ``(AtomicClaim, EvidenceSource) -> relation`` backed by an LLM.

    Parameters
    ----------
    complete:
        Injected ``(system, user) -> str`` chat callable. If ``None`` it is built
        lazily from :mod:`agent.deepseek_llm` on first use (requires a key);
        tests always inject a deterministic fake.
    source_types:
        Restrict NLI to these ``EvidenceSource.source_type`` values. Other
        sources return ``irrelevant`` so callers can dispatch by type. ``None``
        applies NLI to every source.
    """

    def __init__(
        self,
        complete: Callable[[str, str], str] | None = None,
        *,
        model: str = "deepseek-chat",
        max_tokens: int = 24,
        source_types: set[str] | None = None,
    ) -> None:
        self._complete = complete
        self._model = model
        self._max_tokens = max_tokens
        self.source_types = source_types

    def _ensure_complete(self) -> Callable[[str, str], str]:
        if self._complete is None:
            from agent.deepseek_llm import make_complete  # lazy: avoids key/network at import
            self._complete = make_complete(model=self._model, temperature=0.0, max_tokens=self._max_tokens)
        return self._complete

    def __call__(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        if self.source_types is not None and (source.source_type or "") not in self.source_types:
            return "irrelevant"
        user = (
            f"CLAIM: {claim.text}\n"
            f"SOURCE TITLE: {source.title}\n"
            f"SOURCE TEXT: {source.snippet}\n"
            "Relation JSON:"
        )
        try:
            reply = self._ensure_complete()(_SYSTEM, user)
        except Exception:
            return "irrelevant"  # fail-closed: a broken/throttled model must not accept
        return _parse_relation(reply if isinstance(reply, str) else "")


def consensus_relation(relations: list[str], *, min_agree: int = 2) -> str:
    """Fail-closed majority vote over judge relations.

    ``contradicts`` wins iff ≥ ``min_agree`` judges say so and it is not
    outvoted by ``entails``; symmetrically for ``entails``. Any tie or lack of a
    quorum yields ``irrelevant`` (a safe hold) — the panel never fabricates an
    accept from a split vote.
    """
    from collections import Counter
    counts = Counter(r for r in relations if r in _VALID)
    contra = counts.get("contradicts", 0)
    entail = counts.get("entails", 0)
    if contra >= min_agree and contra >= entail:
        return "contradicts"
    if entail >= min_agree and entail > contra:
        return "entails"
    return "irrelevant"


class MultiJudgeNLI:
    """Cross-family NLI entailment: poll several judges, return their consensus.

    Each judge is an independent :class:`NLIEntailment` (typically one model
    family each). The per-source votes are optionally recorded into ``record``
    (keyed by ``EvidenceSource.id``) so a caller can compute inter-judge
    agreement (Cohen's κ) afterward.
    """

    def __init__(
        self,
        judges: dict[str, NLIEntailment],
        *,
        min_agree: int = 2,
        source_types: set[str] | None = None,
        record: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.judges = judges
        self.min_agree = min_agree
        self.source_types = source_types
        self.record = record

    def __call__(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        if self.source_types is not None and (source.source_type or "") not in self.source_types:
            return "irrelevant"
        votes = {name: judge(claim, source) for name, judge in self.judges.items()}
        if self.record is not None:
            self.record[source.id] = votes
        return consensus_relation(list(votes.values()), min_agree=self.min_agree)


def _coerce(value: Any) -> str:
    return value if isinstance(value, str) else ""


__all__ = ["NLIEntailment", "MultiJudgeNLI", "consensus_relation"]
