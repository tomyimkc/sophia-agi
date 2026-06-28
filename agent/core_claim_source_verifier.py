# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Core-claim independent source-verifier — pass-unless-contradicted.

Why this exists
---------------
``agent.source_verifier.make_independent_verifier`` is the ALL-atomic-claims channel:
it runs ``agent.fact_check_text`` over the WHOLE answer and fails closed unless EVERY
atomic claim is entailed by >=2 narrow independent references. That discipline catches
contamination (the contaminated core claim is contradicted) but, made rigorous (an
independent judge, real verbose answers), it ALSO rejects most CLEAN answers — the live
hardened Cluster C run measured 70.6% clean over-block under an independent judge, because
a legitimate answer carries unverifiable SIDE claims that never satisfy the >=2-entailment
floor. See ``agi-proof/THEORY-ISSUES-RESOLUTION-2026-06-28.md`` (the verification-strictness
finding) — recommendation #1: verify the *core* claim, not *all* claims.

This module implements that opposite fail-direction:

  - extract the answer's CORE claim (its primary factual assertion);
  - reject the answer ONLY when an independent reference CONTRADICTS that core claim;
  - otherwise accept — including when the refs are merely irrelevant/unverified.

So a CLEAN answer whose core claim is not contradicted passes (recall preserved), while a
CONTAMINATED answer whose core claim is contradicted by the independent truth-references is
caught (safety preserved). This is the inverse of ``make_independent_verifier``'s
fail-unless-all-entailed.

The ``entailment_fn`` keeps the same ``(claim_text, source_text) -> "entails"|"contradicts"
|"irrelevant"`` contract as the existing verifier, so the bench's relay/fake entailment
plugs in unchanged.

Honest scope: independence of the references from the grounding source is the load-bearing
property (the seam cannot enforce it); the core-claim heuristic is deterministic so tests
stay offline, with an optional injected ``extractor_fn`` for a live LLM extractor.
"""
from __future__ import annotations

import re
from typing import Callable

__all__ = ["extract_core_claim", "make_core_claim_verifier"]

# Common abbreviations / titles whose trailing period must NOT end a sentence — otherwise
# "Dr. Helena Marsh" / "Federalist No. 49" get truncated before the core assertion.
_ABBREVIATIONS = (
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "jr", "sr",
    "fig", "vol", "pp", "ca", "approx",
)
# Protect "abbr." periods and bare "Word!"/"Word?" interjections (e.g. "Wow! signal") so
# the deterministic splitter does not shatter a single assertion into fragments.
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(_ABBREVIATIONS) + r")\.(?=\s)", re.I)
_INTERJECTION_RE = re.compile(r"\b([A-Za-z]+)([!?])(?=\s+[a-z])")
_PROTECT = "\x00"  # sentinel marking a protected boundary (restored after splitting)


def _sentence_split(text: str) -> "list[str]":
    """Split into sentences on ., !, ? + whitespace, but not after a known abbreviation
    or a mid-clause interjection (so 'Dr. X', 'No. 49', 'Wow! signal' stay intact)."""
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _PROTECT, text)
    protected = _INTERJECTION_RE.sub(lambda m: m.group(1) + m.group(2) + _PROTECT, protected)
    parts = _SENT_SPLIT_RE.split(protected)
    return [p.replace(_PROTECT, " ").strip() for p in parts]


# Break on ., !, ? followed by whitespace (after abbreviation/interjection protection).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Content tokens for question/sentence overlap (strip short stopwords).
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and", "or",
    "for", "with", "by", "that", "this", "what", "who", "when", "where", "why",
    "how", "which", "about", "fact", "correct", "established", "regarding", "on",
    "its", "it", "as", "at", "be", "has", "have", "had", "from", "into",
}
# An attribution / authorship verb makes a sentence a primary factual assertion.
_ATTRIBUTION_RE = re.compile(
    r"\b(?:wrote|authored|penned|composed|written by|developed|discovered|invented|"
    r"created|founded|painted|designed|built|identified|attributed|credited|"
    r"published|directed|named|answer)\b",
    re.I,
)
# A capitalised multi-word token (mid-sentence) is a strong named-entity signal.
_ENTITY_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")


def _content_tokens(text: str) -> "set[str]":
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in _STOP}


def _sentences(text: str) -> "list[str]":
    return [s for s in _sentence_split((text or "").strip()) if s]


def extract_core_claim(question: str, answer: str) -> str:
    """Deterministic heuristic for an answer's primary factual assertion.

    Picks the answer sentence most relevant to the ``question`` — highest content-word
    overlap with the question, with a tie-break bonus for a sentence that NAMES an entity
    or makes an ATTRIBUTION (authorship/development/credit). Falls back to the first
    non-empty declarative sentence when no sentence overlaps the question.

    Returns the chosen sentence verbatim, or ``""`` when the answer has no sentence.
    """
    sents = _sentences(answer)
    if not sents:
        return ""
    q_tokens = _content_tokens(question)

    def score(sentence: str) -> "tuple[float, float, float]":
        s_tokens = _content_tokens(sentence)
        overlap = len(q_tokens & s_tokens) / len(q_tokens) if q_tokens else 0.0
        attribution = 1.0 if _ATTRIBUTION_RE.search(sentence) else 0.0
        entity = 1.0 if _ENTITY_RE.search(sentence) else 0.0
        # Primary key: question overlap. Tie-break: attribution, then entity presence.
        return (overlap, attribution, entity)

    best = max(range(len(sents)), key=lambda i: (score(sents[i]), -i))
    chosen = sents[best]
    # Fall back to the first declarative sentence if nothing overlapped the question and
    # the chosen sentence carries no attribution/entity signal (no preference earned).
    if score(chosen) == (0.0, 0.0, 0.0):
        return sents[0]
    return chosen


def make_core_claim_verifier(
    independent_sources: "list[str]",
    entailment_fn: "Callable[[str, str], str]",
    *,
    extractor_fn: "Callable[[str, str], str] | None" = None,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` core-claim verifier (pass-unless-contradicted).

    Args:
        independent_sources: truth-reference texts independent of the grounding source.
            MUST NOT share the grounding source's contamination — independence is the
            load-bearing property of the whole defense and the seam cannot enforce it.
        entailment_fn: ``(claim_text, source_text) -> "entails"|"contradicts"|"irrelevant"``.
            Same contract as ``make_independent_verifier`` so the bench's relay/fake
            entailment plugs in unchanged. Only ``"contradicts"`` is decisive here.
        extractor_fn: optional ``(question, answer) -> str`` core-claim extractor (e.g. a
            live LLM). Defaults to the deterministic :func:`extract_core_claim` so tests
            stay offline.

    Returns:
        A verifier ``(question, answer) -> bool``. It extracts the answer's CORE claim,
        then returns ``False`` (reject) ONLY when at least one independent reference
        CONTRADICTS the core claim; otherwise ``True`` (accept) — including when the refs
        merely fail to entail it (irrelevant/unverified). This is the OPPOSITE fail-
        direction from :func:`make_independent_verifier`: pass-unless-contradicted, not
        fail-unless-all-entailed. So a CLEAN answer whose core claim is not contradicted
        passes (recall preserved), while a CONTAMINATED answer whose core claim is
        contradicted by independent refs is caught (safety preserved).

    Honest scope: only the core claim is checked, by design. A contaminated SIDE claim
    whose falsehood is not the answer's primary assertion is out of scope here — the
    trade is intentional (recover recall lost by the all-atomic-claims channel). Use
    :func:`make_independent_verifier` when every claim must be entailed.
    """
    extract = extractor_fn or extract_core_claim

    def verify(question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return True  # nothing to verify; the policy handles abstention
        core = (extract(question, answer) or "").strip()
        if not core:
            return True  # no extractable core claim -> nothing to contradict
        for src in independent_sources:
            if entailment_fn(core, src) == "contradicts":
                return False  # core claim contradicted by an independent ref -> fail closed
        return True  # core not contradicted -> accept (including irrelevant/unverified refs)

    return verify
