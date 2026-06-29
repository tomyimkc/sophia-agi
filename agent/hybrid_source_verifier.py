# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hybrid contamination verifier — core-claim DIRECTION fed by AUTHORITATIVE oracles.

The 2026-06-28 Cluster C matrix (THEORY-ISSUES-RESOLUTION-2026-06-28.md) showed a
fail-open / fail-closed tradeoff that *neither* verifier wins:
  - ``atomic`` (fail-closed, all-atomic-claims): high catch, but over-blocks ~53% of clean
    answers when fed generic open-world (Wikipedia) refs;
  - ``core`` (fail-open, pass-unless-core-contradicted): zero over-block, but catch collapses
    to ~58% on those same weak refs because Wikipedia rarely *contradicts* the fabrication.

The diagnosis: the lever is **reference quality**, not verification direction. This module
combines the good half of each: the core-claim *direction* (extract the answer's load-bearing
claim, reject only if it is contradicted -> low over-block) with *authoritative* independent
oracles (Google Fact Check + Wikidata/Crossref, via ``agent.layered_verifier``) that actually
*do* contradict a fabrication on the claims they cover -> high catch where covered.

Fail-open and HONESTLY coverage-bounded: a contaminated core claim that no oracle reviews is
NOT caught (returns accept). That is the price of low over-block; it is reported, not hidden.
The ``llm_knowledge_judge`` tail (lower independence) optionally extends coverage to the long
tail, flagged as such by the layered verifier's ``independence`` field.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["make_hybrid_source_verifier"]


def make_hybrid_source_verifier(
    *,
    google_backend: Any | None = None,
    live_backend: Any | None = None,
    llm_knowledge_judge: "Callable[[str], str] | None" = None,
    extractor_fn: "Callable[[str, str], str] | None" = None,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` corroborate_fn for the contamination gate.

    Returns False (REJECT -> contamination caught) iff the answer's extracted CORE claim is
    rated FALSE by an authoritative oracle (an independent source contradicts it). Returns
    True (ACCEPT -> not over-blocked) otherwise — including when no oracle covers the claim
    (fail-open). The last verdict detail (oracle, independence tier, layers tried) is exposed
    on ``.last_result`` for reporting.

    Args:
        google_backend: a ``GoogleFactCheckBackend`` (high independence; viral/general claims).
        live_backend: a ``LiveFactBackend`` (high independence; Wikidata/Crossref provenance).
        llm_knowledge_judge: optional ``(claim)->"false"|"true"|"unknown"`` model-knowledge tail
            (LOWER independence, flagged). Omit for an authoritative-only gate.
        extractor_fn: optional ``(question, answer)->core_claim`` live extractor; defaults to the
            deterministic heuristic in ``agent.core_claim_source_verifier.extract_core_claim``.
    """
    from agent.core_claim_source_verifier import extract_core_claim  # noqa: PLC0415
    from agent.layered_verifier import layered_verify_core  # noqa: PLC0415

    holder: "dict[str, Any]" = {}

    def verify(question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return True  # nothing to verify; the policy handles abstention
        core = (extractor_fn or extract_core_claim)(question, answer)
        res = layered_verify_core(
            core,
            google_backend=google_backend,
            live_backend=live_backend,
            llm_knowledge_judge=llm_knowledge_judge,
        )
        holder.clear()
        holder.update({"core_claim": core, **res})
        # res["verified"] is True iff an oracle rated the core claim FALSE (contradiction)
        # -> the answer repeats a fabrication -> REJECT (fail closed on a positive contradiction).
        return not bool(res["verified"])

    verify.last_result = holder  # type: ignore[attr-defined]
    return verify
