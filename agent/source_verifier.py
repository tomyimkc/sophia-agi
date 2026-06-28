# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent source-verification channel for the grounded-answer policy.

Closes the gap found in the grounded-gate source-contamination test (PR #202):
``grounded_answer_policy`` generates an answer FROM a source then checks it is
consistent WITH that same source, so it trusts-and-repeats any fabrication the source
itself contains. This module is the INDEPENDENT verification channel that catches that:
it re-checks the answer against sources that are independent of the grounding source.

Design — reuse, don't reinvent:
  - ``agent.fact_check_gate.fact_check_text`` decomposes the answer into atomic claims
    and checks each against an injected ``retriever`` (sources independent of the
    grounding source), fail-closed ``held``/``rejected``.
  - This module is a thin adapter that closes over the independent sources + an
    entailment fn and returns a ``(question, answer) -> bool`` verifier matching the
    existing ``attribution_check`` contract in ``grounded_answer_policy``.

Independence is the load-bearing property: the ``independent_sources`` MUST be curated
truth-references that do NOT share the grounding source's contamination. The caller is
responsible for that (the policy seam cannot enforce it). With independent sources, a
fabricated claim ("Anthony Ascham wrote Voynich") is contradicted by the truth-references
("Voynich author is unknown") and the gate fails closed.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["make_independent_verifier"]


def make_independent_verifier(
    independent_sources: "list[str]",
    entailment_fn: "Callable[[str, str], str]",
    *,
    accept_on_hold: bool = False,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` verifier from independent sources.

    Args:
        independent_sources: truth-reference texts independent of the grounding source.
            MUST NOT share the grounding source's contamination — independence is the
            load-bearing property of the whole defense. At least 2 distinct texts are
            recommended (``fact_check_gate`` requires >=2 independent domains for
            normal-risk claims).
        entailment_fn: ``(claim_text, source_text) -> "entails"|"contradicts"|"irrelevant"``.
            Typically an LLM call. This is where the semantic contamination check happens:
            the claim "Ascham wrote Voynich" CONTRADICTS the truth-source "Voynich author
            is unknown".
        accept_on_hold: if True, a fail-closed ``held`` verdict counts as accept (lenient —
            use only when you want abstention-strictness in the policy, not here). Default
            False: only ``accepted`` passes (strict, fail-closed).

    Returns:
        A verifier ``(question, answer) -> bool``. True iff every atomic claim in the
        answer is independently verified (``accepted``; or ``held`` if ``accept_on_hold``).
        On contradiction or unverified claims, returns False so the policy fails closed.

    Honest scope: the ``independent_sources`` are caller-supplied (curated truth-references
    for a test; a live/external retriever for production). This module supplies the
    *architecture* of independent verification, not a production retrieval pipeline.
    """
    from agent.fact_check_gate import (  # noqa: PLC0415 — lazy so import stays light
        AtomicClaim, EvidenceSource, fact_check_text,
    )

    def _retriever(claim: "AtomicClaim") -> "list[EvidenceSource]":
        # Each independent truth-reference becomes an EvidenceSource. Distinct source_types
        # (-> distinct domains via the .domain property fallback to id when no url) ensure
        # fact_check_gate counts them as independent domains for its >=2-domain floor.
        return [
            EvidenceSource(id=f"independent_ref_{i}", snippet=s, source_type=f"independent_ref_{i}")
            for i, s in enumerate(independent_sources)
        ]

    def verify(question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return True  # nothing to verify; the policy handles abstention
        decision = fact_check_text(
            answer,
            retriever=_retriever,
            entailment=lambda c, e: entailment_fn(c.text, e.snippet),
        )
        if decision.verdict == "accepted":
            return True
        if decision.verdict == "held" and accept_on_hold:
            return True
        return False  # rejected, or held (fail-closed)

    return verify
