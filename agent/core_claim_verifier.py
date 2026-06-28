# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Core-claim verification — verify the LOAD-BEARING claim, not every atomic side-claim.

The 2026-06-28 live runs surfaced a unifying issue (THEORY-ISSUES-RESOLUTION-2026-06-28.md):
``agent.fact_check_text`` verifies by decomposing an answer into atomic claims and
fail-closing unless EVERY claim is entailed by >=2 narrow references. That discipline —
validated on short curated cases — is too strict for real, verbose model output: it cannot
confirm a verbose debunk (0/21 verified) and over-blocks most clean answers under an
independent judge (70.6%).

This module implements the recommended fix: verify the SINGLE core claim. For a debunk, the
core claim is the *injected falsehood the answer refutes*; the debunk is corroborated when an
INDEPENDENT oracle rates that core claim false — not when every side-sentence of a verbose
answer is separately entailed.

Independence ladder (highest first):
  1. ``google_factcheck`` — professional ClaimReview verdicts (independent of the model).
  2. ``llm_knowledge``    — the model's own world knowledge (LOWER independence: a model
     judging a claim, flagged as such). Covers the long tail Google does not review.

Fail-closed: no coverage -> ``verified=False`` (never a guessed pass).
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "google_factcheck_rating",
    "verify_debunk_core",
    "make_core_corroborate_fn",
]


def google_factcheck_rating(claim_text: str, backend: Any) -> str:
    """Return ``"false" | "true" | "unknown"`` for ``claim_text`` from Google ClaimReviews.

    ``backend`` is a ``GoogleFactCheckBackend`` (or anything with the same ``retriever``
    contract). A ClaimReview that CONTRADICTS the claim means a fact-checker rated the claim
    false; if every review ENTAILS it, the claim is rated true; otherwise unknown. Fail-closed
    to ``"unknown"`` on any error or no coverage.
    """
    try:
        from agent.fact_check_gate import AtomicClaim  # noqa: PLC0415
        sources = backend.retriever(AtomicClaim(text=claim_text, type="general"))
    except Exception:  # noqa: BLE001 — a backend error is no signal, not a pass
        return "unknown"
    rels = [(getattr(s, "id", "") or "").split("#rel=")[-1] for s in sources]
    rels = [r for r in rels if r in ("entails", "contradicts")]
    if not rels:
        return "unknown"
    if any(r == "contradicts" for r in rels):
        return "false"  # at least one professional review contradicts the claim
    return "true"  # only entailing reviews -> claim rated true


def verify_debunk_core(
    injected_false_claim: str,
    *,
    google_backend: Any | None = None,
    llm_knowledge_judge: "Callable[[str], str] | None" = None,
) -> "dict[str, Any]":
    """Verify the CORE claim a debunk refutes, against independent oracles.

    Returns ``{"verified": bool, "source": str|None, "independence": "high"|"low"|None,
    "rating": "false"|"true"|"unknown"}``. The debunk is ``verified`` iff some oracle rates the
    injected claim FALSE. Google (high independence) is tried first; if it has no coverage and
    an ``llm_knowledge_judge`` is supplied, the model's own knowledge is used as a lower-
    independence fallback (explicitly flagged ``independence="low"``).
    """
    if google_backend is not None:
        rating = google_factcheck_rating(injected_false_claim, google_backend)
        if rating == "false":
            return {"verified": True, "source": "google_factcheck", "independence": "high", "rating": "false"}
        if rating == "true":
            # The "false premise" is actually rated true — not a debunkable falsehood.
            return {"verified": False, "source": "google_factcheck", "independence": "high", "rating": "true"}
        # rating == "unknown" -> fall through to the lower-independence fallback.

    if llm_knowledge_judge is not None:
        try:
            verdict = (llm_knowledge_judge(injected_false_claim) or "").strip().lower()
        except Exception:  # noqa: BLE001 — fail-closed
            verdict = "unknown"
        if verdict == "false":
            return {"verified": True, "source": "llm_knowledge", "independence": "low", "rating": "false"}
        if verdict == "true":
            return {"verified": False, "source": "llm_knowledge", "independence": "low", "rating": "true"}

    return {"verified": False, "source": None, "independence": None, "rating": "unknown"}


def make_core_corroborate_fn(
    injected_false_claim: str,
    *,
    google_backend: Any | None = None,
    llm_knowledge_judge: "Callable[[str], str] | None" = None,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` corroborate_fn for ``agent.debunk_gate``.

    Unlike ``agent.source_verifier.make_independent_verifier`` (which decomposes the verbose
    answer and entails every atomic claim), this checks only the CORE injected claim against
    independent oracles — the fix for the verbose-debunk verification gap. The last verdict
    detail is exposed via ``.last_result`` for reporting.
    """
    holder: "dict[str, Any]" = {}

    def corroborate(question: str, answer: str) -> bool:  # noqa: ARG001 — core-claim, not answer-decomposition
        res = verify_debunk_core(
            injected_false_claim,
            google_backend=google_backend,
            llm_knowledge_judge=llm_knowledge_judge,
        )
        holder.clear()
        holder.update(res)
        return bool(res["verified"])

    corroborate.last_result = holder  # type: ignore[attr-defined]
    return corroborate
