# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Layered core-claim verification — try independent oracles in independence order.

This generalizes ``agent.core_claim_verifier.verify_debunk_core`` (Google ClaimReview
then an LLM-knowledge fallback) by inserting a PROVENANCE layer between them. The
provenance layer reads structured authorship/scholarly records (Wikidata + Crossref via
``agent.live_sources.LiveFactBackend``) so claims that cite a study or attribution Google
does not review — e.g. "A 2023 Yale study identified Anthony Ascham as the Voynich author" —
can still be contradicted by the structured record (the attribution is unsupported).

Independence ladder (highest first):
  1. ``google_factcheck``               — professional ClaimReview verdicts (high indep).
  2. ``provenance_wikidata_crossref``   — structured authorship/scholarly records (high indep).
  3. ``llm_knowledge``                  — the model's own world knowledge (LOW indep, flagged).

The verifier tries each layer in order and returns the FIRST decisive verdict. A debunk is
``verified`` iff some oracle rates the core claim FALSE (an independent source contradicts the
claim the debunk refutes). A "true" rating short-circuits to ``verified=False`` (the premise is
rated true — not a debunkable falsehood). Every layer consulted is recorded in ``layers_tried``.

Fail-closed: no decisive coverage anywhere -> ``verified=False`` (never a guessed pass).
"""
from __future__ import annotations

from typing import Any, Callable

from agent.core_claim_verifier import google_factcheck_rating

__all__ = [
    "provenance_rating",
    "layered_verify_core",
    "make_layered_corroborate_fn",
]


def provenance_rating(claim_text: str, live_backend: Any) -> str:
    """Return ``"false" | "true" | "unknown"`` for ``claim_text`` from structured provenance.

    ``live_backend`` is a ``LiveFactBackend`` (or anything with the same ``retriever`` /
    ``entailment`` contract over Wikidata authorship + Crossref scholarly records). Build an
    AtomicClaim, retrieve sources, and read each source's entailment: if any source CONTRADICTS
    the claim -> ``"false"``; if sources only ENTAIL it -> ``"true"``; otherwise ``"unknown"``.
    Fail-closed to ``"unknown"`` on any error or no coverage.
    """
    try:
        from agent.fact_check_gate import AtomicClaim  # noqa: PLC0415
        claim = AtomicClaim(text=claim_text, type="general")
        sources = live_backend.retriever(claim)
        rels = [live_backend.entailment(claim, s) for s in sources]
    except Exception:  # noqa: BLE001 — a backend error is no signal, not a pass
        return "unknown"
    rels = [r for r in rels if r in ("entails", "contradicts")]
    if not rels:
        return "unknown"
    if any(r == "contradicts" for r in rels):
        return "false"  # a structured record contradicts the cited attribution
    return "true"  # only entailing records -> claim rated true


def layered_verify_core(
    claim_text: str,
    *,
    google_backend: Any | None = None,
    live_backend: Any | None = None,
    llm_knowledge_judge: "Callable[[str], str] | None" = None,
) -> "dict[str, Any]":
    """Verify the CORE claim against independent oracles, in independence order.

    Returns ``{"verified": bool, "source": str|None, "independence": "high"|"low"|None,
    "rating": "false"|"true"|"unknown", "layers_tried": [...]}``. Layers are tried in order
    (Google, provenance Wikidata/Crossref, llm_knowledge) and the FIRST decisive verdict is
    returned. A claim is ``verified`` iff some oracle rates it FALSE; a "true" rating short-
    circuits to ``verified=False`` (not a debunkable falsehood). Every layer consulted is
    recorded in ``layers_tried``. Fail-closed: no decisive coverage -> ``verified=False``.
    """
    layers_tried: "list[str]" = []

    if google_backend is not None:
        layers_tried.append("google_factcheck")
        rating = google_factcheck_rating(claim_text, google_backend)
        if rating == "false":
            return {"verified": True, "source": "google_factcheck", "independence": "high",
                    "rating": "false", "layers_tried": layers_tried}
        if rating == "true":
            # The "false premise" is actually rated true — not a debunkable falsehood.
            return {"verified": False, "source": "google_factcheck", "independence": "high",
                    "rating": "true", "layers_tried": layers_tried}
        # rating == "unknown" -> escalate to the next, equally-independent layer.

    if live_backend is not None:
        layers_tried.append("provenance_wikidata_crossref")
        rating = provenance_rating(claim_text, live_backend)
        if rating == "false":
            return {"verified": True, "source": "provenance_wikidata_crossref", "independence": "high",
                    "rating": "false", "layers_tried": layers_tried}
        if rating == "true":
            return {"verified": False, "source": "provenance_wikidata_crossref", "independence": "high",
                    "rating": "true", "layers_tried": layers_tried}
        # rating == "unknown" -> fall through to the lower-independence fallback.

    if llm_knowledge_judge is not None:
        layers_tried.append("llm_knowledge")
        try:
            verdict = (llm_knowledge_judge(claim_text) or "").strip().lower()
        except Exception:  # noqa: BLE001 — fail-closed
            verdict = "unknown"
        if verdict == "false":
            return {"verified": True, "source": "llm_knowledge", "independence": "low",
                    "rating": "false", "layers_tried": layers_tried}
        if verdict == "true":
            return {"verified": False, "source": "llm_knowledge", "independence": "low",
                    "rating": "true", "layers_tried": layers_tried}

    return {"verified": False, "source": None, "independence": None,
            "rating": "unknown", "layers_tried": layers_tried}


def make_layered_corroborate_fn(
    claim_text: str,
    *,
    google_backend: Any | None = None,
    live_backend: Any | None = None,
    llm_knowledge_judge: "Callable[[str], str] | None" = None,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` corroborate_fn over the layered oracle ladder.

    Mirrors ``agent.core_claim_verifier.make_core_corroborate_fn``: it closes over the CORE
    ``claim_text``, ignores the verbose answer, and checks the claim against the layered
    oracles (Google -> provenance -> llm_knowledge). The last verdict detail (including
    ``layers_tried``) is exposed via ``.last_result`` for reporting.
    """
    holder: "dict[str, Any]" = {}

    def corroborate(question: str, answer: str) -> bool:  # noqa: ARG001 — core-claim, not answer-decomposition
        res = layered_verify_core(
            claim_text,
            google_backend=google_backend,
            live_backend=live_backend,
            llm_knowledge_judge=llm_knowledge_judge,
        )
        holder.clear()
        holder.update(res)
        return bool(res["verified"])

    corroborate.last_result = holder  # type: ignore[attr-defined]
    return corroborate
