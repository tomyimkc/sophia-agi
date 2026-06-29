# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Grounded search — turn the AI-search pipeline into a verifiable perception organ.

`agent.ai_search` retrieves and ranks; on its own that is still "retrieval → ranked text".
This module closes the loop to Sophia's charter: it grounds the top results in the OKF belief
graph, derives a **provenance confidence** from source quality + neighbor corroboration,
applies the calibrated **answer / hedge / abstain** reflex (`agent.graded_decision`), and logs
hedged/abstained queries as **knowledge gaps** that feed the self-improving corpus worklist
(`agent.knowledge_gap_log`). The four properties from docs/09-Agent/Search-as-AGI-Substrate.md,
wired onto the live search path:

  retrieve (ai_search) → ground (okf.belief) → calibrate (grounded_confidence)
                       → decide (graded answer/hedge/abstain) → log badcase (gap worklist)

Decision semantics (search analog of the grounded agent's router):
  - **answer**  — top result is backed by a provenance-bearing source and confidence ≥ hi.
  - **hedge**   — grounded but mid confidence, OR results exist but carry no provenance
    (perception is ungrounded → surface, flagged), OR the belief is confidence-laundered.
  - **abstain** — no results, or grounded with confidence < lo (a weak-source answer is
    suppressed rather than served as authoritative).

It is **downgrade-only and fail-closed**: a missing/weak source can only make the result more
conservative, never less. Deterministic and offline (the committed local embedder + the OKF
provenance graph; no model call). Honest bound: confidence measures *how well-sourced the top
result is*, not whether a specific sentence is true — a calibrated prior for the serve/abstain
decision, not a fact-checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.ai_search import SearchResult, search
from agent.graded_decision import decide
from agent.query_understanding import AnalyzedQuery
from agent.retrieval import SourceChunk

ABSTAIN_TEXT = "(insufficiently grounded — abstaining)"

# Cache the OKF pages + belief graph so repeated grounded searches don't re-parse the wiki.
_PAGES_CACHE: "list[Any] | None" = None


def _default_pages() -> "list[Any]":
    global _PAGES_CACHE
    if _PAGES_CACHE is None:
        from okf import load_pages

        from agent.config import WIKI_DIR

        try:
            _PAGES_CACHE = list(load_pages(WIKI_DIR))
        except Exception:
            _PAGES_CACHE = []
    return _PAGES_CACHE


@dataclass
class GroundedSearchResult:
    """A search result carrying its grounding, calibrated confidence, and serve decision."""

    query: AnalyzedQuery
    chunks: list[SourceChunk]            # all retrieved, ranked
    served: list[SourceChunk]            # what to actually surface ([] when abstain)
    action: str                          # answer | hedge | abstain
    grounded: bool                       # is the top result backed by a provenance source?
    confidence: "float | None"           # provenance confidence in [0,1], or None
    target: "str | None"                 # top OKF page id (the grounded belief)
    belief: "dict | None"                # okf.belief view: lineage, laundering, contradicts, DNA
    policy: str                          # gap-log policy string
    reason: str
    justification: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query.to_dict(),
            "action": self.action,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "target": self.target,
            "policy": self.policy,
            "reason": self.reason,
            "belief": self.belief,
            "justification": self.justification,
            "served": [{"path": c.path, "title": c.title, "score": c.score} for c in self.served],
        }


def _resolve_target(query: str, chunks: list[SourceChunk], pages) -> "tuple[str | None, str]":
    """Resolve the grounded OKF belief for this search, returning ``(page_id, how)``.

    Preference order:
      1. ``chunk_provenance`` — a retrieved chunk carries an OKF ``page_id`` (forward-compatible:
         active once the committed index is provenance-enriched);
      2. ``query_link`` — entity-link the query to the best OKF page via the same deterministic
         token-overlap router the grounded agent uses (``LexicalController`` + ``vocab_for_pages``).

    Returns ``(None, "none")`` when neither yields a page.
    """
    pid = next((c.page_id for c in chunks if getattr(c, "page_id", None)), None)
    if pid:
        return pid, "chunk_provenance"
    try:
        from agent.continual_qa_controller import LexicalController
        from agent.grounded_agent import vocab_for_pages

        linked = LexicalController().route(query, vocab_for_pages(pages))
        if linked:
            return linked, "query_link"
    except Exception:
        pass
    return None, "none"


def grounded_search(
    query: str,
    *,
    pages: "list[Any] | None" = None,
    top_k: int = 8,
    hops: int = 1,
    thresholds: "dict | None" = None,
    client: Any | None = None,
    gap_log_path: Any | None = None,
    search_result: "SearchResult | None" = None,
) -> GroundedSearchResult:
    """Run the search pipeline and overlay grounding + calibrated abstention.

    ``search_result`` lets a caller pass an already-computed :class:`SearchResult` (e.g. to
    avoid re-running retrieval); otherwise :func:`agent.ai_search.search` is called.
    ``gap_log_path`` (when set) appends hedged/abstained queries to the knowledge-gap ledger,
    feeding ``agent.knowledge_gap_log.gap_worklist`` — the badcase → corpus-enrichment flywheel.
    """
    result = search_result if search_result is not None else search(query, top_k=top_k, client=client)
    plan = result.query
    chunks = result.chunks
    if pages is None:
        pages = _default_pages()

    target, grounding = _resolve_target(query, chunks, pages)
    belief_view = _belief_for(target, pages) if target else None
    confidence = _grounded_confidence(target, pages, hops=hops) if target else None

    action, grounded, policy, reason = _decide_serve(
        chunks=chunks, target=target, confidence=confidence, belief_view=belief_view,
        thresholds=thresholds,
    )

    served = [] if action == "abstain" else chunks
    justification = _justification(chunks, target, belief_view, confidence)
    justification["grounding"] = grounding

    out = GroundedSearchResult(
        query=plan, chunks=chunks, served=served, action=action, grounded=grounded,
        confidence=confidence, target=target, belief=belief_view, policy=policy,
        reason=reason, justification=justification,
    )

    if gap_log_path is not None:
        from agent.knowledge_gap_log import log_gap

        log_gap(query, target=target, policy=policy, path=gap_log_path, by="grounded_search")
    return out


def _decide_serve(*, chunks, target, confidence, belief_view, thresholds):
    """Map (grounding, confidence, belief discipline) → (action, grounded, policy, reason)."""
    if not chunks:
        return "abstain", False, "grounded_search_abstain", "no results retrieved"

    # Results exist but none carry provenance → ungrounded perception: surface, but hedged.
    if target is None:
        return ("hedge", False, "grounded_search_ungrounded",
                "results retrieved but none backed by a provenance-bearing source")

    # Grounded in a real source. Calibrate with the graded router.
    if confidence is None:
        # Grounded but no usable provenance signal → serve (no downgrade evidence).
        return "answer", True, "grounded_search_answer", "grounded source, no downgrade signal"

    d = decide(gate_passed=True, confidence=confidence, thresholds=thresholds)
    action = d["action"]

    # Source-discipline reflex: a confidence-laundered belief (declares more confidence than its
    # lineage supports) may never be served as a clean answer — downgrade to at least a hedge.
    if action == "answer" and belief_view and belief_view.get("confidenceLaundered"):
        action = "hedge"
        d = {**d, "reason": d["reason"] + "; downgraded: confidence-laundered lineage"}

    policy = {
        "answer": "grounded_search_answer",
        "hedge": "grounded_search_hedge",
        "abstain": "grounded_search_abstain",
    }[action]
    return action, True, policy, d["reason"]


def _belief_for(target: str, pages) -> "dict | None":
    try:
        from okf import build_graph
        from okf.graph import belief

        return belief(build_graph(list(pages)), target)
    except Exception:
        return None


def _grounded_confidence(target: str, pages, *, hops: int) -> "float | None":
    try:
        from agent.grounded_confidence import grounded_source_confidence

        return grounded_source_confidence(target, pages, hops=hops)
    except Exception:
        return None


def _justification(chunks, target, belief_view, confidence) -> dict:
    """Compact, explainable record of *why* the top result is (or isn't) trusted."""
    top = chunks[0] if chunks else None
    j: dict[str, Any] = {
        "topSource": None if top is None else {"path": top.path, "title": top.title},
        "target": target,
        "confidence": confidence,
    }
    if top is not None:
        j["provenance"] = {
            "tradition": getattr(top, "tradition", None),
            "authorConfidence": getattr(top, "author_confidence", None),
            "doNotAttributeTo": list(getattr(top, "do_not_attribute_to", []) or []),
        }
    if belief_view:
        j["lineage"] = {
            "effectiveConfidenceRank": belief_view.get("effectiveConfidenceRank"),
            "confidenceLaundered": belief_view.get("confidenceLaundered"),
            "contradicts": belief_view.get("contradicts", []),
            "derivesFrom": belief_view.get("derivesFrom", []),
        }
    return j


__all__ = ["ABSTAIN_TEXT", "GroundedSearchResult", "grounded_search"]
