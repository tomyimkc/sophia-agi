# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Close the self-correction loop — materialize knowledge gaps into reviewable draft stubs.

`agent.grounded_search` / `agent.verified_search` log a knowledge gap whenever perception fails
(an ungrounded query, a weak-source hedge, a withheld answer). `agent.knowledge_gap_log`
ranks those into an enrichment worklist. This module takes the last step: it turns the
*missing-topic* gaps into **provenance-skeleton draft pages** so the corpus scaffolding grows
exactly where perception actually failed.

The charter constraint is absolute: **never fabricate provenance**. A generated stub therefore
carries *no claims* — only:
  - ``authorConfidence: none_extant`` (the weakest tier → the grounded router auto-abstains on
    it, so a stub can never be served as authoritative; it is fail-closed by its own confidence);
  - ``needsReview: true`` and ``provenance: knowledge_gap`` markers;
  - a body that records the queries that triggered the gap and an explicit "needs sourcing" note.

So the loop is: *gap → "I know this is a thing I have no source for" stub → (human/sourced fill)
→ canonical page → better grounding next time.* The stub makes the unknown a **known unknown**:
the router can now route to it and abstain, instead of silently missing it. Stubs are written to
the quarantined ``draft`` tier (`agent.wiki_store`, ``wiki/drafts/``) and pass the same hard
provenance gate as any agent write. Materialization is **opt-in** (dry-run by default); nothing
is written unless a caller asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words stripped when slugifying a query into a candidate page id (interrogative frame +
# attribution scaffolding). Kept small and explainable; a curator renames the draft anyway.
_STOP = frozenset({
    "who", "what", "when", "where", "which", "why", "how", "whose", "whom",
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "the", "a", "an",
    "of", "to", "by", "in", "on", "for", "about", "tell", "me", "us", "explain",
    "describe", "define", "list", "wrote", "write", "writes", "written", "author",
    "authored", "compose", "composed", "and", "or",
})


@dataclass
class IngestItem:
    """A planned draft stub for one missing topic."""

    page_id: str
    queries: list[str]
    hits: int
    reason: str = "missing-topic gap (ungrounded or unrouted)"


@dataclass
class EnrichItem:
    """An existing (thin/weak) page a gap points at — flagged for enrichment, not auto-created."""

    target: str
    hits: int


@dataclass
class IngestPlan:
    create: list[IngestItem] = field(default_factory=list)
    enrich: list[EnrichItem] = field(default_factory=list)


def candidate_id(query: str) -> str:
    """Slugify a query into a candidate snake_case page id, stripping the interrogative frame.

    "Who wrote the Dao De Jing?" -> "dao_de_jing". Falls back to a stable token of the raw
    query when nothing survives (e.g. a non-Latin query), so an id is always produced.
    """
    toks = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if t not in _STOP]
    slug = "_".join(toks)[:60].strip("_")
    if slug:
        return slug
    raw = re.sub(r"[^a-z0-9]+", "_", (query or "").lower()).strip("_")
    return raw[:60] or "untitled_gap"


def draft_stub(page_id: str, *, queries: "list[str]", gap_hits: int) -> "tuple[dict, str]":
    """Build the (meta, body) for a provenance-skeleton draft stub. No claims, no attribution."""
    meta = {
        "id": page_id,
        "pageType": "concept",
        "authorConfidence": "none_extant",   # weakest tier → router auto-abstains; fail-closed
        "needsReview": True,
        "provenance": "knowledge_gap",
        "gapHits": int(gap_hits),
        "sources": [],                        # no source yet — that is the whole point
        "doNotAttributeTo": [],
    }
    uniq_q = list(dict.fromkeys(q for q in queries if q))
    lines = [
        f"# {page_id.replace('_', ' ')}",
        "",
        "_Auto-generated knowledge-gap stub — Sophia was queried about this topic but had no "
        "grounded source. **No claims are asserted here.** This page exists so the topic is a "
        "known unknown (routable and abstained) until a real source is ingested and reviewed._",
        "",
        f"- Status: **draft, needs sourcing** (authorConfidence `none_extant`, queried {gap_hits}×)",
        "",
        "## Queries that triggered this gap",
    ]
    lines += [f"- {q}" for q in uniq_q] or ["- (none recorded)"]
    return meta, "\n".join(lines) + "\n"


def plan_ingestion(gaps: "list[dict]", *, existing_ids: "set[str]", min_hits: int = 1) -> IngestPlan:
    """Decide which gaps become NEW draft stubs vs flag an EXISTING page for enrichment.

    A gap whose ``target`` is a live page id → enrichment candidate (the page exists but came up
    short). Any other gap (no target, or a target with no live page) → a missing-topic stub,
    keyed by ``target`` or a slug of the query. Ids already live are skipped (idempotent).
    """
    create: "dict[str, IngestItem]" = {}
    enrich: "dict[str, int]" = {}
    for g in gaps:
        target = g.get("target")
        query = g.get("query") or ""
        if target and target in existing_ids:
            enrich[target] = enrich.get(target, 0) + 1
            continue
        pid = target if target else candidate_id(query)
        if pid in existing_ids:
            # A live page already covers this — treat as enrichment, not creation.
            enrich[pid] = enrich.get(pid, 0) + 1
            continue
        item = create.get(pid)
        if item is None:
            item = create[pid] = IngestItem(page_id=pid, queries=[], hits=0)
        if query and query not in item.queries:
            item.queries.append(query)
        item.hits += 1

    create_list = [it for it in create.values() if it.hits >= min_hits]
    create_list.sort(key=lambda it: it.hits, reverse=True)
    enrich_list = [EnrichItem(target=t, hits=h) for t, h in enrich.items()]
    enrich_list.sort(key=lambda e: e.hits, reverse=True)
    return IngestPlan(create=create_list, enrich=enrich_list)


def materialize(plan: IngestPlan, *, write: bool = False, tier: str = "draft") -> dict:
    """Build (and optionally write) draft stubs for the plan's create list. Fail-closed.

    With ``write=False`` (default) nothing is written — the report lists what *would* be created.
    With ``write=True`` each stub is gated + written via `agent.wiki_store.upsert` (draft tier);
    a stub that fails the provenance gate is reported as ``rejected``, never forced.
    """
    created, would_create, rejected = [], [], []
    for item in plan.create:
        meta, body = draft_stub(item.page_id, queries=item.queries, gap_hits=item.hits)
        if not write:
            from agent.wiki_store import gate

            ok, reasons = gate({**meta, "pageType": meta["pageType"]}, body)
            (would_create if ok else rejected).append(
                {"id": item.page_id, "hits": item.hits, "ok": ok, "reasons": reasons})
            continue
        from agent.wiki_store import upsert

        result = upsert(item.page_id, meta=meta, body=body, tier=tier)
        if result.get("ok"):
            created.append({"id": item.page_id, "path": result.get("path"), "hits": item.hits})
        else:
            rejected.append({"id": item.page_id, "reasons": result.get("reasons") or [result.get("error")]})

    return {
        "schema": "sophia.gap_ingest.report.v1",
        "candidateOnly": True,
        "wrote": bool(write),
        "tier": tier,
        "created": created,
        "wouldCreate": would_create,
        "rejected": rejected,
        "enrichTargets": [{"target": e.target, "hits": e.hits} for e in plan.enrich],
    }


def live_page_ids(pages=None) -> "set[str]":
    """Ids of all currently-loaded OKF pages (canonical + memory + draft tiers)."""
    if pages is None:
        from okf import load_pages

        from agent.config import WIKI_DIR

        pages = load_pages(WIKI_DIR)
    return {p.id for p in pages}


__all__ = [
    "EnrichItem", "IngestItem", "IngestPlan", "candidate_id", "draft_stub",
    "live_page_ids", "materialize", "plan_ingestion",
]
