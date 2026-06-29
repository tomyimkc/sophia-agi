# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Provenance-tainted event/entity extraction + provenance-aware multi-hop recall.

This is the OKF answer to graph-RAG retrieval engines (e.g. Zleap-AI/SAG): decompose
wiki bodies into ``(event, entities)`` units and expand recall by traversing shared
entities across pages — but, unlike a recall-only engine, **carry provenance through
every hop**. Each extracted unit inherits its page's effective (min-over-derivesFrom)
confidence rank, and a multi-hop recall path is floored by the *weakest* page it
touches. A 3-hop answer that passes through a ``legendary`` or ``anachronism_risk``
page cannot be laundered into a confident result — it is surfaced with a low
``provenanceFloor`` and a ``capped`` flag.

House rules (matching the rest of ``okf/``): dependency-free, deterministic, offline,
CPU-only. The extractor here is intentionally a transparent lexical splitter, not an
LLM — the point of the spike is the *provenance plumbing*, which is identical whatever
fills the extraction slot. Swap :func:`extract_events` for an LLM event extractor and
the recall / confidence-floor machinery below is unchanged.

Public API:
    extract_events(pages)        -> list[EventUnit]
    build_entity_index(events)   -> dict[entity_slug, list[event_id]]
    multi_hop_recall(query, ...) -> list[RecallHit]
    is_capped(floor)             -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from okf import graph as okf_graph
from okf import wikilinks
from okf.schema import CONFIDENCE_RANK, confidence_rank

# A multi-hop path whose weakest provenance is at or below this rank rests on weak
# ground (legendary / disputed / none_extant / anachronism_risk). Mirrors the
# CONFIDENCE_RANK ladder in okf.schema so a drift there is caught by the tests.
CAPPED_RANK = 1

# Frontmatter edge keys folded into a page's entities (the typed association/lineage
# graph). NEGATIVE signals (doNotAttributeTo / doNotMergeWith) are deliberately excluded:
# they assert "must not associate", so indexing them for recall would be exactly backwards.
_POSITIVE_EDGE_KEYS = okf_graph.LINK_EDGE_KEYS

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9]+")
# Function words stripped from QUERY tokens only (never from indexed text/entities) so a
# query like "what later thinkers extend the teachings of X" does not score every page
# that merely contains "the"/"of". Deliberately small and domain-neutral.
_STOP = frozenset(
    "a an and are as at be by for from in into is it its of on or that the their "
    "these this to was were what which who whom whose with about over later".split()
)
# Lines we never treat as an event sentence: markdown headings, list bullets that are
# pure metadata, and the generated provenance footer.
_SKIP_LINE = re.compile(r"^\s*(#|_Provenance frontmatter|<!--)")


@dataclass(frozen=True)
class EventUnit:
    """One semantic unit extracted from a page, stamped with its provenance taint."""

    id: str  # f"{page_id}::e{n}"
    page_id: str
    text: str
    entities: tuple = ()  # normalized entity slugs (resolved to page ids where possible)
    author_confidence: "str | None" = None
    confidence_rank: int = 0  # EFFECTIVE rank (min over derivesFrom chain), not face value
    tradition: "str | None" = None

    @property
    def capped(self) -> bool:
        return self.confidence_rank <= CAPPED_RANK


@dataclass(frozen=True)
class RecallHit:
    """An event surfaced by recall, with the provenance floor of the path that found it."""

    event: EventUnit
    score: float
    hops: int  # 0 == direct lexical match; >=1 == reached via entity expansion
    provenance_floor: int  # min effective rank over every page on the surfacing path
    path: tuple = ()  # entity slugs traversed, seed -> ... -> reaching entity

    @property
    def capped(self) -> bool:
        """True when the whole multi-hop chain rests on weak provenance."""
        return self.provenance_floor <= CAPPED_RANK


def is_capped(floor: int) -> bool:
    """A provenance floor at or below CAPPED_RANK cannot back a confident claim."""
    return floor <= CAPPED_RANK


def _tokens(text: str) -> "set[str]":
    return set(_TOKEN.findall((text or "").lower()))


def _query_tokens(text: str) -> "set[str]":
    """Query tokens with function words removed (see _STOP)."""
    return _tokens(text) - _STOP


def _page_edge_entities(page) -> "list[str]":
    """Frontmatter-derived entities for a page: typed forward edges + author + tradition.

    Uses *frontmatter* edges only (``edge_targets``), never body links — body links are
    handled per-sentence. The real wiki authors its graph in ``links:``/typed keys rather
    than inline ``[[wikilinks]]``, so without this a frontmatter-linked corpus would
    collapse to disconnected page-level units and lose all multi-hop reach. Negative
    signals (``doNotAttributeTo``/``doNotMergeWith``) are never included.
    """
    ents: list[str] = []

    def _add(slug) -> None:
        s = wikilinks.normalize_target(str(slug))
        if s and s != page.id and s not in ents:
            ents.append(s)

    for key in _POSITIVE_EDGE_KEYS:
        for target in page.edge_targets(key):
            _add(target)
    if page.meta.get("attributedAuthor"):
        _add(page.meta["attributedAuthor"])
    if page.meta.get("tradition"):
        _add(page.meta["tradition"])
    return ents


def _sentences(body: str) -> "list[str]":
    out: list[str] = []
    for raw in _SENTENCE_SPLIT.split(body or ""):
        line = raw.strip()
        if not line or _SKIP_LINE.match(line):
            continue
        out.append(line)
    return out


def extract_events(pages, graph: "okf_graph.Graph | None" = None) -> "list[EventUnit]":
    """Decompose each page body into event units carrying the page's effective rank.

    Every sentence with at least one ``[[wikilink]]`` becomes an event unit whose
    entities are the resolved link targets plus the page's own id. A page with no
    linked sentences still yields one page-level unit (entities = the page itself) so
    nothing becomes unretrievable. The unit's ``confidence_rank`` is the *effective*
    rank from :func:`okf.graph.propagate_confidence` — i.e. already floored by the
    page's weakest ``derivesFrom`` source, so the taint is correct before recall ever
    starts.
    """
    pages = list(pages)
    if graph is None:
        graph = okf_graph.build(pages)
    effective = okf_graph.propagate_confidence(graph)

    events: list[EventUnit] = []
    for page in pages:
        pid = page.id
        rank = effective.get(pid, confidence_rank(page.meta.get("authorConfidence")))
        ac = page.meta.get("authorConfidence")
        trad = page.meta.get("tradition")

        def _mk(n: int, text: str, ents: "list[str]") -> EventUnit:
            resolved = []
            for e in ents:
                rid = okf_graph.resolve(graph, e) or e
                if rid not in resolved:
                    resolved.append(rid)
            return EventUnit(
                id=f"{pid}::e{n}",
                page_id=pid,
                text=text,
                entities=tuple(resolved),
                author_confidence=ac,
                confidence_rank=rank,
                tradition=trad,
            )

        n = 0
        for sentence in _sentences(page.body):
            links = wikilinks.extract_links(sentence)
            if not links:
                continue
            events.append(_mk(n, sentence, [pid] + links))
            n += 1

        # Spine unit carrying the page's frontmatter edges. Emitted when the page has no
        # body-linked sentence (so it stays recallable) OR when its frontmatter contributes
        # entities beyond the page itself (so the typed graph — author, tradition, links —
        # is indexed even on pages that also have body-linked sentences).
        fm = _page_edge_entities(page)
        if n == 0 or fm:
            paras = _sentences(page.body)
            text = paras[0] if paras else (page.meta.get("canonicalTitleEn") or pid)
            events.append(_mk(n, text, [pid] + fm))

    return events


def build_entity_index(events) -> "dict[str, list[str]]":
    """entity slug -> list of event ids that mention it (SAG's entity index, OKF-side)."""
    index: dict[str, list[str]] = {}
    for ev in events:
        for ent in ev.entities:
            index.setdefault(ent, []).append(ev.id)
    return index


def _lexical_score(query_tokens: "set[str]", ev: EventUnit) -> float:
    """Token-overlap score of a query against an event's text + entity slugs (offline)."""
    if not query_tokens:
        return 0.0
    hay = _tokens(ev.text) | {t for ent in ev.entities for t in _tokens(ent)}
    overlap = len(query_tokens & hay)
    return overlap / (len(query_tokens) ** 0.5) if overlap else 0.0


def multi_hop_recall(
    query: str,
    events,
    *,
    index: "dict[str, list[str]] | None" = None,
    max_hops: int = 2,
    top_k: int = 8,
    hop_decay: float = 0.5,
) -> "list[RecallHit]":
    """Recall events for a query, expanding through shared entities, flooring provenance.

    Stage 0 (direct): lexical match of the query against every event.
    Stage 1..max_hops (expansion): from the entities of the matched events, pull
    co-occurring events out of the entity index. Each expansion hop multiplies the
    score by ``hop_decay`` and — crucially — recomputes ``provenance_floor`` as the
    ``min`` of the path so far and the newly reached event's effective rank. The
    weakest page on the surfacing chain therefore dominates: a confident final event
    reached *through* a legendary bridge is reported with a low floor and
    ``hit.capped is True``. This is the property a recall-only engine cannot offer.
    """
    events = list(events)
    if index is None:
        index = build_entity_index(events)
    by_id = {ev.id: ev for ev in events}

    # Inverse-frequency weight per bridge entity: a hub like a tradition (mentioned by
    # many pages) is a weak bridge and must not flood top-k with its siblings; a rare
    # shared entity is a strong, specific bridge. df = distinct pages mentioning the
    # entity; bonus = 1/df in (0, 1]. This is what stops graph expansion from drowning
    # direct lexical hits on an over-connected corpus.
    def _bridge_bonus(ent: str) -> float:
        pages_with = {eid.split("::")[0] for eid in index.get(ent, ())}
        df = len(pages_with) or 1
        return 1.0 / df

    qtokens = _query_tokens(query)
    # Direct matches seed the frontier.
    best: dict[str, RecallHit] = {}
    frontier: dict[str, tuple] = {}  # entity slug -> (floor_so_far, path_so_far)

    for ev in events:
        s = _lexical_score(qtokens, ev)
        if s <= 0:
            continue
        hit = RecallHit(event=ev, score=s, hops=0, provenance_floor=ev.confidence_rank,
                        path=(ev.page_id,))
        best[ev.id] = hit
        for ent in ev.entities:
            floor, path = frontier.get(ent, (ev.confidence_rank, (ev.page_id,)))
            # keep the *strongest* (max-floor) way to reach this seed entity
            if ev.confidence_rank >= floor or ent not in frontier:
                frontier[ent] = (max(floor, ev.confidence_rank) if ent in frontier else ev.confidence_rank,
                                 path if ent in frontier else (ev.page_id,))

    # Expansion hops.
    for hop in range(1, max_hops + 1):
        next_frontier: dict[str, tuple] = {}
        for ent, (floor, path) in frontier.items():
            for eid in index.get(ent, ()):  # events that also mention this entity
                ev = by_id[eid]
                new_floor = min(floor, ev.confidence_rank)  # <-- provenance can only weaken
                new_path = path + (ent,)
                base = _lexical_score(qtokens, ev)
                # connectivity reward, scaled by how *specific* the bridge entity is
                # (1/df). Hub bridges add ~0; rare bridges add up to 1. Then hop-decayed.
                score = (base + _bridge_bonus(ent)) * (hop_decay ** hop)
                prev = best.get(eid)
                # Never downgrade a direct (hop-0) lexical match to an expansion hop, even
                # if the expansion score is nominally higher — a direct hit keeps its own
                # provenance floor and shortest path.
                if prev is None or (prev.hops > 0 and score > prev.score):
                    best[eid] = RecallHit(event=ev, score=score, hops=hop,
                                          provenance_floor=new_floor, path=new_path)
                for nent in ev.entities:
                    cur = next_frontier.get(nent)
                    if cur is None or new_floor > cur[0]:
                        next_frontier[nent] = (new_floor, new_path)
        frontier = next_frontier

    hits = sorted(best.values(), key=lambda h: (h.score, -h.hops, h.event.id), reverse=True)
    return hits[:top_k]
