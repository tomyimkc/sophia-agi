# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Belief graph over OKF pages: edge resolution, contradiction detection, and
min-over-chain confidence propagation (provenance taint-tracking).

This is the executable core of Sophia's thesis: instead of provenance living as
inert prose, the pages form a typed graph the system can reason over — detecting
lineage merges, supersession cycles, and confidence laundering structurally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from okf import wikilinks
from okf.schema import as_list, confidence_rank

# Frontmatter edge keys that resolve to other page ids.
LINK_EDGE_KEYS = ("links", "contradicts", "supersedes", "supersededBy", "derivesFrom")


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)  # id -> {"id","pageType","meta","page"}
    alias_index: dict = field(default_factory=dict)  # alias slug -> id


def build(pages) -> Graph:
    graph = Graph()
    for page in pages:
        nid = page.id
        graph.nodes[nid] = {"id": nid, "pageType": page.page_type, "meta": page.meta, "page": page}
        for alias in page.aliases:
            graph.alias_index.setdefault(alias, nid)
    return graph


def resolve(graph: Graph, target: str):
    """Resolve a link target to a node id, via id then alias, else None."""
    slug = wikilinks.normalize_target(target)
    if slug in graph.nodes:
        return slug
    return graph.alias_index.get(slug)


def _edge_targets(node: dict, key: str) -> "list[str]":
    return [wikilinks.normalize_target(t) for t in as_list(node["meta"].get(key))]


def out_link_targets(node: dict) -> "list[str]":
    page = node["page"]
    seen = list(page.body_links())
    for t in _edge_targets(node, "links"):
        if t not in seen:
            seen.append(t)
    return seen


def dangling_links(graph: Graph) -> "list[dict]":
    """Forward links (body + edges) that resolve to no page."""
    out: list[dict] = []
    for nid, node in graph.nodes.items():
        targets = set(out_link_targets(node))
        for key in LINK_EDGE_KEYS:
            targets.update(_edge_targets(node, key))
        for target in sorted(targets):
            if not target:
                continue
            if resolve(graph, target) is None:
                out.append({"page": nid, "target": target})
    return out


def supersede_cycles(graph: Graph) -> "list[list[str]]":
    """Cycles in the supersedes relation (X supersedes Y supersedes X)."""
    cycles: list[list[str]] = []
    seen_pairs = set()

    def walk(start: str, current: str, path: "list[str]") -> None:
        for raw in as_list(graph.nodes[current]["meta"].get("supersedes")):
            nxt = resolve(graph, raw)
            if nxt is None:
                continue
            if nxt == start and len(path) >= 1:
                cyc = tuple(sorted(path + [nxt]))
                if cyc not in seen_pairs:
                    seen_pairs.add(cyc)
                    cycles.append(path + [nxt])
            elif nxt not in path:
                walk(start, nxt, path + [nxt])

    for nid in graph.nodes:
        walk(nid, nid, [nid])
    return cycles


def self_merges(graph: Graph) -> "list[dict]":
    """Pages whose attributedAuthor appears in their own doNotAttributeTo — the
    canonical lineage-merge, self-inconsistent at the data level."""
    out: list[dict] = []
    for nid, node in graph.nodes.items():
        meta = node["meta"]
        author = meta.get("attributedAuthor")
        if not author:
            continue
        forbidden = {str(a).lower() for a in as_list(meta.get("doNotAttributeTo"))}
        if str(author).lower() in forbidden:
            out.append({"page": nid, "author": author})
    return out


def tradition_merges(graph: Graph, *, dnm_by_tradition: "dict | None" = None) -> "list[dict]":
    """Two linked pages whose traditions are mutually do-not-merge.

    `dnm_by_tradition` maps a tradition id to the set of traditions it must not be
    merged with (typically loaded from data/traditions.json doNotMergeWith).
    """
    dnm = {k: {str(x).lower() for x in v} for k, v in (dnm_by_tradition or {}).items()}
    if not dnm:
        return []
    out: list[dict] = []
    seen = set()
    for nid, node in graph.nodes.items():
        tradition = node["meta"].get("tradition")
        if not tradition:
            continue
        forbidden = dnm.get(str(tradition).lower(), set())
        if not forbidden:
            continue
        for target in out_link_targets(node):
            other_id = resolve(graph, target)
            if not other_id:
                continue
            other_trad = graph.nodes[other_id]["meta"].get("tradition")
            if other_trad and str(other_trad).lower() in forbidden:
                pair = tuple(sorted([nid, other_id]))
                if pair not in seen:
                    seen.add(pair)
                    out.append({"page": nid, "linksTo": other_id, "tradition": tradition, "otherTradition": other_trad})
    return out


def propagate_confidence(graph: Graph) -> "dict":
    """Effective confidence rank = min over the node and its derivesFrom chain."""
    memo: dict = {}

    def eff(nid: str, stack: "set") -> int:
        if nid in memo:
            return memo[nid]
        node = graph.nodes.get(nid)
        if node is None:
            return 0
        best = confidence_rank(node["meta"].get("authorConfidence"))
        for raw in as_list(node["meta"].get("derivesFrom")):
            dep = resolve(graph, raw)
            if dep is None or dep in stack:
                continue
            best = min(best, eff(dep, stack | {nid}))
        memo[nid] = best
        return best

    return {nid: eff(nid, set()) for nid in graph.nodes}


def belief(graph: Graph, entity: str) -> "dict":
    """Belief record for one entity (page id, alias, or [[wikilink]] target).

    Exposes ``effectiveConfidenceRank`` — the min-over-derivesFrom-chain rank, so a
    claim that declares high confidence while resting on a weak source is flagged
    (``confidenceLaundered``) rather than read at face value — alongside the
    attribution/provenance fields a caller needs to respect source discipline.
    """
    nid = resolve(graph, entity)
    if nid is None:
        return {"found": False, "entity": entity, "id": None}
    node = graph.nodes[nid]
    meta = node["meta"]
    own = meta.get("authorConfidence")
    own_rank = confidence_rank(own)
    effective = propagate_confidence(graph).get(nid, own_rank)
    derives = as_list(meta.get("derivesFrom"))
    return {
        "found": True,
        "entity": entity,
        "id": nid,
        "pageType": node["pageType"],
        "attributedAuthor": meta.get("attributedAuthor"),
        "doNotAttributeTo": [str(a) for a in as_list(meta.get("doNotAttributeTo"))],
        "tradition": meta.get("tradition"),
        "authorConfidence": own,
        "confidenceRank": own_rank,
        "effectiveConfidenceRank": effective,
        "confidenceLaundered": bool(own and derives and own_rank > effective),
        "derivesFrom": [resolve(graph, t) or t for t in derives],
        "contradicts": [resolve(graph, t) or t for t in as_list(meta.get("contradicts"))],
        "supersededBy": [resolve(graph, t) or t for t in as_list(meta.get("supersededBy"))],
    }


def confidence_laundering(graph: Graph) -> "list[dict]":
    """Nodes asserting more confidence than their weakest provenance dependency."""
    effective = propagate_confidence(graph)
    out: list[dict] = []
    for nid, node in graph.nodes.items():
        own = node["meta"].get("authorConfidence")
        if not own or not as_list(node["meta"].get("derivesFrom")):
            continue
        if confidence_rank(own) > effective[nid]:
            out.append({"page": nid, "claims": own, "effectiveRank": effective[nid]})
    return out


def contradiction_ledger(graph: Graph, *, dnm_by_tradition: "dict | None" = None) -> "dict":
    """Aggregate structural contradictions into one machine-readable ledger."""
    declared = []
    for nid, node in graph.nodes.items():
        for raw in as_list(node["meta"].get("contradicts")):
            other = resolve(graph, raw)
            declared.append({"page": nid, "contradicts": other or raw, "resolved": other is not None})
    return {
        "declaredContradictions": declared,
        "selfMerges": self_merges(graph),
        "traditionMerges": tradition_merges(graph, dnm_by_tradition=dnm_by_tradition),
        "supersedeCycles": supersede_cycles(graph),
        "confidenceLaundering": confidence_laundering(graph),
    }
