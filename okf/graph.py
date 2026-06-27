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

# Frontmatter edge keys that resolve to other page ids. The last three are the
# concept-TBox edges (subsumption, disjointness, scoped cross-tradition analogy);
# see docs/11-Platform/Ontology-Claim-Boundary.md. They are structural-consistency
# edges, NOT truth claims — a derived violation is about the closed world of axioms
# we wrote down, each of which is itself an untrusted, sourced claim.
LINK_EDGE_KEYS = (
    "links", "contradicts", "supersedes", "supersededBy", "derivesFrom",
    "subClassOf", "disjointWith", "scopedAnalogy",
)


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


def subclass_cycles(graph: Graph) -> "list[list[str]]":
    """Cycles in the subClassOf relation (X ⊑ Y ⊑ X) — a TBox inconsistency.

    Mirrors :func:`supersede_cycles`; a subsumption cycle makes every class in it
    mutually equivalent, which the closed-world reasoner must never silently admit.
    """
    cycles: list[list[str]] = []
    seen_pairs = set()

    def walk(start: str, current: str, path: "list[str]") -> None:
        for raw in as_list(graph.nodes[current]["meta"].get("subClassOf")):
            nxt = resolve(graph, raw)
            if nxt is None:
                continue
            if nxt == start and len(path) >= 2:
                # canonical key by node set (order-independent, ignores traversal start/duplicate close)
                cycle_nodes = path[:]
                cyc_key = frozenset(cycle_nodes)
                if cyc_key not in seen_pairs:
                    seen_pairs.add(cyc_key)
                    cycles.append(cycle_nodes)
            elif nxt not in path:
                walk(start, nxt, path + [nxt])

    for nid in graph.nodes:
        walk(nid, nid, [nid])
    return cycles


def _subclass_ancestors(graph: Graph, nid: str) -> "set[str]":
    """Transitive subClassOf ancestors of a node (excluding itself)."""
    out: set[str] = set()
    stack = [nid]
    while stack:
        cur = stack.pop()
        for raw in as_list(graph.nodes.get(cur, {}).get("meta", {}).get("subClassOf")):
            anc = resolve(graph, raw)
            if anc and anc not in out and anc != nid:
                out.add(anc)
                stack.append(anc)
    return out


def disjointness_violations(graph: Graph) -> "list[dict]":
    """Nodes that are (transitively) subClassOf two classes declared disjointWith.

    ``disjointWith`` is read symmetrically: if C1 declares disjointWith C2 (or vice
    versa) and a node X is subClassOf both, X cannot consistently exist — the
    canonical TBox contradiction (Datalog ``violation`` rule). Also flags a direct
    ``X subClassOf Y`` where X and Y are mutually disjoint.
    """
    # Symmetric disjointness relation over declared edges.
    disjoint: dict[str, set] = {}
    for nid, node in graph.nodes.items():
        for raw in as_list(node["meta"].get("disjointWith")):
            other = resolve(graph, raw)
            if other:
                disjoint.setdefault(nid, set()).add(other)
                disjoint.setdefault(other, set()).add(nid)

    out: list[dict] = []
    seen = set()
    for nid in graph.nodes:
        ancestors = _subclass_ancestors(graph, nid) | {nid}
        for a in ancestors:
            for b in disjoint.get(a, set()):
                if b in ancestors and a != b:
                    pair = tuple(sorted([a, b]))
                    key = (nid, pair)
                    if key not in seen:
                        seen.add(key)
                        out.append({"page": nid, "disjointClasses": list(pair)})
    return out


def unsupported_ontology_edges(graph: Graph) -> "list[dict]":
    """Concept-TBox edges (subClassOf/disjointWith/scopedAnalogy) on a node whose
    effective provenance is weak/absent — an axiom resting on nothing.

    Reuses :func:`propagate_confidence` (min-over-derivesFrom) and the page's
    ``sources``: an ontology edge whose home page has effective rank 0 AND no
    sources is flagged, so a TBox edge cannot be laundered into the graph without
    provenance (the claim-boundary rule: every axiom needs a source).
    """
    _ONTO_KEYS = ("subClassOf", "disjointWith", "scopedAnalogy")
    effective = propagate_confidence(graph)
    out: list[dict] = []
    for nid, node in graph.nodes.items():
        meta = node["meta"]
        edges = [k for k in _ONTO_KEYS if as_list(meta.get(k))]
        if not edges:
            continue
        has_sources = bool(as_list(meta.get("sources")))
        if effective.get(nid, 0) <= 0 and not has_sources:
            out.append({"page": nid, "edges": edges, "effectiveRank": effective.get(nid, 0)})
    return out


def cross_tradition_unscoped_mappings(graph: Graph, *, dnm_by_tradition: "dict | None" = None) -> "list[dict]":
    """A cross-tradition mapping edge that is either a bare-identity subsumption or
    a scopedAnalogy lacking a declared ``analogyScope``.

    Cross-tradition ``subClassOf`` is always flagged (identity across vocabularies
    is the owl:sameAs bulldozer; admissible only as scoped analogy). A
    ``scopedAnalogy`` across traditions is flagged only when it carries no
    ``analogyScope`` (the EDOAL respect-of-comparison). ``dnm_by_tradition`` is
    accepted for symmetry with :func:`tradition_merges` but not required — any
    distinct-tradition mapping qualifies.
    """
    dnm = {k: {str(x).lower() for x in v} for k, v in (dnm_by_tradition or {}).items()}
    out: list[dict] = []
    seen = set()
    for nid, node in graph.nodes.items():
        meta = node["meta"]
        trad = meta.get("tradition")
        if not trad:
            continue
        trad_l = str(trad).lower()
        has_scope = bool(meta.get("analogyScope"))
        for key, requires_scope in (("subClassOf", False), ("scopedAnalogy", True)):
            for raw in as_list(meta.get(key)):
                other = resolve(graph, raw)
                if not other:
                    continue
                other_trad = graph.nodes[other]["meta"].get("tradition")
                if not other_trad or str(other_trad).lower() == trad_l:
                    continue
                # cross-tradition: subClassOf always flagged; scopedAnalogy only if unscoped.
                if requires_scope and has_scope:
                    continue
                explicit = str(other_trad).lower() in dnm.get(trad_l, set())
                pair = (nid, other, key)
                if pair not in seen:
                    seen.add(pair)
                    out.append({
                        "page": nid, "mapsTo": other, "edgeType": key,
                        "tradition": trad, "otherTradition": other_trad,
                        "explicitDoNotMerge": explicit,
                    })
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
        "subclassCycles": subclass_cycles(graph),
        "disjointnessViolations": disjointness_violations(graph),
        "unsupportedOntologyEdges": unsupported_ontology_edges(graph),
        "crossTraditionUnscopedMappings": cross_tradition_unscoped_mappings(graph, dnm_by_tradition=dnm_by_tradition),
    }
