# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Counterfactual belief-graph queries and first-class retraction.

The belief graph (``okf.graph``) already records *who derives a claim from what*
and propagates confidence as a min-over-``derivesFrom``-chain. What it could not
yet answer is the question the Sophia vision names directly:

    "What would I conclude if this source were removed?"

This module adds that interrogative layer, plus a named, auditable **retraction**
operation, without mutating any persisted page. Both are pure functions over an
in-memory ``Graph``: they build a reduced view (the source struck out), recompute
grounding and confidence, and report what changed — fail-closed, so a claim that
rested *only* on the removed source is reported as having **lost its support**
rather than silently falling back to its own face-value confidence.

    from okf import build_graph
    from okf.counterfactual import counterfactual_remove, retract

    g = build_graph(pages)
    counterfactual_remove(g, "legend")          # impact of dropping a source
    retract(g, "legend", reason="forged")       # named retraction + audit entry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from okf.graph import Graph, propagate_confidence, resolve
from okf.schema import as_list


def resolved_derives(graph: Graph, nid: str) -> "list[str]":
    """The ``derivesFrom`` targets of a node that resolve to a node still present."""
    node = graph.nodes.get(nid)
    if node is None:
        return []
    out: list[str] = []
    for raw in as_list(node["meta"].get("derivesFrom")):
        dep = resolve(graph, raw)
        if dep is not None and dep in graph.nodes and dep not in out:
            out.append(dep)
    return out


def derives_closure(graph: Graph, nid: str) -> "set[str]":
    """Transitive set of provenance ancestors reachable via ``derivesFrom``."""
    seen: set[str] = set()
    stack = list(resolved_derives(graph, nid))
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        stack.extend(resolved_derives(graph, dep))
    return seen


def is_grounded(graph: Graph, nid: str, _stack: "frozenset[str] | None" = None) -> bool:
    """Whether a claim still has a provenance ground.

    A claim with no ``derivesFrom`` is a *primary* claim and is grounded in itself.
    A derived claim is grounded only if at least one of its ``derivesFrom`` targets
    resolves to a present, grounded node. A derived claim whose dependencies have
    all been removed (or never resolved) is **orphaned** — ``False`` — which is the
    fail-closed reading: provenance was claimed, and it is now gone.
    """
    node = graph.nodes.get(nid)
    if node is None:
        return False
    declared = as_list(node["meta"].get("derivesFrom"))
    if not declared:
        return True  # primary / self-grounded root
    stack = (_stack or frozenset()) | {nid}
    for dep in resolved_derives(graph, nid):
        if dep in stack:
            continue  # cycle guard — a cycle is not a ground
        if is_grounded(graph, dep, stack):
            return True
    return False


def _reduced(graph: Graph, removed: "set[str]") -> Graph:
    """A view of the graph with ``removed`` node ids (and their aliases) struck out."""
    nodes = {nid: node for nid, node in graph.nodes.items() if nid not in removed}
    aliases = {a: nid for a, nid in graph.alias_index.items() if nid not in removed}
    return Graph(nodes=nodes, alias_index=aliases)


def reduced_without(graph: Graph, removed: "set[str] | list[str]") -> Graph:
    """Public: a graph view with the given node ids struck out (for multi-removal
    counterfactuals and belief revision)."""
    return _reduced(graph, set(removed))


def counterfactual_remove(graph: Graph, source: str, *, query: "str | None" = None) -> "dict":
    """What would change if ``source`` were removed from the belief graph?

    Returns the set of affected claims with, for each, whether it transitively
    depended on the source, its grounding before/after, whether support was lost,
    and its naive min-over-chain confidence rank before/after. ``supportLost`` is
    the headline, fail-closed signal: those are the claims Sophia would no longer
    be entitled to assert. If ``query`` is given, the report also isolates the
    before/after belief for that one entity.
    """
    rid = resolve(graph, source)
    if rid is None:
        return {"found": False, "source": source, "id": None}

    base_conf = propagate_confidence(graph)
    reduced = _reduced(graph, {rid})
    cf_conf = propagate_confidence(reduced)

    affected: list[dict] = []
    for nid in reduced.nodes:
        depends = rid in derives_closure(graph, nid)
        grounded_before = is_grounded(graph, nid)
        grounded_after = is_grounded(reduced, nid)
        rank_before = base_conf.get(nid, 0)
        rank_after = cf_conf.get(nid, 0)
        changed = (
            depends
            or grounded_before != grounded_after
            or rank_before != rank_after
        )
        if not changed:
            continue
        support_lost = grounded_before and not grounded_after
        affected.append({
            "page": nid,
            "dependsOnRemoved": depends,
            "groundedBefore": grounded_before,
            "groundedAfter": grounded_after,
            "supportLost": support_lost,
            "confidenceRankBefore": rank_before,
            # Fail-closed: a claim that lost its only ground is unsupported (0),
            # not whatever face value the naive min-over-chain would fall back to.
            "confidenceRankAfter": 0 if support_lost else rank_after,
        })

    affected.sort(key=lambda r: (not r["supportLost"], r["page"]))
    support_lost = [r["page"] for r in affected if r["supportLost"]]
    out = {
        "found": True,
        "source": source,
        "id": rid,
        "affected": affected,
        "affectedCount": len(affected),
        "supportLost": support_lost,
        "supportLostCount": len(support_lost),
    }
    if query is not None:
        out["query"] = _belief_delta(graph, reduced, query)
    return out


def _belief_delta(graph: Graph, reduced: Graph, entity: str) -> "dict":
    """Before/after belief snapshot for one entity under a graph reduction."""
    from okf.graph import belief

    nid = resolve(graph, entity)
    before = belief(graph, entity)
    if nid is None or nid not in reduced.nodes:
        # the queried entity is itself the removed source (or unknown)
        return {"entity": entity, "id": nid, "removed": nid is not None and nid not in reduced.nodes,
                "groundedBefore": is_grounded(graph, nid) if nid else False, "groundedAfter": False,
                "before": before}
    after = belief(reduced, entity)
    grounded_before = is_grounded(graph, nid)
    grounded_after = is_grounded(reduced, nid)
    return {
        "entity": entity,
        "id": nid,
        "removed": False,
        "groundedBefore": grounded_before,
        "groundedAfter": grounded_after,
        "supportLost": grounded_before and not grounded_after,
        "effectiveBefore": before.get("effectiveConfidenceRank"),
        "effectiveAfter": 0 if (grounded_before and not grounded_after) else after.get("effectiveConfidenceRank"),
    }


@dataclass
class Retraction:
    """A named, auditable retraction of a claim from the belief set.

    Computed, not yet persisted: the graph view is non-destructive, so a caller
    can inspect the downstream impact (``impact``) and record the decision
    (``audit_entry()``) before committing any change to disk.
    """

    target: str
    id: "str | None"
    reason: str
    by: str = "system"
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    found: bool = True
    impact: dict = field(default_factory=dict)

    @property
    def downstream(self) -> "list[str]":
        """Claims that lose their support if this retraction is committed."""
        return list(self.impact.get("supportLost", []))

    def to_dict(self) -> "dict":
        return {
            "target": self.target,
            "id": self.id,
            "reason": self.reason,
            "by": self.by,
            "at": self.at,
            "found": self.found,
            "downstream": self.downstream,
            "impact": self.impact,
        }

    def audit_entry(self) -> "dict":
        """A compact, append-only audit record (e.g. one JSONL line)."""
        return {
            "event": "retraction",
            "at": self.at,
            "by": self.by,
            "target": self.id or self.target,
            "reason": self.reason,
            "downstream": self.downstream,
        }


def retract(graph: Graph, target: str, *, reason: str, by: str = "system") -> Retraction:
    """Retract a claim: name it, give a reason, and compute the downstream impact.

    This is the deliberate, audited counterpart to a counterfactual probe — the
    same reduced-graph analysis, but recorded as an intentional decision with a
    reason and an audit entry. It does not delete any page; persistence (a
    tombstone, a ``supersededBy`` edge, or removal) is the caller's choice, made
    with the impact in hand.
    """
    rid = resolve(graph, target)
    if rid is None:
        return Retraction(target=target, id=None, reason=reason, by=by, found=False,
                          impact={"found": False, "source": target, "id": None})
    impact = counterfactual_remove(graph, target)
    return Retraction(target=target, id=rid, reason=reason, by=by, found=True, impact=impact)
