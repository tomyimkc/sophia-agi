# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Conflict resolution by belief revision — revise or abstain, never clobber.

In a weight model, learning a fact that contradicts an old one silently overwrites
the old weights: that *is* catastrophic forgetting. On the OKF belief graph a
contradiction is a first-class, declared edge, so we can resolve it the way a Truth
Maintenance System does — give up the *weaker* belief on purpose (with its cascade),
keep the stronger, and **abstain** when the two are comparable rather than guessing.

Importance follows a hierarchy (AGM-style, with hierarchical belief importance):

    axiom / constitution  >  user-stated  >  sourced  >  system-inferred

Within the ``sourced`` tier, ties break on effective (min-over-``derivesFrom``)
confidence rank; a true tie yields **abstention** — Sophia asserts neither side and
flags the contest for a human, which is the fail-closed reading.

    from agent.belief_revision_policy import resolve_conflicts
    report = resolve_conflicts(pages)
    report["retracted"]   # weaker beliefs given up, with cascade
    report["abstained"]   # contested, comparable beliefs Sophia will not assert
"""

from __future__ import annotations

from typing import Any

from okf import build_graph, contradiction_ledger, propagate_confidence, revise
from okf.graph import resolve

# Belief importance tiers (higher wins). Read from frontmatter key ``beliefTier``;
# the default is ``sourced`` because most wiki pages are backed by a source record.
TIER_RANK = {
    "axiom": 3,
    "constitution": 3,
    "user": 2,
    "sourced": 1,
    "source": 1,
    "inferred": 0,
    "system": 0,
}
DEFAULT_TIER = "sourced"


def _tier(meta: dict) -> int:
    return TIER_RANK.get(str(meta.get("beliefTier", DEFAULT_TIER)).lower(), TIER_RANK[DEFAULT_TIER])


def _declared_pairs(graph) -> "list[tuple[str, str]]":
    """Unique, resolved (a, b) contradiction pairs from the belief graph."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in contradiction_ledger(graph)["declaredContradictions"]:
        if not entry["resolved"]:
            continue
        a = entry["page"]
        b = resolve(graph, entry["contradicts"])
        if b is None or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))
    return pairs


def _decide(graph, conf, a: str, b: str) -> "dict":
    """Decide one conflict: keep the higher-importance side, or abstain if comparable."""
    ma, mb = graph.nodes[a]["meta"], graph.nodes[b]["meta"]
    ta, tb = _tier(ma), _tier(mb)
    if ta != tb:
        winner, loser = (a, b) if ta > tb else (b, a)
        return {"verdict": "kept", "keep": winner, "retract": loser,
                "reason": f"higher belief tier ({max(ta, tb)} > {min(ta, tb)})"}
    # Same tier. Only the sourced tier breaks ties on evidence strength.
    if ta == TIER_RANK[DEFAULT_TIER]:
        ca, cb = conf.get(a, 0), conf.get(b, 0)
        if ca != cb:
            winner, loser = (a, b) if ca > cb else (b, a)
            return {"verdict": "kept", "keep": winner, "retract": loser,
                    "reason": f"stronger effective confidence ({max(ca, cb)} > {min(ca, cb)})"}
    # Comparable beliefs (equal tier, equal/again-uncomparable confidence): do not guess.
    return {"verdict": "abstain", "keep": None, "retract": None,
            "reason": "comparable importance — Sophia abstains on both rather than overwrite"}


def resolve_conflicts(pages, *, by: str = "belief_revision_policy") -> "dict[str, Any]":
    """Resolve all declared contradictions among ``pages`` by belief revision.

    Returns a decision ledger. For each conflict the policy keeps the higher-importance
    belief and retracts the weaker (with the transitive cascade ``okf.revise`` computes),
    or abstains on both when they are comparable. Non-destructive: it reports what a gate
    must no longer assert; it does not delete pages.
    """
    graph = build_graph(list(pages))
    conf = propagate_confidence(graph)
    pairs = _declared_pairs(graph)

    conflicts: list[dict] = []
    retracted: set[str] = set()
    abstained: set[str] = set()

    for a, b in pairs:
        d = _decide(graph, conf, a, b)
        record = {"a": a, "b": b, "verdict": d["verdict"], "reason": d["reason"]}
        if d["verdict"] == "kept":
            loser = d["retract"]
            rev = revise(graph, [(loser, d["reason"])], by=by)
            record.update({
                "kept": d["keep"],
                "retracted": rev.retracted,
                "cascade": [c["page"] for c in rev.cascade],
                "abstainSet": rev.abstain,
            })
            retracted.update(rev.abstain)
        else:
            record.update({"abstained": [a, b]})
            abstained.update((a, b))
        conflicts.append(record)

    # A belief kept by one conflict but retracted by another is, net, not assertable.
    abstained -= retracted
    kept = sorted(set(graph.nodes) - retracted - abstained)

    return {
        "schema": "sophia.belief_revision_policy.v1",
        "level3Evidence": False,
        "conflictCount": len(conflicts),
        "conflicts": conflicts,
        "kept": kept,
        "retracted": sorted(retracted),
        "abstained": sorted(abstained),
    }


def last_write_wins(pages) -> "dict[str, Any]":
    """Baseline a weight model imitates: the later-arriving belief overwrites the
    earlier one on every conflict, regardless of importance. Provided only to contrast
    with ``resolve_conflicts`` — it silently forgets the older fact."""
    order = {p.id: i for i, p in enumerate(pages)}
    graph = build_graph(list(pages))
    overwritten: list[str] = []
    for a, b in _declared_pairs(graph):
        older = a if order.get(a, 0) <= order.get(b, 0) else b
        overwritten.append(older)
    return {"schema": "sophia.last_write_wins_baseline.v1", "overwritten": sorted(set(overwritten))}


__all__ = ["resolve_conflicts", "last_write_wins", "TIER_RANK"]
