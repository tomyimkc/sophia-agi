# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Belief revision: apply retractions and propagate their consequences.

Counterfactual queries (``okf.counterfactual``) ask *what if* a single source were
removed. Revision is the deliberate, possibly multi-source act of actually removing
claims from the belief set and computing the **cascade**: because grounding is
transitive (``is_grounded`` recurses over ``derivesFrom``), retracting one record
can orphan a claim that derives from it, which can in turn orphan a claim that
derives from *that* — all surfaced in one pass over the reduced graph.

Revision is non-destructive: it computes the revised belief state and an audit
trail without deleting any page. The headline product is ``claims_to_abstain`` —
the set of claims the agent must no longer assert because their provenance ground
is gone. That set is what a runtime gate consults to fail closed after a retraction.

    from okf import build_graph
    from okf.revision import revise, claims_to_abstain

    g = build_graph(pages)
    rev = revise(g, [("dao_de_jing", "single-author attribution discredited")])
    rev.cascade            # claims that lost support (transitively)
    claims_to_abstain(g, ["dao_de_jing"])   # ids the gate must now refuse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from okf.counterfactual import is_grounded, reduced_without
from okf.graph import Graph, propagate_confidence, resolve

# A retraction is either a bare target id/alias or a (target, reason) pair.
Retractable = "str | tuple[str, str]"


def _split(item) -> "tuple[str, str]":
    if isinstance(item, (tuple, list)):
        target = item[0]
        reason = item[1] if len(item) > 1 else "(unspecified)"
        return target, reason
    return item, "(unspecified)"


@dataclass
class Revision:
    """Result of revising the belief set by a set of retractions (non-destructive)."""

    retracted: "list[str]" = field(default_factory=list)      # resolved ids removed
    notFound: "list[str]" = field(default_factory=list)       # targets that did not resolve
    cascade: "list[dict]" = field(default_factory=list)       # claims that lost support
    reasons: "dict" = field(default_factory=dict)             # id -> reason
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    by: str = "system"

    @property
    def abstain(self) -> "list[str]":
        """Claims the agent must no longer assert: the retracted ids + the cascade."""
        return sorted(set(self.retracted) | {c["page"] for c in self.cascade})

    def to_dict(self) -> "dict":
        return {
            "retracted": self.retracted,
            "notFound": self.notFound,
            "cascade": self.cascade,
            "cascadeCount": len(self.cascade),
            "abstain": self.abstain,
            "reasons": self.reasons,
            "at": self.at,
            "by": self.by,
        }

    def audit_log(self) -> "list[dict]":
        """One append-only audit record per retraction (plus the cascade it caused)."""
        cascade_ids = [c["page"] for c in self.cascade]
        return [
            {"event": "retraction", "at": self.at, "by": self.by, "target": rid,
             "reason": self.reasons.get(rid, "(unspecified)"), "cascade": cascade_ids}
            for rid in self.retracted
        ]


def revise(graph: Graph, retractions: "list", *, by: str = "system") -> Revision:
    """Apply a set of retractions and return the revised belief state + cascade.

    ``retractions`` is a list of target ids/aliases, or ``(target, reason)`` pairs.
    Targets that do not resolve are reported in ``notFound`` (fail-closed: an
    unknown retraction never silently no-ops). The cascade is computed over the
    graph with *all* resolved targets removed at once, so transitive support loss
    is captured in a single consistent pass.
    """
    resolved: list[str] = []
    not_found: list[str] = []
    reasons: dict = {}
    for item in retractions:
        target, reason = _split(item)
        rid = resolve(graph, target)
        if rid is None:
            not_found.append(target)
            continue
        if rid not in resolved:
            resolved.append(rid)
        reasons[rid] = reason

    removed = set(resolved)
    reduced = reduced_without(graph, removed)
    base_conf = propagate_confidence(graph)
    cf_conf = propagate_confidence(reduced)

    cascade: list[dict] = []
    for nid in reduced.nodes:                       # nodes still present (not retracted)
        grounded_before = is_grounded(graph, nid)
        grounded_after = is_grounded(reduced, nid)
        if grounded_before and not grounded_after:  # lost its only provenance ground
            cascade.append({
                "page": nid,
                "confidenceRankBefore": base_conf.get(nid, 0),
                "confidenceRankAfter": 0,            # fail-closed: unsupported -> 0
            })
    cascade.sort(key=lambda r: r["page"])
    return Revision(retracted=resolved, notFound=not_found, cascade=cascade,
                    reasons=reasons, by=by)


def claims_to_abstain(graph: Graph, retractions: "list") -> "list[str]":
    """The ids a runtime gate must refuse to assert after the given retractions:
    the retracted claims plus everything that transitively loses support."""
    return revise(graph, retractions).abstain
