# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Belief-revision consistency — assert that a source retraction leaves no orphans.

``okf.counterfactual`` and ``okf.revision`` compute *what* loses support when a
source is retracted. This module states the **invariant** those computations must
satisfy for the belief set to be internally consistent:

    After retracting a set of sources, NO belief that was grounded before but has
    lost ALL of its provenance support may still be asserted.

An *orphan* is exactly such a belief: it was grounded (``is_grounded`` was True in
the base graph), and in the reduced graph — with the retracted sources struck out —
it is no longer grounded, yet it is neither one of the retracted nodes nor already
reported in the revision cascade. If any orphan exists, the revision machinery has
under-reported the fallout and the agent could keep asserting a claim whose ground
is gone. ``check_no_orphans_after_retraction`` cross-checks the *actual* reduced-graph
grounding against the revision's declared cascade and fails closed on any mismatch.

This is a pure, non-destructive check over an in-memory ``Graph`` (no page is
mutated, no clock is read). It is the consistency test a runtime gate runs after any
retraction to guarantee the abstain-set is complete.

    from okf import build_graph
    from okf.belief_revision_consistency import check_no_orphans_after_retraction

    g = build_graph(pages)
    check_no_orphans_after_retraction(g, ["dao_de_jing"])   # {"ok": True, "orphans": []}
"""
from __future__ import annotations

from okf.counterfactual import is_grounded, reduced_without
from okf.graph import Graph, resolve
from okf.revision import revise


def orphaned_beliefs(graph: Graph, retract_ids: "set[str]") -> "list[str]":
    """Beliefs that were grounded before the retraction and are un-grounded after it,
    excluding the retracted nodes themselves. This is the ground-truth orphan set,
    computed directly from grounding in the reduced graph (not from any cascade)."""
    reduced = reduced_without(graph, retract_ids)
    orphans: list[str] = []
    for nid in reduced.nodes:                     # nodes still present (not retracted)
        if is_grounded(graph, nid) and not is_grounded(reduced, nid):
            orphans.append(nid)
    return sorted(orphans)


def check_no_orphans_after_retraction(graph: Graph, retract_targets: "list") -> "dict":
    """Assert the no-orphaned-belief invariant after retracting ``retract_targets``.

    Resolves the targets, computes (a) the ground-truth orphan set directly from
    grounding in the reduced graph and (b) the revision's declared cascade, and
    reports ``ok`` only when the abstain-set (retracted ∪ cascade) covers every
    orphan. Any belief that lost all support but is NOT in the abstain-set is a
    consistency violation — surfaced in ``uncovered`` and forces ``ok=False``.

    Fail-closed: a target that does not resolve is reported in ``notFound`` and does
    NOT silently reduce the retraction to a no-op.

    Returns::

        {
          "ok": bool,               # invariant holds: every orphan is covered
          "orphans": [ids...],      # beliefs that lost ALL support (ground truth)
          "cascade": [ids...],      # what revision() reported losing support
          "uncovered": [ids...],    # orphans NOT in the abstain-set (violations)
          "abstain": [ids...],      # ids the gate must now refuse (retracted ∪ cascade)
          "retracted": [ids...],
          "notFound": [targets...],
        }
    """
    resolved: list[str] = []
    not_found: list[str] = []
    for target in retract_targets:
        rid = resolve(graph, target)
        if rid is None:
            not_found.append(target)
        elif rid not in resolved:
            resolved.append(rid)

    retract_ids = set(resolved)
    orphans = orphaned_beliefs(graph, retract_ids)

    rev = revise(graph, list(resolved))
    cascade = sorted(c["page"] for c in rev.cascade)
    abstain = set(rev.abstain)

    # Every ground-truth orphan MUST be in the abstain-set. Any that is not is a
    # belief the agent would keep asserting despite having lost all support.
    uncovered = sorted(o for o in orphans if o not in abstain)

    return {
        "ok": not uncovered,
        "orphans": orphans,
        "orphanCount": len(orphans),
        "cascade": cascade,
        "uncovered": uncovered,
        "abstain": sorted(abstain),
        "retracted": sorted(retract_ids),
        "notFound": not_found,
        "detail": ("no orphaned belief survives the retraction"
                   if not uncovered
                   else f"{len(uncovered)} belief(s) lost all support but are not "
                        f"in the abstain-set: {uncovered}"),
    }


__all__ = ["orphaned_beliefs", "check_no_orphans_after_retraction"]
