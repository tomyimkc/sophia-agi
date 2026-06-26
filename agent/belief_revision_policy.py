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


def _base_retraction_targets(graph, conf) -> "list[str]":
    """The losers a single pass of the policy would retract — the consequence-aware
    loop's round-0 "move". Shares ``_decide``/``_declared_pairs`` with
    ``resolve_conflicts`` so the two can never disagree on which side yields.
    Abstain-verdict conflicts contribute no target (nothing is retracted)."""
    targets: set[str] = set()
    for a, b in _declared_pairs(graph):
        d = _decide(graph, conf, a, b)
        if d["verdict"] == "kept":
            targets.add(d["retract"])
    return sorted(targets)


def resolve_conflicts_consequence_aware(
    pages,
    *,
    by: str = "belief_revision_policy.consequence_aware",
    escalate_threshold: "float | None" = None,
    ko_max_rounds: "int | None" = None,
) -> "dict[str, Any]":
    """Consequence-aware contradiction resolution — the LIVE consumer of
    ``reasoning.consequence.run_revise_loop`` (the loop's first runtime caller).

    Single-pass ``resolve_conflicts`` decides each conflict and retracts the losers
    once, with no view of the *consequence* of doing so. This variant keeps that
    ledger but additionally asks: what does retracting those losers cost? It drives
    ``run_revise_loop`` with a consequence-aware policy that reacts to the loop's
    own observed abstain set:

      round 0  — retract the policy's chosen losers (the base targets).
      if the resulting abstain cascade is WITHIN the flip-severity threshold, the
        retraction is a bounded consequence: accept it and stop (``allow``).
      else the cascade is too severe to commit autonomously, so the policy
        HESITATES — it reasserts (backs off) the retraction to avoid orphaning a
        large fraction of the belief set, but the contradiction is still declared,
        so on re-examination it re-retracts. That hesitation IS a ko oscillation
        (retract -> observe severe abstain set -> reassert -> contradiction
        persists -> re-retract -> recur), which ``run_revise_loop`` detects and
        routes to ``escalate`` — the load-bearing answer: a contradiction whose
        only autonomous resolutions are "overwrite catastrophically" or "leave it
        unresolved" needs a human or a new source, NEVER a silent abstain.

    Returns the ``resolve_conflicts`` ledger (kept/retracted/abstained/conflicts —
    unchanged) augmented with ``consequenceVerdict`` (the loop's recommendation)
    and ``loop`` (the full ``ReviseLoopState``). Non-destructive: it reports; it
    does not write pages or mutate the graph.

    ``candidateOnly`` stays true and ``level3Evidence`` stays false: this wires the
    live decision surface; it earns ``level3Evidence: true`` only after a real OKF
    corpus run routes contradiction decisions through it with empirical evidence
    that escalate-on-ko is the right operator response.
    """
    # Lazy import keeps the threshold/window resolution co-located with the loop's
    # own config-backed defaults so the policy's severity check and the loop's
    # per-round verdict use the SAME threshold (no drift).
    from agent.consequence_gate import flip_severity_escalate, ko_max_rounds as _cfg_ko_rounds
    from reasoning.consequence import run_revise_loop

    # Materialize once: pages may be a one-shot iterable and we read it twice
    # (resolve_conflicts + the graph we drive the loop over).
    pages = list(pages)
    base = resolve_conflicts(pages, by=by)

    graph = build_graph(pages)
    conf = propagate_confidence(graph)
    base_targets = _base_retraction_targets(graph, conf)

    n = max(1, len(graph.nodes))
    thr = flip_severity_escalate if escalate_threshold is None else escalate_threshold
    kmr = _cfg_ko_rounds if ko_max_rounds is None else ko_max_rounds
    state = {"severe": False}

    def policy(round_index: int, prev_abstain: "frozenset[str]") -> "list[str] | None":
        if round_index == 0:
            return list(base_targets)
        if not state["severe"]:
            # The round-0 cascade is bounded -> the retraction is a safe consequence;
            # accept it and stop. Otherwise enter hesitation (too severe to commit).
            if len(prev_abstain) / n < thr:
                return None
            state["severe"] = True
        # Hesitation: alternate reassert (back off) / re-retract until the loop sees
        # the round-0 abstain set recur and escalates. Even rounds re-retract
        # (recreating that severe set); odd rounds back off.
        return list(base_targets) if round_index % 2 == 0 else []

    loop = run_revise_loop(
        graph, policy=policy, escalate_threshold=thr, ko_max_rounds=kmr, by=f"{by}.loop"
    )

    return {
        **base,
        "schema": "sophia.belief_revision_policy.consequence_aware.v1",
        "consequenceVerdict": loop.finalVerdict,
        "loop": loop.to_dict(),
        "candidateOnly": True,
        "level3Evidence": False,
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


__all__ = [
    "resolve_conflicts",
    "resolve_conflicts_consequence_aware",
    "last_write_wins",
    "TIER_RANK",
]
