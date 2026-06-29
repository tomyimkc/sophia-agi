# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Incremental belief revision — recompute only the affected subgraph, not the world.

The limitations ledger flags that classic Truth-Maintenance Systems did not scale:
re-resolving the *whole* belief graph on every new fact is O(N) per update, so a
stream of N facts is O(N^2). ``agent.belief_revision_scaling`` measures that
full-recompute cost honestly. This module is the answer: an ``IncrementalReviser``
that, when a fact is added, touches ONLY the affected closure — the new page, the
beliefs it contradicts (and that contradict it), and the transitive ``derivesFrom``
dependents that could cascade — and re-resolves conflicts within that frontier alone.

The honest discipline (Sophia, not magic): the incremental belief state is asserted
to be **provably equivalent** to the full-recompute oracle
(``agent.belief_revision_policy.resolve_conflicts``) at every step — same kept,
retracted, and abstained sets. Correctness is the deterministic, asserted property;
wall-time is only *reported*. Sub-linearity is an emergent, *structural* claim: the
count of nodes actually re-examined per update (``affectedNodes``) stays bounded by
the local conflict neighbourhood, not by total N. The harness reports that curve;
the verdict ``subLinear`` is decided by the affected-node curve being bounded, never
by noisy timing.

    from agent.incremental_revision import IncrementalReviser, incremental_sweep
    r = IncrementalReviser()
    for page in pages:
        r.add_fact(page)              # touches only the affected subgraph
    r.kept_ids()                      # == set(resolve_conflicts(pages)["kept"])

Fail-closed: every emitted verdict/report dict carries ``"candidateOnly": True``;
ambiguous (comparable) conflicts abstain rather than silently pick a side.
"""

from __future__ import annotations

import time
from typing import Any

from agent.belief_revision_scaling import make_pages, scaling_sweep
from okf.schema import as_list, confidence_rank
from okf import wikilinks

# Re-use the oracle's tier vocabulary so the incremental decision is provably the
# same decision the full recompute would make (single source of truth for ordering).
from agent.belief_revision_policy import TIER_RANK, DEFAULT_TIER

__all__ = [
    "IncrementalReviser",
    "incremental_sweep",
    "compare_to_full",
    "SUBLINEAR_TOLERANCE",
]

# The affected-node curve at the largest N is judged sub-linear iff it is within this
# constant factor of the curve at the smallest N. Bounded (constant) per-update work,
# independent of N, is the structural sub-linearity claim — full recompute touches ~N.
SUBLINEAR_TOLERANCE = 3.0


def _tier(meta: dict) -> int:
    return TIER_RANK.get(str(meta.get("beliefTier", DEFAULT_TIER)).lower(), TIER_RANK[DEFAULT_TIER])


class IncrementalReviser:
    """Maintains a resolved belief state under a growing page set, re-resolving only
    the affected subgraph on each ``add_fact``.

    Maintained indices (built incrementally, O(edges of the new page) per add — never
    O(N)):
      * ``_pages``        : id -> Page (the growing page set, in insertion order)
      * ``_contradicts``  : id -> set(ids) symmetric contradiction adjacency
      * ``_dependents``   : id -> set(ids) reverse ``derivesFrom`` edges (who derives
                            FROM this node) — the cascade-frontier map
      * ``_derives``      : id -> set(ids) forward ``derivesFrom`` edges (resolved)

    The current belief partition (``_retracted`` / ``_abstained`` / kept-by-difference)
    is kept up to date by recomputing the local closure only.
    """

    def __init__(self) -> None:
        self._pages: dict[str, Any] = {}
        self._order: "list[str]" = []
        self._contradicts: "dict[str, set]" = {}
        self._dependents: "dict[str, set]" = {}
        self._derives: "dict[str, set]" = {}
        # Belief partition over the full id set, kept current incrementally.
        self._retracted: "set[str]" = set()
        self._abstained: "set[str]" = set()

    # ---- index construction (incremental, local) ----------------------------------

    def _norm(self, target: str) -> str:
        return wikilinks.normalize_target(target)

    def _index_page(self, page) -> None:
        """Add one page to the indices. Work is bounded by the new page's own edge
        count plus the (already-present) endpoints it links to — never a full scan."""
        nid = page.id
        meta = page.meta
        self._pages[nid] = page
        self._order.append(nid)
        self._contradicts.setdefault(nid, set())
        self._dependents.setdefault(nid, set())
        self._derives.setdefault(nid, set())

        # Symmetric contradicts edges (resolve only to ids already present OR equal to
        # this id; dangling targets are recorded so a later-arriving endpoint links up).
        for raw in as_list(meta.get("contradicts")):
            other = self._norm(raw)
            self._contradicts[nid].add(other)
            self._contradicts.setdefault(other, set()).add(nid)

        # Forward derivesFrom edges + reverse dependent edges.
        for raw in as_list(meta.get("derivesFrom")):
            dep = self._norm(raw)
            self._derives[nid].add(dep)
            self._dependents.setdefault(dep, set()).add(nid)

    # ---- affected-subgraph computation --------------------------------------------

    def _present(self, nid: str) -> bool:
        return nid in self._pages

    def _affected_closure(self, nid: str) -> "set[str]":
        """The set of nodes that must be re-examined when ``nid`` is added.

        Built locally (bounded by the conflict neighbourhood + provenance chain depth,
        never by N):
          1. seeds = the new node PLUS its transitive ``derivesFrom`` ANCESTORS — a
             retraction of any ancestor cascades down to ``nid``, so a late-arriving
             derived fact must re-examine the conflicts sitting above it;
          2. add the contradiction neighbours (symmetric — both directions are kept in
             the adjacency at index time) of every seed — those are the conflicts whose
             losers could cascade into this neighbourhood;
          3. take the transitive ``derivesFrom`` DEPENDENTS (reverse edges) of all of
             the above — the downward cascade frontier.

        A new fact with no contradiction and no contested ancestor touches only itself
        (+ its own dependents)."""
        # 1. provenance ancestors of the new node (chain upward).
        seeds: "set[str]" = {nid}
        up_stack = [d for d in self._derives.get(nid, set()) if self._present(d)]
        while up_stack:
            cur = up_stack.pop()
            if cur in seeds:
                continue
            seeds.add(cur)
            for dep in self._derives.get(cur, set()):
                if self._present(dep) and dep not in seeds:
                    up_stack.append(dep)

        # 2. contradiction neighbours of every seed (symmetric adjacency).
        for s in list(seeds):
            for other in self._contradicts.get(s, set()):
                if self._present(other):
                    seeds.add(other)

        # 3. transitive derivesFrom dependents (who derives from any seed, recursively).
        closure: "set[str]" = set()
        stack = list(seeds)
        while stack:
            cur = stack.pop()
            if cur in closure or not self._present(cur):
                continue
            closure.add(cur)
            for dependent in self._dependents.get(cur, set()):
                if self._present(dependent) and dependent not in closure:
                    stack.append(dependent)
        return closure

    # ---- local conflict resolution -------------------------------------------------

    def _local_pairs(self, closure: "set[str]") -> "list[tuple[str, str]]":
        """Unique resolved contradiction pairs with at least one endpoint in the
        affected closure (the conflicts that could have changed)."""
        seen: "set[tuple[str, str]]" = set()
        pairs: "list[tuple[str, str]]" = []
        for a in closure:
            for b in self._contradicts.get(a, set()):
                if not self._present(b) or a == b:
                    continue
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((a, b))
        return pairs

    def _is_grounded(self, nid: str, _stack: "frozenset | None" = None) -> bool:
        """Mirror of okf.counterfactual.is_grounded over the maintained derives index,
        scoped to currently-present nodes."""
        if not self._present(nid):
            return False
        deps = [d for d in self._derives.get(nid, set()) if self._present(d)]
        declared = as_list(self._pages[nid].meta.get("derivesFrom"))
        if not declared:
            return True  # self-grounded root
        if not deps:
            return False  # declared provenance, none present -> orphaned
        stack = (_stack or frozenset()) | {nid}
        for dep in deps:
            if dep in stack:
                continue
            if self._is_grounded(dep, stack):
                return True
        return False

    def _effective_conf(self, nid: str, _stack: "frozenset | None" = None) -> int:
        """min-over-derivesFrom-chain confidence rank, mirroring propagate_confidence,
        scoped to present nodes (matches the oracle's per-node value)."""
        if not self._present(nid):
            return 0
        stack = (_stack or frozenset())
        if nid in stack:
            return confidence_rank(self._pages[nid].meta.get("authorConfidence"))
        best = confidence_rank(self._pages[nid].meta.get("authorConfidence"))
        for dep in self._derives.get(nid, set()):
            if self._present(dep) and dep not in stack:
                best = min(best, self._effective_conf(dep, stack | {nid}))
        return best

    def _decide(self, a: str, b: str) -> "dict":
        """Decide one conflict exactly as belief_revision_policy._decide does:
        higher tier wins; sourced-tier ties break on effective confidence; comparable
        => abstain. Fail-closed: comparable beliefs abstain, never silent-pick."""
        ma, mb = self._pages[a].meta, self._pages[b].meta
        ta, tb = _tier(ma), _tier(mb)
        if ta != tb:
            winner, loser = (a, b) if ta > tb else (b, a)
            return {"verdict": "kept", "keep": winner, "retract": loser}
        if ta == TIER_RANK[DEFAULT_TIER]:
            ca, cb = self._effective_conf(a), self._effective_conf(b)
            if ca != cb:
                winner, loser = (a, b) if ca > cb else (b, a)
                return {"verdict": "kept", "keep": winner, "retract": loser}
        return {"verdict": "abstain", "keep": None, "retract": None}

    def _cascade_for(self, loser: str) -> "set[str]":
        """Claims that lose support when ``loser`` is retracted: the transitive
        derivesFrom dependents that become ungrounded once ``loser`` is gone. Computed
        over the maintained dependents map, scoped to the affected frontier — this is
        the same set okf.revise's cascade computes for a single retraction."""
        abstain: "set[str]" = {loser}
        # BFS over reverse derivesFrom edges; a dependent is in the cascade only if it
        # was grounded before but is no longer grounded once `removed` are gone.
        removed: "set[str]" = {loser}
        frontier = [d for d in self._dependents.get(loser, set()) if self._present(d)]
        while frontier:
            cur = frontier.pop()
            if cur in abstain:
                continue
            if self._is_grounded(cur) and not self._grounded_without(cur, removed):
                abstain.add(cur)
                removed.add(cur)
                for nxt in self._dependents.get(cur, set()):
                    if self._present(nxt) and nxt not in abstain:
                        frontier.append(nxt)
        return abstain

    def _grounded_without(self, nid: str, removed: "set[str]", _stack: "frozenset | None" = None) -> bool:
        """is_grounded as if ``removed`` ids were struck from the graph."""
        if not self._present(nid) or nid in removed:
            return False
        declared = as_list(self._pages[nid].meta.get("derivesFrom"))
        if not declared:
            return True
        deps = [d for d in self._derives.get(nid, set()) if self._present(d) and d not in removed]
        if not deps:
            return False
        stack = (_stack or frozenset()) | {nid}
        for dep in deps:
            if dep in stack:
                continue
            if self._grounded_without(dep, removed, stack):
                return True
        return False

    def _resolve_closure(self, closure: "set[str]") -> "tuple[set, set, int]":
        """Re-resolve every conflict touching ``closure`` and return the local
        (retracted, abstained, conflictsTouched). Mirrors resolve_conflicts' merge:
        a node retracted by any conflict is retracted (with its cascade); abstain is
        net of retracted."""
        pairs = self._local_pairs(closure)
        local_retracted: "set[str]" = set()
        local_abstained: "set[str]" = set()
        for a, b in pairs:
            d = self._decide(a, b)
            if d["verdict"] == "kept":
                local_retracted |= self._cascade_for(d["retract"])
            else:
                local_abstained.update((a, b))
        local_abstained -= local_retracted
        return local_retracted, local_abstained, len(pairs)

    # ---- public API ----------------------------------------------------------------

    def add_fact(self, page) -> "dict":
        """Add one fact and re-resolve ONLY the affected subgraph. Returns a delta
        report. ``affectedNodes`` is the count of nodes actually re-examined — the
        sub-linearity measure (bounded by the local conflict neighbourhood, not N)."""
        before_retracted = set(self._retracted)

        self._index_page(page)
        nid = page.id
        closure = self._affected_closure(nid)

        # The conflicts within the closure are the only ones whose decision can have
        # changed. We recompute the local partition over the closure's conflict pairs,
        # then splice it into the maintained global partition: every id that the local
        # pass touches (as an endpoint or cascade member) has its membership overwritten
        # by the local result, leaving the rest of the graph untouched.
        local_retracted, local_abstained, conflicts_touched = self._resolve_closure(closure)

        # Determine the universe of ids the local pass is authoritative over: closure
        # nodes plus any cascade members it produced.
        touched = set(closure) | local_retracted | local_abstained

        # Splice: clear prior membership for touched ids, then apply local decision.
        self._retracted -= touched
        self._abstained -= touched
        self._retracted |= local_retracted
        self._abstained |= local_abstained
        # Net: a retracted id is never also abstained.
        self._abstained -= self._retracted

        after_retracted = set(self._retracted)
        delta_retracted = sorted(after_retracted - before_retracted)
        delta_restored = sorted(before_retracted - after_retracted)

        return {
            "schema": "sophia.incremental_revision.v1",
            "added": nid,
            "affectedNodes": len(closure),
            "affectedIds": sorted(closure),
            "conflictsTouched": conflicts_touched,
            "deltaRetracted": delta_retracted,
            "deltaRestored": delta_restored,
            "candidateOnly": True,
        }

    def kept_ids(self) -> "set[str]":
        """Current assertable belief: all present ids minus retracted and abstained."""
        return set(self._pages) - self._retracted - self._abstained

    def belief_state(self) -> "dict":
        """The current resolved belief partition, kept up to date incrementally."""
        return {
            "schema": "sophia.incremental_revision.belief_state.v1",
            "kept": sorted(self.kept_ids()),
            "retracted": sorted(self._retracted),
            "abstained": sorted(self._abstained),
            "candidateOnly": True,
        }


def incremental_sweep(sizes, *, contradiction_every: int = 10) -> "list[dict]":
    """For each N, replay make_pages(N) through a fresh IncrementalReviser one page at
    a time and record the affected-node curve + wall-time (reported, never asserted).

    ``maxAffectedNodes`` staying bounded as N grows is the sub-linear evidence: full
    recompute would touch ~N every step."""
    out: "list[dict]" = []
    for n in sizes:
        pages = make_pages(n, contradiction_every=contradiction_every)
        reviser = IncrementalReviser()
        affected_counts: "list[int]" = []
        start = time.perf_counter()
        for page in pages:
            rep = reviser.add_fact(page)
            affected_counts.append(rep["affectedNodes"])
        elapsed = time.perf_counter() - start
        total = sum(affected_counts)
        out.append({
            "n": n,
            "maxAffectedNodes": max(affected_counts) if affected_counts else 0,
            "meanAffectedNodes": round(total / len(affected_counts), 4) if affected_counts else 0.0,
            "totalAffected": total,
            "seconds": round(elapsed, 4),
            "candidateOnly": True,
        })
    return out


def compare_to_full(sizes, *, contradiction_every: int = 10) -> "dict":
    """Run incremental_sweep AND the full-recompute scaling_sweep, returning both
    curves plus a DETERMINISTIC, structural sub-linearity verdict.

    ``subLinear`` is decided by the AFFECTED-NODE curve being bounded — maxAffectedNodes
    at the largest N within SUBLINEAR_TOLERANCE x maxAffectedNodes at the smallest N —
    NOT by wall-time (which is noisy and only reported)."""
    sizes = list(sizes)
    incremental = incremental_sweep(sizes, contradiction_every=contradiction_every)
    full = scaling_sweep(sizes, contradiction_every=contradiction_every)

    sub_linear = False
    if len(incremental) >= 2:
        by_n = sorted(incremental, key=lambda r: r["n"])
        smallest = by_n[0]["maxAffectedNodes"]
        largest = by_n[-1]["maxAffectedNodes"]
        # Bounded growth: the largest-N affected-node count must not exceed a small
        # constant factor of the smallest-N count. Guard against a zero baseline.
        baseline = max(smallest, 1)
        sub_linear = largest <= baseline * SUBLINEAR_TOLERANCE
    elif len(incremental) == 1:
        sub_linear = True  # single point — trivially bounded

    return {
        "schema": "sophia.incremental_revision.compare.v1",
        "sizes": sizes,
        "incremental": incremental,
        "full": full,
        "tolerance": SUBLINEAR_TOLERANCE,
        "subLinear": bool(sub_linear),
        "candidateOnly": True,
    }
