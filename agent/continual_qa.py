"""Continual Provenance QA (CPQA): the integrated dual-store benchmark.

Runs the full "LLM-as-control-flow, knowledge-in-the-graph" loop over a *streamed*
sequence of episodes and scores it against a frozen weight-model baseline. Each episode
may teach new OKF pages, retract sources, and ask questions about any fact (old or new).
The contrast is the thesis in one number:

    graph_backed   — learns by page write, revises conflicts (belief_revision_policy),
                     unlearns on demand (Unlearner), answers ONLY from the grounded
                     belief state, abstains fail-closed. Forgets 0 facts, fabricates 0.
    parametric_baseline — knowledge frozen after episode 0 (a weight model without
                     retraining): permanently knows t0 facts, but cannot learn new ones
                     and cannot unlearn/correct stale ones.

This integrates Experiments 1–4. The controller here is deterministic (retrieval-only)
so the benchmark isolates the *knowledge substrate*; an LLM controller can plug in behind
the same `answer(target)` contract later. Offline, deterministic, dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.belief_revision_policy import resolve_conflicts
from agent.continual_retention import Snapshot, build_report
from okf import build_graph, claims_to_abstain, is_grounded, propagate_confidence
from okf.graph import resolve
from okf.page import Page

# Frontmatter keys we DON'T copy into a page's meta (they are harness fields).
_NON_META = {"q", "expect", "type", "queries", "learn", "retract", "episode"}


@dataclass(frozen=True)
class Query:
    id: str
    target: str                 # the fact id the question is about
    expect: str                 # "assert" | "abstain"
    type: str = "recall"
    text: str = ""


@dataclass(frozen=True)
class Episode:
    id: str
    learn: tuple = ()           # OKF Page objects taught this episode
    retract: tuple = ()         # source ids to forget this episode
    queries: tuple = ()         # Query objects asked at this step


def _page_from_dict(d: dict) -> Page:
    meta = {k: v for k, v in d.items() if k not in _NON_META}
    return Page(path=Path(f"{meta['id']}.md"), meta=meta)


def load_episodes(path) -> "list[Episode]":
    episodes: list[Episode] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        learn = tuple(_page_from_dict(p) for p in d.get("learn", []))
        queries = tuple(
            Query(id=q["id"], target=q["target"], expect=q["expect"],
                  type=q.get("type", "recall"), text=q.get("q", ""))
            for q in d.get("queries", [])
        )
        episodes.append(Episode(id=d["episode"], learn=learn,
                                retract=tuple(d.get("retract", [])), queries=queries))
    return episodes


class GraphBackedSystem:
    """Knowledge lives in the OKF graph. Learning = page write; unlearning = tombstone;
    conflicts = belief revision. An answer asserts a fact only if it is *grounded* in the
    current belief state, else abstains (fail-closed)."""

    def __init__(self) -> None:
        self._pages: list[Page] = []
        self._tombstoned: set[str] = set()
        self._cascade: set[str] = set()        # ids deliberately abstained via retraction

    def learn(self, pages) -> None:
        self._pages.extend(pages)

    def retract(self, ids) -> None:
        graph = build_graph(self._active())
        for target in ids:
            rid = resolve(graph, target)
            if rid is None:
                continue
            # claims_to_abstain returns the target + every claim that loses support (cascade).
            self._cascade.update(claims_to_abstain(graph, [target]))
            self._tombstoned.add(rid)

    def _active(self) -> "list[Page]":
        return [p for p in self._pages if p.id not in self._tombstoned]

    def _conflict_suppressed(self) -> "set[str]":
        active = self._active()
        if not active:
            return set()
        revision = resolve_conflicts(active)
        return set(revision["retracted"]) | set(revision["abstained"])

    def suppressed_ids(self) -> "set[str]":
        """Everything not assertable because of a *deliberate* operation: retraction
        (+ its cascade) and conflict resolution. Used to separate intended unlearning
        from catastrophic forgetting."""
        return set(self._tombstoned) | set(self._cascade) | self._conflict_suppressed()

    def grounded_state(self) -> "dict":
        """Assertable facts -> effective confidence rank: grounded, minus anything a
        retraction or conflict deliberately removed."""
        active = self._active()
        if not active:
            return {}
        suppressed = self._cascade | self._conflict_suppressed()
        graph = build_graph(active)
        conf = propagate_confidence(graph)
        return {nid: conf.get(nid, 0) for nid in graph.nodes
                if is_grounded(graph, nid) and nid not in suppressed}

    def answer(self, target: str) -> str:
        return "assert" if target in self.grounded_state() else "abstain"


class ParametricBaseline:
    """A frozen weight model: it permanently knows the facts present at t0 and nothing
    else. It cannot learn later facts and cannot unlearn/correct — the catastrophic-
    forgetting / staleness failure modes, modeled fairly (it is correct on all t0 facts)."""

    def __init__(self, frozen) -> None:
        self._frozen = set(frozen)

    def answer(self, target: str) -> str:
        return "assert" if target in self._frozen else "abstain"


def _score(answer: str, expect: str) -> str:
    if expect == "assert":
        return "correct" if answer == "assert" else "miss"          # miss == forgetting/never-learned
    return "correct" if answer == "abstain" else "fabrication"      # asserted a stale/unknown fact


def _aggregate(rows, key: str) -> "dict":
    n = len(rows)
    correct = sum(1 for r in rows if r[key] == "correct")
    fabrication = sum(1 for r in rows if r[key] == "fabrication")
    miss = sum(1 for r in rows if r[key] == "miss")
    return {
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "fabrications": fabrication,
        "fabricationRate": round(fabrication / n, 4) if n else 0.0,
        "misses": miss,
        "missRate": round(miss / n, 4) if n else 0.0,
    }


def run_benchmark(episodes) -> "dict[str, Any]":
    """Stream the episodes through both systems and score retention + fabrication."""
    gb = GraphBackedSystem()
    baseline: "ParametricBaseline | None" = None
    snapshots: list[Snapshot] = []
    rows: list[dict] = []
    intentional: set[str] = set()        # ids removed on purpose (retraction/revision)

    for i, ep in enumerate(episodes):
        gb.learn(ep.learn)
        gb.retract(ep.retract)
        state = gb.grounded_state()
        intentional |= gb.suppressed_ids()
        introduced = tuple(p.id for p in ep.learn if p.id in state)
        snapshots.append(Snapshot(task_id=ep.id, grounded=dict(state), introduced=introduced))
        if i == 0:
            baseline = ParametricBaseline(state.keys())   # freeze the weight model at t0

        for q in ep.queries:
            gb_ans = gb.answer(q.target)
            bl_ans = baseline.answer(q.target) if baseline else "abstain"
            rows.append({
                "episode": ep.id,
                "query": q.id,
                "target": q.target,
                "expect": q.expect,
                "type": q.type,
                "graph_backed": _score(gb_ans, q.expect),
                "parametric_baseline": _score(bl_ans, q.expect),
            })

    retention = build_report(snapshots)
    # Separate catastrophic (unintended) forgetting from deliberate unlearning/revision.
    unintended = [d for d in retention["forgottenDetail"] if d["fact"] not in intentional]
    deliberate = [d for d in retention["forgottenDetail"] if d["fact"] in intentional]
    return {
        "schema": "sophia.continual_qa.v1",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "episodes": [e.id for e in episodes],
        "queryCount": len(rows),
        "systems": {
            "graph_backed": _aggregate(rows, "graph_backed"),
            "parametric_baseline": _aggregate(rows, "parametric_baseline"),
        },
        "retention": {
            "unintendedForgetting": len(unintended),         # headline: catastrophic forgetting
            "deliberateUnlearning": len(deliberate),         # retraction/revision (expected)
            "unintendedDetail": unintended,
            "backwardTransfer": retention["backwardTransfer"],
            "retentionMatrix": retention["retentionMatrix"],
        },
        "rows": rows,
    }


__all__ = ["Query", "Episode", "load_episodes", "GraphBackedSystem", "ParametricBaseline", "run_benchmark"]
