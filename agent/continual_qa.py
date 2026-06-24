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
from agent.continual_qa_controller import OracleController
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


def build_vocab(episodes) -> "dict[str, str]":
    """Map every taught fact id to its searchable text (id words + title + aliases).

    This is the namespace a controller routes a question into — the wiki's catalog.
    """
    vocab: dict[str, str] = {}
    for ep in episodes:
        for p in ep.learn:
            parts = [p.id.replace("_", " ")]
            title = p.meta.get("canonicalTitleEn")
            if title:
                parts.append(str(title))
            for alias in p.meta.get("aliases", []) or []:
                parts.append(str(alias).replace("_", " "))
            # Type/domain disambiguate near-duplicate entries (e.g. the dao_de_jing *text*
            # vs the dao_de_jing_daoist_scripture *figure_source_seat*) for the controller.
            tag = " ".join(str(p.meta.get(k)) for k in ("pageType", "domain") if p.meta.get(k))
            vocab[p.id] = " ".join(parts) + (f" [{tag}]" if tag else "")
    return vocab


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


def _score_routed(routed, target: str, expect: str, asserted: bool) -> str:
    """Score one query given where the controller routed and whether the store asserted.

    assert-expected: correct only if the store asserts the *right* fact; asserting the
    wrong routed fact is ``wrong`` (control-flow error), asserting nothing is ``miss``
    (forgetting / never-learned). abstain-expected: asserting anything is a fabrication.
    """
    if expect == "assert":
        if not asserted:
            return "miss"
        return "correct" if routed == target else "wrong"
    return "correct" if not asserted else "fabrication"


def _aggregate(rows, key: str) -> "dict":
    n = len(rows)
    correct = sum(1 for r in rows if r[key] == "correct")
    fabrication = sum(1 for r in rows if r[key] == "fabrication")
    miss = sum(1 for r in rows if r[key] == "miss")
    wrong = sum(1 for r in rows if r[key] == "wrong")
    return {
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "fabrications": fabrication,
        "fabricationRate": round(fabrication / n, 4) if n else 0.0,
        "misses": miss,
        "missRate": round(miss / n, 4) if n else 0.0,
        "wrong": wrong,
        "wrongRate": round(wrong / n, 4) if n else 0.0,
    }


def run_benchmark(episodes, controller=None) -> "dict[str, Any]":
    """Stream the episodes through both systems, routing each question with ``controller``.

    The controller (the "LLM as control flow" layer) decides which fact a question is
    about; both stores then answer the *same* routed fact, so the only variable between
    systems is the knowledge store. Defaults to ``OracleController`` (perfect routing),
    which isolates the knowledge substrate.
    """
    controller = controller or OracleController()
    vocab = build_vocab(episodes)
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
            routed = controller.route(q.text, vocab, gold=q.target)
            gb_asserted = routed is not None and routed in state
            bl_asserted = bool(baseline and routed is not None and baseline.answer(routed) == "assert")
            rows.append({
                "episode": ep.id,
                "query": q.id,
                "target": q.target,
                "routed": routed,
                "expect": q.expect,
                "type": q.type,
                "graph_backed": _score_routed(routed, q.target, q.expect, gb_asserted),
                "parametric_baseline": _score_routed(routed, q.target, q.expect, bl_asserted),
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
        "controller": controller.name,
        "rows": rows,
    }


def control_flow_report(episodes, controller) -> "dict[str, Any]":
    """Quantify the control-flow error: graph_backed accuracy under a perfect oracle
    router vs under ``controller``. The gap is what the LLM-as-control-flow layer costs.
    """
    oracle = run_benchmark(episodes, OracleController())
    routed = run_benchmark(episodes, controller)
    sub = oracle["systems"]["graph_backed"]["accuracy"]
    end = routed["systems"]["graph_backed"]["accuracy"]
    return {
        "schema": "sophia.continual_qa_control_flow.v1",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "controller": controller.name,
        "substrateAccuracy": sub,                       # oracle routing == knowledge store alone
        "endToEndAccuracy": end,                         # store + this controller
        "controlFlowGap": round(sub - end, 4),           # error introduced by routing
        "routingErrors": [r["query"] for r in routed["rows"] if r["graph_backed"] in ("wrong", "miss")],
        "oracle": oracle["systems"]["graph_backed"],
        "routedSystem": routed["systems"]["graph_backed"],
    }


__all__ = ["Query", "Episode", "load_episodes", "build_vocab", "GraphBackedSystem",
           "ParametricBaseline", "run_benchmark", "control_flow_report"]
