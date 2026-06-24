# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sequential-retention benchmark over the OKF belief graph.

Measures catastrophic forgetting of *declarative* knowledge when that knowledge is
stored non-parametrically as OKF pages instead of in shared weights. A **task** is a
batch of pages; learning the stream means adding batches in sequence and rebuilding
the belief graph after each one. Because adding page *N+1* cannot mutate page *N*, a
purely additive stream forgets nothing — this module turns that structural claim into
a *measured* one (the "forgetting is hard to even measure" problem), and is sensitive
enough to detect forgetting when knowledge actually is removed or weakened.

A "fact" is a page id; it is **retained** at a later step if it is still grounded
(``okf.is_grounded``) and its effective confidence rank (min-over-``derivesFrom``
chain) has not dropped below what it was when introduced. The headline metric is
``forgottenGroundedClaims`` — for an additive stream it is 0.

    from agent.continual_retention import Task, run_stream
    report = run_stream([Task("t1", pages_a), Task("t2", pages_b)])
    report["forgottenGroundedClaims"]   # 0 for additive learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from okf import build_graph, is_grounded, propagate_confidence


@dataclass(frozen=True)
class Task:
    """One step of the continual stream: a batch of OKF ``Page`` objects to learn."""

    id: str
    pages: tuple = ()


@dataclass(frozen=True)
class Snapshot:
    """Belief state after learning a task: grounded fact -> effective confidence rank,
    plus the ids this task introduced."""

    task_id: str
    grounded: dict  # fact id -> effective confidence rank (only grounded facts)
    introduced: tuple


def belief_state(graph) -> "dict":
    """Grounded facts and their effective (min-over-chain) confidence rank.

    A fact absent from this mapping is *not currently assertable* — it either was
    never added or lost its provenance ground. That is the fail-closed reading the
    runtime gate uses, so retention is measured against it.
    """
    conf = propagate_confidence(graph)
    return {nid: conf.get(nid, 0) for nid in graph.nodes if is_grounded(graph, nid)}


def stream_snapshots(tasks) -> "list[Snapshot]":
    """Replay the stream, snapshotting the belief state after each cumulative task."""
    cumulative: list = []
    snapshots: list[Snapshot] = []
    for task in tasks:
        cumulative.extend(task.pages)
        graph = build_graph(cumulative)
        state = belief_state(graph)
        introduced = tuple(p.id for p in task.pages)
        snapshots.append(Snapshot(task_id=task.id, grounded=state, introduced=introduced))
    return snapshots


def _origin_confidence(snapshots: "list[Snapshot]") -> "dict":
    """For each introduced fact, the effective confidence it had when introduced."""
    origin: dict = {}
    for snap in snapshots:
        for fid in snap.introduced:
            if fid not in origin and fid in snap.grounded:
                origin[fid] = snap.grounded[fid]
    return origin


def build_report(snapshots: "list[Snapshot]") -> "dict[str, Any]":
    """Compute the retention matrix and forgetting metrics from stream snapshots.

    ``retentionMatrix[i][j]`` (for j >= i) = fraction of the *grounded* facts that
    task *i* introduced that are still retained after task *j*. Cells with j < i are
    ``None`` (a task cannot be evaluated before it is learned).
    """
    n = len(snapshots)
    origin = _origin_confidence(snapshots)

    # Grounded facts each task contributed (a page that arrived ungrounded is not a fact).
    facts_per_task: list[list[str]] = [
        sorted(fid for fid in snap.introduced if fid in origin) for snap in snapshots
    ]

    def retained(fid: str, state: dict) -> bool:
        return fid in state and state[fid] >= origin[fid]

    matrix: list[list] = []
    for i in range(n):
        row: list = [None] * n
        facts = facts_per_task[i]
        for j in range(i, n):
            state = snapshots[j].grounded
            if not facts:
                row[j] = 1.0  # vacuously retained
            else:
                row[j] = round(sum(retained(f, state) for f in facts) / len(facts), 4)
        matrix.append(row)

    # Per-fact forgetting at the final snapshot.
    final = snapshots[-1].grounded if snapshots else {}
    forgotten: list[dict] = []
    for i, facts in enumerate(facts_per_task):
        for fid in facts:
            if fid not in final:
                forgotten.append({"fact": fid, "introducedInTask": snapshots[i].task_id, "reason": "lost_grounding"})
            elif final[fid] < origin[fid]:
                forgotten.append({
                    "fact": fid, "introducedInTask": snapshots[i].task_id, "reason": "confidence_dropped",
                    "from": origin[fid], "to": final[fid],
                })

    # Backward transfer: mean over earlier tasks of (retention_at_end - 1.0).
    # 0.0 == nothing forgotten; negative == forgetting.
    deltas = [matrix[i][n - 1] - 1.0 for i in range(n - 1) if facts_per_task[i]]
    backward_transfer = round(sum(deltas) / len(deltas), 4) if deltas else 0.0

    total_facts = sum(len(f) for f in facts_per_task)
    return {
        "schema": "sophia.continual_retention.v1",
        "level3Evidence": False,
        "tasks": [s.task_id for s in snapshots],
        "factsPerTask": {snapshots[i].task_id: facts_per_task[i] for i in range(n)},
        "totalGroundedFacts": total_facts,
        "retentionMatrix": matrix,
        "forgottenGroundedClaims": len(forgotten),
        "forgottenDetail": forgotten,
        "backwardTransfer": backward_transfer,
        "perfectRetention": len(forgotten) == 0,
    }


def run_stream(tasks) -> "dict[str, Any]":
    """End-to-end: replay the task stream and return the retention report."""
    return build_report(stream_snapshots(tasks))


def write_report(tasks, out) -> "dict[str, Any]":
    """Run the benchmark and persist the JSON report."""
    import json
    from pathlib import Path

    report = run_stream(tasks)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = ["Task", "Snapshot", "belief_state", "stream_snapshots", "build_report", "run_stream", "write_report"]
