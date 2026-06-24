#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.continual_retention — sequential-retention over the OKF graph.

Proves the core thesis: an additive stream of OKF pages forgets nothing (the
retention matrix is all 1.0, ``forgottenGroundedClaims == 0``), while the metric is
sensitive enough to *detect* forgetting when a fact loses its provenance ground or
its confidence is weakened. Dependency-free, offline, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_retention import Snapshot, build_report, run_stream, Task  # noqa: E402
from okf.page import Page  # noqa: E402


def _page(pid: str, **meta) -> Page:
    meta = {"id": pid, "pageType": "concept", **meta}
    return Page(path=Path(f"{pid}.md"), meta=meta)


def _additive_tasks() -> "list[Task]":
    # Three tasks, each adding self-grounded (primary) facts. Nothing overlaps.
    return [
        Task("t1", (_page("a", authorConfidence="consensus"), _page("b", authorConfidence="attributed"))),
        Task("t2", (_page("c", authorConfidence="attributed"),)),
        Task("t3", (_page("d", authorConfidence="consensus"), _page("e", authorConfidence="attributed"))),
    ]


def test_additive_stream_forgets_nothing() -> None:
    report = run_stream(_additive_tasks())
    assert report["forgottenGroundedClaims"] == 0
    assert report["perfectRetention"] is True
    assert report["backwardTransfer"] == 0.0
    assert report["totalGroundedFacts"] == 5


def test_retention_matrix_is_complete_and_full() -> None:
    report = run_stream(_additive_tasks())
    matrix = report["retentionMatrix"]
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if j < i:
                assert matrix[i][j] is None          # cannot evaluate before learning
            else:
                assert matrix[i][j] == 1.0           # fully retained, additive stream


def test_derived_fact_retained_across_tasks() -> None:
    # task1 introduces a primary; task2 adds a fact deriving from it — both stay grounded.
    tasks = [
        Task("t1", (_page("primary", authorConfidence="consensus"),)),
        Task("t2", (_page("derived", derivesFrom=["primary"], authorConfidence="attributed"),)),
    ]
    report = run_stream(tasks)
    assert report["forgottenGroundedClaims"] == 0
    assert "derived" in report["factsPerTask"]["t2"]


def test_metric_detects_lost_grounding() -> None:
    # Hand-built snapshots: 'x' is grounded after t1, then gone after t2 (an overwrite
    # / retraction). The benchmark must catch this, proving it is not vacuously zero.
    snaps = [
        Snapshot("t1", {"x": 3, "y": 2}, ("x", "y")),
        Snapshot("t2", {"y": 2, "z": 4}, ("z",)),
    ]
    report = build_report(snaps)
    assert report["forgottenGroundedClaims"] == 1
    assert report["perfectRetention"] is False
    detail = report["forgottenDetail"][0]
    assert detail == {"fact": "x", "introducedInTask": "t1", "reason": "lost_grounding"}
    assert report["backwardTransfer"] < 0.0          # negative == forgetting


def test_metric_detects_confidence_drop() -> None:
    # 'x' survives but its effective confidence is laundered down — also forgetting.
    snaps = [
        Snapshot("t1", {"x": 4}, ("x",)),
        Snapshot("t2", {"x": 1}, ()),
    ]
    report = build_report(snaps)
    assert report["forgottenGroundedClaims"] == 1
    assert report["forgottenDetail"][0]["reason"] == "confidence_dropped"
    assert report["forgottenDetail"][0]["from"] == 4
    assert report["forgottenDetail"][0]["to"] == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
