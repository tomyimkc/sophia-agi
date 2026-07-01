#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Throughput / wall-clock scheduling simulation tests.

Covers: determinism, monotonic speedup + decreasing efficiency, the Mac-judge contention effect,
that job distribution is the SAME assignment `cluster_scheduler.assigned_node` produces, and flat
scaling once node count exceeds job count. Pure, offline, deterministic — no GPU, no network, no
clock. The sim schedules the owner's GPU-time ESTIMATES; it makes no hardware claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cluster_scheduler import assigned_node  # noqa: E402
from tools.cluster_schedule_sim import (  # noqa: E402
    DEFAULT_NODE_COUNTS,
    Job,
    _node_ids,
    forecast_jobs,
    offline_invariants,
    report,
    scaling_table,
    simulate,
    sweep_jobs,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_determinism() -> None:
    jobs = forecast_jobs()
    for n in DEFAULT_NODE_COUNTS:
        assert simulate(jobs, n) == simulate(jobs, n)
    # the whole table reproduces byte-for-byte
    assert scaling_table(jobs, DEFAULT_NODE_COUNTS) == scaling_table(jobs, DEFAULT_NODE_COUNTS)
    assert report(jobs, DEFAULT_NODE_COUNTS) == report(jobs, DEFAULT_NODE_COUNTS)


def test_assignment_is_from_cluster_scheduler() -> None:
    """Every job's owning node in the sim is exactly assigned_node(job.id, node_ids) — the sim does
    not re-implement assignment, it reuses the deterministic single-owner hash."""
    jobs = forecast_jobs()
    for n in (2, 4, 8, 16):
        nodes = _node_ids(n)
        res = simulate(jobs, n)
        owners = {jid: owner for owner, jids in res["assignment"].items() for jid in jids}
        assert len(owners) == len(jobs)  # single owner per job, nothing dropped
        for j in jobs:
            assert owners[j.id] == assigned_node(j.id, nodes)


def test_monotonic_speedup_and_decreasing_efficiency() -> None:
    jobs = forecast_jobs()
    rows = scaling_table(jobs, DEFAULT_NODE_COUNTS)
    speeds = [r["speedupVs1"] for r in rows]
    effs = [r["efficiency"] for r in rows]
    assert speeds[0] == 1.0 and effs[0] == 1.0
    # speedup non-decreasing, efficiency non-increasing
    for i in range(len(rows) - 1):
        assert speeds[i] <= speeds[i + 1] + 1e-9, (i, speeds)
        assert effs[i] >= effs[i + 1] - 1e-9, (i, effs)
    # all speedups at least 1x (never slower than a single node)
    assert all(s >= 1.0 - 1e-9 for s in speeds)


def test_mac_judge_contention_raises_wallclock() -> None:
    """With >1 judge job and concurrency=1 the judge serializes => more wall-clock + judge wait than
    a generous-concurrency run on the same judge-heavy workload."""
    judge_heavy = sweep_jobs(seeds=4, disciplines=2, judged=True)  # 8 judged jobs
    serial = simulate(judge_heavy, 8, mac_judge_concurrency=1)
    parallel = simulate(judge_heavy, 8, mac_judge_concurrency=8)
    assert serial["wallClockMinutes"] > parallel["wallClockMinutes"]
    assert serial["macJudgeWaitMinutes"] > 0.0
    assert parallel["macJudgeWaitMinutes"] == 0.0
    assert serial["bottleneck"] == "mac-judge"


def test_fully_judged_workload_is_judge_bound() -> None:
    """A workload where EVERY job needs the judge, concurrency=1: the single judge serializes all of
    it, so speedup stays ~1x no matter how many nodes — the Mac judge is the ceiling."""
    judged = sweep_jobs(seeds=3, disciplines=2, judged=True)  # 6 judged jobs
    rows = scaling_table(judged, (1, 2, 4, 8), mac_judge_concurrency=1)
    walls = {r["nNodes"]: r["wallClockMinutes"] for r in rows}
    # wall-clock does not improve past 1 node (judge is fully serialized)
    assert walls[8] == walls[1]
    assert all(r["speedupVs1"] <= 1.0 + 1e-9 for r in rows)
    assert all(r["bottleneck"] == "mac-judge" for r in rows if r["nNodes"] > 1)


def test_independent_nomac_scales_then_flat() -> None:
    """A pure independent no-mac workload scales (more nodes => more parallelism) until jobs < nodes,
    then is flat (no extra job to place)."""
    indep = sweep_jobs(seeds=4, disciplines=2, judged=False)  # 8 independent no-mac jobs
    rows = {r["nNodes"]: r for r in scaling_table(indep, (1, 2, 4, 8, 16), mac_judge_concurrency=1)}
    # scaling improves up to the job count
    assert rows[8]["speedupVs1"] > rows[4]["speedupVs1"] + 1e-9
    assert rows[4]["speedupVs1"] > rows[2]["speedupVs1"] + 1e-9
    # 16 nodes for 8 jobs: identical wall-clock to 8 nodes (flat past job count)
    assert rows[16]["wallClockMinutes"] == rows[8]["wallClockMinutes"]
    # never any judge wait in a no-mac workload, never judge-bound
    for r in rows.values():
        assert r["macJudgeWaitMinutes"] == 0.0
        assert r["bottleneck"] == "compute"


def test_forecast_queue_shape() -> None:
    """The seeded forecast queue carries the T1-T4 jobs (their kinds + the one mac-judge gate) plus
    an independent sweep batch."""
    jobs = forecast_jobs()
    by_id = {j.id: j for j in jobs}
    assert {"T1-nvfp4-cert", "T2-faithfulness", "T3-sophrosyne-virtues", "T4-council-train"} <= set(by_id)
    assert by_id["T1-nvfp4-cert"].kind == "cert" and not by_id["T1-nvfp4-cert"].needs_mac_judge
    assert by_id["T3-sophrosyne-virtues"].needs_mac_judge  # the virtue judge gate routes to the Mac
    assert not by_id["T4-council-train"].needs_mac_judge   # LoRA train does not judge
    # has more than the 4 queue jobs (the independent batch was added)
    assert len(jobs) > 4


def test_job_is_immutable_estimate() -> None:
    """Job is a frozen dataclass — gpu_minutes is the owner's estimate, not mutated by the sim."""
    j = Job("x", "cert", 15.0, needs_mac_judge=False)
    raised = False
    try:
        j.gpu_minutes = 99.0  # type: ignore[misc]
    except Exception:
        raised = True
    assert raised


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} cluster_schedule_sim tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
