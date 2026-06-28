#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fault-injection + checkpoint/restart recovery tests."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustersim.faults import Fault, by_id_dev, inject_faults, simulate_with_faults
from clustersim.job import JobState, synthetic_trace
from clustersim.scheduler import get_policy
from clustersim.topology import homogeneous_cluster


def test_dev_node_membership() -> None:
    assert by_id_dev("n3-g5", "n3") is True
    assert by_id_dev("n3-g5", "n30") is False
    assert by_id_dev("n12-g0", "n12") is True


def test_inject_faults_reproducible_and_in_window() -> None:
    ids = [f"n{i}" for i in range(8)]
    a = inject_faults(ids, seed=4, mtbf_s=300, horizon_s=3600)
    b = inject_faults(ids, seed=4, mtbf_s=300, horizon_s=3600)
    assert [f.__dict__ for f in a] == [f.__dict__ for f in b]  # deterministic
    assert all(0 <= f.at_s < 3600 for f in a)
    assert all(f.node_id in ids for f in a)


def test_no_faults_matches_clean_completion() -> None:
    cl = homogeneous_cluster(nodes=4, gpus_per_node=8, islands_per_node=2)
    trace = synthetic_trace(n_jobs=60, seed=2, horizon_s=1800)
    res = simulate_with_faults(cl, trace, get_policy("backfill-topo"), faults=[])
    assert res.n_faults == 0
    assert res.total_restarts == 0
    assert res.wasted_gpu_seconds == 0.0
    assert res.completed == res.n_jobs
    # No failures → no failure-waste, but busy time still exceeds nominal useful work by
    # the network tax of scattered collective jobs, so goodput <= raw utilization.
    assert res.goodput <= res.raw_utilization + 1e-9

    # With the network tax zeroed, busy == useful, so goodput == raw utilization exactly.
    cl0 = homogeneous_cluster(nodes=4, gpus_per_node=8, islands_per_node=2)
    res0 = simulate_with_faults(cl0, synthetic_trace(n_jobs=60, seed=2, horizon_s=1800),
                                get_policy("backfill-topo"), faults=[],
                                island_tax=0.0, node_tax=0.0)
    assert abs(res0.goodput - res0.raw_utilization) < 1e-6


def test_failures_cause_restarts_and_waste() -> None:
    cl = homogeneous_cluster(nodes=6, gpus_per_node=8, islands_per_node=2)
    trace = synthetic_trace(n_jobs=80, seed=3, horizon_s=2400)
    faults = inject_faults([n.id for n in cl.nodes], seed=3, mtbf_s=200, horizon_s=2400)
    assert len(faults) > 0
    res = simulate_with_faults(cl, deepcopy(trace), get_policy("backfill-topo"), faults,
                               checkpoint_s=300, recovery_s=60)
    assert res.total_restarts >= 1
    assert res.wasted_gpu_seconds > 0.0
    # goodput must be <= raw utilization: some busy time was doomed work
    assert res.goodput <= res.raw_utilization + 1e-9
    assert 0.0 <= res.wasted_fraction <= 1.0


def test_frequent_checkpoints_reduce_waste() -> None:
    # Shorter checkpoint interval loses less work per failure → less wasted compute.
    cl0 = homogeneous_cluster(nodes=6, gpus_per_node=8, islands_per_node=2)
    trace = synthetic_trace(n_jobs=80, seed=7, horizon_s=2400)
    faults = inject_faults([n.id for n in cl0.nodes], seed=7, mtbf_s=180, horizon_s=2400)

    def waste(ckpt):
        cl = homogeneous_cluster(nodes=6, gpus_per_node=8, islands_per_node=2)
        return simulate_with_faults(cl, deepcopy(trace), get_policy("backfill-topo"),
                                    list(faults), checkpoint_s=ckpt, recovery_s=60).wasted_gpu_seconds

    assert waste(60) <= waste(1200) + 1e-6  # frequent checkpoints waste no more


def test_recovered_jobs_eventually_complete() -> None:
    cl = homogeneous_cluster(nodes=8, gpus_per_node=8, islands_per_node=2)
    trace = synthetic_trace(n_jobs=50, seed=1, horizon_s=1800)
    faults = [Fault(at_s=300.0, node_id="n0"), Fault(at_s=600.0, node_id="n1")]
    res = simulate_with_faults(cl, trace, get_policy("backfill-topo"), faults,
                               checkpoint_s=120, recovery_s=30)
    done = [j for j in trace if j.state is JobState.DONE]
    # With only 2 isolated node losses and plenty of spare capacity, all jobs finish.
    assert len(done) == 50
    restarted = [j for j in trace if j.restarts > 0]
    for j in restarted:
        assert j.committed_s >= j.duration_s - 1e-6  # finished work was fully re-done


def main() -> int:
    test_dev_node_membership()
    test_inject_faults_reproducible_and_in_window()
    test_no_faults_matches_clean_completion()
    test_failures_cause_restarts_and_waste()
    test_frequent_checkpoints_reduce_waste()
    test_recovered_jobs_eventually_complete()
    print("test_cluster_faults: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
