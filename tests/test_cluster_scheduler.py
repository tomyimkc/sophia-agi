#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cluster topology + placement + simulator tests (pure stdlib, deterministic)."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustersim.job import Job, JobState, synthetic_trace
from clustersim.scheduler import (
    fragmentation,
    get_policy,
    pack_topo,
    placement_span,
)
from clustersim.simulator import effective_runtime, simulate
from clustersim.topology import heterogeneous_cluster, homogeneous_cluster


def test_topology_shape() -> None:
    cl = homogeneous_cluster(nodes=4, gpus_per_node=8, islands_per_node=2)
    assert cl.total_gpus == 32
    assert cl.free_count == 32
    # 8 GPUs / 2 islands -> g0-3 island0, g4-7 island1
    assert cl.island_of("n0-g0") == ("n0", 0)
    assert cl.island_of("n0-g7") == ("n0", 1)


def test_allocate_release_roundtrip() -> None:
    cl = homogeneous_cluster(nodes=2, gpus_per_node=4)
    cl.allocate(["n0-g0", "n0-g1"])
    assert cl.free_count == 6
    assert abs(cl.utilization - 0.25) < 1e-9
    cl.release(["n0-g0", "n0-g1"])
    assert cl.free_count == 8


def test_topo_packing_prefers_single_island() -> None:
    cl = homogeneous_cluster(nodes=2, gpus_per_node=8, islands_per_node=2)
    job = Job(id="j", gpus=4, duration_s=100, submit_s=0)
    devs = pack_topo(cl, job)
    assert devs is not None and len(devs) == 4
    nodes, islands = placement_span(cl, devs)
    assert islands == 1  # a 4-GPU job fits one 4-GPU island → zero fragmentation
    assert fragmentation(cl, devs) == 0.0


def test_topo_packing_spills_minimally() -> None:
    cl = homogeneous_cluster(nodes=1, gpus_per_node=8, islands_per_node=2)
    # occupy one full island so an 8-GPU... use a 6-GPU job that must spill across both islands
    cl.allocate(["n0-g0", "n0-g1"])  # 2 gone from island0, 2 free there + 4 in island1
    job = Job(id="j", gpus=6, duration_s=100, submit_s=0)
    devs = pack_topo(cl, job)
    assert devs is not None and len(devs) == 6
    _, islands = placement_span(cl, devs)
    assert islands == 2  # forced to use both islands, but no more


def test_firstfit_scatters_more_than_topo() -> None:
    # A trace of multi-GPU colocate jobs: topo should keep fragmentation <= firstfit.
    trace = synthetic_trace(n_jobs=120, seed=5)
    frag = {}
    for name in ("fifo-firstfit", "topology-aware"):
        cl = homogeneous_cluster(nodes=8, gpus_per_node=8, islands_per_node=2)
        res = simulate(cl, deepcopy(trace), get_policy(name))
        frag[name] = res.mean_fragmentation
    assert frag["topology-aware"] <= frag["fifo-firstfit"] + 1e-9


def test_network_tax_penalizes_scatter() -> None:
    job = Job(id="j", gpus=8, duration_s=1000, submit_s=0, colocate=True)
    one_island = effective_runtime(job, node_span=1, island_span=1)
    scattered = effective_runtime(job, node_span=4, island_span=8)
    assert one_island == 1000.0
    assert scattered > one_island
    # non-colocate (eval) jobs pay no tax
    ev = Job(id="e", gpus=8, duration_s=1000, submit_s=0, colocate=False)
    assert effective_runtime(ev, node_span=4, island_span=8) == 1000.0


def test_simulator_conserves_jobs_and_runs() -> None:
    cl = homogeneous_cluster(nodes=4, gpus_per_node=8, islands_per_node=2)
    trace = synthetic_trace(n_jobs=80, seed=2)
    res = simulate(cl, trace, get_policy("backfill-topo"))
    assert res.completed == res.n_jobs == 80  # finite trace, all drain
    assert 0.0 < res.utilization <= 1.0
    assert res.wait_p99_s >= res.wait_p50_s >= 0.0
    assert res.makespan_s > 0
    # every job ends after it starts after it was submitted
    for j in trace:
        assert j.state is JobState.DONE
        assert j.submit_s <= j.start_s <= j.end_s


def test_backfill_cuts_queue_latency() -> None:
    # Isolate the backfill effect: same topology-aware packing, with vs without
    # backfilling small jobs past a blocked head-of-line job. Backfill should
    # substantially reduce mean queue wait (small jobs stop starving behind big ones).
    trace = synthetic_trace(n_jobs=150, seed=9)
    wait = {}
    for name in ("topology-aware", "backfill-topo"):
        cl = homogeneous_cluster(nodes=6, gpus_per_node=8, islands_per_node=2)
        wait[name] = simulate(cl, deepcopy(trace), get_policy(name)).wait_mean_s
    assert wait["backfill-topo"] < wait["topology-aware"]


def test_heterogeneous_cluster_and_class_pinning() -> None:
    cl = heterogeneous_cluster([
        {"nodes": 1, "gpus_per_node": 4, "vram_gb": 80, "klass": "H100"},
        {"nodes": 1, "gpus_per_node": 4, "vram_gb": 64, "klass": "domestic-x1"},
    ])
    assert cl.total_gpus == 8
    klasses = {d.klass for d in cl.gpus()}
    assert klasses == {"H100", "domestic-x1"}
    # a job pinned to a class only lands on that class
    job = Job(id="j", gpus=2, duration_s=10, submit_s=0, klass="domestic-x1")
    devs = pack_topo(cl, job)
    assert devs is not None
    assert all(cl.device(d).klass == "domestic-x1" for d in devs)


def test_determinism() -> None:
    def run():
        cl = homogeneous_cluster(nodes=4, gpus_per_node=8, islands_per_node=2)
        return simulate(cl, synthetic_trace(n_jobs=60, seed=11), get_policy("topology-aware")).as_dict()
    assert run() == run()


def main() -> int:
    test_topology_shape()
    test_allocate_release_roundtrip()
    test_topo_packing_prefers_single_island()
    test_topo_packing_spills_minimally()
    test_firstfit_scatters_more_than_topo()
    test_network_tax_penalizes_scatter()
    test_simulator_conserves_jobs_and_runs()
    test_backfill_cuts_queue_latency()
    test_heterogeneous_cluster_and_class_pinning()
    test_determinism()
    print("test_cluster_scheduler: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
