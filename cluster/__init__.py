# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia Cluster — a measured, first-principles model of AI-supercomputer scheduling.

A pure-stdlib sandbox for the questions a 10k→100k-GPU cluster team lives in: how to
place heterogeneous, collective-heavy jobs without scattering them across the network,
how scheduling policy trades utilization against queue latency, and what a node-failure
rate plus a checkpoint cadence actually cost in wasted compute. It turns the repo's
single-pod RunPod tooling (tools/runpod_train.py, the gpu-orchestration skill) into an
analyzable cluster, in the repo's measured-not-claimed style: deterministic, seeded,
and emitted as *.public-report.json.

    from cluster import homogeneous_cluster, synthetic_trace, get_policy, simulate
    cl = homogeneous_cluster(nodes=8, gpus_per_node=8, islands_per_node=2)
    res = simulate(cl, synthetic_trace(n_jobs=200, seed=1), get_policy("backfill-topo"))
    res.utilization, res.wait_p99_s, res.mean_fragmentation

Scope: this is a *simulator* for reasoning about policy and resilience trade-offs, not
a production scheduler and not a claim about any real fleet's numbers. The network-tax
and failure constants are illustrative; the deliverable is the honest machinery and the
shape of the trade-off curves it produces. See docs/11-Platform/Cluster-Engineering-Roadmap.md.
"""

from cluster.topology import (
    Cluster,
    Device,
    Node,
    heterogeneous_cluster,
    homogeneous_cluster,
)
from cluster.job import Job, JobState, synthetic_trace
from cluster.scheduler import (
    BackfillTopo,
    FifoFirstFit,
    Policy,
    TopologyAware,
    fragmentation,
    get_policy,
)
from cluster.simulator import SimResult, simulate
from cluster.observability import straggler_report, summarize
from cluster.faults import Fault, FaultResult, inject_faults, simulate_with_faults

__all__ = [
    "Cluster", "Device", "Node", "homogeneous_cluster", "heterogeneous_cluster",
    "Job", "JobState", "synthetic_trace",
    "Policy", "FifoFirstFit", "TopologyAware", "BackfillTopo", "get_policy", "fragmentation",
    "SimResult", "simulate",
    "summarize", "straggler_report",
    "Fault", "FaultResult", "inject_faults", "simulate_with_faults",
]
