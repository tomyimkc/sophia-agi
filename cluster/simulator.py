# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Discrete-event cluster simulator — replays a job trace under a placement policy.

This is the measured core: feed it a Cluster (cluster/topology.py), a job trace
(cluster/job.py), and a Policy (cluster/scheduler.py); it advances an event clock
(arrivals + completions), asks the policy to place pending jobs whenever resources
change, and records the trade-off the JD cares about:

  utilization   — GPU-seconds used / GPU-seconds available over the makespan
  queue latency — wait_s = start - submit, reported as p50 / p99 (tail)
  throughput    — jobs completed per hour
  fragmentation — mean island-span penalty, i.e. how much locality the policy preserved

Network tax: a collective-heavy (`colocate`) job placed across N islands runs slower —
all-reduce traffic that would stay on NVLink now crosses the NIC. effective_runtime =
duration * (1 + island_tax*(islands-1) + node_tax*(nodes-1)). So a policy that scatters
jobs is punished with longer runtimes *and* worse utilization, which is physically honest.

Deterministic given (cluster, trace, policy) — no wall-clock, no unseeded RNG.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from cluster.job import Job, JobState
from cluster.observability import summarize
from cluster.scheduler import Policy, fragmentation, placement_span
from cluster.topology import Cluster


# Network-tax coefficients (fraction of runtime added per extra island / node hop for
# collective-heavy jobs). These are the FALLBACK defaults used only when no calibration
# file is present; cluster/netcalib.json (written by tools/calibrate_network_tax.py)
# overrides them with coefficients derived from all-reduce bandwidth. Overridable per run.
ISLAND_TAX = 0.06
NODE_TAX = 0.12


def calibrated_taxes() -> tuple[float, float]:
    """(island_tax, node_tax) from cluster/netcalib.json if present, else the fallbacks."""
    from cluster.netcalib import load_calibration

    calib = load_calibration()
    if calib is None:
        return ISLAND_TAX, NODE_TAX
    return calib.island_tax, calib.node_tax


def effective_runtime(job: Job, node_span: int, island_span: int,
                      island_tax: float = ISLAND_TAX, node_tax: float = NODE_TAX) -> float:
    """Runtime inflated by the *slowest* link the collective is forced to cross.

    Worst-tier (not linear-in-span): a ring all-reduce runs at the speed of its bottleneck
    hop, so the penalty is set once by the coarsest boundary the placement straddles —
    cross-node (NIC) dominates cross-island (NVSwitch) dominates intra-island (NVLink, free).
    This matches the per-tier semantics of cluster/netcalib.py, so calibrated coefficients
    plug in directly. Non-collective (eval) jobs pay nothing.
    """
    if not job.colocate:
        return job.duration_s
    if node_span > 1:
        tax = node_tax
    elif island_span > 1:
        tax = island_tax
    else:
        tax = 0.0
    return job.duration_s * (1.0 + tax)


@dataclass
class SimResult:
    policy: str
    n_jobs: int
    completed: int
    makespan_s: float
    utilization: float
    busy_gpu_seconds: float
    capacity_gpu_seconds: float
    throughput_jobs_per_hr: float
    wait_p50_s: float
    wait_p99_s: float
    wait_mean_s: float
    mean_fragmentation: float
    mean_network_tax: float          # mean effective/nominal runtime ratio for colocate jobs
    jobs: list[Job] = field(default_factory=list)

    def as_dict(self, with_jobs: bool = False) -> dict:
        d = {
            "policy": self.policy,
            "n_jobs": self.n_jobs,
            "completed": self.completed,
            "makespan_s": round(self.makespan_s, 1),
            "utilization": round(self.utilization, 4),
            "throughput_jobs_per_hr": round(self.throughput_jobs_per_hr, 2),
            "wait_p50_s": round(self.wait_p50_s, 1),
            "wait_p99_s": round(self.wait_p99_s, 1),
            "wait_mean_s": round(self.wait_mean_s, 1),
            "mean_fragmentation": round(self.mean_fragmentation, 4),
            "mean_network_tax": round(self.mean_network_tax, 4),
        }
        if with_jobs:
            d["jobs"] = [
                {
                    "id": j.id, "gpus": j.gpus, "submit_s": j.submit_s,
                    "start_s": j.start_s, "end_s": j.end_s,
                    "wait_s": round(j.wait_s, 1) if j.wait_s is not None else None,
                    "node_span": j.node_span, "island_span": j.island_span,
                    "restarts": j.restarts, "state": j.state.value,
                }
                for j in self.jobs
            ]
        return d


# Event types in the priority queue. Tie-break: completions (0) before arrivals (1) at
# the same timestamp so freed GPUs are visible to that instant's scheduling pass.
_COMPLETE, _ARRIVE = 0, 1


def simulate(
    cluster: Cluster,
    trace: list[Job],
    policy: Policy,
    *,
    island_tax: float | None = None,
    node_tax: float | None = None,
) -> SimResult:
    """Run `trace` on `cluster` under `policy`. Mutates the jobs' runtime state.

    island_tax/node_tax default to the calibrated coefficients (cluster/netcalib.json)
    when None, so a fresh calibration automatically flows into the simulation.
    """
    if island_tax is None or node_tax is None:
        ci, cn = calibrated_taxes()
        island_tax = ci if island_tax is None else island_tax
        node_tax = cn if node_tax is None else node_tax
    jobs = trace
    by_id = {j.id: j for j in jobs}
    pending: list[str] = []
    running: list[str] = []

    events: list[tuple[float, int, int, str]] = []
    for seq, j in enumerate(jobs):
        j.state = JobState.PENDING
        j.start_s = j.end_s = None
        heapq.heappush(events, (j.submit_s, _ARRIVE, seq, j.id))

    busy_gpu_seconds = 0.0
    last_t = 0.0
    makespan = 0.0

    def _accrue(now: float) -> None:
        nonlocal busy_gpu_seconds, last_t
        busy = cluster.total_gpus - cluster.free_count
        busy_gpu_seconds += busy * max(0.0, now - last_t)
        last_t = now

    def _try_schedule(now: float) -> None:
        if not pending:
            return
        waiting = [by_id[jid] for jid in pending]
        decisions = policy.schedule(cluster, waiting, now)
        for dec in decisions:
            job = dec.job
            node_span, island_span = placement_span(cluster, dec.devices)
            eff = effective_runtime(job, node_span, island_span, island_tax, node_tax)
            job.state = JobState.RUNNING
            job.start_s = now
            job.end_s = now + eff
            job.devices = dec.devices
            job.node_span = node_span
            job.island_span = island_span
            pending.remove(job.id)
            running.append(job.id)
            seq = list(by_id).index(job.id)
            heapq.heappush(events, (job.end_s, _COMPLETE, seq, job.id))

    while events:
        now, kind, _seq, jid = heapq.heappop(events)
        _accrue(now)
        makespan = max(makespan, now)
        job = by_id[jid]
        if kind == _ARRIVE:
            pending.append(jid)
            _try_schedule(now)
        else:  # _COMPLETE
            if job.state is not JobState.RUNNING or abs((job.end_s or 0) - now) > 1e-6:
                continue  # stale completion (e.g. job was restarted by the fault model)
            cluster.release(job.devices)
            running.remove(jid)
            job.state = JobState.DONE
            _try_schedule(now)  # freed GPUs may unblock the queue

    completed = [j for j in jobs if j.state is JobState.DONE]
    waits = [j.wait_s for j in completed if j.wait_s is not None]
    colo = [j for j in completed if j.colocate and j.gpus > 1]
    frags = [fragmentation(cluster, j.devices) if j.devices else 0.0 for j in colo]
    taxes = [
        effective_runtime(j, j.node_span, j.island_span, island_tax, node_tax) / j.duration_s
        for j in colo
    ]
    capacity = cluster.total_gpus * makespan if makespan else 0.0
    wsum = summarize(waits)

    return SimResult(
        policy=policy.name,
        n_jobs=len(jobs),
        completed=len(completed),
        makespan_s=makespan,
        utilization=(busy_gpu_seconds / capacity) if capacity else 0.0,
        busy_gpu_seconds=busy_gpu_seconds,
        capacity_gpu_seconds=capacity,
        throughput_jobs_per_hr=(len(completed) / makespan * 3600.0) if makespan else 0.0,
        wait_p50_s=wsum.p50,
        wait_p99_s=wsum.p99,
        wait_mean_s=wsum.mean,
        mean_fragmentation=(sum(frags) / len(frags)) if frags else 0.0,
        mean_network_tax=(sum(taxes) / len(taxes)) if taxes else 1.0,
        jobs=jobs,
    )
