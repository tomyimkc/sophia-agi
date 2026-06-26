# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fault injection + checkpoint/restart recovery — the resilience half of the JD.

"故障发现与自动容灾 … 大规模训练下的故障诊断" at 10k+ GPUs is dominated by one fact:
the bigger the job, the more likely *some* node dies mid-step, and a synchronous
training job dies with it. The only defence is checkpoint/restart, and its cost is
the work done since the last checkpoint — wasted, re-run, multiplied across the fleet.

This simulator injects node failures into a trace and models job-level auto-recovery:

  * progress is committed only at checkpoints (every `checkpoint_s` wall-seconds);
  * a node failure kills every job touching it; each loses work back to its last
    checkpoint (`wasted_s`), is requeued, and pays `recovery_s` (reschedule + reload);
  * the report separates raw busy time from *goodput* (useful work that actually
    survived) and quantifies the wasted-compute tax of a given MTBF + checkpoint cadence.

Use it to answer: at this failure rate, what checkpoint interval maximizes goodput?
— the canonical large-scale-training resilience trade-off. Deterministic given a seed.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

from cluster.job import Job, JobState
from cluster.observability import summarize
from cluster.scheduler import Policy, placement_span
from cluster.simulator import effective_runtime


@dataclass(frozen=True)
class Fault:
    at_s: float
    node_id: str


def inject_faults(node_ids: list[str], *, seed: int, mtbf_s: float, horizon_s: float) -> list[Fault]:
    """Sample node failures as a Poisson process with mean time-between-failures `mtbf_s`.

    Returns a time-sorted list of (at_s, node_id). A real fleet's failure rate scales
    with node count; pass a per-fleet mtbf_s already reflecting that.
    """
    rng = random.Random(seed)
    faults: list[Fault] = []
    t = 0.0
    while True:
        t += rng.expovariate(1.0 / mtbf_s)
        if t >= horizon_s or not node_ids:
            break
        faults.append(Fault(at_s=round(t, 1), node_id=rng.choice(node_ids)))
    return faults


@dataclass
class FaultResult:
    policy: str
    n_jobs: int
    completed: int
    n_faults: int
    makespan_s: float
    raw_utilization: float           # busy GPU-seconds (incl. doomed work) / capacity
    goodput: float                   # useful (surviving) GPU-seconds / capacity
    wasted_gpu_seconds: float
    wasted_fraction: float           # wasted / busy — the failure tax
    total_restarts: int
    recovery_p50_s: float
    recovery_p99_s: float
    checkpoint_s: float
    jobs: list[Job] = field(default_factory=list)

    def as_dict(self, with_jobs: bool = False) -> dict:
        d = {
            "policy": self.policy,
            "n_jobs": self.n_jobs,
            "completed": self.completed,
            "n_faults": self.n_faults,
            "makespan_s": round(self.makespan_s, 1),
            "raw_utilization": round(self.raw_utilization, 4),
            "goodput": round(self.goodput, 4),
            "wasted_gpu_seconds": round(self.wasted_gpu_seconds, 1),
            "wasted_fraction": round(self.wasted_fraction, 4),
            "total_restarts": self.total_restarts,
            "recovery_p50_s": round(self.recovery_p50_s, 1),
            "recovery_p99_s": round(self.recovery_p99_s, 1),
            "checkpoint_s": self.checkpoint_s,
        }
        if with_jobs:
            d["jobs"] = [
                {"id": j.id, "gpus": j.gpus, "state": j.state.value,
                 "restarts": j.restarts, "wasted_s": round(j.wasted_s, 1)}
                for j in self.jobs
            ]
        return d


_COMPLETE, _FAIL, _ARRIVE = 0, 1, 2  # tie-break at equal t: completions, then faults, then arrivals


def simulate_with_faults(
    cluster,
    trace: list[Job],
    policy: Policy,
    faults: list[Fault],
    *,
    checkpoint_s: float = 300.0,
    recovery_s: float = 60.0,
    island_tax: float | None = None,
    node_tax: float | None = None,
) -> FaultResult:
    """Replay `trace` under `policy` while `faults` kill nodes; recover via checkpoints.

    Progress model: a running job commits nominal work at each `checkpoint_s` wall-second
    boundary. On a failure that touches it, it loses the uncommitted tail (`wasted_s`),
    is requeued, and its next run starts after `recovery_s`. `committed_s` carries the
    durable progress across restarts so re-runs only redo the remaining nominal work.
    """
    from cluster.simulator import calibrated_taxes

    ci, cn = calibrated_taxes()
    itax = ci if island_tax is None else island_tax
    ntax = cn if node_tax is None else node_tax

    by_id = {j.id: j for j in trace}
    seqof = {jid: i for i, jid in enumerate(by_id)}
    pending: list[str] = []

    # per-job run bookkeeping
    run_start: dict[str, float] = {}
    tax_factor: dict[str, float] = {}
    recovery_latencies: list[float] = []

    events: list[tuple[float, int, int, str]] = []
    for j in trace:
        j.state = JobState.PENDING
        j.start_s = j.end_s = None
        j.committed_s = 0.0
        j.restarts = 0
        j.wasted_s = 0.0
        heapq.heappush(events, (j.submit_s, _ARRIVE, seqof[j.id], j.id))
    for k, f in enumerate(faults):
        heapq.heappush(events, (f.at_s, _FAIL, k, f.node_id))

    busy_gpu_seconds = 0.0
    capacity_gpu_seconds = 0.0
    last_t = 0.0
    makespan = 0.0

    def _accrue(now: float) -> None:
        nonlocal busy_gpu_seconds, capacity_gpu_seconds, last_t
        dt = max(0.0, now - last_t)
        busy = cluster.total_gpus - cluster.free_count
        busy_gpu_seconds += busy * dt
        capacity_gpu_seconds += cluster.total_gpus * dt
        last_t = now

    def _remaining_nominal(j: Job) -> float:
        return max(0.0, j.duration_s - j.committed_s)

    def _start(j: Job, devices: list[str], now: float) -> None:
        node_span, island_span = placement_span(cluster, devices)
        f = effective_runtime(j, node_span, island_span, itax, ntax) / j.duration_s if j.duration_s else 1.0
        rem = _remaining_nominal(j)
        j.state = JobState.RUNNING
        if j.start_s is None:
            j.start_s = now
        j.end_s = now + recovery_s * (1 if j.restarts else 0) + rem * f
        j.devices = devices
        j.node_span, j.island_span = node_span, island_span
        run_start[j.id] = now
        tax_factor[j.id] = f
        heapq.heappush(events, (j.end_s, _COMPLETE, seqof[j.id], j.id))

    def _try_schedule(now: float) -> None:
        if not pending:
            return
        waiting = [by_id[jid] for jid in pending]
        for dec in policy.schedule(cluster, waiting, now):
            _start(dec.job, dec.devices, now)
            if dec.job.id in pending:
                pending.remove(dec.job.id)

    def _commit_progress(j: Job, now: float) -> float:
        """Advance committed_s to the last checkpoint before `now`; return wasted nominal."""
        rs = run_start[j.id]
        f = tax_factor[j.id]
        work_wall = now - rs - (recovery_s if j.restarts else 0)
        if work_wall <= 0:
            return 0.0  # died during recovery; nothing computed
        nominal_done = work_wall / f
        n_ckpt = int(work_wall // checkpoint_s)
        committed_now = min(nominal_done, (n_ckpt * checkpoint_s) / f)
        rem_at_start = _remaining_nominal(j)
        committed_now = min(committed_now, rem_at_start)
        j.committed_s += committed_now
        return max(0.0, nominal_done - committed_now)

    while events:
        now, kind, _k, ref = heapq.heappop(events)
        _accrue(now)
        makespan = max(makespan, now)

        if kind == _ARRIVE:
            pending.append(ref)
            _try_schedule(now)

        elif kind == _COMPLETE:
            j = by_id[ref]
            if j.state is not JobState.RUNNING or abs((j.end_s or 0) - now) > 1e-6:
                continue  # stale (job was failed/restarted)
            j.committed_s = j.duration_s
            cluster.release(j.devices)
            j.state = JobState.DONE
            _try_schedule(now)

        else:  # _FAIL
            node_id = ref
            if not any(n.id == node_id for n in cluster.nodes):
                continue  # already gone
            victims = [j for j in trace if j.state is JobState.RUNNING
                       and any(by_id_dev(d, node_id) for d in j.devices)]
            cluster.fail_node(node_id)  # frees + removes the node
            for j in victims:
                wasted = _commit_progress(j, now)
                j.wasted_s += wasted * j.gpus
                # release any of its devices that lived on *surviving* nodes
                cluster.release([d for d in j.devices if not by_id_dev(d, node_id)])
                j.restarts += 1
                j.state = JobState.PENDING
                j.end_s = None
                recovery_latencies.append(recovery_s)
                if j.id not in pending:
                    pending.append(j.id)
            _try_schedule(now)

    completed = [j for j in trace if j.state is JobState.DONE]
    useful = sum(j.gpus * j.duration_s for j in completed)
    wasted_total = sum(j.wasted_s for j in completed) + sum(
        j.wasted_s for j in trace if j.state is not JobState.DONE)
    rsum = summarize(recovery_latencies)

    return FaultResult(
        policy=policy.name,
        n_jobs=len(trace),
        completed=len(completed),
        n_faults=len(faults),
        makespan_s=makespan,
        raw_utilization=(busy_gpu_seconds / capacity_gpu_seconds) if capacity_gpu_seconds else 0.0,
        goodput=(useful / capacity_gpu_seconds) if capacity_gpu_seconds else 0.0,
        wasted_gpu_seconds=wasted_total,
        wasted_fraction=(wasted_total / busy_gpu_seconds) if busy_gpu_seconds else 0.0,
        total_restarts=sum(j.restarts for j in trace),
        recovery_p50_s=rsum.p50,
        recovery_p99_s=rsum.p99,
        checkpoint_s=checkpoint_s,
        jobs=trace,
    )


def by_id_dev(dev_id: str, node_id: str) -> bool:
    """True if device id `dev_id` belongs to node `node_id` (ids are '<node>-g<k>')."""
    return dev_id.rsplit("-g", 1)[0] == node_id
