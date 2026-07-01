# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Placement policies — the scheduling brain the simulator calls each event.

Each policy answers one question: given the jobs waiting and the GPUs free *right now*,
which jobs start and on which devices? The three policies span the trade-off the JD
names — task throughput vs. queue latency vs. resource utilization:

  FifoFirstFit   — strict FIFO, first free GPUs in id order. Simple, fair, but scatters
                   collective jobs across nodes (high network tax) and head-of-line blocks.
  TopologyAware  — still priority/FIFO ordered, but *packs* each colocate job onto the
                   fewest NVLink islands / nodes, minimizing cross-node traffic. Skips a
                   job it cannot place compactly enough rather than fragmenting it.
  BackfillTopo   — TopologyAware placement + EASY backfilling: a small job behind a
                   blocked big job may jump ahead *iff* it cannot delay the blocked job's
                   reservation. Trades a little fairness for utilization + lower latency.

`fragmentation(job)` quantifies placement quality: 0.0 = all GPUs in one island (ideal
for collectives), →1.0 = maximally scattered. The simulator integrates this into the
network-tax model so bad placement shows up as longer effective runtime.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from clustersim.job import Job
from clustersim.topology import Cluster, Device


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------
def _by_island(devices: list[Device]) -> dict[tuple[str, int], list[Device]]:
    groups: dict[tuple[str, int], list[Device]] = defaultdict(list)
    for d in devices:
        groups[(d.node_id, d.island)].append(d)
    return groups


def placement_span(cluster: Cluster, dev_ids: list[str]) -> tuple[int, int]:
    """Return (#distinct nodes, #distinct islands) a placement touches."""
    nodes, islands = set(), set()
    for did in dev_ids:
        d = cluster.device(did)
        nodes.add(d.node_id)
        islands.add((d.node_id, d.island))
    return len(nodes), len(islands)


def fragmentation(cluster: Cluster, dev_ids: list[str]) -> float:
    """0.0 (one island) .. 1.0 (one GPU per island). Single-GPU jobs are 0."""
    n = len(dev_ids)
    if n <= 1:
        return 0.0
    _, islands = placement_span(cluster, dev_ids)
    return (islands - 1) / (n - 1)


def _eligible_free(cluster: Cluster, job: Job) -> list[Device]:
    free = cluster.free_gpus()
    if job.klass is not None:
        free = [d for d in free if d.klass == job.klass]
    return free


def pack_topo(cluster: Cluster, job: Job) -> list[str] | None:
    """Pick `job.gpus` free GPUs minimizing island/node span (best-fit packing).

    Greedy: fill the island that leaves the smallest remainder first, walking islands
    sorted by (free-capacity that still covers need? , then largest). This favours
    consolidating a job into one island/node when possible and only spills when forced.
    Returns device ids, or None if it cannot satisfy the request at all.
    """
    free = _eligible_free(cluster, job)
    if len(free) < job.gpus:
        return None

    groups = _by_island(free)
    need = job.gpus

    # 1) Try to fit entirely inside a single island (zero fragmentation): pick the
    #    tightest island that still holds the job (best-fit → less external frag).
    fits = sorted(
        (g for g in groups.values() if len(g) >= need),
        key=len,
    )
    if fits:
        chosen = fits[0][:need]
        return [d.id for d in chosen]

    # 2) Spill across islands, largest-island-first to minimize island count, and
    #    prefer islands on the same node / rack to keep the network hop cheap.
    order = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]),
    )
    picked: list[str] = []
    for _key, devs in order:
        take = min(need - len(picked), len(devs))
        picked.extend(d.id for d in devs[:take])
        if len(picked) >= need:
            break
    return picked if len(picked) >= need else None


def pack_firstfit(cluster: Cluster, job: Job) -> list[str] | None:
    free = _eligible_free(cluster, job)
    if len(free) < job.gpus:
        return None
    free.sort(key=lambda d: d.id)
    return [d.id for d in free[: job.gpus]]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    job: Job
    devices: list[str]


class Policy:
    name = "base"

    def schedule(self, cluster: Cluster, pending: list[Job], now: float) -> list[Decision]:
        raise NotImplementedError

    @staticmethod
    def _order(pending: list[Job]) -> list[Job]:
        # higher priority first, then FIFO by submit time
        return sorted(pending, key=lambda j: (-j.priority, j.submit_s, j.id))


class FifoFirstFit(Policy):
    name = "fifo-firstfit"

    def schedule(self, cluster, pending, now):
        out: list[Decision] = []
        for job in self._order(pending):
            devs = pack_firstfit(cluster, job)
            if devs is None:
                break  # strict FIFO: head-of-line block
            cluster.allocate(devs)
            out.append(Decision(job, devs))
        return out


class TopologyAware(Policy):
    name = "topology-aware"

    def schedule(self, cluster, pending, now):
        out: list[Decision] = []
        for job in self._order(pending):
            devs = pack_topo(cluster, job)
            if devs is None:
                break  # head-of-line block (no backfill)
            cluster.allocate(devs)
            out.append(Decision(job, devs))
        return out


class BackfillTopo(Policy):
    name = "backfill-topo"

    def schedule(self, cluster, pending, now):
        out: list[Decision] = []
        queue = self._order(pending)
        i = 0
        # Schedule the head of the queue greedily; when it blocks, try to backfill
        # *smaller* jobs that fit in the leftover GPUs without consuming the shortfall
        # the head job is waiting for (EASY backfill, single reservation).
        while i < len(queue):
            job = queue[i]
            devs = pack_topo(cluster, job)
            if devs is not None:
                cluster.allocate(devs)
                out.append(Decision(job, devs))
                i += 1
                continue
            # head-of-line job j blocked: reserve its shortfall, backfill the rest.
            shortfall = job.gpus - cluster.free_count
            for cand in queue[i + 1 :]:
                if cand.gpus <= max(0, cluster.free_count - max(0, shortfall)):
                    d = pack_topo(cluster, cand)
                    if d is not None:
                        cluster.allocate(d)
                        out.append(Decision(cand, d))
            break
        return out


POLICIES: dict[str, type[Policy]] = {
    FifoFirstFit.name: FifoFirstFit,
    TopologyAware.name: TopologyAware,
    BackfillTopo.name: BackfillTopo,
}


def get_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}") from exc
