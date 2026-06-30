#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Throughput / wall-clock scheduling simulation for the DGX-Spark cluster (1 vs 2 vs 4 vs 8 vs 16).

Answers the owner's standing ROI question: **for MY actual job queue, how much faster is 1 vs 2 vs 4
vs 8 vs 16 Sparks, and where does the shared Mac judge become the bottleneck?**

This is a PLANNING TOOL, exactly like `tools/run_cluster_sim.py`. It is **pure, offline, and
deterministic** (no `random`, no clock, no GPU, no network). It makes **NO capability or throughput
CLAIM about real hardware** — it schedules the owner's *own GPU-time ESTIMATES* (from
`docs/06-Roadmap/Spark-Theory-Test-Forecast.md`) across N nodes and reports the resulting wall-clock,
speedup, efficiency, and Mac-judge contention. The estimates are the owner's forecasts; the sim only
does the arithmetic of distributing them. `canClaimAGI` stays false.

How it is COMPLEMENTARY to `tools/run_cluster_sim.py`:
  * `run_cluster_sim.py` models the **network "node tax"** (all-reduce hops / island / switch) on a
    *synthetic GPU-job trace* — i.e. how much a single data-parallel run loses to interconnect.
  * THIS sim models the **scheduler/throughput** level: many *independent* jobs (the cluster's real
    sweet spot per `Spark-Cluster-Capacity.md`) distributed one-job-per-node, plus the contention on
    the **single shared Mac judge** that v2 (`Cluster-Scheduler-v2.md`) leaves as an OPEN concern.
  It does NOT re-implement the node-tax model; it references `run_cluster_sim`'s default node count
  set for the report. Independent jobs need no gradient sync, so there is no node-tax term here —
  that is the whole point of the "many independent jobs" regime.

Job distribution reuses `tools/cluster_scheduler.assigned_node` verbatim (the deterministic
single-owner sha1 hash) — the assignment is NOT re-implemented here. Each node then runs its assigned
jobs SERIALLY (the one-GPU-job-per-node invariant), and jobs that need the Mac judge contend on a
shared judge modeled as a semaphore of size `mac_judge_concurrency`.

The single Mac judge is generalized by the JUDGE POOL (`tools/judge_pool.py`): the
`mac_judge_concurrency` scalar IS the number of LANES (endpoint replicas) serving the mac-bound
judge family. Pass `--mac-lanes N` or `--judge-pool config/inference.local.judge-pool.json` to set
it from a real pool, so the sim reports the true ROI of adding judge replicas (e.g. 1 lane ~1.36x
vs 4 lanes ~2.81x at 8 nodes for the forecast queue). `--mac-concurrency` is kept as a back-compat
alias. design/infra; no capability claim; canClaimAGI stays false.

CLI:
    python tools/cluster_schedule_sim.py --self-test
    python tools/cluster_schedule_sim.py --jobs forecast --nodes 1,2,4,8,16
    python tools/cluster_schedule_sim.py --jobs forecast --nodes 1,2,4,8,16 --mac-lanes 4
    python tools/cluster_schedule_sim.py --jobs forecast --judge-pool config/inference.local.judge-pool.json
    python tools/cluster_schedule_sim.py --jobs sweep:4x3 --nodes 1,2,4,8,16 --mac-concurrency 1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the deterministic single-owner assignment verbatim — do NOT fork it.
from tools.cluster_scheduler import assigned_node  # noqa: E402
# Reuse the judge-pool lane accounting — the sim's `mac_judge_concurrency` scalar IS, semantically,
# the number of LANES serving the mac-bound judge family. With a judge pool, that scalar becomes the
# real lane count for that family (Mac + spare-Spark replicas), so the sim reports the true speedup
# of adding judge replicas. Do NOT fork judge_pool — import its accounting.
from tools.judge_pool import load_pool, families as _pool_families, lanes_for_family  # noqa: E402

# The node-count set the cluster docs / run_cluster_sim.py reason about (1..16 Sparks).
DEFAULT_NODE_COUNTS: "tuple[int, ...]" = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class Job:
    """One unit of GPU work. `gpu_minutes` is the OWNER'S ESTIMATE (not a measurement). `kind` is
    descriptive; `needs_mac_judge` marks a job that must hold the single shared Mac judge lane."""

    id: str
    kind: str  # 'cert' | 'judge' | 'train' | 'faith'
    gpu_minutes: float
    needs_mac_judge: bool


# --- workloads ----------------------------------------------------------------------------------
def forecast_jobs() -> "list[Job]":
    """The real pre-registered queue from `docs/06-Roadmap/Spark-Theory-Test-Forecast.md` (T1-T4),
    using the owner's own GPU-time estimates, PLUS a small independent experiment batch so the
    cluster's actual sweet spot (many independent jobs) is represented, not just the 4-job queue.

    T1 NVFP4 cert ~15m (no mac);  T2 faithfulness ~30m (no mac);
    T3 sophrosyne virtues ~55m (needs mac judge);  T4 council-train ~120m (no mac).
    The judge gate (T3, virtues) is the one that routes to the Mac judge farm; the deterministic
    LowRamGate cert (T1), the faithfulness measurement (T2) and the LoRA train (T4) do not.
    """
    jobs = [
        Job("T1-nvfp4-cert", "cert", 15.0, needs_mac_judge=False),
        Job("T2-faithfulness", "faith", 30.0, needs_mac_judge=False),
        Job("T3-sophrosyne-virtues", "judge", 55.0, needs_mac_judge=True),
        Job("T4-council-train", "train", 120.0, needs_mac_judge=False),
    ]
    # Independent experiment batch: a 3-seed x 2-discipline judged virtue sweep (the kind of
    # "run the whole no-overclaim matrix in one pass" workload the capacity doc calls the on-charter
    # value of more Sparks). Each judged shard contends on the Mac judge.
    jobs.extend(sweep_jobs(seeds=3, disciplines=2))
    return jobs


def sweep_jobs(seeds: int, disciplines: int, *, judged: bool = True) -> "list[Job]":
    """A pure independent N-seed x M-discipline sweep: `seeds*disciplines` jobs with NO cross-job
    dependency (the cluster's best-scaling case). Deterministic GPU-minute estimate seeded by the
    job index (no random): a fixed base plus a small index-driven spread, so different shards have
    slightly different lengths (realistic) yet the trace is identical every run. When `judged`, every
    shard needs the Mac judge (a virtue/bench-A style sweep) so judge contention is exercised."""
    if seeds <= 0 or disciplines <= 0:
        raise ValueError("seeds and disciplines must be positive")
    out: "list[Job]" = []
    idx = 0
    for d in range(disciplines):
        for s in range(seeds):
            # Deterministic per-index variation (no random, no clock): 40m base, +0..18m spread.
            gpu_minutes = 40.0 + float((idx * 7) % 19)
            out.append(Job(
                id=f"sweep-d{d}-s{s}",
                kind="judge" if judged else "train",
                gpu_minutes=gpu_minutes,
                needs_mac_judge=judged,
            ))
            idx += 1
    return out


def _node_ids(n_nodes: int) -> "list[str]":
    """Stable node ids spark-00..spark-(n-1). Zero-padded so the sorted order assigned_node uses is
    numeric-stable (spark-02 < spark-10)."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    width = max(2, len(str(n_nodes - 1)))
    return [f"spark-{i:0{width}d}" for i in range(n_nodes)]


# --- the simulation -----------------------------------------------------------------------------
def simulate(jobs: "list[Job]", n_nodes: int, mac_judge_concurrency: int = 1) -> dict:
    """Distribute `jobs` across `n_nodes` and compute wall-clock under the one-GPU-job-per-node
    invariant plus a shared Mac-judge semaphore of size `mac_judge_concurrency`.

    Model (deterministic, event-free closed form — no clock):
      * Assignment: each job's owning node is `assigned_node(job.id, node_ids)` — the SAME function
        the live scheduler uses, so the sim's distribution == the cluster's distribution.
      * Each node runs its jobs serially. We order a node's jobs deterministically (mac-judge jobs
        first, then by id) so the judge demand is front-loaded the way a real fan-out hits it.
      * The Mac judge is a semaphore of `mac_judge_concurrency` lanes shared across ALL nodes. We
        track each lane's running total; a judge job's judge-phase cannot start before a lane frees.
        The added wait a node accrues waiting for a lane is `macJudgeWaitMinutes` (per-node, summed).
      * Wall-clock = max over nodes of (sum of that node's job minutes + that node's judge wait).
      * bottleneck = 'mac-judge' if total judge wait is the binding term (removing it would shorten
        the critical node), else 'compute'.

    Returns: {wallClockMinutes, perNodeMinutes, speedupVs1, efficiency, macJudgeWaitMinutes,
              bottleneck, assignment}. `speedupVs1`/`efficiency` are filled by `report`/callers that
      have the 1-node baseline; `simulate` reports speedupVs1=1.0 placeholder when n_nodes==1."""
    if not jobs:
        raise ValueError("jobs must be a non-empty list")
    if mac_judge_concurrency <= 0:
        raise ValueError("mac_judge_concurrency must be positive")
    nodes = _node_ids(n_nodes)

    # 1) deterministic single-owner assignment (reused, not re-implemented)
    by_node: "dict[str, list[Job]]" = {n: [] for n in nodes}
    for job in jobs:
        owner = assigned_node(job.id, nodes)
        by_node[owner].append(job)

    # 2) per-node serial schedule with a shared judge semaphore.
    # Judge lanes hold a running "free-at" minute; a node processing in node-local time must, for a
    # judge job, wait until a lane is free in GLOBAL judge time. We advance global judge time as the
    # union of demand. To stay closed-form + deterministic we walk nodes in a fixed order and assign
    # each judge job to the earliest-free lane, accruing the wait to that node.
    lanes = [0.0] * mac_judge_concurrency  # global judge-busy-until per lane (minutes)
    per_node_minutes: "dict[str, float]" = {}
    per_node_wait: "dict[str, float]" = {}

    # Process nodes in sorted order; within a node, judge jobs first (front-loaded), then by id.
    for n in nodes:
        node_jobs = sorted(by_node[n], key=lambda j: (not j.needs_mac_judge, j.id))
        local_time = 0.0   # this node's own wall-clock cursor
        wait = 0.0
        for job in node_jobs:
            if job.needs_mac_judge:
                # earliest-free lane in global judge time
                li = min(range(len(lanes)), key=lambda i: lanes[i])
                lane_free = lanes[li]
                start = max(local_time, lane_free)
                wait += start - local_time          # idle the node spent waiting for the judge lane
                end = start + job.gpu_minutes
                lanes[li] = end                      # lane busy until this judge job ends
                local_time = end
            else:
                local_time += job.gpu_minutes
        per_node_minutes[n] = local_time
        per_node_wait[n] = wait

    wall = max(per_node_minutes.values()) if per_node_minutes else 0.0
    total_wait = sum(per_node_wait.values())

    # bottleneck classification (stable, not tie-sensitive): the cluster is mac-judge-bound when the
    # serialized judge work is the binding term. The judge floor = (total judge-minutes /
    # mac_judge_concurrency): the minimum wall-clock the judge lanes alone impose regardless of node
    # count. The compute floor = the heaviest single node's PURE (non-wait) compute. Whichever floor
    # is larger is what the wall-clock is bound on. This is tie-stable: adding nodes past the job
    # count cannot move either floor, so 8 and 16 nodes report the SAME bottleneck for the same queue.
    judge_minutes = sum(j.gpu_minutes for j in jobs if j.needs_mac_judge)
    judge_floor = judge_minutes / mac_judge_concurrency
    compute_floor = max((per_node_minutes[k] - per_node_wait[k] for k in per_node_minutes),
                        default=0.0)
    bottleneck = "mac-judge" if (total_wait > 0.0 and judge_floor >= compute_floor) else "compute"

    return {
        "nNodes": n_nodes,
        "wallClockMinutes": round(wall, 4),
        "perNodeMinutes": {k: round(v, 4) for k, v in per_node_minutes.items()},
        "speedupVs1": 1.0,          # filled in by report() against the real 1-node baseline
        "efficiency": 1.0,          # idem
        "macJudgeWaitMinutes": round(total_wait, 4),
        "bottleneck": bottleneck,
        "assignment": {k: [j.id for j in by_node[k]] for k in nodes},
    }


def scaling_table(jobs: "list[Job]", node_counts="(1,2,4,8,16)", mac_judge_concurrency: int = 1):
    """Run `simulate` for each node count and fill in speedup/efficiency against the 1-node baseline.
    Returns a list of result dicts in node-count order. Deterministic."""
    counts = list(node_counts)
    baseline = simulate(jobs, 1, mac_judge_concurrency)["wallClockMinutes"]
    rows = []
    for n in counts:
        r = simulate(jobs, n, mac_judge_concurrency)
        wall = r["wallClockMinutes"]
        r["speedupVs1"] = round(baseline / wall, 4) if wall > 0 else 0.0
        r["efficiency"] = round(r["speedupVs1"] / n, 4)
        rows.append(r)
    return rows


# --- report -------------------------------------------------------------------------------------
def report(jobs: "list[Job]", node_counts="(1,2,4,8,16)", mac_judge_concurrency: int = 1) -> str:
    """A plain-text table: wall-clock, speedup, efficiency, mac-judge wait, bottleneck per node count
    — so the owner sees diminishing returns and the Mac-judge ceiling for THEIR queue. Returns the
    table as a string (printed by main). PLANNING ONLY — these are scheduled estimates, not a
    hardware measurement."""
    rows = scaling_table(jobs, node_counts, mac_judge_concurrency)
    n_judge = sum(1 for j in jobs if j.needs_mac_judge)
    lines = [
        f"Throughput sim — {len(jobs)} jobs ({n_judge} need the Mac judge), "
        f"mac_judge_concurrency={mac_judge_concurrency}",
        "  (PLANNING: schedules the owner's GPU-time ESTIMATES; no hardware claim. "
        "canClaimAGI=false)",
        "",
        f"  {'nodes':>5} | {'wall(min)':>9} | {'speedup':>7} | {'eff':>5} | "
        f"{'judgeWait(min)':>14} | bottleneck",
        f"  {'-'*5}-+-{'-'*9}-+-{'-'*7}-+-{'-'*5}-+-{'-'*14}-+-{'-'*10}",
    ]
    for r in rows:
        lines.append(
            f"  {r['nNodes']:>5} | {r['wallClockMinutes']:>9.1f} | {r['speedupVs1']:>7.2f} | "
            f"{r['efficiency']:>5.2f} | {r['macJudgeWaitMinutes']:>14.1f} | {r['bottleneck']}"
        )
    return "\n".join(lines)


# --- offline invariants / self-test -------------------------------------------------------------
def offline_invariants() -> "tuple[bool, dict]":
    checks: "dict[str, bool]" = {}

    # 1) assignment uses cluster_scheduler.assigned_node (single owner per job)
    jobs = forecast_jobs()
    nodes = _node_ids(4)
    res = simulate(jobs, 4)
    owners = {jid: owner for owner, jids in res["assignment"].items() for jid in jids}
    checks["assignment_single_owner"] = (
        len(owners) == len(jobs) and
        all(owners[j.id] == assigned_node(j.id, nodes) for j in jobs)
    )

    # 2) deterministic across runs (same trace, same numbers)
    a = simulate(jobs, 8)
    b = simulate(jobs, 8)
    checks["deterministic"] = (a == b)

    # 3) speedup monotonic non-decreasing in n_nodes; efficiency decreasing (non-increasing)
    rows = scaling_table(jobs, DEFAULT_NODE_COUNTS)
    speeds = [r["speedupVs1"] for r in rows]
    effs = [r["efficiency"] for r in rows]
    checks["speedup_monotonic"] = all(speeds[i] <= speeds[i + 1] + 1e-9 for i in range(len(speeds) - 1))
    checks["efficiency_decreasing"] = all(effs[i] >= effs[i + 1] - 1e-9 for i in range(len(effs) - 1))
    checks["speedup_ge_1"] = all(s >= 1.0 - 1e-9 for s in speeds)

    # 4) mac-judge contention raises wall-clock when >1 judge job and concurrency=1.
    #    Compare concurrency=1 vs a generous concurrency on a judge-heavy workload at high node count
    #    (so compute is NOT the binding term and the judge serialization is exposed).
    judge_heavy = sweep_jobs(seeds=4, disciplines=2, judged=True)  # 8 judged jobs
    serial = simulate(judge_heavy, 16, mac_judge_concurrency=1)
    parallel = simulate(judge_heavy, 16, mac_judge_concurrency=8)
    checks["mac_contention_raises_wall"] = (
        serial["wallClockMinutes"] > parallel["wallClockMinutes"] and
        serial["macJudgeWaitMinutes"] > 0.0 and
        parallel["macJudgeWaitMinutes"] == 0.0
    )
    checks["mac_bottleneck_flagged"] = (serial["bottleneck"] == "mac-judge")

    # 5) a pure-independent NO-MAC workload scales near-linearly until jobs < nodes, then flat.
    indep = sweep_jobs(seeds=4, disciplines=2, judged=False)  # 8 independent no-mac jobs
    irows = {r["nNodes"]: r for r in scaling_table(indep, (1, 2, 4, 8, 16), mac_judge_concurrency=1)}
    # 8 jobs across 8 nodes: speedup should be well above the 4-node speedup (more parallelism used)
    checks["independent_scales"] = irows[8]["speedupVs1"] > irows[4]["speedupVs1"] + 1e-9
    # 16 nodes for 8 jobs: flat vs 8 (no extra job to place — at most one job per node already)
    checks["independent_flat_past_jobcount"] = (
        abs(irows[16]["wallClockMinutes"] - irows[8]["wallClockMinutes"]) < 1e-9
    )
    # no judge wait anywhere in a no-mac workload
    checks["independent_no_judge_wait"] = all(r["macJudgeWaitMinutes"] == 0.0 for r in irows.values())

    # 6) judge-pool lane generalization: the mac-bound family's lane count is what the sim uses, and
    #    more lanes raise the forecast speedup at 8 nodes (the whole point of the pool). The example
    #    pool has a 3-lane mac-bound family; 4 lanes must beat 1 lane.
    example_pool = {"qwen": ["vllm:Qwen/Qwen2.5-7B-Instruct@http://h0:8000/v1"],
                    "mlx-community": [f"vllm:mlx-community/m@http://h{i}:8001/v1" for i in (1, 2, 3)]}
    checks["pool_lane_count"] = (mac_lanes_from_pool(example_pool) == 3)
    fc = forecast_jobs()
    s1 = {r["nNodes"]: r for r in scaling_table(fc, (8,), mac_judge_concurrency=1)}[8]["speedupVs1"]
    s4 = {r["nNodes"]: r for r in scaling_table(fc, (8,), mac_judge_concurrency=4)}[8]["speedupVs1"]
    checks["more_lanes_raise_speedup"] = (s4 > s1 + 1e-9)

    return all(checks.values()), {"checks": checks}


# --- CLI ----------------------------------------------------------------------------------------
def _parse_jobs_arg(spec: str) -> "list[Job]":
    spec = (spec or "").strip()
    if spec == "forecast":
        return forecast_jobs()
    if spec.startswith("sweep:"):
        body = spec[len("sweep:"):]
        try:
            n, m = body.lower().split("x")
            return sweep_jobs(seeds=int(n), disciplines=int(m))
        except ValueError:
            raise ValueError(f"bad sweep spec {spec!r}; use sweep:NxM (e.g. sweep:3x2)")
    raise ValueError(f"unknown --jobs {spec!r}; use 'forecast' or 'sweep:NxM'")


def mac_lanes_from_pool(pool: "dict[str, list[str]]") -> int:
    """The number of LANES serving the mac-bound judge family in a judge pool. The mac-bound family
    is the bottleneck family — the one the pool scales out — taken here as the family with the MOST
    lanes (ties broken by family name for determinism). This is exactly the scalar the sim has been
    calling `mac_judge_concurrency`: how many replicas can serve the serialized judge demand at once.
    A 1-lane pool reproduces the old fully-serialized Mac judge; a 3-lane pool is the worked example
    (Mac + 2 spare Sparks)."""
    fams = _pool_families(pool)
    if not fams:
        raise ValueError("judge pool has no families")
    # most-lanes family wins; deterministic tie-break by family name (no random)
    best = max(fams, key=lambda f: (lanes_for_family(pool, f), f))
    return lanes_for_family(pool, best)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default="forecast",
                    help="workload: 'forecast' (the T1-T4 queue + a sweep) or 'sweep:NxM'")
    ap.add_argument("--nodes", default="1,2,4,8,16", help="comma-separated node counts")
    ap.add_argument("--mac-concurrency", type=int, default=1,
                    help="shared Mac-judge lanes (semaphore size); 1 = fully serialized judge. "
                         "Back-compat alias kept; --mac-lanes / --judge-pool are the generalized form.")
    ap.add_argument("--mac-lanes", type=int, default=None,
                    help="number of LANES (replicas) serving the mac-bound judge family — the "
                         "judge-pool generalization of --mac-concurrency. e.g. --mac-lanes 4 = the "
                         "Mac + 3 spare-Spark 70B replicas. Overrides --mac-concurrency when set.")
    ap.add_argument("--judge-pool", type=Path, default=None,
                    help="judge-pool config (config/inference.local.judge-pool.json); the sim uses "
                         "that pool's lane count for the mac-bound family as the judge concurrency, "
                         "so it reports the REAL speedup of adding judge replicas.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        ok, detail = offline_invariants()
        print("cluster_schedule_sim invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1

    try:
        jobs = _parse_jobs_arg(args.jobs)
        counts = tuple(int(c) for c in args.nodes.split(",") if c.strip())
        if not counts:
            raise ValueError("--nodes produced no node counts")
        # Resolve the judge concurrency (lanes). Precedence: --judge-pool > --mac-lanes >
        # --mac-concurrency (back-compat). All three feed the SAME semaphore-size scalar.
        lanes = args.mac_concurrency
        if args.judge_pool is not None:
            pool = load_pool(json.loads(args.judge_pool.read_text(encoding="utf-8")))
            lanes = mac_lanes_from_pool(pool)
        elif args.mac_lanes is not None:
            lanes = args.mac_lanes
        if lanes <= 0:
            raise ValueError("judge lanes (--mac-lanes/--mac-concurrency) must be positive")
    except (ValueError, json.JSONDecodeError, OSError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    print(report(jobs, counts, lanes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
