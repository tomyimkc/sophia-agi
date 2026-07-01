# Cluster scheduler — git-bridge PROTOCOL-v2 (multi-node)

> **Status: design/infra; no capability claim; canClaimAGI stays false.**
> Offline, deterministic plumbing that lets 2/4/8/16 DGX Sparks share ONE job queue over GitHub —
> **no peer-to-peer network needed**. Extends the single-node bridge (`bridge/PROTOCOL.md`,
> `tools/spark_bridge.py`, `docs/11-Platform/Spark-Bridge-Cloud-Operator.md`); reuses its safety
> rules verbatim and does not widen them. Code: `tools/cluster_scheduler.py`; tests:
> `tests/test_cluster_scheduler_bridge.py`.

## Why v2

PROTOCOL-v1 drives ONE Spark: the cloud writes `bridge/commands/<id>.json`, the Spark polls,
executes allowlisted flags, writes `bridge/results/<id>.json`. With N Sparks on the same branch we
need three things v1 lacks, **without** a central dispatcher or any Spark-to-Spark networking (the
egress policy forbids it — see the v1 doc): (1) per-node liveness, (2) a way for exactly one node to
own each command (a mutex), and (3) fan-out/aggregate for sweeps. GitHub stays the only shared
channel.

The single-node invariants are preserved, just made per-node:

- the **allowlist** (`--dry-run --bench-a --bench-b --all --execute --run-train`) is reused
  unchanged (`tools/cluster_scheduler` imports `spark_bridge.ALLOWLIST`/`GATED`);
- **no AI self-approval of execute**: `--execute`/`--run-train` still require a non-empty human
  `approvedBy`, and that requirement is INHERITED by every fan-out shard;
- the **one-GPU-job invariant** now holds **per node** via `node_is_free`, mirroring
  `spark_bridge.gpu_is_free`.

## 1. Per-node status

Each node publishes `bridge/nodes/<node_id>/status.json`:

```json
{
  "nodeId": "spark-2f2d",
  "updatedAt": "2026-06-30T05:00:00Z",
  "running": "cmd-train-42",          // a command id, or null when idle
  "gpuFreeBytes": 121000000000,
  "heartbeatAt": "2026-06-30T05:00:03Z"
}
```

`node_is_free(status)` is the per-node one-GPU-job guard: free iff `running` is null **and** there
is nothing pending — the exact shape of `spark_bridge.gpu_is_free`. `node_id` is validated with the
SAME unsafe-id rule `spark_bridge` applies to command ids (filesystem-safe, no `/ \t`, non-empty).

## 2. Deterministic, decentralized assignment

`assigned_node(cmd_id, node_ids)` answers "who owns this command?" with **no dispatcher**:

```
owner = sorted(unique(node_ids))[ int(sha1(cmd_id), 16) % len(nodes) ]
```

Properties (enforced by `offline_invariants` and the tests):

- **single owner** — exactly one node per command, so two nodes never both decide to claim;
- **stable** — pure function of `cmd_id` + the node set, independent of input order (the list is
  de-duplicated and sorted internally), recomputed identically by every node;
- **balanced** — sha1 spreads commands within ±40% of an even split across 2/4/8 nodes;
- **no `random`, no clock** — determinism is load-bearing for the tests and for cross-node agreement.

Every node independently runs this over the same membership list and only attempts the commands that
hash to itself. That is the first line of defence against a stampede; the claim mutex below is the
second.

## 3. Claim / lease — git-push as compare-and-swap (the mutex)

Assignment says which node *should* claim a command; the **git ref is the actual mutex**. To claim
`cmd_id`, a node:

1. creates `bridge/claims/<cmd_id>.json` = `build_claim(cmd_id, node_id, leased_at, ttl_seconds)`:

   ```json
   { "cmdId": "cmd-train-42", "nodeId": "spark-a", "leasedAt": "1719723600", "ttlSeconds": 3600 }
   ```

2. **pushes** the branch. The first push **fast-forwards and wins**. A racing node's push is
   rejected **non-fast-forward**; it re-fetches, sees the winner's claim file, and backs off.

No lock server, no consensus protocol — just the property that GitHub accepts exactly one
fast-forward for a given parent commit. This is the compare-and-swap.

### Worked 2-node race

Two nodes, command `cmd-train-42`. Both fetch the branch at commit `C0`.

- `assigned_node("cmd-train-42", ["spark-a","spark-b"])` → both compute the **same** owner, say
  `spark-a`. `spark-b` defers immediately (it isn't the owner) — no push, no contention.
- Suppose membership is momentarily inconsistent (a node mid-join) and both *do* try. `spark-a`
  writes the claim file on top of `C0` → `C1a`, pushes → **fast-forward, accepted**. `spark-b`
  writes its claim on `C0` → `C1b`, pushes → server rejects (`! [rejected] (non-fast-forward)`).
  `spark-b` re-fetches, sees `claims/cmd-train-42.json` already owned by `spark-a`, and runs nothing.

Result: at most one node ever executes a command. The deterministic assignment makes the common case
contention-free; the git CAS makes the worst case **safe**, not merely unlikely.

### Lease + requeue

`lease_is_expired(claim, now)` is true once `leasedAt + ttlSeconds < now` (`now` passed in; epoch
seconds). If the owner crashes mid-run, its lease expires and the command becomes claimable again —
another node (or the same one after recovery) can re-create the claim and push. A malformed/empty
claim is treated as expired (safe to requeue).

## 4. Fan-out and aggregate

`split_command(parent_cmd, shards)` turns one parent (e.g. `--bench-a --execute` over SEEDS
`[1,2,10]`) into one sub-command per shard:

```json
{ "id": "bench-a-exec--shard-1", "args": "--bench-a --execute",
  "createdBy": "claude", "approvedBy": "user: 'go' (2026-06-30)",
  "parentId": "bench-a-exec", "shard": 1 }
```

Each sub-command **inherits** the parent's `args`, `createdBy`, and human `approvedBy`. The gated
rule is inherited verbatim from `spark_bridge`: **a gated parent (`--execute`/`--run-train`) with an
empty `approvedBy` cannot fan out — `split_command` raises `ValueError`.** Non-allowlisted parent
args are likewise rejected before any shard is produced. The scheduler never self-approves GPU work.

Each shard is then assigned (`assigned_node`) and claimed (the CAS above) independently, so a sweep
spreads across the cluster.

`collect_shard_results(parent_id, results)` → `{parentId, shards, complete, missing}` lets a
collector poll until every shard has a result and know exactly which shards are still outstanding.

## 5. What stays a thin git wrapper

All scheduling logic is pure and unit-tested; the only git touch is `_git_show_claim` (a read-only
`git show origin/spark-bridge:bridge/claims/<id>.json`), isolated exactly like
`spark_bridge._git_show`. The winning push itself is done by the node's poller / the GitHub API, not
by the core logic — keeping the tested surface offline and deterministic.

## Open scheduling concern — Mac-judge contention

Several v1/v2 jobs route their outputs to a single Mac Studio acting as the LLM judge. With N Sparks
fanning out a sweep, they can hit that **one Mac judge** concurrently, creating a serialization
bottleneck (or, worse, judge overload changing scoring latency mid-run). v2 schedules the GPU work;
it does **not** yet schedule judge capacity. Judge-side admission control (a separate claim/lease on
a `bridge/judges/<id>` lane, or a token bucket) is left as an OPEN concern, tracked here so a future
session does not assume the GPU mutex also protects the judge.

## Throughput simulation

The Mac-judge contention flagged above is now *quantified* by a sibling planning tool,
`tools/cluster_schedule_sim.py` (tests: `tests/test_cluster_schedule_sim.py`). It answers the
owner's standing ROI question — **for MY actual queue, how much faster is 1 vs 2 vs 4 vs 8 vs 16
Sparks, and where does the shared Mac judge bottleneck?**

It is a **pure, offline, deterministic** planning tool, exactly like `tools/run_cluster_sim.py`, and
makes **no capability or throughput claim about real hardware**: it schedules the owner's *own
GPU-time ESTIMATES* (the T1-T4 queue in `docs/06-Roadmap/Spark-Theory-Test-Forecast.md`, plus a
configurable independent N-seed × M-discipline sweep) and reports the resulting wall-clock, speedup,
efficiency, and Mac-judge wait. The arithmetic of distribution is the only thing it does; the
estimates are forecasts, not measurements. `canClaimAGI` stays false.

What it models, and how it stays complementary to `run_cluster_sim.py`:

- **Distribution reuses `assigned_node` verbatim** — each job's owner is the same deterministic
  single-owner sha1 hash this protocol uses, so the sim's placement == the cluster's placement. It
  does not re-implement assignment.
- **One-GPU-job-per-node** — each node runs its assigned jobs serially (this protocol's per-node
  invariant), so independent jobs across N nodes is the good case.
- **The shared Mac judge as a semaphore** of size `mac_judge_concurrency` (the OPEN concern from the
  section above, made concrete): every `needs_mac_judge` job must hold a judge lane, so with
  concurrency=1 they serialize through the one Mac regardless of node count.
- It deliberately carries **no network "node tax"** — that is `run_cluster_sim.py`'s job (the loss a
  single *data-parallel* run takes to all-reduce/switch). This sim is the *independent-job*
  throughput regime, which needs no gradient sync, so the two sims are non-overlapping.

The headline finding for this repo's queue: because most judged jobs route to **one** Mac judge,
**independent-job scaling hits a Mac-judge ceiling fast** — adding Sparks past ~2 drops efficiency
without moving wall-clock until the judge lane count is raised. The fix the tool lets you explore is
`--mac-concurrency K` (more judge lanes), not more Sparks. Run:

```bash
python tools/cluster_schedule_sim.py --self-test
python tools/cluster_schedule_sim.py --jobs forecast --nodes 1,2,4,8,16
python tools/cluster_schedule_sim.py --jobs sweep:4x3 --nodes 1,2,4,8,16 --mac-concurrency 4
```

`--self-test` asserts the invariants: assignment comes from `cluster_scheduler` (single owner per
job), speedup is monotonic non-decreasing in node count while efficiency decreases, Mac-judge
contention raises wall-clock when >1 judge job and concurrency=1, and a pure independent no-mac
workload scales near-linearly until jobs < nodes (then flat). Deterministic across runs. This is
design/infra; **no capability claim; canClaimAGI stays false.**

## Invariants checklist (machine-checked)

`python tools/cluster_scheduler.py --self-test` asserts: allowlist/gating reused unchanged; single +
stable assignment (no double-claim); lease live/expired boundary; gated fan-out refused without
approval and inherited with it; unsafe node ids refused; `node_is_free` mirrors `gpu_is_free`;
aggregation completeness. This is design/infra; **no capability claim; canClaimAGI stays false.**
