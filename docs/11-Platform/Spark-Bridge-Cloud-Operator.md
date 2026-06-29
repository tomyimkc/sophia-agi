# Spark bridge — cloud operator envelope

> How a cloud Claude-Code session operates the DGX Spark cluster, and the safety line it holds.
> Pairs with `bridge/PROTOCOL.md` (the Spark side) and the `spark-cluster-ops` skill.

## Why a bridge at all

A cloud session cannot reach the Spark directly — its egress policy rejects the Spark's Tailscale
Funnel (`connect_rejected 502`). The only shared channel is **GitHub**. So the Spark runs
`tools/github_bridge_poll.py` (publishes `bridge/STATUS.json`, executes approved commands, writes
`bridge/results/<id>.json`), and the cloud session uses `tools/spark_bridge.py` to **compose +
validate** commands and **read** status/results from the `spark-bridge` branch.

## What the cloud operator may do autonomously

- **Read** live status and results (`spark_bridge.py status` / `result --id ...`).
- **Submit `--dry-run`** commands freely (no GPU load, no approval) — plan validation, wiring checks.
- **Compose + stream** any job for review.

## The line it holds: no AI self-approval of GPU work

`--execute` and `--run-train` are GATED. The poller runs them ONLY when `approvedBy` is a non-empty
**human handle**, and `spark_bridge.build_command()` refuses to even compose a gated command without
one. This is deliberate, not a limitation to route around:

- the cluster is shared with other live sessions (one checkout, one 128 GB GPU);
- the GPU is one-job-exclusive — `spark-cluster-ops` records that a stray third load **killed a
  training seed mid-run** (empty checkpoint, no `rlvr.json`);
- so a GPU job is a high-consequence, hard-to-reverse action that stays with a human.

When a human (the repo owner) instructs a specific job, that instruction is quoted into `approvedBy`
(e.g. `"user: 'execute bench-a' (2026-06-29)"`) — the same pattern already in `bridge/commands/`.

## The operating ritual for any execute

1. `gpu_is_free(status)` — submit an execute ONLY when `running` is null AND `pendingCommands` is
   empty (the one-GPU-job invariant). Otherwise wait or coordinate.
2. Carry the human approval handle.
3. Stream before (the command) and after (the result), and on a NO-GO report it plainly.
4. Never push noise to `spark-bridge` while a peer session is mid-job; the commands/ dir is the only
   thing the cloud writes.

## Allowlist (enforced cloud-side AND by the poller)

`--dry-run --bench-a --bench-b --all --execute --run-train`. Anything else is rejected before
submit. Only `scripts/run_local_benchmarks.sh` is ever executed on the Spark; arbitrary code is not
runnable through the bridge (so new jobs — e.g. council-adapter training — require extending that
script on the Spark side, or the RunPod GitHub Action).

## Honest limits

- This is a **control plane**. Real metered GPU (RunPod) still goes through GitHub Actions per the
  repo guardrail; the bridge runs only owned-hardware (Spark/Mac) benchmarks.
- The cloud session cannot see `SESSION-COORDINATION.md` (untracked, local to the cluster), so it
  relies on `STATUS.json` + the owner for live ownership. `canClaimAGI` stays false.
