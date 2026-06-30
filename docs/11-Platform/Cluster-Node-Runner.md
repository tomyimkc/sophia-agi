# Cluster node runner — the per-Spark daemon (git-bridge PROTOCOL-v2)

> **Status: design/infra; no capability claim; canClaimAGI stays false.**
> The daemon that runs ON each DGX Spark so the cluster self-coordinates over GitHub with NO Claude
> session attached. It generalizes the single-node poller (`tools/github_bridge_poll.py`, branch
> `spark-bridge`) to many nodes sharing ONE job queue, on top of the already-landed
> `tools/cluster_scheduler.py`. Every node talks only to GitHub; coordination is git-mediated.
> Reuses the `spark_bridge` allowlist + gating verbatim (not widened). Code:
> `tools/cluster_node_runner.py`; tests: `tests/test_cluster_node_runner.py`.

## What it is

`tools/cluster_node_runner.py` is the node-side loop. It splits cleanly into:

* a **PURE core** (`decide`) — no git, no exec, no GPU, no clock (`now`/timestamps are passed in) —
  that, given `(node_id, all_node_ids, pending_command_ids, existing_claims, my_node_status, now,
  commands_by_id)`, returns the next ACTION; and
* **thin impure wrappers** (`_sync`, `_read_pending`, `_read_claims`, `_write_claim`,
  `_publish_node_status`, `_run`) modelled on the single-node poller, isolated exactly like
  `spark_bridge._git_show`. Only these touch git / Popen; they are exercised live on the Spark, not
  in CI.

`tick()` composes them. The decision rule (from the PURE core):

| Situation | Action |
| --- | --- |
| A pending cmd whose `assigned_node(cmd_id, sorted node ids) == me`, no live claim, I am free | `claim` |
| I hold my own live (unexpired) claim on an assigned, still-pending cmd, and I am free | `run` |
| That assigned cmd carries `--execute`/`--run-train` but has no human `approvedBy` | `refuse-gated` |
| I am busy (one GPU job already running), or nothing is eligible for me | `idle` |

A crashed node's claim re-queues automatically via `lease_is_expired` (the lease has a TTL; once
`leasedAt + ttlSeconds < now`, the owner re-claims).

## How it preserves every single-node invariant — PER NODE

* **One GPU job per node.** `node_is_free(my status)` mirrors `spark_bridge.gpu_is_free`. A busy
  node `idle`s even on its own assigned, unclaimed command, so it never starts a second job.
* **Allowlist, not widened.** `_run` re-validates with `spark_bridge.validate_args`; only
  `scripts/run_local_benchmarks.sh` is ever executed, only with allowlisted flags.
* **No AI self-approval of gated work.** A command carrying `--execute`/`--run-train` with no human
  `approvedBy` is `refuse-gated` in the PURE core AND re-checked at exec in `_run`. The runner
  NEVER self-approves a GPU job — the cloud operator (`tools/spark_bridge.py`) only ever COMPOSES
  human-approved commands; the runner merely executes what a human already approved.
* **Exclusive ownership.** `assigned_node` (deterministic sha1-mod over the SORTED node list) means
  each command has exactly one owner. Only that owner acts on it; everyone else idles. No central
  dispatcher, no `random`, no clock.
* **Claim = git compare-and-swap.** `_write_claim` creates `bridge/claims/<cmd_id>.json` and pushes;
  the first push fast-forwards (wins), a loser gets a non-ff rejection, re-syncs ff-only, and sees
  the winner's claim on the next `decide`.

## Deploy on each Spark (one process per node, unique `--node-id`)

Every node runs ONE runner pointing at the **`spark-bridge`** branch with a unique `--node-id`. The
node-id set can be passed explicitly with `--nodes` or discovered from `bridge/nodes/`.

`nohup` (quickest):

```bash
# On spark-2f2d:
cd /home/tomyimkc/sophia-bridge
nohup env PYTHON=/home/tomyimkc/sophia-agi/.venv/bin/python \
  python3 tools/cluster_node_runner.py \
    --node-id spark-2f2d \
    --branch spark-bridge \
    --nodes spark-2f2d,spark-aaaa,spark-bbbb,spark-cccc \
    --interval 30 \
    --trainwatch http://127.0.0.1:8420/api/runs \
  >> ~/node-runner.log 2>&1 &
```

`systemd` unit (`/etc/systemd/system/sophia-node-runner.service`, one per node with its own
`--node-id`):

```ini
[Unit]
Description=Sophia cluster node runner (spark-2f2d)
After=network-online.target

[Service]
WorkingDirectory=/home/tomyimkc/sophia-bridge
ExecStart=/home/tomyimkc/sophia-agi/.venv/bin/python3 tools/cluster_node_runner.py \
  --node-id spark-2f2d --branch spark-bridge \
  --nodes spark-2f2d,spark-aaaa,spark-bbbb,spark-cccc \
  --interval 30 --trainwatch http://127.0.0.1:8420/api/runs
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
```

Dry-run a node's next decision without any side effects (no sync, no claim, no run):

```bash
python tools/cluster_node_runner.py --node-id spark-2f2d --nodes spark-2f2d,spark-aaaa --once
```

Verify the pure core offline at any time: `python tools/cluster_node_runner.py --self-test`.

## Mac-judge contention (pair with the judge pool)

When several Sparks each finish a gated train/bench and request judgments, they contend for the same
Mac Studio judge. The runner's per-node claim/lease keeps GPU work serialized PER NODE, but it does
NOT itself serialize the downstream judge. Pair this runner with the **judge pool**: route judge
requests through the same git-bridge claim discipline (one judge lease at a time), so N nodes
completing in parallel do not stampede a single Mac judge. The lease TTL also bounds a hung judge —
an expired judge lease re-queues just like a crashed node's command.

## Deployment is an owner-applied patch

The cloud operator CANNOT push to `spark-bridge` (egress to the Spark's Funnel returns
`connect_rejected 502`; the cloud only drives the bridge through GitHub command files). Therefore
**deploying this runner requires landing `tools/cluster_node_runner.py` on the `spark-bridge`
branch** — a patch the repo owner applies. This file is delivered on the feature branch
`claude/sophia-positioning-gaps-84kb0v`; the owner cherry-picks / applies it onto `spark-bridge`,
then starts one runner per Spark with the lines above. The runner does not, and cannot, self-deploy.
