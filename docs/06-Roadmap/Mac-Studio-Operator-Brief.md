# Mac Studio operator brief — utilize the Mac + drive the Spark GPU 24/7

You have **Claude Code on the Mac Studio, signed into this account**. That unlocks something the cloud
session **cannot** do: the Mac is on your LAN and **can SSH to the Spark**, while the cloud session
can only reach it through the git bridge. So the Mac becomes the **local operator** for every action
the cloud is structurally blocked from. `canClaimAGI` stays false.

## 1. What the Mac Studio is FOR (three concurrent roles)
1. **Spark operator (the big unlock).** SSH to the Spark to do what the bridge can't: pin/repair the
   `peft`/`transformers` version (fixes the fused-expert merge — the v5 cert blocker), apply staged
   poller/script patches, restart the Hermes poller, and **recover the bridge if it ever wedges**
   (the cloud can't, since it has no SSH).
2. **Judge / eval lane (free, parallel to the Spark).** 512 GB unified + MLX runs the 70B judge and
   the eval harnesses *concurrently* with the Spark GPU lane — the second free heavy lane in the
   30-day plan.
3. **A second worker (CPU lane).** Run a Claude Code agent on the Mac for builds/tests/docs/curation
   in parallel with the cloud — and as a **continuity orchestrator** if the cloud session ends.

Install the **OKF logging bundle** here too (hook + skill) so the Mac's agent logs to the same corpus
(set `OKF_DEVICE=mac-studio`).

## 2. Drive the Spark GPU 24/7 — deploy the idle backlog
The poller now has an **idle-backlog** feature (patch
`scripts/spark/2026-06-30-poller-idle-backlog.patch`): when the queue is empty, nothing is running,
and no `bridge/JUDGE_HOLD` file exists, it materializes ONE job from `bridge/backlog.jsonl` into a
normal command — so the GPU stays fed, but **any real command or a JUDGE_HOLD always preempts it**.

**Why the Mac, not the cloud, deploys this:** it changes the poller's control loop, and if a control-
loop change misbehaves only a machine that can SSH can recover it. The cloud can't. So the Mac applies
+ restarts + watches the first few cycles.

Deploy steps (on the Mac, SSH'd to the Spark checkout `/home/tomyimkc/sophia-bridge`):
```bash
# 1. sync spark-bridge, apply the verified patch (already --check-clean against spark-bridge)
cd /home/tomyimkc/sophia-bridge && git fetch origin spark-bridge && git checkout spark-bridge && git pull
git apply scripts/spark/2026-06-30-poller-idle-backlog.patch   # or fetch the patch from the feature branch
python -m py_compile tools/github_bridge_poll.py               # sanity
# 2. seed the backlog (bridge/backlog.jsonl is pre-staged on spark-bridge; edit to taste)
# 3. restart the poller so it loads the new code, then WATCH a few ticks
#    (confirm: it seeds only when STATUS.running==null && no pending && no JUDGE_HOLD)
# 4. reserve the GPU for judging any time with:  touch bridge/JUDGE_HOLD   (rm to resume)
```

### Safety rules baked in (do not remove)
- **Judge priority:** backlog seeds ONLY when the queue is empty; a real command always wins. Before
  a judge run, `touch bridge/JUDGE_HOLD` to stop the backlog filling the gap; `rm` it after.
- **Bounded jobs only:** the GPU is serial, so a long backlog job delays a judge job until it finishes.
  Keep backlog entries SHORT (cert-only `--bench-b` n≈64–256, short evals — minutes, not the 2.5 h
  train). Put the long v6 train through the normal queue when you can watch it, not the backlog.
- **Consumed once:** each entry is removed when seeded, so nothing re-runs unless it re-enqueues.
- **Fail-open:** a malformed backlog line is dropped, never wedges the loop.

## 3. The highest-value Mac job right now (unblocks the v5 cert)
SSH to the Spark and fix the merge instrument so the cert measures the FULL adapter:
- **Option A (cheapest):** pin a `peft`/`transformers` pair where `PeftModel.merge_and_unload` works
  on OLMoE fused experts (kills the `WeightConverter ... 'distributed_operation'` crash). Then re-cert
  v5 → the first TRUE v5 top1 (96/96 merged). No retrain.
- **Option B:** train v6 with experts loaded as per-expert `nn.Linear` (so the adapter is 2-D,
  name-matched, `.weight`-bearing → the manual merge handles it natively), per the v6 recipe.
Apply the staged `--bench-virtues` merge patch + the faithfulness allowlist patch too, then those
benches become runnable (good backlog entries afterward).

## 4. One-line org
Cloud = bridge-only planner/dispatcher. **Mac Studio = the hands** (SSH operator + judge lane + 2nd
worker + bridge recovery). The idle backlog keeps the Spark at ~0% idle; JUDGE_HOLD + bounded jobs
keep judging first-class.
