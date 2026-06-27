# long_horizon_30m — live GRPO RunPod run: in-flight status (2026-06-27)

**Owner:** Claude (Opus) takeover session. `canClaimAGI` stays **False**. This file
tracks an IN-PROGRESS attempt to clear the `long_horizon_30m` (1800s `short-30min`)
tier with a GENUINE GPU run; it is NOT a clearance claim.

## Decision (human-approved this session)
- Path: **GPU RunPod dispatch, live GRPO** (real training, naturally 30+ min).
- Credentials: human added a fresh **RUNPOD_API_KEY repo secret** (confirmed).
- Dispatch mechanism: **merge the wrapper workflow to main** (GitHub only registers
  `workflow_dispatch` workflows that live on the default branch), then dispatch.

## What is built and committed (this branch)
- `.github/workflows/long-horizon-runpod.yml` — dispatch-only, `confirm=RUN` + secret
  gated. Runs `tools/run_long_horizon.py` **on the GitHub runner** (which has SSH
  egress to the rented pod; the CPU box does not), wrapping the real live
  `tools/runpod_rlvr.py` GRPO launch as its long step. The harness's own monotonic
  clock therefore times the real GPU compute end-to-end → a `>=1800s` tier is genuine,
  not padded. This is the honest path routed by `HONEST-WALL-2026-06-27.md` §4.
  The workflow commits the harness JSONL+report back to the dispatch branch (durable).
- `agi-proof/long-horizon-runs/runpod-grpo-30m.spec.json` — the harness spec
  (provenance → dry-run → LIVE GRPO → SSIL Layer-1 ingest → objective gate).

## Planned dispatch parameters
- task=`provenance`, epochs=`1.0`, seed=`0`, gpu=`NVIDIA A100 80GB PCIe`,
  interruptible=`false` (on-demand; spot gets preempted mid-pip-install),
  remote_mode=`live`. Est. ~30–60 min, ~$1.5–3 on-demand. Pod is ALWAYS deleted by
  `runpod_rlvr.py` (finally block + remote watchdog).

## Current blocker (environment, not the plan)
- The **GitHub MCP server is intermittently/fully disconnected** this session. It is
  required to (a) open+merge the workflow PR to main and (b) dispatch the workflow.
  Raw `gh`/GitHub API is disallowed in this environment; direct push to main is
  rejected (protected + main advancing). Waiting for GitHub MCP to return.
- The **RunPod MCP server returns 401** (its configured key is stale/compromised), so
  independent pod-hygiene checks via RunPod MCP are unavailable. Pod cleanup relies on
  `runpod_rlvr.py`'s always-delete + remote watchdog (run inside the Action with the
  repo secret) and will be verified from the Action logs.

## Next steps when GitHub MCP returns
1. Open PR (this branch → main) for the two workflow/spec files; merge it.
2. Dispatch `long-horizon-runpod.yml` on this branch with the parameters above
   (`confirm=RUN`).
3. Monitor the Action to completion; the workflow auto-commits the harness report here.
4. Verify the committed report tiers `short-30min`; confirm pod termination from logs;
   update `agi-proof/failure-ledger.md` (`long-horizon-not-run`) honestly.
