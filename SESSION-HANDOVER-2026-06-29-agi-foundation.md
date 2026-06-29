# Session Handover — 2026-06-29 (AGI-foundation advice → machinery → GitHub-Spark bridge)

> Continuation point for the next session/device. This session turned "advise on making the
> repo an AGI foundation" into committed, gated machinery + a working GitHub-mediated control
> bridge to the DGX Spark. `canClaimAGI` stays **false**. Nothing here promotes a result.

## 0. Branches / where things are
- **Feature branch `claude/agi-foundation-repo-advice-ssosit`** (tip `f7cb6b2`): all the code/docs
  below. Pushed, all gates green. **Not** merged to `main` (open a PR when ready — not done yet).
- **`spark-bridge`** branch: the GitHub-mediated control plane (see §3). The Spark's bridge
  worktree runs on it. As of handover the poller is live and ticking (~30s).
- Local `main` is the usual stale container lineage; `origin/main` (7259ac2) is source of truth.

## 1. What was delivered (committed on the feature branch)
- **Advice docs:** `agi-proof/AGI-FOUNDATION-ROADMAP.md` (P0–P4 priorities) +
  `agi-proof/independence-eval-plan.md` (the third-party hidden-eval that gates canClaimAGI).
- **P1 — instrument debt:** `tools/audit_fabrication_scorer.py` (+14 tests) — audits the
  deterministic fabrication scorer (`tools/run_seib.py::score_answer` + `agent/seib_contested_score.py`)
  vs a human-gold slice; confusion matrix + per-class P/R/F1 + bootstrap CI + threshold sweep.
- **P2 — independence machinery:** `agi-proof/benchmark-results/independence/measurement_spec.json`
  (required N=356 @ MDE 0.105), `tools/seal_eval_pack.py` (sha256 seal/verify),
  `tools/run_independence_eval.py` (raw-vs-gated × ≥3 families × two-box judges; `--emit-pending`
  writes a NO-GO/not_run artifact that gates NO-GO through `claim_gate --prefix independence-eval`)
  (+11 tests).
- **P3 — long-horizon harness:** `eval/long_horizon/{tasks,harness}.py` (deterministic per-step
  checkpoints, completion-rate / step-success / horizon-length + CIs), `tools/run_long_horizon_eval.py`,
  `agi-proof/benchmark-results/long-horizon/measurement_spec.json` (+14 tests).
- **P0 — one-command benchmarks:** `scripts/run_local_benchmarks.sh` (+ README), Makefile
  `bench-local` target. Found & handles 5 handover-vs-tool command mismatches (train_lora default
  base/scheme/output, `judge_pilot_answers` has no `--seed`, the A2→A3 schema assembly step).
- **NVFP4 v5 lever:** `tools/certify_lowram.py --keep-suffixes` (mixed precision — hold e.g.
  `down_proj` bf16; internals already supported `suffixes=`). Runbook Benchmark B now defaults to
  v5 (epochs 2→3 at the **stable λ=0.001**, output `olmoe-qat-spark-v5`, artifact
  `certify-lowram-olmoe-nvfp4-v5.json`), accepts `--dry-run` as an explicit no-op (+5 tests).
- **Failure ledger:** 4 new OPEN "machinery built — not yet run" entries (independence,
  long-horizon, fabrication-scorer audit, nvfp4-v5).

## 2. Live findings read through the bridge (not just docs)
- **GPU idle**; trainwatch shows 4 completed runs. **NVFP4 v4 is a NO-GO and a regression** vs v3:
  v4 `mean_kl 0.0537 (>0.05)`, `top1 0.926 (<0.97)`, `protected_max_kl 0.71` (over-fit at λ=0.01).
  **v3 remains best** (`mean_kl 0.045 ✓`, `top1 0.906 ✗`). Confirmed from the live cert artifacts.
- The MoE router gate, embeddings, norms, lm_head are **already** kept bf16 (so v5 is about
  mixed precision on the served projections + not over-training, not "keep the router bf16").

## 3. The GitHub-mediated Spark bridge (how the cloud agent reaches the Spark)
- **Why:** the cloud session's egress policy blocks the Spark's Tailscale Funnel
  (`*.tail9d1c70.ts.net` → `connect_rejected 502`); only allowlisted hosts (GitHub) are reachable.
  So both sides talk only to GitHub.
- **Branch `spark-bridge`** message queue (see `bridge/PROTOCOL.md`):
  `bridge/STATUS.json` (Spark→cloud snapshot), `bridge/commands/<id>.json` (cloud→Spark),
  `bridge/results/<id>.json` (Spark→cloud). Poller `tools/github_bridge_poll.py` runs on the
  Spark (tmux `bridge-poller`), ff-only sync, allowlist-enforced, `--execute`/`--run-train`
  require a non-empty `approvedBy`. Runs ONLY `scripts/run_local_benchmarks.sh`.
- **Proven:** dry-run command `2026-06-29-dryrun-all-01` round-tripped `status:ok exit 0`.
- **Cloud-side usage:** read `bridge/STATUS.json` / `bridge/results/*.json` via the GitHub MCP
  (`get_file_contents`, ref `refs/heads/spark-bridge`); write `bridge/commands/*.json` via
  `push_files`. No funnel/token needed (the funnel + its bearer token should be revoked).

## 4. ▶ Next steps (in order)
1. **Sync v5 code into the Spark bridge worktree** (Hermes, poller paused): remove the untracked
   `scripts/run_local_benchmarks.sh` shadow, `git merge origin/claude/agi-foundation-repo-advice-ssosit`
   into `spark-bridge`, restart poller. Verify `grep -c keep-suffixes` ≥1 in runbook + certify and
   `olmoe-qat-spark-v5` present. (The dry-run showed the worktree was running a *copied* runbook,
   not the tracked v5 one — this is why the sync is required before any v5 run.)
2. **Run NVFP4 v5** via the bridge: queue `bridge/commands/*.json` with `args:"--bench-b --run-train
   --execute"` + non-empty `approvedBy`. Reads back via the result's `exitCode`/`stdoutTail`. If
   top1 still < 0.97 at full quant, re-certify with `KEEP_SUFFIXES=down_proj` (conservative). On
   pass write the v5 artifact; else log NO-GO (claim only the aggregate mean_kl bound).
3. **Benchmark A (real ≥2-family judging)** — only once the **Llama-3.3-70B** judge (Mac, mlx :8081)
   is confirmed live alongside Qwen2.5-7B (Spark, vLLM :8000). The weak 8B judge caused the κ
   artifact; do not repeat it. Queue `--bench-a --execute` with `approvedBy`.
4. **Independence eval (P2, the canClaimAGI gate):** obtain an externally-authored sealed pack,
   plug ≥3 subject families + the two-box judges + reviewer signature, run `claim_gate --prefix
   independence-eval --assert-prereg`.
5. **PR the feature branch to `main`** when ready (not yet opened).

## 5. Guardrails (unchanged)
No-overclaim gate decides validity, never prose. Before commit/push: `make claim-check` + the
drift gates. Owned hardware is free; RunPod only via GitHub Actions + read `wisdom-gpu-prebaked`
first. `canClaimAGI` stays **false** until a third-party hidden eval is beaten (step 4).
