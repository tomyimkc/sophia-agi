# sophia-7b-train-verify

Pre-registered Qwen2.5-7B QLoRA SFT (Stage 2) and DPO (Stage 3) for the local Sophia wisdom
model — **not** an AGI claim. See `preregistration.json`, `oracle-split.md`, and
`heldout-seal.manifest.json`.

## Policy: RunPod via GitHub Actions only

**RunPod GPU training for this experiment MUST run through GitHub Actions**, not from a local
Mac shell or Cursor agent terminal.

Local attempts failed with `runpod_pod_ssh_egress_timeout`: RunPod API reported SSH port mapping,
but outbound SSH login to mapped `ip:high-port` timed out from the Cursor agent host (while
TCP/22 to `github.com` and `ssh.runpod.io` succeeded). Evidence:

- `stage2-runpod-blocker.public-report.json`
- `agi-proof/benchmark-results/runpod-train/ssh-smoke-20260625-111011.log`
- `agi-proof/failure-ledger.md` (section `sophia-7b-train-verify-data-flywheel-2026-06-25`)

GitHub Actions `ubuntu-latest` runners have reliable egress to RunPod mapped pod ports (same
pattern as `train-runpod`, `speedup-runpod`, `rlvr-runpod`).

## How to trigger

1. Ensure repo secret **`RUNPOD_API_KEY`** is set (Settings → Secrets and variables → Actions).
2. Open **Actions** → **runpod-sophia-7b-sft** → **Run workflow**.
3. Set **confirm** to `RUN` (acknowledges GPU cost).
4. Choose **stage**: `sft` (Stage 2, three seeds by default) or `dpo` (Stage 3, needs SFT tarballs).
5. Optional: **seed** (`0`, `1`, or `2`) for a single seed; leave empty for all three.
6. For **dpo**: supply **sft_workflow_run_id** — the Actions run ID of a completed SFT workflow
   whose `runpod-sophia-7b-artifacts` were uploaded.

Artifacts (14-day retention): train logs, `eval_ladder` JSON, `promotion.public-report.json`,
`repo-head.txt`, holdout seal manifest. Adapter tarballs are included when present (~hundreds of
MB per seed; not multi-GB full weights).

## Local scripts (GHA runners only)

These shell wrappers are invoked by the workflow; do not run them from Cursor/local Mac:

- `runpod-sft-3seed.sh` — QLoRA SFT, seeds 0–2, `training/local_sophia_7b/mlx/train.jsonl`
- `runpod-dpo-3seed.sh` — DPO on `dpo_hard_negatives.jsonl` atop per-seed SFT adapters

Preflight (also run in the workflow): `build_local_sophia_dataset.py --check`,
`seal_sophia_7b_holdout.py --check`, `lint_claims.py`.

`canClaimAGI`: **false** on all artifacts until external evidence gates pass.
