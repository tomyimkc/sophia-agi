---
name: spark-cluster-ops
description: >
  Run BEFORE launching any GPU job, opening a worktree/cherry-pick on this git-crypt repo,
  pushing to a shared feature branch, creating or deleting a RunPod pod, or coordinating with
  peer agents on the sophia-agi cluster (DGX Spark + Mac Studio + RunPod x86 + the bridge
  serializer). This repo runs 5+ concurrent Claude/Copilot/ops sessions on ONE checkout and ONE
  shared 128 GB Spark GPU; the local checkout is stale the moment it is read, and a stray heavy
  GPU load can silently KILL an in-flight cert/train. Use to: pick the right box for a workload,
  hold the one-GPU-job invariant, recover from git-crypt-broken worktrees, reconcile shared-branch
  rebases, and avoid the pod-deletion / cost-gate traps. Augments (does not replace) git-discipline,
  ci-artifact-drift, session-handover.
metadata:
  short-description: "Operate the Spark+Mac+RunPod multi-agent cluster: box selection, GPU gating, git-crypt worktrees, shared-branch rebase, pod discipline"
---

# spark-cluster-ops

Operational playbook for the multi-agent sophia-agi cluster. Cross-references **git-discipline**,
**ci-artifact-drift**, **session-handover** — read those for the basics; this skill carries the
net-new, hard-won cluster addenda. canClaimAGI stays false; every headline number is honest-or-NO-GO.

## 0. Cluster asset map — pick the right box FIRST
- **DGX Spark** (GB10 Blackwell, aarch64, 128 GB unified, native FP4, **273 GB/s**) = ITERATION +
  capacity tier. bf16 LoRA + NVFP4 QAT, low-RAM serving/cert (the deliverable runs here), 24/7
  gate-clean DATA REFINERY (70B NVFP4 teacher), one Qwen vLLM judge. Bandwidth-poor → batch-1
  decode is ALWAYS memory-bound. **Never the registered-number tier**; aarch64 forces
  `--quant bf16 --vllm none` (no bitsandbytes / x86 flash-attn / unsloth-prebuilt / mlx).
- **Mac Studio M3 Ultra** (MLX/Metal, **~819 GB/s ≈ 3× Spark**) = the FAST decode box → SECOND
  judge family (Llama via `mlx_lm.server`) + control plane (SSH into the Spark over Cat6). A
  different engine gives less-correlated judge errors → clears the ≥2-family κ≥0.40 gate with zero
  metered cloud. Don't try to cluster Mac+Spark (no shared collective backend).
- **RunPod x86** (H100/A100, GitHub-Actions-dispatched) = the ONLY registered / headline /
  source-of-record tier. Gated behind `environment: runpod-paid` (human approval) + `--yes`.
- **Bridge orchestrator** (`serve_benchmark_bridge.py` + Tailscale Funnel + `trainwatch:8420`) =
  the GPU-EXCLUSIVE poller that guarantees ONE GPU job at a time, allowlisted `POST /run`,
  auto-certify → `bridge/results/`. **It only protects work routed THROUGH it.**

Refs: `docs/11-Platform/{DGX-Spark-Maximization.md, Spark-Cluster-Capacity.md, Spark-Local-GPU-Lane.md, Mac-Spark-Judge-Farm.md}`, `bridge-info.txt`.

## 1. Multi-agent coordination — `SESSION-COORDINATION.md` is the live ownership ledger
- It is **UNTRACKED** (`??` in status), "ephemeral — not for commit", delete when parallel work ends.
  It is the real-time CLAIM/OWNS/DONE layer; the committed `SESSION-HANDOVER-YYYY-MM-DD.md` is the
  durable dated state. Don't conflate them.
- Before launching expensive work: **read it**, then append a block with your session id + pid and
  explicit OWNS / CLAIM / do-not-touch lists and anti-double-launch lines
  (e.g. "train-08 TRAINING under bridge poller PID 461280 — DO NOT disturb, 8th retry";
  "OPS agent X: don't also run the math-physics seeds — I've got it").
- Shared rules (`docs/12-Setup/Concurrent-Sessions-Worktrees.md`): keep the main checkout clean;
  feature work in worktrees; NEVER `git checkout`/`switch` the shared branch out from under another
  session; `--force-with-lease`, never `--force`.

## 2. The one-GPU-job invariant + 128 GB coexist gating  (biggest time-sink today)
- 128 GB unified fits TWO ~7B LoRA/QAT trainings without OOM, but they share compute+bandwidth
  (both slower). A THIRD heavy load broke it: a `qwen2.5:32b` ollama judge (~27 GB) loading mid-run
  coincided with a math-physics RLVR **seed killed mid-train** (empty checkpoint, no `rlvr.json`).
- There is **no canonical pgrep gate in the repo**. Real guards: (a) the bridge poller's single-job
  serialization, (b) a manual `nvidia-smi` headroom check.
- **RULE:** route ALL heavy Spark GPU work THROUGH the bridge poller. If you must launch directly in
  tmux/ollama, check `nvidia-smi` VRAM/util headroom FIRST and never start a heavy judge/teacher
  while a cert/train is live. (A local self-gating tmux orchestrator — pgrep `train_lora.py|certify_lowram.py`
  before each step — is a fine *session-local* mechanism, but the bridge is the cluster-wide lock.)

## 3. git-crypt worktrees / cherry-pick — the `-c` bypass + collision recovery  (→ fold into git-discipline)
- `.gitattributes` routes `secret/**, AGENTS.md, CONTRACT.md, .claude/skills/**, .grok/**,
  docs/superpowers/**` through `filter=git-crypt`; any checkout/worktree/cherry-pick re-runs the
  smudge filter and aborts (`external filter '...' failed` / `smudge filter git-crypt failed`) when
  the key isn't loaded. Only git-discipline / ci-artifact-drift / session-handover skills are
  exempted (`!filter !diff`), so they stay readable when locked.
- **Bypass that ONE git op** (cat = passthrough, leaves ciphertext as-is):
  ```
  git -c filter.git-crypt.smudge=cat -c filter.git-crypt.clean=cat -c filter.git-crypt.required=false \
      worktree add -b feat/<x> <dir> origin/main
  git -c filter.git-crypt.smudge=cat -c filter.git-crypt.required=false cherry-pick <sha>
  git -c filter.git-crypt.required=false add <file> && git -c filter.git-crypt.required=false commit -m ...
  ```
- **COLLISION RECOVERY** — a *failed* `worktree add -b` STILL created the branch ref, and your
  subsequent cherry-pick landed on the SHARED checkout (the ref is created before the dir is
  materialized). Recover:
  ```
  git branch -f feat/<x> <your-commit-sha>        # 1. save your work onto the feature branch
  git reset --hard <shared-branch-PRIOR-HEAD>      # 2. restore the shared checkout (NOT necessarily origin/main)
  git push --force-with-lease origin feat/<x>      # 3. push from a clean isolated worktree
  ```
  Prevention: ALWAYS pass the `-c` overrides on `worktree add` so create+materialize is atomic.

## 4. Shared-branch rebase reconcile + the never-stage run-result  (→ fold into ci-artifact-drift)
- Publish on a shared feature branch: stage ONLY your explicit files (**never `git add -A`**) →
  `git commit` → `git pull --rebase origin <branch>` → `git push`.
- `agi-proof/benchmark-results/rlvr.public-report.json` is a **TRACKED** file that every RLVR run
  rewrites locally and is **NOT gitignored**, so it perpetually shows `M` and blocks the rebase
  ("cannot rebase: you have unstaged changes"). **NEVER stage it** (that commits a machine-specific
  run result). Unblock: `git checkout -- agi-proof/benchmark-results/rlvr.public-report.json` then re-rebase.
- Also never stage: `.venv/`, `training/rlvr/checkpoints/`.

## 5. RunPod cost-gate + pod-deletion discipline
- **CORRECTION (don't repeat the wrong lore):** PR #276 adds GitHub Environment `runpod-paid`
  (required reviewer @tomyimkc) to the paid pod-CREATING jobs. It **PENDS** a paid job for a human
  approval BEFORE pod creation — it is **NOT a reaper and CANNOT kill running pods**.
- The real 3-pod + bridge mass-termination was a **PEER AGENT manually deleting pods it wrongly
  thought leaked** from its own run (balance was fine, $71). The only automated reaper
  (`tools/runpod_connect.py reap_exited`) terminates ONLY `verdict=='stopped'`/EXITED leaks — never running pods.
- **RULES:** launch ONE paid pod at a time and **announce its pod id** in SESSION-COORDINATION.md
  (`[runpod] created pod <id>`) so peers don't mistake it for a leak. **Delete ONLY pods whose id YOUR
  run printed** — never reap by inference. Reap real leaks with `python tools/runpod_connect.py
  --reap-exited --yes`. Agents must NEVER self-approve `runpod-paid`. Prefer the pre-baked image over
  a persistent `--network-volume-id` (volume break-even ~90 paid runs/mo).

## 6. Merge mechanics  (cross-ref git-discipline; don't restate)
- `main-protection` requires checks `fast` + `ci-complete`, `required_linear_history`,
  `required_review_thread_resolution:true` (the SILENT blocker — green checks but unresolved threads
  still block), `required_approving_review_count:0`; direct pushes blocked.
- Two-phase: queue `gh pr merge <N> --squash --auto --delete-branch`; if it stalls on the
  review-thread gate once checks are green, override `gh pr merge <N> --squash --admin --delete-branch`.
  Always `--squash` (linear history).

## 7. Training arrangement — division of labor for max productivity
**Iterate cheap on the Spark → register once on RunPod → judge independently on the Mac.**
1. **DGX Spark = always-busy iteration+capacity engine** (serialize via the bridge): (a) the low-RAM
   NVFP4 QAT subject under test — `train_lora.py --qat --target-modules attn-mlp` + `certify_lowram.py`
   (the deliverable runs HERE, not RunPod); (b) when otherwise idle, the always-on DATA REFINERY
   (70B NVFP4 teacher, gate-clean SFT/RFT 24/7 — highest ROI, since the capability lever is data
   quality not train speed); (c) one Qwen vLLM judge. Never a registered number here (aarch64 numerics).
2. **Mac Studio = fast decode** → second judge family (Llama via mlx) + control plane. Bring-up over ssh:
   `nohup mlx_lm.server --model mlx-community/<...> --port <P> >/tmp/mlx-judge-<P>.log 2>&1 & disown`
   (**no `setsid` on macOS**) and set **`OPENAI_API_KEY=none`** for the keyless OpenAI-compatible
   transport (else the `openai:` client sends empty auth and the judge silently yields **n=0** — the
   real root cause of the earlier "70B judge n=0").
3. **RunPod x86 = the only source-of-record tier.** Dispatch from GHA behind `runpod-paid` + `--yes`,
   SPARINGLY, only for the result-of-record arm, with the harness PINNED (see rlvr-harness-traps §C).
   `runpod_rlvr.py --task step --step-domain math` copies back `{pod_id}.rlvr.adapter-eval.json` and
   auto-deletes the pod unless `--keep-pod`.
4. **Bridge + trainwatch = connective tissue.** Keep binds localhost, expose only via Tailscale Funnel
   + bearer token, only allowlisted flags reach `run_local_benchmarks.sh`, rotate any token pasted in chat.

**Avoid the four failure modes that cost time today:** (i) one-GPU-job invariant — route everything
heavy through the bridge, `nvidia-smi` headroom-check before any direct tmux/ollama launch;
(ii) own-only pod deletion + explicit SESSION-COORDINATION claims — never double-launch a seed,
never reap by inference; (iii) pin numerics for comparability — temp=0 headline judging, freeze the
harness across a sweep, headline numbers ONLY from x86 RunPod, re-run clean rather than clearing a
candidate on mixed metrics; (iv) Spark as resilient fallback — default latency-tolerant iteration to
the free Spark (reuse already-done seeds) and reserve RunPod for the registered arm, so a cost-gate
pend or a peer reap just reverts to Spark.
