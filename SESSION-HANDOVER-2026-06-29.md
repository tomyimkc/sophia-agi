# Session Handover — 2026-06-29 (DGX Spark + Mac Studio)

Continuation of the OLMoE NVFP4 + M3-SFT ≥2-family benchmark plan. **All work is committed on
branch `claude/olmoe-qat-certification-dgx-ctv737`** (NOT yet on `main` — `main` is heavily
contended by other sessions' merges; land via PR when quiet). `canClaimAGI` stays **false**.

## Hardware / environment (now live)
- **DGX Spark** (this box): GB10 Grace-Blackwell, aarch64, CUDA 13, 128 GB unified. torch 2.14.0.dev,
  transformers **4.46.3** (NO gemma-3 support — see Task A), peft 0.19.1. Venv `.venv`.
  **Cost-guard rule:** owned hardware → free; RunPod is the only paid path (gated workflow).
- **Mac Studio** `TomdeMac-Studio.local` — **M3 Ultra, 96 GB**. Reached from the Spark over a direct
  **10GbE Cat wire**: Spark `enP7s7` = `169.254.26.170/16` ↔ Mac `169.254.26.171/16` (link-local,
  set via `nmcli`, no sudo). SSH: `ssh -i ~/.ssh/id_ed25519 tom@169.254.26.171`. mlx-lm 0.31.2 in
  `/Users/tom/.pyenv/versions/3.10.6/bin`.
- **Tailscale** up: Spark = `spark-2f2d` / `100.119.221.44` (reachable from anywhere on the tailnet).
- **git-crypt** unlocked (GPG key `FFF449113309A1EF` imported; `git-crypt` 0.7.0 staged in `~/.local/bin`).

## Results this session

### Task B — OLMoE-1B-7B NVFP4 low-RAM certification → honest NO-GO (v3 best)
- **Root-cause bug fixed (`77a1076d`):** `training/qat.attach_qat` wrapped PEFT's `lora.Linear`
  (also class-named `Linear`), replacing its forward with base-only fake-quant → the adapter got
  zero gradient → **every prior OLMoE cert measured an untrained no-op** (all `lora_B` == 0). Fix +
  regression test `tests/test_qat.py::test_attach_qat_does_not_bypass_lora_adapter`.
- **v3** (expert-co-adapted, 2 ep, λ=0.001): **mean_kl 0.045 (PASSES ≤0.05)**, top1 0.906 (< 0.97).
- **v4** (3 ep, λ=0.01): top1 0.926 but mean_kl 0.054 (regressed, overfit) — not better. **v3 is best.**
- Verdict: NO-GO on the strict gate (top1 ≥ 0.97 unreached), but expert-QAT makes NVFP4 serving
  substantially more faithful than baseline; the mean-KL bound is achievable. Docs: cert report
  `docs/11-Platform/OLMoE-NVFP4-Certification.md` + ledger; artifacts `certify-lowram-olmoe-nvfp4-v{3,4}.json`.

### Task A — M3-SFT two-box ≥2-family judging → strong CANDIDATE (not VALIDATED)
- **Two-box judge farm:** Spark `ollama:qwen2.5:7b-instruct` (family `ollama`) + Mac Studio
  `mlx_lm.server` reached via **`openai:`-transport** (family `mlx-community`). 2 distinct families.
- **mlx-judge bug fixed:** agent.model's `mlx:` provider is a *local* mlx_lm loader → returns empty/
  all-TIE on the Spark. Use `openai:<model>@http://169.254.26.171:PORT/v1` to hit a remote mlx server.
- **Result (3 full seeds, 268 source cases each):** with a capable 2nd judge (**Llama-3.3-70B-4bit**),
  **both families significantly favor the adapter every seed** — pooled **Qwen 0.72 [0.687,0.750]**,
  **Llama-70B 0.81 [0.777,0.832]** (CIs exclude 0.5). A weak Llama-8B-4bit had shown that family null
  (~0.50) — a judge-capability artifact.
- **κ = 0.24 < 0.40 floor** → formal gate unmet (prevalence-deflation; PABAK ≈ 0.42). So **CANDIDATE,
  not VALIDATED.** Promoted to `published-results.json` → `illustrative[]` (labeled candidate) + RESULTS.md.
  Artifacts: `agi-proof/benchmark-results/wisdom-market/m3-2family-judge{,-70b}/`.
- **M3-SFT adapter is reproduce-by-design** (registry `sophia-wisdom-4b-pilot-seed0`, weights not
  committed). Local retrain impossible (transformers 4.46.3 lacks gemma-3). Reproduced full answer
  seeds via the **`wisdom-pilot-runpod` workflow on the quiet `claude/sophia-wisdom-4b-roadmap-jyesip`
  branch** (NOT `main` — `main`'s contention silently drops the pod's push-back). Full seeds: 1, 2, 3.

## Tooling built (committed)
- **`/trainwatch`** Claude Code command (local sqlite OR remote HTTP over Tailscale; installed on the
  Spark + Mac `~/.claude/`). `tools/trainwatch_stats.py` (pure stdlib) + `tools/trainwatch_bridge.py`
  (streams any `train_lora` log into TrainWatch). Dashboard: `trainwatch serve` → `http://spark-2f2d:8420`.

## Servers / processes left RUNNING
- Spark: `ollama serve` (:11434, `~/ollama-local/bin/ollama`), `trainwatch serve` (:8420).
- Mac: `mlx_lm.server` Llama-3.1-8B-4bit (:8080) + Llama-3.3-70B-4bit (:8081). Kill via `lsof -ti:PORT|xargs kill`.

## Open / next
1. **Land this branch on `main`** via PR when `main` is quiet (much committed value).
2. **Do NOT chase formal VALIDATED for M3-SFT** — κ=0.24 is mathematical prevalence-deflation, not
   fixable without goalpost-moving. The win-rate panel (both families significant) is the honest headline.
3. **Task C (untouched):** retention/transfer at seeds 1,2 (`tools/runpod_wisdom_pilot_selfreport.py
   --mode retention|transfer`); external-independence via SimpleQA Verified.
4. The other session `claude/workflow-skills-mcp-setup-d1yaoy` adds Claude-Code hooks/process-skills
   (auto-unlock, git-discipline, session-handover) — complementary, no file overlap; let it land.

One rule held throughout: never report a number without its CI, seeds, judge families, and
candidate/validated label. Everything above is candidate/NO-GO-labelled; nothing claims VALIDATED.
