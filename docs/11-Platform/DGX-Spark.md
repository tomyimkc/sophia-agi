# DGX Spark integration — the iteration tier

**Status:** machinery implemented, NOT yet run on an actual Spark (OPEN in the failure
ledger). `canClaimAGI = false` throughout — a Spark changes iteration velocity, not the
claim ladder.

## The one rule (governs everything)

An NVIDIA DGX Spark (GB10 Grace Blackwell, **aarch64/sm_121a**, 128 GB unified LPDDR5x)
on the desk is the **iteration / judge / distillation / CI-smoke tier** — NOT the
headline-evidence tier. Per `agi-proof/third-party-replication/REPLICATION.md`, **headline
numbers stay on x86 RunPod**: GB10 is not datacenter Blackwell, and sm_121a-specific
builds can diverge. A Spark removes the "blocked on cloud" pain for *iteration*; it does
not license citing its numbers as the registered result.

## Why this matters for the verifier-as-reward work

The live GRPO / code-RLVR / multi-generation compounding runs are OPEN precisely because
there is no persistent local NVIDIA box (the dev is on Apple Silicon). A Spark turns those
from "blocked on RunPod SSH/cost" into "runs on the desk" — including overnight compounding
generations and a local ≥2-family judge farm.

## The four uses (mapped to what's implemented)

| Tier | Use | Seam |
|---|---|---|
| **Inference** | vLLM orchestrator + SGLang constrained tool-caller; cloud Claude for escalation | `config/inference.local.spark.json` → `agent/inference_topology.py` router; `scripts/spark_serve.sh` |
| **Iteration training** | RLVR + LoRA SFT + the compounding loop, locally | `tools/runpod_rlvr.py --local` (RLVR); `tools/runpod_train.py --local` (SFT, bf16) |
| **Local judge farm** | ≥2 distinct local judges (Qwen + Llama) for the no-overclaim gate, free | `scripts/spark_judge_farm.sh`; `--judge vllm:…@url --judge vllm:…@url` |
| **Distillation** | local teacher → disciplined student adapter | `tools/run_distill_local.py` (+ `build_distill_dpo_pairs.py` v2) |

Supporting: `tools/spark_vs_runpod_ab.py` (data-backs the iteration/headline rule);
`.github/workflows/spark-smoke.yml` (self-hosted CI lane).

## aarch64 / GB10 reality (and what to re-test)

Known blockers (use the workarounds):
- **vLLM colocate + QLoRA-4bit**: risky on aarch64 → `--vllm none --quant bf16` (or `--vllm server`).
- **bitsandbytes**: aarch64 wheel pain → bf16, not 4-bit (the 128 GB pool fits it).
- **flash-attn**: prebuilt wheels are x86 → build-from-source or skip.

**Re-test, don't blanket-avoid:**
- **Unsloth**: now ships an official DGX Spark path (1.5–7× LoRA speedups reported). The
  old "avoid unsloth (no aarch64 build)" note is likely outdated — `train_lora.py --backend
  unsloth` already exists; probe it on the Spark before committing to peft-only.

**Never on a Spark:** `engine: mlx` (Apple-only — `agent.inference_topology` refuses it
fail-closed), prebuilt `linux_x86_64` flash-attn wheels.

## The honest boundary (a Spark does NOT change these)

1. **Headline numbers stay x86 RunPod.** Measure the divergence with
   `tools/spark_vs_runpod_ab.py` so the rule is data-backed.
2. **A Spark does not close the third-party gap.** You own the Spark; ownership is the
   contamination risk, not the hardware. The `agi-proof/third-party-heldout/` pack is still
   the only clean-external claim.
3. **`canClaimAGI` stays false.**
4. **The Spark CI lane is `candidateOnly`** if it ever runs real training — never auto-promote.

## Getting started (once you have a Spark)

```bash
# 1. serve the two local tiers
bash scripts/spark_serve.sh
# 2. point Sophia at them
cp config/inference.local.spark.json config/inference.local.json   # then edit
python -m agent.inference_topology                                   # show resolved tiers
# 3. iterate RLVR locally (no RunPod)
python tools/runpod_rlvr.py --local --task code --quant bf16 --vllm none --epochs 1 --seed 0 --yes
# 4. distill a disciplined student
python tools/run_distill_local.py --teacher vllm:Qwen/Qwen2.5-14B-Instruct@http://localhost:8000/v1 \
    --student-model Qwen/Qwen2.5-3B-Instruct --yes
# 5. local 2-family judge farm for the no-overclaim gate
bash scripts/spark_judge_farm.sh
```

See the failure ledger for the OPEN items each of these still needs (a real run on the box).
