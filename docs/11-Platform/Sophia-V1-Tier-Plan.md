# Sophia-V1 tier plan — low → top-notch, and where the DGX Spark fits

**Status:** implementation plan. No capability claim; governed by `Cheap-Compute-Boundary.md`.
Launcher: `tools/runpod_qat_lowram.py` (`--tier {low,mid,high,top} --target {local-spark,runpod}`).

> One honest thread through every tier: at frontier scale you **adapt + serve** an existing open
> MoE — you do **not** pretrain one (671B from scratch ≈ thousands of H100-days, ~$5–10M). The
> **low-RAM win is real and certifiable at every tier** — that is the deliverable.

---

## The four tiers

| Tier | Open MoE base (`--tier`) | Total / Active | GPU (RunPod, in stock) | ~Cost/run | Supported today? |
|---|---|---|---|---|---|
| **low** | `allenai/OLMoE-1B-7B-0924-Instruct` | 7B / 1B | 1× RTX 4090/5090, **or your Spark** | **~$3–10 / FREE on Spark** | ✅ single-GPU |
| **mid** | `mistralai/Mixtral-8x7B-Instruct` | 47B / 13B | 1× A100/H100 80 GB, **or your Spark** | ~$20–60 / FREE on Spark | ✅ single-GPU |
| **high** | `mistralai/Mixtral-8x22B-Instruct` | 141B / 39B | 1× H200 141 GB / B200 / MI300X, or 2–4× H100 | ~$150–500 | ⚠️ needs expert-parallel/FSDP |
| **top** | `deepseek-ai/DeepSeek-V3` | 671B / 37B | multi-node 8× H200/B200 | ~$1k–5k (adapt+certify) | ⚠️ needs multi-node sharding |

Resident-RAM payoff (from `serving/lowram_runtime.plan_ram`, NVFP4): a 141B MoE serves in **~tens of
GB**; the 671B target in **~25 GB (one H200) / ~5 GB streamed**. That is the whole point.

### Per-tier engineering state
- **low / mid (single GPU):** runnable **now**. `tools/train_lora.py --qat` + QLoRA (RunPod) or bf16
  (Spark). The merged code does this.
- **high / top (multi-GPU / multi-node):** the remaining code investment is **expert-parallel +
  FSDP sharding** for `train_lora.py` (it is single-GPU today). This is the gating work before a
  top-tier run is one click away — tracked as the next milestone.

---

## How to use your DGX Spark (GB10 Blackwell, 128 GB unified, aarch64)

The Spark is a **great fit for three of the four jobs**, and a poor fit for one — be precise about
which (per `config/inference.local.spark.json` + `REPLICATION.md`):

**✅ Use the Spark for:**
1. **Serving + benchmarking the low-RAM stack** — its **128 GB unified memory** is exactly the
   target `lowram_runtime` accounts for (fits 7–34B bf16, or up to ~200B quantized). It is
   **Blackwell → native FP4**, so the `moe/quant.py` NVFP4 path is *Spark-native*. This is your
   **benchmark machine** for the certification numbers.
2. **Local bf16 LoRA + QAT iteration** — `train_lora.py --qat --qat-scheme nvfp4` runs on the Spark
   because **QAT needs no bitsandbytes** (the penalty is pure torch; only `--4bit` pulls
   bitsandbytes, which has no aarch64 wheel). Use bf16, sdpa attention.
3. **Running `lowram_eval` certification** — the gate is pure numpy; runs anywhere, including the
   Spark, right after a local train.

**⚠️ Do NOT use the Spark for:**
4. **Registered / headline training numbers** — those stay on **x86 RunPod** (`REPLICATION.md`
   reproducibility discipline). Spark numbers are for *iteration and benchmarking*, not the result
   of record. Also blocked on aarch64: `--4bit`/bitsandbytes, `unsloth`, prebuilt flash-attn.

**Net:** the Spark is where Sophia-V1 actually **runs at low RAM** (the deliverable) and where you
**iterate QAT cheaply for free**; RunPod x86 is where the **registered training numbers** are made.

---

## Start here — the low tier (this is set up and ready)

**On your Spark (free):**
```
python tools/runpod_qat_lowram.py --target local-spark --tier low      # see the plan
# then run the emitted commands; drop --dry-run on the train step to actually train:
python tools/run_calibration.py --out training/lora/calibration_datasheet.json --target-bits 4.5
python tools/train_lora.py --model allenai/OLMoE-1B-7B-0924-Instruct --qat --qat-scheme nvfp4 \
    --epochs 1 --dtype bf16 --attn sdpa
# then certify the quantized adapter vs fp16 with serving.lowram_eval (Boundary-3 evidence)
```

**On RunPod (registered numbers), once you set a budget:**
```
python tools/runpod_qat_lowram.py --target runpod --tier low --budget-usd 10
# emits a runnable runpod_train.py command using its REAL flags, e.g.:
#   python tools/runpod_train.py --model allenai/OLMoE-1B-7B-0924-Instruct \
#       --gpu-type "NVIDIA A100-SXM4-80GB" --gpu-count 1 --epochs 1 \
#       --extra-train-args "--qat --qat-scheme nvfp4" --dry-run
# review, then re-run WITHOUT --dry-run and WITH --yes to spend.
```

`--extra-train-args` is a real passthrough on `runpod_train.py` that forwards `--qat` (and, for
high/top tiers, `--shard fsdp --expert-parallel`) to `train_lora.py` on the pod — so QAT actually
reaches training. The launcher provisions nothing on its own; the Spark path is free; the RunPod
path is gated on an explicit budget + `--yes`. High/top tiers auto-set `--gpu-count` (Mixtral-8x22B
→ 2, DeepSeek-V3 → 8) and add the sharding flags.

---

## The roadmap to top-notch (sequenced)

1. **Low tier on the Spark** *(ready now — benchmark this first)* → first real `lowram_eval` vs FP16.
2. **Low/Mid on RunPod** *(ready now)* → the registered Boundary-3 evidence number.
3. **Expert-parallel + FSDP sharding for `train_lora.py`** *(next code milestone — the gate to high/top)*.
4. **High tier** (Mixtral-8x22B 141B) → headline RAM number at near-frontier scale.
5. **Top-notch** (DeepSeek-V3 671B) → the Sophia-V1 claim: 25 GB resident, certified vs FP16, to the
   `RESULTS.md` bar.

Steps 1–2 are achievable today; 3 is the engineering that unlocks 4–5. None of it pretrains a
frontier model from scratch — it adapts + serves an open one at a fraction of the RAM, measured.
