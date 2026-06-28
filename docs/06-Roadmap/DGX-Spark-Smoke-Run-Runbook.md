# Phase 3 Smoke Run — DGX Spark Runbook (≤8B RLVR/GRPO)

**Status:** execution runbook. Target: a **bounded smoke run** of the verifier-as-reward
GRPO loop (`tools/run_rlvr.py`) on a single **NVIDIA DGX Spark** (GB10 Grace-Blackwell,
128 GB unified memory, aarch64), training a **≤8B** base. Goal of the smoke: prove the
loop runs end-to-end on Spark hardware and the mean verifier reward *rises* — not a
capability claim. The rigorous before/after (Tier B) and the published numbers come
after the smoke passes.

> **No-overclaim reminder.** A capability claim requires a *gated* run (≥2 judge
> families, κ, ≥3 seeds, sealed-holdout CI). The smoke produces a reward curve and a
> training-config artifact only. The "before" is already sealed in
> `agi-proof/benchmark-results/baseline-{math,physics,code}.json` (Phase 0).

---

## 0. Why these choices on Spark

| Spark fact | Consequence for the smoke |
|---|---|
| **128 GB unified (coherent) memory** | An 8B model in **bf16** (~16 GB) + LoRA + GRPO rollouts fits with huge headroom. **Skip 4-bit** — `bitsandbytes` on aarch64/Blackwell is the most fragile dep and you don't need it. |
| **Blackwell GB10 (sm_121), brand-new** | Use a **CUDA-13 / Blackwell aarch64** PyTorch build. Don't build `flash-attn` (ARM/Blackwell build pain) — Transformers falls back to SDPA, which is fine at this scale. |
| **vLLM on GB10 aarch64 is young** | For the smoke, use **`--vllm none`** (HF generation). Removes vLLM as a variable; slower but bulletproof. Try `--vllm colocate` only *after* the smoke is green. |
| **Apache-2.0 hygiene** | Use **`Qwen/Qwen2.5-7B-Instruct`** (Apache-2.0, strong math/code ≤8B) rather than the repo default GLM-4-9B (commercial-license caveat). `run_rlvr` now auto-picks the right LoRA target modules per family. |

---

## 1. Environment (DGX OS / Ubuntu aarch64)

```bash
# On the Spark
git clone <your sophia-agi remote> && cd sophia-agi
git checkout claude/deepseek-reasonix-integration-y83cmc

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip

# PyTorch for Blackwell aarch64 — use NVIDIA's CUDA-13 sbsa/aarch64 wheels.
# (Confirm the exact index URL against the build shipped on your Spark.)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# RL stack WITHOUT bitsandbytes/vllm/flash-attn for the smoke:
pip install "transformers>=4.46.2" "trl>=0.16" "peft>=0.19.1" \
            "datasets>=2.18" "accelerate>=1.14.0"
pip install -r requirements-math.txt   # sympy, for the math reward oracle
```

Verify the GPU is visible and is Blackwell:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

---

## 2. Offline sanity (no GPU) — do this first

```bash
python tools/run_rlvr.py --task math --dry-run     # -> RLVR REWARD WIRING VERIFIED ✓
python tools/run_rlvr.py --task physics --dry-run   # -> VERIFIED ✓
```

If these don't pass, stop — the reward machinery is broken independent of hardware.

---

## 3. Tier A — the bounded smoke GRPO run (~20–40 min)

Bounded by `--max-steps` so it stops fast. Math first (sympy reward; cheapest signal):

```bash
python tools/run_rlvr.py \
  --task math \
  --model Qwen/Qwen2.5-7B-Instruct \
  --quant bf16 \
  --vllm none \
  --max-steps 30 \
  --batch-size 2 --grad-accum 2 \
  --num-generations 4 \
  --max-prompt-len 256 --max-completion-len 256 \
  --lora-r 16 --lora-alpha 32 \
  --seed 0 \
  --output training/rlvr/checkpoints/qwen7b-math-smoke
```

**What to watch:** TRL logs every 5 steps (`logging_steps=5`). The smoke **passes** if
`reward` (mean over the group) trends **up** across the 30 steps and the run completes
without OOM. LoRA target modules auto-resolve to the Qwen split set
(`q,k,v,o_proj,gate_proj,up_proj,down_proj`) — confirm that in the printed config.

If memory is tight (it shouldn't be at 7B/bf16 on 128 GB): drop `--num-generations` to
2, `--max-completion-len` to 192. If you want a faster pulse, `--max-steps 10`.

Then repeat for physics (dimensional reward — the new domain):

```bash
python tools/run_rlvr.py --task physics --model Qwen/Qwen2.5-7B-Instruct \
  --quant bf16 --vllm none --max-steps 30 --batch-size 2 --grad-accum 2 \
  --num-generations 4 --max-prompt-len 256 --max-completion-len 256 \
  --output training/rlvr/checkpoints/qwen7b-physics-smoke
```

---

## 4. Tier B — rigorous before/after on the sealed eval split

Only after Tier A is green. This produces the first honest delta.

**Before** is already computed for the *mock* floor; recompute it against the **base
model** by serving it and pointing the baseline at it:

```bash
# Terminal 1 — serve the BASE model (try vLLM now; fall back to any OpenAI-compatible server)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --served-model-name local
# Terminal 2
python tools/run_baseline.py --task math --model vllm \
  --out agi-proof/benchmark-results/baseline-math.qwen7b-base.json
```

**Train a longer run** (drop `--max-steps`, let it do `--epochs 1`), then **merge** the
LoRA adapter and serve the trained model:

```bash
python tools/run_rlvr.py --task math --model Qwen/Qwen2.5-7B-Instruct \
  --quant bf16 --vllm none --epochs 1 --num-generations 6 \
  --output training/rlvr/checkpoints/qwen7b-math-v1

python - <<'PY'
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
p = "training/rlvr/checkpoints/qwen7b-math-v1"
m = AutoPeftModelForCausalLM.from_pretrained(p).merge_and_unload()
m.save_pretrained(p + "-merged"); AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained(p + "-merged")
PY

# Serve the merged model and measure AFTER on the IDENTICAL sealed eval split
vllm serve training/rlvr/checkpoints/qwen7b-math-v1-merged --port 8000 --served-model-name local
python tools/run_baseline.py --task math --model vllm \
  --out agi-proof/benchmark-results/baseline-math.qwen7b-v1.json
```

The two `baseline-math.*.json` carry the **same `evalSealed` hash** → the delta is on
the identical, unpeeked held-out families. Compare `passAt1` + Wilson CIs. A real uplift
needs the CIs to separate (and, for a *claim*, ≥3 seeds + a second judge family).

---

## 5. Generating cheap training data with the rollout factory (optional, parallel)

The Phase-1 factory (`pipeline/rollout/`) can manufacture extra verifier-passing traces
for SFT cold-start before RL. Point it at the locally-served model:

```bash
export SOPHIA_MODEL_PROVIDER=vllm   # uses http://localhost:8000/v1, model "local"
python - <<'PY'
from pipeline.rollout import RolloutFactory, DEFAULT_TOOLS
from provenance_bench import physics_dataset, physics_reward
f = RolloutFactory()                      # real vLLM backend via agent.model
probs = physics_dataset.load_problems()
out = f.generate_traces(probs, reward_for=physics_reward.reward_for_problem)
print("passRate:", out["passRate"], "savings:", out["aggregateSavingsRatio"])
PY
```

Keep only `reward == 1.0` traces (gate-filtered) as SFT data — the R1 cold-start recipe,
every trace machine-checked.

---

## 6. Troubleshooting (Spark-specific)

| Symptom | Fix |
|---|---|
| `bitsandbytes` import/CUDA error | You're on the 4-bit path — use **`--quant bf16`** (default-safe on Spark). |
| vLLM fails to start on GB10 | Use **`--vllm none`** for training; for serving, fall back to `sglang`/`llama.cpp`/`ollama` presets in `agent/model.py`. |
| `flash_attn` import error | Don't install it; set `attn_implementation="sdpa"` is the HF default fallback. |
| OOM (unlikely at 7B) | Lower `--num-generations`, `--max-completion-len`, `--batch-size`. |
| LoRA "target modules not found" | You overrode `--lora-target-modules` wrongly, or used a non-Qwen/Llama/GLM family — pass the correct names explicitly. |
| Reward flat at smoke | Expected sometimes at 30 steps; raise `--max-steps`, `--num-generations`, or `--lr` slightly. Confirm sympy is installed (math) so the reward isn't all −1 `sympy_unavailable`. |

---

## 7. Exit criteria

- **Smoke (Tier A):** run completes on Spark, mean reward trends up. → mark the
  failure-ledger "no live RL run" item *in progress* (not yet closed).
- **First delta (Tier B):** `passAt1` after > before with separated Wilson CIs on the
  sealed split, for ≥1 domain. → the failure-ledger item moves toward closed; queue the
  multi-seed, second-judge gated run for the actual capability claim.

*Cross-refs:* [DeepSeek-Reasonix-Integration-Roadmap.md](./DeepSeek-Reasonix-Integration-Roadmap.md),
[../09-Agent/RLVR-Experiment.md](../09-Agent/RLVR-Experiment.md),
[../../agi-proof/failure-ledger.md](../../agi-proof/failure-ledger.md).
