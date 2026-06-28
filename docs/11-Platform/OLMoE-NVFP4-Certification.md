# OLMoE-1B-7B NVFP4 Low-RAM Certification (DGX Spark)

Boundary-3 evidence run (cf. `Cheap-Compute-Boundary.md`): QAT-train a LoRA adapter on a
sparse-MoE base, then certify the NVFP4-served artifact against full bf16 with the no-overclaim
gate (`serving/lowram_eval.py`). Tooling: `tools/train_lora.py --qat` and `tools/certify_lowram.py`.

> **UPDATE 2026-06-28 — the FAIL below was an INSTRUMENT artifact, not a model limit.** The
> `0.0648 / 0.867` numbers in this report were measured on an adapter that **never trained**:
> `training/qat.attach_qat` wrapped PEFT's `lora.Linear` (also class-named `Linear`), replacing
> its forward with a base-only `F.linear(x, fake_quant(weight))` and silently bypassing the
> `lora_A`/`lora_B` path — so the adapter got **zero gradient** and every `lora_B` stayed at its
> zero init. Direct check: `olmoe-qat-spark-v2` has **all 3136 `lora_B` == 0.0**, and its certify
> is **bit-identical** to the no-adapter base. So the "Why it fails — root cause" section below
> (experts not QAT-co-adapted) is the *second-order* story; the *first-order* story is that
> **no LoRA was applied at all** — this was a measurement of base-OLMoE NVFP4 degradation. Fixed
> in `77a1076d` (skip the LoRA wrapper, fake-quant its inner `base_layer`; regression test
> `tests/test_qat.py::test_attach_qat_does_not_bypass_lora_adapter`). With the fix, v3 train loss
> drops `2.331→1.752` over 20 steps (vs v2 flat ~2.4) and QAT coverage is `3136` not the doubled
> `6272`. **A v3 re-train + re-certify is in progress; the table below stays until v3 lands a
> measured verdict.** See `agi-proof/failure-ledger.md` → `qat-lora-forward-bypass-2026-06-28`.

## Verdict: FAIL (honest, complete measurement) — SUPERSEDED, see UPDATE above

| Metric | Value | Contract | Pass? |
|---|---|---|---|
| `mean_kl` (full ‖ nvfp4) | **0.0648** | ≤ 0.05 | ✗ (marginal) |
| `top1_agreement` | **0.8672** | ≥ 0.97 | ✗ |
| `mem_ratio` (whole-model vs bf16) | **3.30×** | reported | — |
| `per_tensor_mem_ratio` | 3.56× | — | — |
| `quantized_fraction` | 0.9699 (6.71B / 6.92B params) | — | — |
| `n_eval` | 256 next-token positions | — | — |

`protected_*` equals the aggregate (no protected slice was configured, so the whole set is the
"protected" set). The binding failures are `top1_agreement` and `mean_kl`.

## Environment & artifact

- **Device:** NVIDIA DGX Spark — GB10 Blackwell, aarch64, 128 GB unified memory (NVFP4-native).
- **Base:** `allenai/OLMoE-1B-7B-0924-Instruct` (6.92B total / ~1.3B active MoE, Apache-2.0).
- **Adapter:** LoRA + QAT, scheme `nvfp4`, bf16 + sdpa, `--target-modules attn-mlp`, dropout 0,
  1 epoch, 439 train rows. **Best val loss 1.7655 @ step 25.** `adapter_model.safetensors` ≈ 755 MB.
- **Stack (working):** torch 2.14.0.dev (cu130, aarch64), transformers 4.46.3, **peft 0.19.1**,
  safetensors 0.8.0. aarch64 forbids bitsandbytes / unsloth / flash-attn → bf16 + sdpa only;
  Triton JIT needs `TRITON_INTERPRET=1` (missing `python3-dev` headers).
- **Quantized set:** attention q/k/v/o + all per-expert gate/up/down projections (3136 tensors);
  embeddings, norms, MoE router gate, and `lm_head` kept bf16 (the real NVFP4 serving recipe).

## Why it fails — root cause

The degradation is **not** a tooling artifact; it is the adapter's genuine limitation:

1. **The MoE experts were never QAT-co-adapted.** At training time, under peft 0.14, OLMoE's
   experts presented as a *fused* module, invisible to `training/qat.attach_qat` (which wraps
   `type=='Linear'`). So QAT's fake-quant only co-adapted **attention**; the ~6.4B expert weights
   never saw their own NVFP4 error during training. Certifying now quantizes those un-QAT'd experts
   → the `top1` drop and high worst-case KL come from there.
2. **The adapter's expert LoRA is orphaned.** Its config targets *fused* expert names
   (`target_parameters=['gate_up_proj','down_proj']`), which match nothing on this *split*-expert
   model build (peft warns "no parameter was matched"). So even the native merge applies only the
   **attention** LoRA; the expert LoRA silently does not apply (surfaced as `incomplete_merge` in
   the report).

Net: this is a fair, complete test of *attention-only* QAT/LoRA served at full-model NVFP4.

## Debugging history (what made the number trustworthy)

| Run | mean_kl | top1 | mem_ratio | quantized | Problem |
|---|---|---|---|---|---|
| 1 | 0.4145 | 0.797 | 3.56 (claimed) | base-only | PEFT load failed → fell back to **base-only** (no LoRA); quantized everything incl. `lm_head`. |
| 2 | 0.0363 | 0.902 | 1.26 | 64 attn only | `type=='Linear'` scan missed **fused experts** (~6B params left bf16). |
| 3 | 0.0373 | 0.918 | 2.87 | 32 experts, attn skipped | `PeftModel.from_pretrained` mutated the model **in place before raising** → half-wrapped, contaminated. |
| **4** | **0.0648** | **0.867** | **3.30** | **3136 (97%)** | **Clean.** peft 0.14→0.19.1 fixes the skew; native merge; experts + attention quantized. |

Fixes landed in `tools/certify_lowram.py`: robust LoRA load (peft → clean-reload manual-merge
fallback), parameter-based quantization (catches fused + per-expert weights, excludes
head/embeddings/norms/router), honest whole-model `mem_ratio`, low-coverage + incomplete-merge
warnings, and `--inspect-adapter` / `--selftest` (GPU-free invariants). Regression-tested in
`tests/test_lowram_e2e.py`.

## To actually clear the gate (next step, not yet run)

Re-train QAT so the **experts co-adapt** to NVFP4 (now viable: under transformers 4.46.3 the
experts load as per-expert `nn.Linear`, so `attach_qat` and PEFT `attn-mlp` targeting reach them):
extend `attach_qat`/`qat_penalty` to also fake-quant fused expert Parameters (defensive), ensure
`--target-modules attn-mlp` resolves to the split per-expert projections, then re-train ≥1 epoch
and re-certify. Levers if still short: more epochs, higher `--qat-lambda`, or a deployment-justified
contract. One pass is evidence, not the headline (RESULTS.md bar still applies).
