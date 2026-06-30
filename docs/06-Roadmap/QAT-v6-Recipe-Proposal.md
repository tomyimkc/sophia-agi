# QAT NVFP4 low-RAM cert — current v5 recipe + proposed v6 recipe

`canClaimAGI` false. Cert bar: **mean_kl ≤ 0.05 AND top1 ≥ 0.97** (quantized-vs-FP next-token
agreement); protected slice **KL ≤ 0.10 AND agreement ≥ 0.95**.

## 1. Current v5 recipe (as run for `olmoe-qat-spark-v5`)
| Knob | v5 value | Source |
|---|---|---|
| Base | `allenai/OLMoE-1B-7B-0924-Instruct` (6.92B MoE, 64 experts) | runner default |
| Adapter | LoRA `r=16`, `alpha=32`, `dropout=0.05`, `target-modules=all-linear` | `tools/train_lora.py` defaults |
| Optimizer | `lr=5e-5`, `batch-size=1`, `epochs=3` | runner `QAT_EPOCHS=3` |
| QAT objective | task CE under STE fake-quant (fwd `dequant(quant(W))` NVFP4, bwd identity) **+ weight-space penalty** `λ·mean((W−fake_quant(W))²)`, **λ=0.001** | `training/qat.py`, runner `QAT_LAMBDA=0.001` |
| Kept bf16 | embeddings, norms, MoE router gate (`mlp.gate`), `lm_head` | `training/qat.py` |
| **Result** | **mean_kl 0.0506 ✗ / top1 0.8828 ✗ → FAIL** | `bridge/results/2026-06-29-nvfp4-v5-train-08.json` |

**Why it's stuck (first principles).** The objective optimizes task loss + a **weight-space** grid-
proximity penalty. The cert scores **output-space** agreement. The recipe never trains the quantized
model to match *its own FP outputs*, so `mean_kl` lands near the bar while **`top1` argmax flips on
~8% of tokens** — a ranking failure the loss does not target. v4 pushed λ→0.01 and over-regularized
(broke mean_kl + the protected slice). **You cannot reach top1 ≥ 0.97 by tuning λ alone — wrong
objective space.** (Measurement caveat: the committed v5 cert is also contaminated — the merge
dropped the 32 fused-expert LoRA modules; that is being fixed in parallel and does not change the
recipe diagnosis below.)

## 2. Proposed v6 recipe (concrete — change the objective, not just the knobs)
**Headline: train the cert's own metrics.** Replace "task loss + weight penalty" with
"task loss + output-space distillation + a top1-margin term", back off the weight penalty, and give
the experts more capacity + epochs.

| Knob | v5 → **v6** | Rationale |
|---|---|---|
| **Loss** | add **KD** `α·KL(FP_logits ‖ quant_logits)` at temp **T=2** **+ top1-margin hinge** `β·hinge(FP_argmax vs quant_logits)` | directly drives the two cert metrics (mean_kl, top1). **The core fix.** |
| **α, β, T** | `α=1.0`, `β=0.5`, `T=2.0` | KD carries distribution match; margin term carries the argmax. Start here, sweep β if top1 still short. |
| **λ (weight penalty)** | `0.001 → 0.0005` | KD now carries quant-fidelity; back off the grid penalty that over-regularizes (v4 lesson). |
| **epochs** | `3 → 5` | v5 under-trained the experts (single-pass MoE specialization). |
| **LoRA rank** | uniform `r=16` → **`--lora-rank-alloc`, experts r≈32 / attn r≈16** | MoE capacity lives in the experts; a rank-16 slice may not co-adapt them enough to be quant-robust. |
| **calibration** | widen the cert calib set (more diverse rows) | stops the quant scales from over-tuning to a narrow distribution. |
| **serve-time mixed precision** (if still short) | extend keep-list `keep_suffixes → keep_layers`: hold the **last 1–2 transformer blocks** bf16 | the argmax is decided by the final layers; the session showed holding by *projection type* didn't help — hold by *depth*. |

### The repo-coherent fallback (ship even if the cert never hits 0.97)
**Conformal abstention on the flips.** The repo's whole thesis is *abstain instead of fabricate*. The
cert demands the model NEVER flips (top1 ≥ 0.97); a system that **abstains** doesn't need that. Serve
the v6 model at top1≈0.92 + a **conformal gate** (`agent/conformal_gate.py`) that abstains on the ~8%
low-margin tokens where quant disagrees with FP. Honest metric = "top1 *on the tokens it answers*"
(≥0.99). This changes the **gate, not the model**, and de-risks the iPhone goal regardless of the cert.

## 3. Sequence
0. **Fix the measurement first** (merge the 32 fused-expert LoRA modules at cert time / pin a
   compatible peft on the Spark) — else v6's expert co-adaptation won't even be measured. *(in flight)*
1. **v6 train:** loss change (KD + top1-margin) + λ=0.0005 + epochs=5 + expert-rank alloc. Re-cert.
2. If short: add depth-mixed precision (hold last 1–2 blocks bf16).
3. **Parallel, no GPU:** build the conformal-abstention serve path so a top1≈0.92 model is honestly
   shippable now.
4. After a PASS: verifier-gated process supervision on the shipped low-RAM model.

**One line:** the cert is stuck because v5 optimizes weight-space, not the output-space agreement the
cert scores — so **v6 trains the cert's own metrics (KD + top1-margin), backs off λ, and gives the
experts more rank + epochs**; and in parallel we let the model **abstain on the flips** instead of
forcing it never to flip. Compute is not the blocker; the objective is.
