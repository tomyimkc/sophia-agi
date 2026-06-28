# Sophia-V1 — frontier *total* params at a fraction of the RAM (the honest plan)

**Status:** plan + accounting doc. No capability claim; `canClaimAGI` stays `false`. Governed by
`Cheap-Compute-Boundary.md` (Boundaries 1–3) — read it before quoting any number here.

> Goal, as stated: *"train a top-notch frontier model that uses less RAM — GLM-5.2 / DeepSeek
> scale total params, but much less RAM than the original."* This doc states the **honest,
> achievable** version of that goal, **quantifies** it (`serving/lowram_runtime.py`), and lists
> **every remaining step** — separating what is buildable now (CI, free) from what needs real GPU
> compute (RunPod, costs money).

---

## The reframe that makes the goal real

Two things hide in "train a frontier model with less RAM," and only one is achievable here:

- **Pretraining a GLM-5.2 / DeepSeek-V4-scale model from scratch, cheaply — NOT achievable.**
  GLM-5.2's 744B MoE cost *thousands of H100-days* on *28.5T tokens*. There is no honest cheap
  version, and `VISION.md` declines the race. This doc does **not** promise it.
- **A released artifact with frontier-class *total* params but a *fraction* of the resident RAM —
  achievable and certifiable.** This is the GLM-5.2 thesis itself: *total* params cost storage,
  only *active* params cost resident fast memory, and quantization + offload + streaming shrink
  every resident byte. Sophia's contribution is to do it **with a measured error bound**, not an
  asserted one.

So the deliverable is: **Sophia-V1 = a sparse MoE adapted onto a pretrained backbone, with
GLM-5.2-class *total* parameters and single-digit-to-tens-of-GB resident RAM, certified against
FP16 by `serving/lowram_eval.py`.**

## The number, quantified (`serving/lowram_runtime.py`, `plan_ram`)

Resident RAM at three operating points, NVFP4 weights (~4.5 bits), KV+activations included:

| Model (total / active) | dense fp16 | expert-offload (fast) | full-stream (AirLLM-max) |
|---|---|---|---|
| **GLM-5.2** (744B / 40B) | ~1,488 GB | **25.5 GB** (58× — one GPU/Spark) | **5.3 GB** (4GB-class device, slow) |
| **DeepSeek-V3** (671B / 37B) | ~1,342 GB | ~23 GB | ~5 GB |
| **Sophia-V1 target** (744B / 40B) | ~1,488 GB | **25.5 GB** | **5.3 GB** |

The mechanism stack that delivers each column is already merged (`serving/`, `moe/`):
sparsity (MoE) → expert offload (`expert_offload.py`) → layer streaming (`layer_stream.py`) →
quantization (`quant.py` + sensitivity allocation `adapt.py`). `lowram_runtime.py` composes them
in one GPU budget and CI-checks that a full decode step stays within it.

> These are **byte-accounting guarantees with a bounded-error serving path** — real and
> defensible — **not** a capability claim. The capability claim needs the two floors below.

---

## All possible steps (the roadmap)

### ✅ Done (merged / this branch — CI-tested mechanism + accounting)
- Layer streaming, expert offload, KV-quant, adaptive quantization (`serving/`, `moe/`).
- QAT, distillation-into-sparsity study, calibration stage (`training/qat.py`, `pretraining/distill/`, `tools/run_calibration.py`).
- Low-RAM measurement gate (`serving/lowram_eval.py`).
- **This branch:** integrated runtime + frontier RAM planner (`serving/lowram_runtime.py`) — quantifies the table above.

### ⏭ Buildable now (CI, free — no GPU)
1. **Trainable end-to-end MoE LM** (the **Boundary-1 floor**). `moe/` is a numpy reference; connect router + experts into a small *real* torch MoE LM (CPU-runnable tiny config) so "large total params via sparsity" is a delivered artifact, not a reference. This is the keystone that unblocks every capability claim.
2. **MLA / low-rank KV** (DeepSeek's lever) as a reference in `serving/` — shrinks the KV term in the table further.
3. **Shared-expert + fine-grained-expert accounting** in `lowram_runtime.py` to match the DeepSeek-V3 / GLM-5.2 expert topology exactly.
4. **End-to-end wiring test**: `shard_checkpoint` → `lowram_runtime` → `lowram_eval` on a tiny real model, proving the whole pipeline composes.

### 💰 Needs real GPU compute (RunPod — costs money, needs your go-ahead)
5. **Real QAT run** on a pretrained sparse-MoE base (`tools/train_lora.py --qat`), then **clear `lowram_eval` vs FP16** on a held-out, decontaminated set — the **Boundary-3 floor**, the first real "low-RAM, capability-retained" evidence (the P6 artifact).
6. **Distillation onto the sparse backbone** at real scale (council/teacher → MoE student).
7. **Headline replication** to the `RESULTS.md` bar (≥2 judge families, κ ≥ 0.40, ≥3 seeds, 95% CIs excluding zero).

---

## Boundary placement (no overclaim)

This doc + `lowram_runtime.py` deliver the **mechanism and the accounting** for "frontier total
params, fraction of the RAM." The **capability** claim is gated on two floors, both named in
`Cheap-Compute-Boundary.md`:

- **Boundary 1:** no "large parameter" claim until `moe/` is a *trainable end-to-end LM* (step 1).
- **Boundary 3:** no "low-RAM, capability-retained" claim until the quantized artifact clears
  `serving/lowram_eval.py` vs FP16 on a held-out, decontaminated set (step 5).

Until then, the defensible statement is exactly the table above: *a frontier-total-param model's
resident RAM, byte-accounted under the merged low-RAM stack, with a bounded-error serving path.*
That is real, narrower than a capability claim, and does not cross any boundary.
