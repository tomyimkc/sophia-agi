# Cheap-Compute Boundary — what "few computing resources" can and cannot mean

**Status:** scope-setting doc (no capability claim; `canClaimAGI` stays `false`).

> A goal was set for this repo: build a *highly intelligent* model that is *large in
> parameters*, *trainable with few computing resources*, and *low in RAM at release*.
> This doc states the honest boundary that goal can occupy without repeating the
> overclaim that `Training-Efficiency-Feasibility.md` was written to kill. **Read it
> before writing or quoting any "cheap" / "large" / "low-RAM" claim.**

---

## The tension with the charter, stated plainly

`VISION.md` says: *"Don't try to out-train frontier labs. Sophia's contribution is
provenance, verification, calibration, and fail-closed reasoning."*
`Training-Efficiency-Feasibility.md` already measured the "10–50× faster" claim down to
**~2.0× (dynamic padding alone, RTX 4090)** — and found QLoRA/Unsloth *slower* on
micro-runs. `Governed-Scaling.md` defines the program as *"scale that carries its own
proof,"* not scale for capacity.

The new goal is therefore **not** a license to become a frontier pretraining lab. It is a
**restatement of the governed-scaling thesis in three concrete dimensions** (parameters,
training compute, serving RAM), each of which has a hard floor this doc names so no future
claim crosses it.

---

## The three honest boundaries

### Boundary 1 — "Large parameters" = large *total* params, achieved via sparsity, **not** dense scale

**Allowed claim.** A model with a high total/active parameter ratio (MoE: many experts,
few active per token). Total params cost *storage*; active params cost *per-token compute
and bandwidth*. The GLM-5.2 ratio (744B total / 40B active ≈ 18.6×) is the precedent.
Sophia's existing `moe/router.py` (top-k gating, capacity, Switch-Transformer aux loss)
and `pretraining/architecture/moe.py` are the seeds.

**Disallowed claim.** A large *dense* model trained cheaply. Dense scale is the frontier
lab's game; it cannot be won cheaply and the charter declines to try.

**Floor:** *you cannot claim a "large parameter" model until `moe/` is connected to a
trainable end-to-end LM, not just a numpy reference.* Until then, "large parameters via
sparsity" is a *design intent*, documented in this doc, not a delivered artifact.

### Boundary 2 — "Trainable with few computing resources" = cheap *adaptation*, **never** cheap *pretraining*

This is the boundary the GLM-5.2 precedent *refutes*, not supports.

**The precedent cuts against cheap pretraining.** GLM-5.2's 744B MoE cost Zhipu
**thousands of H100-days** and was trained on **28.5T tokens**. "Few compute resources"
does not describe pretraining a frontier-scale model, by any honest reading. The Unsloth
Dynamic 2.0 work that made GLM-5.2 locally *runnable* is a **serving/quantization** result,
not a training result — it did not make the *training* cheap, it made the *weights small
enough to load*. Conflating the two is the exact error this doc exists to prevent.

**Allowed claim.** Cheap **adaptation** of a pre-trained sparse backbone:
- QLoRA / expert-level LoRA on an MoE base (the `training/` track; the RTX 4090 run was
  ~$0.67 for a gate-disciplined adapter).
- Quantization-aware fine-tuning (QAT) on a pre-trained base.
- Council distillation + gate-filtered SFT on a small (LIMA-scale, 500–2000 row) corpus.

**Disallowed claim.** Pretraining a large model from scratch "with few compute resources."
There is no honest version of this claim at frontier scale, and Sophia does not make it.

**Honest reframe of the goal:** *"a model adapted cheaply onto a pre-trained sparse
backbone, with provenance discipline baked in by gate-filtered data and external
verifiers."* That is true to the charter and supported by measured artifacts.

### Boundary 3 — "Low RAM at release" = quantized + sparse + offloaded, **with measured error bounds**

This is the boundary Sophia is best positioned to honor, because it is exactly the
**equivalence-proof / bounded-error bar** of `Governed-Scaling.md`.

**Allowed claim.** A served model whose memory footprint is reduced by:
- **Adaptive per-tensor quantization** (`moe/adapt.py`) — bits allocated by measured
  output-KL sensitivity, with a CI-checked budget + protected-floor invariant.
- **MoE expert offloading** (`serving/`) — only active experts resident in fast memory.
- **KV-cache quantization** (`serving/`) — INT4/INT8 KV for long context.
- Each governed by an `offline_invariants()` proof and a no-overclaim measurement gate.

**Disallowed claim.** "Low RAM" asserted without a measured error bound against the
full-precision reference. A quantized model that has not been measured against FP16 on the
**deployment distribution** (decontaminated from the eval set) cannot claim to "retain"
capability. Unsloth's GLM-5.2 "76%/82%" numbers are demo-sourced and under-validated; they
are a *target*, not a template. Sophia's bar is stricter: ≥2 judge families, κ ≥ 0.40,
≥3 seeds, 95% CIs excluding zero (`RESULTS.md`).

**Floor:** *no "low-RAM, capability-retained" claim ships until the quantized artifact is
evaluated against the FP16 reference on a held-out, decontaminated set, to the no-overclaim
gate.* Until then, reductions are "size/byte reductions with a bounded round-trip error"
(the `moe/quant.py` guarantee) — a real, defensible, narrower claim.

---

## What this changes about the roadmap

The seven planned artifacts (P1–P7) are all **within** these boundaries:

| Artifact | Boundary it honors | Claim it may eventually support |
|---|---|---|
| P1 adaptive quant (`moe/adapt.py`) | B1, B3 | "a CI-checked adaptive-quant policy — the allocation rule is reproducible, not opaque" |
| P2 calibration-matching (`moe/calibrate.py`) | B3 | "quantization calibrated on the deployment distribution, decontaminated from eval" |
| P3 QAT study (`pretraining/qat/`) | B2 | "closed-form-floor evidence that importance-concentrating training lowers the quant floor" |
| P4 IndexShare (`kernels/indexshare.py`) | B1 | "a reproducible study of attention-index amortization with a measured quality/compute curve" |
| P5 expert-offload + KV-quant (`serving/`) | B3 | "tiered expert memory, promote-on-route governed, low-RAM serving" |
| P6 real-GPU validation | B3 | the first claim that *clears the no-overclaim gate* for a served quantized model |
| P7 trainable nano MoE (`pretraining/`) | B1 | the floor artifact that makes B1's "large parameters" claim deliverable |

None of these claims cheap frontier pretraining. All honor the charter's "innovate at the
trust layer" by making the *efficiency primitive carry its own proof*.

---

## The one-line test for any future "cheap/large/low-RAM" claim

> *Does this claim survive substituting "adapted" for "trained" and "within a measured
> error bound on a decontaminated eval" for "retained capability"?*

If yes — it is honest and ships. If no — it is the 10–50× error in new clothing, and it
does not ship. This doc is the standing instruction to apply that test.
