# Layer Streaming — running a model larger than fast memory (the AirLLM technique)

**Status:** mechanism doc. Adds the dense-weight analog of expert offloading. No capability
claim; `canClaimAGI` stays `false`. Read `Cheap-Compute-Boundary.md` (Boundary 3) first — it
governs every "low-RAM" word here.

> Prompted by [lyogavin/airllm](https://github.com/lyogavin/airllm), which runs a 70B model
> on a 4GB GPU. This doc states *why* that works, *what* of it Sophia adopts, and *what it
> does not buy* — so no future claim crosses the boundary `Cheap-Compute-Boundary.md` names.

---

## Why AirLLM fits a 70B model in 4GB

A dense transformer is a stack of ~80 near-identical layers run in sequence. At any instant
only **one** layer's weights must be resident to compute its output. AirLLM exploits this:

1. **Decompose to per-layer shards on disk** (once, ahead of time).
2. **Stream one layer at a time:** load layer *i* → compute → free → load *i+1*. Peak weight
   memory is `max(single layer) + activations + KV`, not the whole model. A 70B layer is
   ~1–2% of the model (see `tools/shard_checkpoint.py --selftest`: 1.2% for Llama-70B).
3. **Prefetch:** overlap loading *i+1* with computing *i* (AirLLM v2.5+).
4. **Block-wise 4/8-bit quantization:** shrink each shard, so less to move per layer.

**The honest caveat (load-bearing for our boundary).** This is **inference-only** and **does
not change the trained artifact**: 70B params still do 70B params' worth of FLOPs. It trades
**RAM for wall-clock + disk I/O** (every token re-streams every layer → seconds/token). It
makes a big model *runnable* on tiny hardware; it does **not** make it *intrinsically smaller*.
So it cannot, by itself, deliver "train a frontier model whose release artifact needs little
RAM" — that requires the property be **baked into training** (sparsity/MoE, QAT, distillation;
Boundaries 1–2). Layer streaming is the **serving safety net** that lets *any* such artifact
run on a 4GB / Spark-class device. The two are complementary, not substitutes.

---

## What Sophia adopts (and where it lives)

| AirLLM idea | Sophia module | Notes |
|---|---|---|
| Per-layer disk shards + manifest | `tools/shard_checkpoint.py` | `--plan` (config-only, CI-testable) or `--materialize` (real safetensors). Manifest schema `sophia.layer_shard_manifest.v1`. |
| Stream one layer at a time, GPU→CPU→disk | `serving/layer_stream.py` | `StreamingLayerStore` — the **dense analog** of `serving/expert_offload.py`; same tiering/LRU/byte accounting, promotion signal is sequential layer order. |
| Prefetch / compute–load overlap | `serving/layer_stream.py` | `prefetch_depth` window; `prefetch_hits` measures the overlap. |
| Block-wise quantization | `moe/quant.py` + `serving.layer_stream.plan_layer_bits` | Quant-aware sizing: a layer at *b* bits costs `b/16` of its fp16 residence. `plan_layer_bits` delegates to `moe/adapt.py`'s sensitivity allocator (protected floor on embed/head/first/last). |
| "It still runs at quality" — the part AirLLM asserts and we measure | `serving/lowram_eval.py` | The **no-overclaim gate**: streamed+quantized vs FP16 on a held-out set, bounded mean-KL / top-1 agreement, protected floor. This is the Boundary-3 measurement, not the mechanism. |

Each module ships a deterministic `offline_invariants()` (CI-gated, `tests/test_layer_stream.py`),
exactly like the rest of `serving/` and `moe/`. Payloads are opaque sizes — the policy is
proven offline; the real mmap/safetensors loader + CUDA-stream prefetch kernel is the
deployment artifact, out of scope for the CI reference.

---

## How it composes with what was already here

- **`serving/expert_offload.py`** tiers *experts* (MoE, promote-on-route). **`layer_stream.py`**
  tiers *dense layers* (promote-on-sequence). An MoE's dense trunk streams in the latter, its
  experts in the former — same `GPU→CPU→disk` bookkeeping object, two promotion signals.
- **`moe/adapt.py`** decides *how many bits* each layer gets; **`layer_stream.py`** decides
  *which layers are resident when*; **`lowram_eval.py`** decides *whether the result is allowed
  to claim it kept quality*. Mechanism → mechanism → measurement.

## Track B — making the *trained artifact* intrinsically cheaper (not just streamable)

Layer streaming runs a big model in little memory but does **not** shrink the artifact. The
training-side levers that do — so the *released* weights need less RAM *at quality* — live
alongside it:

| Lever | Module | What it buys |
|---|---|---|
| **QAT** — co-adapt weights to their serving quantization | `training/qat.py` (+ `tools/train_lora.py --qat`) | The released checkpoint serves at INT8/NVFP4 with little measured loss (fake-quant STE forward + quant-pushing penalty built on `moe/quant.py`). |
| **Distillation into sparsity** — teacher quality at small *active* cost | `pretraining/distill/study.py` | Nano-substrate, known-floor evidence that a sparse MoE student distilled from a dense teacher beats an equally-active dense student — the "large total params / small active RAM" thesis, measured. |
| **Calibration** — quantize on the deployment distribution, decontaminated | `tools/run_calibration.py` (plan stage `calibrate`, `moe/calibrate.py`) | The audit trail (datasheet + disjoint-from-eval proof) a quantized artifact must carry before any capability-retention claim. |

These are the B1/B2/B3 artifacts of `Cheap-Compute-Boundary.md` Boundaries 1–3. QAT + distillation
shrink the artifact; layer streaming + calibration + `lowram_eval` make the *served* result
cheap *and certified*. None of them claims cheap frontier pretraining.

## Boundary-3 placement (no overclaim)

This delivers the **mechanism** for "low RAM at release," and the **measurement gate** that a
real claim must clear. It does **not** itself clear that gate — that needs the streamed+quantized
artifact evaluated against FP16 on a held-out, decontaminated deployment set (`RESULTS.md` bar:
≥2 judge families, κ ≥ 0.40, ≥3 seeds, 95% CIs excluding zero). Until then the defensible claim
is the narrower one `moe/quant.py` already supports: *byte/size reduction with a bounded
round-trip error, plus a tiered loader that holds peak fast-memory to one layer window.* This
is the dense-weight completion of the P5 artifact and the harness for the P6 validation in
`Cheap-Compute-Boundary.md`.
