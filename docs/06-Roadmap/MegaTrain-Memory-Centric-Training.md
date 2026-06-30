# MegaTrain — memory-centric LARGE-model *training* (the training mirror of `layer_stream.py`)

**Status:** research/design analysis. No capability claim; `canClaimAGI` stays **false**. Pairs with
`docs/11-Platform/Cheap-Compute-Boundary.md`, `Spark-Cluster-Capacity.md`, and the existing
*serving*-side `docs/11-Platform/Layer-Streaming.md`. Source: **MegaTrain**, Yuan/Sun/Sun/Ye,
arXiv:2604.05091 (Apr 2026).

## 1. What MegaTrain does
A **memory-centric** training system: parameters + optimizer states live in **host memory**, the GPU
is a **transient compute engine**. Per layer it streams params *in* and gradients *out*, minimizing
persistent device state. Two levers beat the CPU↔GPU bandwidth wall:
1. a **pipelined double-buffered engine** overlapping param-prefetch / compute / gradient-offload
   across multiple CUDA streams → continuous GPU execution;
2. **stateless layer templates** replacing persistent autograd graphs — weights bind dynamically as
   they stream in, killing graph metadata and freeing the scheduler.

Results: one **H200 + 1.5 TB host RAM** trains **120B params at full precision**; **1.84× the
throughput of DeepSpeed ZeRO-3 CPU-offload** on 14B; **7B @ 512k-token context on a single GH200**.

## 2. Why this is THE paper for *our* hardware (not a generic systems paper)
- **GB10 (DGX Spark) and GH200 (the paper's testbed) are the same family** — a Grace-class CPU + an
  NVIDIA GPU sharing **coherent unified LPDDR5x**. MegaTrain's GH200 numbers transfer to the GB10.
- On **unified memory** the very bottleneck MegaTrain fights — CPU↔GPU **PCIe** copies — is *far
  milder*: there is one coherent pool, no PCIe hop. So the technique is **easier here**, and the
  capacity win (fit a bigger model / longer context) is the whole point.
- **We already shipped the serving mirror.** `serving/layer_stream.py` (`StreamingLayerStore`,
  `prefetch_depth`, byte accounting, block-wise quant via `moe/quant.py`, CI'd `offline_invariants`)
  streams layers to *serve* a model bigger than fast memory. **MegaTrain is the same idea for
  *training* — the half we have not built.** The tiering / LRU / prefetch-overlap / byte-accounting
  machinery transfers almost 1:1.

## 3. What it unlocks here — with the honest wall
Full-precision Adam ≈ **16 bytes/param** (fp32 weight 4 + grad 4 + m 4 + v 4). Memory-fit ceilings:

| Box | Unified RAM | ~Full-precision train ceiling (memory-fit) | vs today |
|---|---|---|---|
| 1 Spark (GB10) | 128 GB | **~8B** | repo trains ≤3B full / ≤70B 4-bit *LoRA* |
| Mac Studio (M3 Ultra) | up to 512 GB | **~32B** | Mac is only a *judge* today |
| 8 Sparks | ~1 TB | **~64B** (multi-node, link-gated) | — |

Plus **long context**: MegaTrain's 512k-context result → a capability the repo's short-context
adapters lack.

> **The wall that does NOT move (be honest):** MegaTrain solves **memory**, not **FLOPs**. Training
> FLOPs ≈ `6·params·tokens` is unchanged — from-scratch *frontier* pretraining stays FLOP-walled
> (`Cheap-Compute-Boundary.md` Boundary 2 holds). The win is concrete and large but bounded:
> **full-precision train/fine-tune a several×-larger base on owned hardware**, not frontier
> pretraining. Memory-fit ≠ trained-to-convergence.

## 4. Complementary theses (compose, don't pick one)
- **ZeRO-Offload / ZeRO-Infinity** (DeepSpeed) — the CPU/NVMe-offload baseline MegaTrain beats 1.84×.
- **GaLore / LoMo** — low-rank / fused-backward optimizers that shrink the 16 B/param → fit an even
  bigger model in the same RAM. *Stacks* with MegaTrain.
- **FlashAttention / Ring Attention** — the activation/KV math that makes 512k context tractable.
- **QLoRA + AirLLM** — the repo's existing 4-bit + serving-stream lineage (`serving/`, `moe/`).
- **Activation recomputation** — trade FLOPs for activation memory (helps the long-context arm).
- **The novel combo:** MegaTrain layer-streaming × GaLore optimizer-shrink × **the repo's
  verifier-gated loss** — memory-centric training with a provenance reward in the objective.

## 5. Creative plan — five directions the repo has NOT developed
1. **Own a trainable larger base** (stop only *adapting* others'): an **~8B council base** trained
   full-precision on a Spark via a training-side layer-stream — the repo's first self-trained base.
2. **Verifier-gated MID-training (the repo's unique angle).** Today the verifier reward is *post-hoc*
   (DPO / reasoning-distill / test-time-thinking). With real training feasible, put the **machine-
   verifier reward into the training loss** — a base provenance-aware *from pretraining*, gated by
   the same no-overclaim machinery. **Nobody else is doing verifier-gated memory-centric training.**
3. **Long-context provenance.** A **512k-context wisdom gate** (whole-book / full-case-law
   provenance, multi-document attribution) — MegaTrain-style activation streaming makes the context
   fit; the repo's abstention/attribution gate is currently short-context.
4. **Mac-as-trainer (MLX).** The 512 GB / 819 GB/s Mac is only a judge today; an MLX memory-centric
   streamer turns it into a **~32B full-precision trainer** on owned hardware.
5. **Cluster-MegaTrain fabric.** Shard host-memory across Sparks (the landed `cluster_scheduler` /
   `cluster_node_runner` already coordinate them) to train beyond one box's RAM — honest about the
   25 GB/s inter-node tax (`Spark-Cluster-Capacity.md`).

## 6. The build (reuse what's already CI'd)
Extend `serving/layer_stream.py`'s proven pattern into **`training/layer_stream_train.py`**:
params + optimizer tier in unified memory; **double-buffered** prefetch / compute / gradient-offload;
**stateless layer templates** (no persistent autograd graph). Reuse the existing tiering, byte
accounting, `prefetch_depth` overlap metric, and `offline_invariants()` CI gate.

## 7. The single highest-value next step (on-charter: prove it offline first)
Before any GPU run, build a **GPU-free memory model + offline prototype** that *proves the byte
accounting*: params + optimizer + activations for an **8B full-precision** train **fit in 128 GB**
under double-buffered streaming, with a `prefetch_hits`-style overlap bound — tested exactly like
`tests/test_layer_stream.py`. That is the pre-registered, falsifiable first deliverable; only then
does a real Spark train earn a (still candidate-only) throughput claim. `canClaimAGI` stays false.
