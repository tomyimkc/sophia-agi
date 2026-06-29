# DGX Spark Cluster — Capacity & Training-Time, 1 vs 2 vs 4 vs 8

**Status:** capacity-planning doc (no capability claim; `canClaimAGI` stays `false`). Read with
`Cheap-Compute-Boundary.md` — the standing rule is **cheap *adaptation*, never cheap *pretraining***.
The numbers here say *how much* adaptation/serving more Sparks buy, and where the wall is.

## The one hardware fact that governs everything
A DGX Spark GB10 is a **bandwidth-starved unified-memory box**: **125 TFLOP/s BF16**, **273 GB/s**
LPDDR5x (NOT HBM), 500 TFLOP/s FP4, **128 GB**. For comparison an H100 is **989 TFLOP/s / 3.35 TB/s**
(`kernels/bench/roofline.py` device specs). Training throughput on the Spark is gated by the **273 GB/s
memory bandwidth**, not its FLOPs. The inter-Spark link is **ConnectX-7 200 GbE ≈ 25 GB/s** — ~10× slower
than on-package memory. That gap is why the parallelism *mode* matters more than the node count.

## Two parallelism modes (this is the whole story)
| Mode | What spans nodes | Bound by | Use it for |
|---|---|---|---|
| **Data-parallel** | each node holds a FULL copy; different data batches; sync only the gradients | compute (sync is cheap) | **LoRA/adapter training** — gradients are tiny, scales near-linearly |
| **Model-parallel** (tensor/pipeline) | the model is SPLIT across nodes | the 25 GB/s link | **fitting a model too big for one node** — adds capacity, link-bound, little speedup |

**Adapter training is data-parallel** → it is the *good* scaling case. Serving/fine-tuning a model that
doesn't fit on one Spark is model-parallel → you get **capacity, not speed**.

## Scaling table (measured + extrapolated)
| Cluster | Unified memory | Interconnect | Serve ceiling (4-bit) | 1-node fine-tune ceiling | **Data-parallel adapter-train speedup** |
|---|---|---|---|---|---|
| **1 Spark** | 128 GB | — | ~200B | ~70B | **1×** (baseline) |
| **2 Sparks** | 256 GB | direct cable | ~405B | ~120B | **~1.8×** |
| **4 Sparks** | 512 GB | NCCL ring (3) / 200 Gb switch (4) | ~800B | ~200B | **3.44× (measured)** |
| **8 Sparks** | **1 TB** | **200 Gb managed switch (required)** | **~1.6T** | ~400B | **~6× (extrapolated, ~75% eff.)** |

(4-node 3.44× is a measured NVIDIA/partner figure — ~86% efficiency; efficiency declines with node count
as all-reduce hops + the switch add the repo's "node tax" — see `tools/run_cluster_sim.py`. 8-node ≈ ~6×.)

## So: can 8 Sparks train an adapter faster? Yes — with two big caveats.
1. **Yes, ~6× faster** than one Spark for a data-parallel LoRA/QAT run **whose base fits on each node**
   (≤~70B at 4-bit). A run that's ~2.5–3 h on 1 Spark (e.g. this repo's OLMoE-1B-7B QAT, 439 rows /
   220 steps) becomes **~30 min on 8**.
2. **Caveat A — the small-corpus floor.** This repo's adaptation corpora are *tiny* (LIMA-scale, ~400–2000
   rows). At 8-way data-parallel the per-node batch starves: each node sees ~50–250 rows/epoch, and the
   fixed per-step + sync overhead dominates. **For small-corpus LoRA the sweet spot is ~2–4 Sparks; 8 is
   over-provisioned** — you can't make a 730-row fine-tune meaningfully "8× faster" because it's already
   small and fast. Diminishing returns set in hard past ~4 nodes for this workload.
3. **Caveat B — bigger base, not bigger speedup.** 8-way data-parallel does NOT reduce per-node memory
   (every node holds the whole base), so it does not let you fine-tune a *larger* model — it processes
   *more data* in parallel. To fine-tune a bigger base you need model-parallel (link-bound, slow).

## The significance of 8 vs 2 vs 1 — for THIS repo
The most on-charter value of 8 Sparks is **not single-run speed**, it's **parallel throughput for the
measurement discipline**:
- **Run the whole no-overclaim matrix in one wall-clock pass.** The gate needs ≥3 seeds × ≥2 judge
  families. With 8 Sparks you run **8 independent fine-tune/eval/judge jobs at once** instead of serially —
  the ≥3-seed + ≥2-family validation (which took this session *hours* serially) finishes in *one* pass.
- **Serve giant sparse models locally** (1 TB → ~405B–1.6T quantized MoE) for the `serving/` low-RAM
  frontier (expert-offload, KV-quant, adaptive per-tensor quant) — exactly Boundary 3 work.
- **~6× throughput for LARGE-base (70B+) data-parallel fine-tunes** — where the corpus is big enough to
  keep 8 nodes fed.

8 Sparks is a **huge-memory / modest-compute / low-bandwidth** cluster (~1 TB RAM but only ~1000 TFLOP/s
BF16 ≈ *one* H100 of compute). Buy it for **capacity, local serving scale, and parallel experiments** —
not for raw training throughput.

## The wall that no node count moves: from-scratch pretraining
Training FLOPs ≈ `6 × active_params × tokens`. Even at a generous 25% utilization:

| Model | Train FLOPs | 1 Spark | 8 Sparks (ideal) | What it really took |
|---|---|---|---|---|
| DeepSeek-V3 (37B act, 14.8T tok) | 3.3×10²⁴ | ~3,300 yr | ~410 yr | 2.79 M H800-h (~2 mo on ~2,048 H800) |
| GLM-5.2 (40B act, 28.5T tok) | 6.8×10²⁴ | ~7,000 yr | ~875 yr | "thousands of H100-days," 28.5T tok |

8 Sparks turn "thousands of years" into "hundreds of years." **From-scratch frontier pretraining is a
rented-datacenter task (thousands of GPUs, $millions), not a desk cluster — at any Spark count.** And note
DeepSeek-V3/GLM-5.2 are 671–744B: even 4-bit (~335–372 GB) they need ~3–4 Sparks just to *fit*, so you
can't even LoRA-fine-tune *those specific models* until ~4 nodes.

## Recommendation
- **1 → 2 Sparks:** best marginal value — doubles memory to 256 GB (serve ~405B), ~1.8× on fitting
  fine-tunes, direct-cable (no switch). On-charter and cheap.
- **2 → 4 Sparks:** 512 GB + measured 3.44×; needs a ring/switch. Justified if you fine-tune 70B+ bases
  or want the seed×family matrix in parallel.
- **8 Sparks:** justified by **local serving of ~405B–1.6T sparse models** and **8-wide parallel
  experiments**, *not* by speeding up small-corpus adapters (over-provisioned there) and *not* by enabling
  frontier pretraining (still infeasible). Needs a 200 Gb managed switch + the power/space for 8 units.
- **To actually *pretrain* a large model:** rent cloud GPUs (the `tools/runpod_wisdom_pilot_selfreport.py` launcher + sibling `tools/runpod_*.py` scripts); no Spark count
  makes it feasible locally.

The Mac Studio (M3 Ultra, 819 GB/s — 3× a Spark's bandwidth, up to 512 GB) does **not** join the Spark CUDA
cluster (Metal/MLX ≠ CUDA); keep it as an independent inference/judge node (it served the Llama-3.3-70B
judge this session) or for MLX-native fine-tuning.

**Sources:** NVIDIA DGX Spark hardware overview; DGX Spark multi-node clustering (2-node direct / 3-node
NCCL ring / 4-node 200 Gb switch; 4-node ≈ 3.44×); DeepSeek-V3 Technical Report (arXiv:2412.19437);
repo `kernels/bench/roofline.py`, `tools/run_cluster_sim.py`, `docs/11-Platform/Cheap-Compute-Boundary.md`.
