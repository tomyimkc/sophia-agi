# HPC Operator / Communication / Compiler — development roadmap

> **What this is.** A concrete, honest roadmap for growing a *high-performance operator,
> collective-communication, and DSL-compiler* skill track on top of Sophia's existing
> NVIDIA RunPod GPU harness and no-overclaim measurement gate. It is written against the
> DeepSeek 「高性能算子/通信/编译器工程师」 job description (AI 核心系统研发).
>
> **Scope, stated plainly.** Sophia is a provenance-aware reasoning layer, *not* an
> operator library, and this roadmap does not pretend otherwise. The JD lives in a
> different domain (CUDA / Ascend C / TileLang kernels, NCCL/HCCL/DeepEP comms, MLIR).
> What follows is a *portfolio* track — a `kernels/` workstream that reuses Sophia's GPU
> rig and measurement discipline to demonstrate the JD's skills against the **hardware's
> physical limit**, with every number honestly bounded. Nothing here is an AGI claim, and
> no perf number ships without a roofline/theoretical-peak comparison and error bars.

---

## Why this repo is a credible launch point (and why it isn't a kernel library)

The JD's single standard is unusual and worth quoting:

> 团队的唯一标准：硬件的物理极限。我们不和任何 baseline 或者其他实现比较性能差异，只关心当前设计和实现距离理论上限还有多远，对每一个 cycle 和每一瓦功耗有着近乎偏执的追求。

Three genuine bridges from the existing repo:

1. **Measurement culture is a direct match.** Sophia's headline rule — *every public number
   clears a no-overclaim gate (CIs, ≥3 runs, multi-judge)* — is the same instinct as
   roofline analysis. `tools/bench_lora_speedup.py` already refuses the inflated number
   ("the 10-50x claim overstates… realistically ~2-4x"). Port that instinct to kernels:
   report **% of theoretical FLOP/s peak and % of peak HBM bandwidth**, never "Nx faster
   than a strawman."
2. **A real NVIDIA GPU rig already exists.** `tools/runpod_rlvr.py` /
   `tools/runpod_train.py` rent a CUDA pod, run over SSH, copy artifacts back, and *always*
   tear the pod down. That lifecycle is the substrate for running and profiling kernels —
   most kernel portfolios have no GPU at all.
3. **Real workloads to profile.** The JD wants *对真实训练/推理负载持续进行性能分析*.
   Sophia's own LoRA / RLVR runs are real workloads to Nsight-profile and attribute to
   operators — not a synthetic microbenchmark.

What is **out of scope / honest gaps**: Ascend NPU hardware (no access via RunPod — study
the API and design a portable abstraction, but mark NPU numbers as un-run); production
NCCL/HCCL maintenance; and IBGDA/RDMA on real InfiniBand fabric (single-node multi-GPU
NVLink is reachable; cross-node RoCE/IB generally is not on commodity rentals).

---

## Mapping the JD to buildable milestones

Priority is **M1 → M5** for fastest credible signal; M6 is continuous.

| JD responsibility | Roadmap milestone |
|---|---|
| 设计/实现 GEMM、Attention 等算子（CUDA / TileLang） | **M1** — roofline-gated GEMM + FlashAttention kernels |
| 面向 DL 的 DSL 编译器研发（TileLang / Triton / MLIR），扩展 IR/CodeGen | **M2** — Triton/TileLang DSL track + a toy MLIR pass |
| 通信算子（DeepEP 中的 EP/CP/DP/PP），NCCL/HCCL 调优排障 | **M3** — NCCL collective microbenchmarks + DeepEP study |
| GPU/NPU 多硬件体系（Ascend C、memory model、ISA） | **M4** — Ascend C study + portable kernel abstraction (no-HW) |
| 对真实训练/推理负载持续性能分析 | **M5** — Nsight-profile Sophia's own LoRA/RLVR runs |
| 加分项：LLVM / NCCL / PyTorch / TileLang / Triton 深度贡献 | **M6** — upstream open-source contributions |

---

## M1 — Roofline-gated GEMM + FlashAttention kernels *(do first)*

**Goal.** A `kernels/` directory with a small set of kernels written *twice* — once in
Triton (fast to iterate, JD-relevant), once in raw CUDA/CUTLASS (closer to the metal) —
each gated by a roofline harness rather than a baseline comparison.

- **Kernels (smallest-first):** fused bias+GELU → softmax → **tiled FP16/BF16 GEMM** with
  shared-memory staging → **FlashAttention-style** fused attention (online softmax, no
  materialized scores).
- **The gate (the point):** `kernels/bench/roofline.py` computes, per kernel, achieved
  FLOP/s and HBM GB/s, the device's theoretical peaks (from `list-gpu-types` /
  `cudaDeviceProp`), the arithmetic intensity, and **% of the relevant roofline ceiling**.
  Output mirrors `RESULTS.md`: ≥3 runs, dispersion, and an explicit "distance to physical
  limit." No "Nx vs naive" headline.
- **Run path:** extend the existing pod lifecycle — `tools/runpod_kernels.py --dry-run`
  prints the exact SSH commands offline; with `RUNPOD_API_KEY` it rents a pod, builds,
  runs Nsight Compute (`ncu`) for SM/memory utilization, copies the report back, and
  deletes the pod. Reuse `PodConnection` / teardown from `tools/runpod_rlvr.py`.
- **Honesty bound:** start single-precision-tier and one GPU arch (e.g. Ada/Hopper as
  rented). Tensor-core MMA + warp-specialization is a stretch goal; mark un-tuned kernels
  as "naive tiling, X% of peak" rather than hiding the gap.

**Skills it demonstrates:** CUDA/Triton authoring, memory hierarchy, MMA units, roofline
reasoning — the core of the JD.

## M2 — DSL/compiler track: Triton/TileLang + a toy MLIR pass

**Goal.** Show compiler-stack fluency, not just kernel authoring.

- **TileLang/Triton:** re-express the M1 GEMM in TileLang; write up the IR → schedule →
  CodeGen path and where autotuning lives. Capture how the same algorithm maps to
  different `tl` schedules and what each does to occupancy.
- **A minimal MLIR pass:** a standalone out-of-tree LLVM/MLIR project with one real
  rewrite (e.g. a tiling or fusion pass on a `linalg`/`affine` snippet), with lit tests.
  This is the most transferable single artifact for the "扩展 DSL/IR/CodeGen" line.
- **Deliverable:** `kernels/dsl/` + `docs/.../mlir-pass-notes.md` explaining the pass and
  the parallel-execution / memory-level model it assumes.

**Honesty bound:** a toy pass is a toy pass — label it a learning artifact, not a
production optimization.

## M3 — Collective-communication microbenchmarks + DeepEP study

**Goal.** Touch the communication half of the JD on reachable hardware.

- **Multi-GPU on one node:** rent a 2×/4×/8× NVLink pod; run **NCCL all-reduce /
  all-gather / reduce-scatter** microbenchmarks; report achieved bus bandwidth vs NVLink
  theoretical, and the all-reduce ring/tree algorithm crossover.
- **Tuning + triage:** sweep `NCCL_ALGO`, `NCCL_PROTO`, message size; document a
  diagnosis workflow (`NCCL_DEBUG=INFO`, topology dump) — this maps to *调优与排障*.
- **DeepEP / EP read-out:** a written study of expert-parallel dispatch/combine and where
  CP/DP/PP communication sits relative to compute; design (not necessarily implement) an
  overlap schedule. Mark **NVSHMEM/IBGDA and cross-node RDMA/RoCE as study-only** unless
  an IB-capable cluster becomes available — do not fabricate fabric numbers.

**Skills it demonstrates:** collective-comm internals, bandwidth roofline, NCCL triage.

## M4 — Ascend C / NPU study + portable kernel abstraction (no hardware)

**Goal.** Engage the *国产硬件 / 多硬件体系* dimension honestly, without an NPU.

- Write up the Ascend C programming model, memory model, and Cube/Vector unit differences
  vs CUDA SM; produce a side-by-side mental-model doc.
- Sketch a thin **portable tile abstraction** so an M1 kernel's structure could target
  either backend, and enumerate what would actually differ at CodeGen time.
- **Honesty bound:** clearly labelled as design + API study; **no NPU perf numbers** are
  claimed until hardware is available.

## M5 — Profile Sophia's own real training/inference workloads

**Goal.** Satisfy *对真实训练/推理负载持续进行性能分析* with workloads already in-repo.

- Run `tools/runpod_train.py` / `runpod_rlvr.py` under Nsight Systems; produce an operator
  time-breakdown (attention vs GEMM vs comms vs Python overhead) and a bottleneck report.
- Feed one finding back as a concrete optimization (e.g. a fused kernel from M1 swapped
  into the hot path), measured end-to-end through the existing gate.
- This closes the loop the JD prizes: kernels exist *to serve real model workloads*.

## M6 — Upstream open-source contributions *(continuous; the strongest signal)*

The JD's 加分项 explicitly values *在 LLVM / NCCL / PyTorch / Triton 等大型开源软件中有深度贡献*.
A few landed PRs outweigh any private portfolio:

- **Triton/TileLang:** docs fixes → small kernel/tutorial → a real autotuning or lowering
  improvement.
- **PyTorch:** an inductor/ATen kernel or a perf fix with a benchmark.
- **NCCL / LLVM:** start with reproducible issue triage; graduate to a patch.

Track these in this doc as they land, with PR links and the measured effect.

---

## Honest non-goals

- Not turning Sophia into an operator library; the `kernels/` track is a sibling portfolio
  that reuses the GPU rig, not a pivot of the trust-layer mission.
- No Ascend NPU perf claims without NPU hardware.
- No cross-node RDMA/IBGDA numbers without a real IB fabric.
- No "Nx vs naive baseline" headline numbers — only **% of theoretical peak**, with the
  same CI/≥3-run discipline as `RESULTS.md`.

## First concrete step

Land **M1's roofline harness skeleton** (`kernels/bench/roofline.py` + a `--dry-run`
`tools/runpod_kernels.py`) before writing any kernel — so that the *first* kernel is born
already measured against the physical limit, exactly as the JD demands.
