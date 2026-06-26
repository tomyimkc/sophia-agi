# Plan 02 — Real Kernel-Engineering Track for `kernels/`

> **Goal.** Take the `kernels/` workstream from "Python/PyTorch-level reproductions
> with a roofline harness" to **real, fused, low-level GPU kernels** that meet a
> frontier-lab kernel-engineering bar (Anthropic *TPU Kernel Engineer* / *Performance
> Engineer, GPU* / *Inference Runtime*; DeepMind *JAX/Pallas*). Preserve the repo's
> hardest-won asset: the **no-overclaim, roofline-gated, ≥3-runs-with-CIs** measurement
> ethos. Every kernel is born measured against the hardware's physical limit, never
> against a strawman baseline.

Author role: staff GPU performance engineer. Scope: this is a sibling *portfolio* track,
not a pivot of Sophia's trust-layer mission. Nothing here is an AGI claim.

---

## 1. Thesis & references

### 1.1 What "top-tier kernel work" actually requires

A frontier-lab kernel engineer is judged on the ability to take a numerically-defined
operator and drive it to a high, *measured* fraction of the hardware's speed-of-light
(SOL) — bounded by compute peak or HBM bandwidth — by controlling the memory hierarchy,
the instruction mix (tensor-core / MMA issue), occupancy, and IO. The concrete skill set:

1. **GPU kernel DSLs — Triton (OpenAI).** Tile-level Python that lowers through
   Triton-IR → LLVM → PTX/`ptxas`. The engineer reasons about `BLOCK_M/N/K`, `num_warps`,
   `num_stages` (software pipelining / async-copy depth), shared-memory budget, and
   `tl.dot` tensor-core issue. Autotuning (`triton.autotune`) over block/stage configs is
   table stakes. Ref: Tillet, Kung, Cox, *"Triton: An Intermediate Language and Compiler
   for Tiled Neural Network Computations"* (MAPL 2019); the OpenAI Triton tutorials
   (matmul, fused-softmax, layernorm, fused-attention).

2. **CUDA / CUTLASS / CuTe — closer to the metal.** Hierarchical GEMM via CUTLASS
   threadblock→warp→MMA tiling; `cp.async` / TMA (Hopper) bulk copies; `mma.sync`/`wgmma`
   tensor-core instructions; CuTe `Layout`/`Tensor` algebra for index/swizzle reasoning;
   bank-conflict-free shared-memory swizzles; double-buffered software pipelines. Ref:
   NVIDIA CUTLASS docs and the CuTe layout-algebra tutorials; the CUDA C++ Programming
   Guide (memory model, `cp.async`, async pipelines).

3. **FlashAttention v1/v2/v3 — the canonical IO-aware fused kernel.**
   - **v1** (Dao, Fu, Ermon, Rudra, Ré, *"FlashAttention: Fast and Memory-Efficient Exact
     Attention with IO-Awareness"*, NeurIPS 2022, arXiv:2205.14135): never materialize the
     N×N scores matrix `S = QKᵀ`; tile over K/V and maintain per-query running
     `(m, ℓ, acc)` via the **online-softmax** recurrence (rescale by `exp(m_old − m_new)`),
     turning score memory from O(N²) → O(tile) and HBM traffic from O(N²) → O(N²/M) — an
     *IO-complexity* argument, not just a constant-factor win.
   - **v2** (Dao, *"FlashAttention-2: Faster Attention with Better Parallelism and Work
     Partitioning"*, 2023, arXiv:2307.08691): fewer non-matmul FLOPs (defer the `1/ℓ`
     rescale to the epilogue), parallelize over the sequence (query) dimension as well as
     batch×heads, and better warp work-partitioning (split-Q across warps, keep K/V
     shared) to keep the tensor cores fed. ~2× over v1.
   - **v3** (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, *"FlashAttention-3"*, 2024,
     arXiv:2407.08608): Hopper-specific — overlap `wgmma` (warpgroup MMA) with softmax via
     warp-specialized producer/consumer pipelines and TMA, plus **FP8** attention with
     incoherent processing for accuracy. Pushes H100 utilization toward 70–75% of FP16 SOL
     / ~1.2 PFLOP/s FP8.

4. **Fused kernels — eliminating launch + HBM round-trips.** Fuse memory-bound epilogues
   into the producing kernel: **RMSNorm** (fused reduction + normalize + scale; the
   Llama-family norm), **LayerNorm** (Welford / two-pass), fused **bias+activation**
   (GELU/SiLU), and **fused-MLP / SwiGLU** (gate·up GEMM → activation → down GEMM with the
   activation fused into the first GEMM's epilogue). The win is bandwidth: a norm reads N
   bytes and writes N bytes, so it lives at the HBM roofline and fusion removes whole
   passes. Ref: Zhang & Sennrich, *RMSNorm* (NeurIPS 2019); the Triton fused-softmax /
   layernorm tutorials.

5. **Quantization kernels — FP8 / INT4 GEMM.** Low-precision matmul where the engineering
   is in scaling and packing, not just the MMA: **FP8** (E4M3/E5M2) GEMM with per-tensor /
   per-row / per-block (e.g. 1×128, 128×128) scaling and FP32 accumulation; **INT4 / INT8**
   weight-only GEMM for decode-time inference (dequantize-in-the-mainloop, packed `int4`
   weights, group-wise scales). Ref: NVIDIA Transformer Engine (FP8 recipes); the
   *LLM.int8()* (Dettmers et al., 2022) and *GPTQ* (Frantar et al., 2023) / *AWQ* (Lin et
   al., 2023) lineages for the quantization side; Marlin / Machete (FP16×INT4) and CUTLASS
   mixed-input GEMM for the kernel side; DeepSeek-V3's fine-grained FP8 block scaling.

6. **KV-cache layouts — paged vs contiguous.** **PagedAttention** (Kwon et al., *"Efficient
   Memory Management for Large Language Model Serving with PagedAttention"*, SOSP 2023 —
   the vLLM paper): a block-table indirection so KV lives in non-contiguous fixed-size
   pages, near-eliminating fragmentation and enabling sharing (prefix reuse, beam search).
   The kernel must gather K/V through the block table inside the attention inner loop.
   Contrast with contiguous KV (simpler, faster gather, but fragments). Decode-phase
   attention is *memory-bound* (batch-1 GEMV-like), so the metric is **% of HBM SOL**, not
   FLOP/s.

7. **Roofline / occupancy / bandwidth analysis — the discipline.** Williams, Waterman &
   Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore
   Architectures"* (CACM 2009). Per kernel: arithmetic intensity (FLOP/byte), the ridge
   point, memory-bound vs compute-bound regime, and **% of the attainable roofline
   ceiling** `min(peak_compute, I·peak_bw)`. Tools: Nsight Compute (`ncu`) for SM /
   tensor-core / memory utilization, achieved occupancy, warp-stall reasons, L2 hit rate;
   Nsight Systems (`nsys`) for the end-to-end timeline. This repo *already encodes this
   ethos* (`kernels/bench/roofline.py`) — the new kernels plug straight into it.

8. **TPU side — Pallas / Mosaic / JAX.** Pallas is JAX's kernel-authoring DSL; on TPU it
   lowers through the **Mosaic** compiler to the systolic MXU + VPU, reasoning about VMEM
   tiles, the `BlockSpec`/`grid` mapping, and pipelining over HBM↔VMEM. The canonical
   exercise is a Pallas FlashAttention (`jax.experimental.pallas.ops.tpu`). On GPU, Pallas
   lowers via a Triton/Mosaic-GPU path. Ref: the JAX Pallas design notes and TPU
   FlashAttention reference kernel; relevant to the **Anthropic TPU Kernel Engineer** and
   **DeepMind JAX/Pallas** bars. Treated here as a **study + one portable kernel** track
   (no TPU on RunPod — honestly marked un-run, mirroring the roadmap's NPU stance).

### 1.2 Reference list (cite by name in write-ups)

- Williams, Waterman, Patterson 2009 — Roofline model (CACM).
- Tillet et al. 2019 — Triton (MAPL).
- Dao et al. 2022 — FlashAttention v1 (arXiv:2205.14135).
- Dao 2023 — FlashAttention-2 (arXiv:2307.08691).
- Shah et al. 2024 — FlashAttention-3 (arXiv:2407.08608).
- Zhang & Sennrich 2019 — RMSNorm.
- Kwon et al. 2023 — PagedAttention / vLLM (SOSP).
- Dettmers et al. 2022 — LLM.int8(); Frantar et al. 2023 — GPTQ; Lin et al. 2023 — AWQ.
- NVIDIA — CUTLASS / CuTe docs; Transformer Engine FP8; CUDA C++ Programming Guide.
- JAX — Pallas / Mosaic docs and the TPU FlashAttention reference kernel.

---

## 2. Current repo state (file-level, honest)

| Path | What it actually is today | Honest level |
|---|---|---|
| `kernels/bench/roofline.py` | A genuinely good **roofline harness**: device datasheet peaks (`DEVICE_SPECS`), FLOP/byte accounting (`gemm_flops`, `gemm_bytes`, `attention_flops`), `analyze()` → median-of-≥3 timings, regime, ridge point, **% of roofline / compute peak / HBM peak**, an **over-100% guard**, `--self-test`, `--demo`. Offline; no GPU needed for the math. | **Strong & reusable.** This is the asset to build on. |
| `kernels/flash_attention.py` | A **numpy** O(N²) `naive_attention` reference + a **numpy** `flash_attention_numpy` online-softmax tiled reference (matches naive, tracks max score-tile to *prove* O(tile) memory, causal tile-skip), CI-gated `offline_invariants()`. Plus a **gated Triton `flash_attention_triton`** kernel that only runs with Triton+CUDA (i.e. never in CI) and is **not yet measured on a GPU**. | numpy = **algorithm reproduction, not a kernel.** Triton fn = real but **fp32, unprofiled, fixed `BLOCK=64`, non-autotuned, no v2 work-partitioning, no `num_stages`/`num_warps` tuning, CPU↔GPU copies each call.** |
| `kernels/bench.py` | Offline correctness + **score-memory** benchmark (flash-tile vs full N×N) across seqlens; explicitly refuses to report numpy wall-time as a speedup. Runs the Triton path only if present. | Honest; **memory-traffic argument only**, no GPU timing. |
| `kernels/src/run_kernel.py` | The **one real kernel**: a Triton **tiled BF16 GEMM**, FP32 accumulate, grouped super-block ordering for L2 reuse, correctness vs `torch.matmul`, CUDA-event timing (≥3 iters, warmup excluded), self-rooflines via `analyze()`. **Single fixed block config (128/128/64/group 8), no autotune, no split-K, no warp-spec, no async pipeline.** | Real Triton, but **un-profiled on hardware and un-tuned** — README is honest that its % of roofline is "the M1 number to report and then close." |
| `tools/runpod_kernels.py` | Orchestrator: reuses the proven `runpod_rlvr.py` pod lifecycle (create → SSH → run → copy artifacts → **always delete pod**), `--dry-run` by default, runs roofline self-test + timed kernel + **`ncu --set full`** profile, copies `kernels/reports/**` back. GitHub Actions `kernels-runpod` workflow with a `RUN` confirm gate. | **Excellent plumbing.** GPU access exists and is safe/cheap-by-default. |
| `docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md` | M1–M6 roadmap (GEMM+Flash → DSL/MLIR → NCCL → Ascend → profile real workloads → upstream PRs), written against a DeepSeek HPC JD, with explicit honesty bounds. | Good north star; this plan executes & extends **M1** and feeds **M2/M5**. |

**Honest one-liner.** Today the `kernels/` track has (a) a first-class roofline/measurement
harness, (b) excellent safe GPU-rental plumbing, (c) **one** real (un-tuned, un-profiled)
Triton GEMM, and (d) FlashAttention as a **numpy algorithm reproduction** plus an unprofiled
gated Triton draft. There is **no measured GPU number yet**, no fused norm/softmax kernel, no
quantized GEMM, no autotuning, no `ncu`-attributed optimization loop. That is exactly the gap
between "I reproduced the algorithm in Python" and "I am a kernel engineer."

---

## 3. Top-tier target end-state

A `kernels/` track that demonstrates, with **measured, roofline-gated, reproducible**
numbers on rented NVIDIA GPUs:

1. **A library of fused Triton kernels**, each with: a numerically-checked reference
   (numpy/torch), a tolerance-bounded correctness gate, an **autotuned** Triton
   implementation, a CUDA-event timing harness, an `ncu`-attributed roofline report
   (**% of SOL bandwidth for memory-bound ops; % of compute roofline for GEMM/attention**),
   and a short write-up citing the technique by name.
   - Memory-bound, fused: **fused softmax**, **RMSNorm**, **LayerNorm**, **fused
     bias+SiLU/GELU**, **fused SwiGLU MLP**.
   - Compute-bound: **autotuned BF16/FP16 GEMM** (close the gap left open in `run_kernel.py`),
     **FlashAttention v2-style fused attention** (forward; causal + non-causal).
   - Quantized: **FP8 (E4M3) GEMM** with block scaling (H100), **INT4 weight-only GEMM**
     (dequant-in-mainloop) for decode.
2. **A KV-cache study + kernel**: a contiguous-KV decode attention kernel and a
   **paged** variant (block-table gather), benchmarked as **% of HBM SOL** in the decode
   (memory-bound) regime — citing PagedAttention.
3. **An `ncu`-driven optimization narrative** for at least the GEMM and FlashAttention:
   baseline → identify the binding stall (tensor-core issue, shared-memory bank conflicts,
   occupancy, uncoalesced loads) → fix → re-report % of roofline at each step. This *loop*
   is the single most senior signal.
4. **A Pallas/JAX study + one portable kernel** (RMSNorm or attention) written in Pallas,
   run on **GPU** via Pallas's Triton/Mosaic-GPU path, with TPU lowering **designed and
   honestly marked un-run** (no TPU hardware).
5. **Reproducibility**: every reported number regenerated by `tools/runpod_kernels.py --yes`
   (or the `kernels-runpod` Actions workflow), with `kernels/reports/**` artifacts and the
   roofline block, under the same ≥3-runs/CI gate as `RESULTS.md`.

---

## 4. Phased plan — concrete kernels to write

Each kernel follows the **same five-file pattern** so it plugs into the existing harness:
`kernels/src/<name>.py` (reference + autotuned Triton + a `run_<name>()` that times,
correctness-checks, and calls `roofline.analyze`), correctness exercised offline in
`kernels/tests/`, profiled by extending `tools/runpod_kernels.py`'s remote script, written
up in `kernels/reports/` / a short doc. **No "Nx vs naive" headline — only % of roofline.**

### Harness prerequisites (do first, ~0.5 day, offline)

- **P0.1** Generalize `roofline.py` accounting helpers for memory-bound ops: add
  `elementwise_bytes(numel, dtype_bytes, reads, writes)` and a
  `reduction_flops/​bytes(rows, cols)` pair so softmax/RMSNorm get a correct HBM-traffic
  denominator (they are **bandwidth-bound**, so the headline is **% of HBM peak**, and
  `analyze()` already computes `pct_of_bandwidth_peak`). Add FP8/INT4 byte sizes
  (`dtype_bytes ∈ {1, 0.5}`) and an FP8 tensor-peak field to `DeviceSpec` for H100
  (`fp8_tensor_tflops ≈ 1979` dense).
- **P0.2** Add a tiny `kernels/src/_bench.py` shared helper: CUDA-event timing loop
  (warmup excluded, ≥`iters` trials), correctness check with explicit tolerance, and a
  one-call `report(flops, bytes, times, dtype)` wrapper. Refactor `run_kernel.py` to use
  it (keeps one timing path, prevents per-kernel drift).
- **P0.3** Extend `tools/runpod_kernels.py` remote script to discover and run **every**
  `kernels/src/run_*.py` (loop over a manifest), each emitting its own
  `kernels/reports/<name>_roofline.txt`, with a single `ncu` pass per kernel.

### Phase A — Fused memory-bound kernels (START HERE; cheapest GPU, highest learning/$) — ~3–4 days

These are the right first real kernels: they are bandwidth-bound (so the metric is clean —
**% of HBM SOL**), they teach the Triton reduction/tiling model, and they run on the
**cheapest** GPU tier (T4/A10). They mirror the OpenAI fused-softmax / layernorm tutorials
but are measured against the roofline, not "faster than torch."

1. **Fused softmax** — `kernels/src/fused_softmax.py`.
   - Triton row-wise softmax: one program per row block, `tl.max`/`tl.sum` reductions in
     SRAM, single HBM read + single write (the fusion win — eager torch does max, sub, exp,
     sum, div as separate passes). Autotune `BLOCK_SIZE` over the row length; handle rows
     wider than one block (online two-pass).
   - Reference: `torch.softmax`. Tolerance: fp32 1e-5, bf16 ~1e-2.
   - **Benchmark vs**: `torch.softmax` (eager) and `F.scaled_dot_product_attention`'s
     internal softmax is N/A — compare **wall-time AND % of HBM SOL**. Headline = % of HBM
     peak; a good fused softmax should clear ~70–85% of HBM SOL on an A10.

2. **RMSNorm** — `kernels/src/rmsnorm.py`.
   - Fused: read x, compute `1/rms = rsqrt(mean(x²)+eps)` reduction, multiply by weight,
     write — one read, one write. Autotune `BLOCK_SIZE`. Optional fused residual-add
     (`x + attn_out` then norm) — the real transformer hot path.
   - Reference: a numpy/torch RMSNorm; cite Zhang & Sennrich 2019.
   - **Benchmark vs**: torch eager (`x * rsqrt(mean(x²)) * w`) — report % of HBM SOL;
     show the fused-residual variant removes an extra HBM round-trip.

3. **(stretch in Phase A) Fused bias+activation / SwiGLU gate** —
   `kernels/src/fused_act.py`: fused `SiLU(x·W_gate) ⊙ (x·W_up)` epilogue concept at the
   elementwise level first (the GEMM fusion comes in Phase C). Memory-bound; % of HBM SOL.

**Phase A exit criteria**: fused softmax + RMSNorm each correct vs reference, autotuned,
and reporting a measured **% of HBM SOL ≥ ~70%** on an A10/A100, with an `ncu` memory-
utilization screenshot in `kernels/reports/`. This is the first **measured GPU number** the
track has ever produced — the highest-signal early milestone.

### Phase B — Triton FlashAttention v2-style forward — ~4–6 days

Promote the gated draft in `kernels/flash_attention.py` into a real, tuned kernel:
`kernels/src/flash_attention_fwd.py`.

- Start from the existing recurrence (it is correct), then:
  - **v2 work-partitioning**: parallelize the grid over `(batch·heads, query-tiles)`;
    defer the `1/ℓ` normalization to the epilogue (fewer non-matmul FLOPs); split Q across
    warps, keep K/V tiles shared. Cite Dao 2023.
  - Support **causal** (whole-tile skip above the diagonal — already prototyped in numpy),
    multiple `head_dim ∈ {64, 128}`, and BF16 in / FP32 accumulate.
  - **Autotune** `(BLOCK_M, BLOCK_N, num_warps, num_stages)`; `num_stages` enables Triton's
    async-copy software pipeline (the `cp.async` analogue).
- **FLOP/byte accounting**: reuse `attention_flops(...)`; HBM bytes = read Q,K,V once +
  write O (the *whole point* is O(N²)→O(N) traffic). Attention is compute-bound at long
  context, so the headline = **% of compute roofline**; report HBM % too.
- **Benchmark vs**: `torch.nn.functional.scaled_dot_product_attention` (which dispatches to
  the real FlashAttention-2 backend) **and** the numpy reference for correctness. Be honest:
  our hand-written Triton kernel will land **below** the cuDNN/FA2 backend — report the gap
  as **% of roofline for both**, never "we beat torch."
- **`ncu` loop**: identify the stall (tensor-core issue vs softmax exp throughput vs shared
  memory), apply one fix, re-report. This narrative is the Phase B deliverable.

**Phase B exit criteria**: forward attention correct vs reference (causal + non-causal,
head_dim 64/128), autotuned, measured % of compute roofline on A100/H100, with an
`ncu`-attributed before/after on at least one optimization, and an honest table vs SDPA.

### Phase C — Quantized GEMM (FP8 / INT4) — ~5–7 days (needs A100/H100)

1. **FP8 (E4M3) GEMM** — `kernels/src/fp8_gemm.py` (H100; Ada FP8 also works).
   - `tl.dot` with FP8 inputs → FP32 accumulate; per-row (or 128×128 block) scaling factors
     applied in the epilogue; correctness vs an FP16 reference within an FP8-appropriate
     tolerance. Cite Transformer Engine / DeepSeek-V3 block scaling.
   - Add `fp8_tensor_tflops` to `DeviceSpec`; headline = **% of FP8 compute roofline**.
   - **Benchmark vs**: the BF16 Triton GEMM from `run_kernel.py` (same M/N/K) and
     `torch._scaled_mm` (the FP8 path) — report % of FP8 roofline for each; FP8 doubles the
     compute ceiling, so the honest comparison is FP8 % of FP8 SOL, not "2× the BF16 kernel."

2. **INT4 weight-only GEMM** — `kernels/src/int4_gemm.py` (decode-time inference path).
   - Packed `int4` weights + group-wise (e.g. group=128) FP16 scales; **dequantize in the
     mainloop** (load packed int4 → unpack → scale → `tl.dot` against FP16 activations).
     This is the Marlin/AWQ-kernel shape. Cite GPTQ/AWQ for the quant scheme, Marlin for the
     kernel.
   - This is the **memory-bound decode** regime (weights dominate traffic) — headline =
     **% of HBM SOL**; the win is reading 4-bit weights, not FLOPs.
   - **Benchmark vs**: an FP16 GEMV/skinny-GEMM (batch 1–8, the decode shape) — report
     achieved weight-read bandwidth as % of HBM SOL.

**Phase C exit criteria**: FP8 GEMM correct + measured % of FP8 roofline on H100; INT4
weight-only GEMM correct + measured % of HBM SOL on A100, each with `ncu` artifacts.

### Phase D — KV-cache & Pallas studies (feeds M2/M5; partly study-only) — ~4–6 days

1. **Decode attention + paged KV** — `kernels/src/decode_attention.py`: batch-1/few
   decode-step attention (GEMV-like, memory-bound) over (a) **contiguous** KV and (b) a
   **paged** block-table gather. Headline = **% of HBM SOL** in decode. Cite PagedAttention
   (Kwon et al. 2023). A correctness gate vs the numpy reference; benchmark page-size effect.
2. **Pallas portable kernel** — `kernels/pallas/rmsnorm_pallas.py`: RMSNorm (and/or a Pallas
   FlashAttention adapted from the JAX reference) run on **GPU** via Pallas's Triton/Mosaic-GPU
   backend, measured through a JAX timing path. **TPU lowering designed and documented but
   marked un-run** (no TPU on RunPod) — mirrors the roadmap's NPU honesty bound. This is the
   direct artifact for the Anthropic TPU / DeepMind JAX-Pallas bars.

### How benchmarking plugs into the existing harness (all phases)

- Each `run_<name>()` builds `times_s` from CUDA events (warmup excluded, ≥3 iters) and
  calls `roofline.analyze(flops=…, bytes_moved=…, times_s=…, device=resolve_device(detect_device()), dtype=…)`,
  printing `format_report(...)`. The **% of roofline / % of HBM peak** block is the headline.
- **PyTorch baselines are reported side-by-side as a second roofline row**, not as the
  headline ratio: e.g. "Triton fused softmax: 78% of HBM SOL; `torch.softmax` eager: 41% of
  HBM SOL" — both measured, both % of the *physical* limit. For attention, the baseline is
  `F.scaled_dot_product_attention` (real FA2 backend); for GEMM, `torch.matmul`/`_scaled_mm`.
- `tools/runpod_kernels.py` runs the manifest, `ncu --set full` profiles each kernel once,
  and copies `kernels/reports/**` back; the `kernels-runpod` Actions workflow does the same
  with the `RUN` confirm gate. Reproducible by construction.

---

## 5. Compute / budget tiers (Triton needs a GPU)

Triton **cannot** be exercised without a CUDA GPU — CI stays offline (numpy references +
roofline self-test). RunPod tiers, cheapest-capable-first:

| Tier | GPU (RunPod) | ~$/hr (community/secure, indicative) | What it unlocks | When |
|---|---|---|---|---|
| **Min / cheap** | **1× NVIDIA T4 (16GB)** or **1× A10 / L4 (24GB)** | ~$0.20–0.50/hr | Phase A fused softmax/RMSNorm, the Triton GEMM, FlashAttention fwd correctness + a first % of HBM/compute SOL. Tensor cores present (T4 = Turing FP16; A10/L4 = Ampere/Ada). | A, start of B |
| **Standard** | **1× A100 80GB (SXM/PCIe)** | ~$1.2–2.0/hr | Real FlashAttention tuning, INT4 weight-only GEMM, BF16 GEMM at scale, clean roofline at high intensity, big `ncu` profiles. The default in `runpod_kernels.py`. | B, C (INT4), D |
| **FP8 / frontier** | **1× H100 80GB HBM3** | ~$2.5–4.0/hr | **FP8 GEMM** (E4M3, the only tier with the FP8 tensor path that matters), FlashAttention-3-style overlap study, ~1 PFLOP/s-class roofline. | C (FP8), B stretch |
| **Multi-GPU (later)** | 2×/4×/8× NVLink (A100/H100) | scales with count | NCCL collectives (roadmap M3) — out of scope for this kernel plan. | future |

Budget discipline (preserve the repo's cheap-by-default stance): develop kernels **offline**
against numpy/torch-CPU references, get correctness green in CI, then rent the **smallest
capable** GPU and run the manifest in a single pod session (all kernels, one `ncu` pass each)
to amortize boot/clone cost. A full Phase-A measurement run is well under **$1**; a tuned
Phase-B/C H100 session is a few **$**. The pod **always self-deletes** (existing watchdog).

---

## 6. Honest success metrics

Per kernel, a number ships only if it clears the **same gate as `RESULTS.md`**:

1. **Correctness** vs an independent reference (numpy fp64 or torch), with an **explicit,
   dtype-appropriate tolerance** (fp32 ~1e-5; bf16 ~1e-2 rel; fp8/int4 a stated, looser
   bound). Recorded, not assumed.
2. **Measured speedup vs the PyTorch baseline** — reported as **wall-time with ≥3 runs +
   dispersion (median/min/stdev)**, but framed as *both* kernels' **% of the physical
   limit**, never a bare "Nx." (e.g. "fused RMSNorm 80% of HBM SOL vs torch eager 45%".)
3. **% of SOL**: **% of HBM bandwidth peak** for memory-bound kernels (softmax, RMSNorm,
   INT4 decode, paged-KV); **% of compute roofline** for GEMM and long-context attention.
   This is *the headline*, bounded ≤100% by construction.
4. **`ncu`-attributed regime**: achieved SM / tensor-core / memory utilization, occupancy,
   and the dominant warp-stall reason — so the % gap to roofline is *explained*, not just
   stated.
5. **Reproducibility**: regenerable via `tools/runpod_kernels.py --yes` / the Actions
   workflow, with the `kernels/reports/**` artifact committed-by-reference (git-ignored
   binaries, text reports summarized in the doc).

Concrete (honest, achievable) targets — *aspirational, to be confirmed by measurement*:
fused softmax / RMSNorm **≥70% HBM SOL** (A10/A100); autotuned BF16 GEMM **≥60–70% compute
roofline** (won't match cuBLAS — state the gap); FlashAttention fwd **≥40–55% compute
roofline** and honestly **below** the SDPA/FA2 backend; FP8 GEMM **≥50% FP8 roofline**
(H100). **Anything >~95% of roofline is treated as a FLOP/byte accounting bug until proven
otherwise** — the existing over-100% guard stays.

---

## 7. Risks / overclaim guards

- **"I wrote a kernel" ≠ "it's fast."** Guard: no kernel is "done" until it has a *measured*
  % of SOL and an `ncu` attribution. The numpy FlashAttention is a *reference*, never quoted
  as a kernel result.
- **Beating a strawman.** Guard: keep the no-"Nx vs naive" rule; report % of roofline for
  *both* our kernel and the PyTorch baseline. Explicitly state when torch's backend (FA2,
  cuBLAS) beats us.
- **Cherry-picked shapes.** Guard: sweep a small grid of shapes (seqlen, head_dim, M/N/K)
  and report the curve, not the best point.
- **FLOP/byte miscounting** (inflates % of SOL). Guard: the existing over-100% warning;
  audited `gemm_flops`/`attention_flops`/new elementwise helpers; cross-check intensity
  against the regime `ncu` reports.
- **Precision laundering.** Guard: FP8/INT4 numbers always carry their tolerance and the
  reference dtype; never compare an FP8 kernel's TFLOP/s to a BF16 roofline.
- **TPU/Pallas overreach.** Guard: TPU lowering is **designed and marked un-run** (no
  hardware), exactly as the roadmap marks Ascend NPU — no fabricated TPU numbers.
- **Autotune overfit / non-determinism.** Guard: pin the autotuned config in the report,
  fix seeds, re-run ≥3×; report dispersion.
- **Cost creep.** Guard: offline-first, smallest capable GPU, single batched pod session,
  always-delete watchdog.

---

## 8. Effort estimate

| Phase | Scope | Est. (focused) | GPU $ (indicative) |
|---|---|---|---|
| P0 | Harness extensions (elementwise/FP8 accounting, shared bench helper, manifest runner) | 0.5–1 day | $0 (offline) |
| A | Fused softmax + RMSNorm (+act stretch), first measured % of HBM SOL | 3–4 days | < $1 (T4/A10) |
| B | FlashAttention v2-style fwd, autotuned, `ncu` loop vs SDPA | 4–6 days | $2–6 (A100) |
| C | FP8 GEMM (H100) + INT4 weight-only GEMM (A100) | 5–7 days | $5–15 (H100/A100) |
| D | Paged/contiguous decode KV + Pallas portable kernel + TPU study | 4–6 days | $2–5 (A100) |
| — | Write-ups, roadmap update, reproducibility pass | ongoing | — |

**Total: ~3–4 focused weeks**, < ~$30 of GPU rental for the full measured set, to go from
"Python reproductions + one un-profiled GEMM" to "a measured, roofline-gated, `ncu`-
attributed fused-kernel + quantized-GEMM portfolio." **Phase A alone** (≈1 week, < $1)
produces the track's first real GPU number and is the highest learning-per-dollar starting
point.

---

### Appendix — first-week concrete checklist

1. P0.1–P0.3 harness extensions (offline, CI-green).
2. `kernels/src/fused_softmax.py` — Triton + autotune + reference + `run_fused_softmax()`.
3. `kernels/src/rmsnorm.py` — Triton + autotune + reference + `run_rmsnorm()`.
4. `kernels/tests/` correctness for both (CI, CPU/numpy reference).
5. Extend `tools/runpod_kernels.py` manifest to run both; one `runpod_kernels.py --yes` on a
   1× A10/A100 pod → first **measured % of HBM SOL** in `kernels/reports/`.
6. Short write-up citing the OpenAI fused-softmax tutorial + Zhang & Sennrich (RMSNorm),
   reporting % of HBM SOL for the Triton kernels *and* the torch-eager baselines.
