# Maximizing the DGX Spark — a bandwidth-honest playbook

_How to get the most out of an NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64,
128 GB unified LPDDR5x) inside Sophia's provenance-first discipline._

> **Companion doc.** This is the *capability* playbook for the Spark; the
> *provenance boundary* (why a Spark number is never a registered result) lives in
> [`Spark-Local-GPU-Lane.md`](./Spark-Local-GPU-Lane.md) and is **load-bearing** —
> read it first. Nothing here weakens that boundary.

## 1. Read the hardware honestly

The Spark's headline is "1 PFLOP of AI compute" — but that is **sparse FP4**, and
it is not the number that governs day-to-day LLM work. The number that governs it
is **memory bandwidth: 273 GB/s** of unified LPDDR5x (confirmed by NVIDIA's DGX
Spark hardware overview), shared by the 20-core Arm CPU and the Blackwell GPU.

| Property | Value | Consequence |
|---|---|---|
| Unified memory | **128 GB** LPDDR5x | Hold models that don't fit an 80 GB H100 / 24 GB 4090 — the superpower. |
| Bandwidth | **273 GB/s** | ~12× *less* than an H100's ~3.35 TB/s. Token **decode is memory-bound**. |
| Compute | ~1 PFLOP **sparse** FP4 (~500 TFLOP/s dense FP4) | Plenty — but you rarely reach it, because you hit the bandwidth wall first. |
| Arch | **aarch64** | Breaks the x86 ML wheel stack (flash-attn, bitsandbytes, vLLM-colocate, unsloth). Rust cross-compiles cleanly. |

**The one sentence that should drive every decision:** the Spark is
**memory-capacity-rich and bandwidth-poor.** Design *for* the 128 GB pool and *around*
the 273 GB/s wall — never the reverse. Put concretely in our own roofline harness
([`kernels/bench/roofline.py`](../../kernels/bench/roofline.py), now carrying a
`NVIDIA DGX Spark GB10` profile with a `fp4_tensor_tflops` tier): even a dense FP4
GEMM needs arithmetic intensity **> ~1830 FLOP/byte** to escape the memory wall, so
batch-1 decode is *always* memory-bound here. Bytes-per-weight (i.e. 4-bit) is the
dominant lever, not FLOPs.

## 2. What the Spark is *not*

- **Not a pretraining box.** Bandwidth-bound and single-node; pretraining is the one
  thing it is worst at.
- **Not a registered-results producer.** aarch64 forces `--quant bf16 --vllm none`,
  different numerics from the x86 RunPod path — see the lane doc. Source of record
  stays x86 RunPod.
- **Not a CI replacement.** 17/18 of our workflows are CPU/lint/test and gain nothing
  from Grace-Blackwell compute; moving them onto a personal box that reboots would make
  CI flaky and drop the `windows-latest` leg.

## 3. The five highest-ROI uses (in order)

### 3.1 The always-on **data refinery** — highest leverage
Our own [`Training-Efficiency-Feasibility.md`](./Training-Efficiency-Feasibility.md)
proves the capability lever is **data quality**, not training speed (the honest LoRA
speedup is ~2×, from dynamic padding; the "10–50×" is the corpus-shrink lever). The
Spark's 128 GB lets it hold a **70B-class teacher in NVFP4** and generate
council-distillation + RFT targets **24/7 for free**, each passed through the
**intrinsic, fail-closed** gate (`check_response(text, mode="advisor")["violations"]`,
*no question* — passing a question invokes the trap-grader that wrongly deletes clean
rows). This *is* roadmap phase **P2** (gate-filtered RFT data engine), and it feeds the
lever that actually moves capability. It is latency-tolerant, so the bandwidth wall
doesn't hurt. **Do this first.**

### 3.2 The **NVFP4 MoE inference lane** — the Spark-native capability
MoE is the one architecture where "huge memory + low bandwidth" is a *feature*: all
experts live in the 128 GB pool, but only top-k activate per token, so bytes-moved-per-token
stays low. We already own [`moe/router.py`](../../moe/router.py) +
[`moe/quant.py`](../../moe/quant.py); this doc adds an **NVFP4 backend** (E2M1 + per-block
FP8-E4M3 micro-scale, ~4.5 effective bits, ~3.6× smaller than FP16) to `quant.py`, proven
against its error bound in CI alongside INT8/FP8. The micro-scale is the load-bearing
trick — a single global FP4 scale is unusable (CI demonstrates block-scaled error ≪
global-scaled). The deployment artifact is the fused dequant-in-the-GEMM kernel (§3.3).
Branch to land it through: `claude/spark-moe-workflow`.

### 3.3 **Roofline everything on the Spark's own ceiling**
The Spark will tempt misleading "Nx vs my laptop" numbers. The roofline gate forbids
that: every Spark kernel reports **% of the Spark's 273 GB/s (or FP4) roofline**. The
device profile is now in `DEVICE_SPECS`; a Spark NVFP4-dequant-GEMM should report its %
of roofline with ≥3 runs and dispersion, exactly like the H100 GEMM. Most kernels here
will read "memory-bound, N% of 273 GB/s" — and that is the honest headline.

### 3.4 **Overnight closed-loop agent runs**
`agent/long_horizon.py`, `agent/lifelong_accumulation.py`, and the self-evolving
evolve→promote→retain loop are long-horizon and latency-tolerant — ideal for a free,
always-on box. Run them overnight; wake to gate-checked accumulated memory / candidate
adapters you then **register** on RunPod. Free compute on work that doesn't care about
throughput.

### 3.5 **Two independent local backends = a provenance feature**
We already run a Mac Studio (M3 Ultra / MLX) and now the Spark (CUDA aarch64). A result
that **reproduces across two independent local numerics paths** is more trustworthy than
one. Fold "cross-backend agreement" into the evidence story — while keeping the rule that
*neither* produces a registered number; RunPod x86 remains the source of record.

## 4. Languages: Rust now, Mojo as a gated pilot

### Rust (already here — consolidate it)
The Spark's aarch64 arch *breaks* the x86 Python ML wheels but Rust cross-compiles to
aarch64 cleanly — so **every hot path moved from fragile Python into Rust increases Spark
usability.** We already have ~9 crates (`storage/{kvcache,lsm,raftkv,miniraft,diskstore,
infcache}`, `sophia-storage/`, `services/ann_serving`), but they are sprawled and partly
duplicated (`storage/kvcache` vs `sophia-storage/crates/sophia-kvcache`). Optimized move:
**one Cargo workspace, one `Cargo.lock`, one CI lane**, and keep pushing the durable, hot,
safety-critical paths (KV cache, vector store, dataflow firewall, agent state) behind PyO3
bridges into it. Python stays for research/orchestration.

### Mojo (officially Spark-supported — pilot it, don't bet on it)
As of **Modular Platform 26.2**, Mojo/MAX added **explicit DGX Spark support** (with B300
and Jetson Thor); Mojo 1.0 targets Summer 2026, and Modular's GTC 2026 demo ported NVIDIA's
CUTLASS Blackwell conv2d kernel to **130.7 TFLOP/s on B200 in ~770 lines of Mojo vs ~3k of
CUDA C++**. So Mojo can write competitive Blackwell kernels *and* targets our exact device.
Adopt it the repo-idiomatic way: **one bounded, roofline-gated pilot** — write the fused
NVFP4-dequant-GEMM (or MoE dispatch) in Mojo, measure it against the *same* Spark roofline as
the Triton version, adopt only if it wins and is stable on aarch64. If it doesn't, the
Triton path already works. Falsifiable, measured against a ceiling, promote-only-what-verifies
— the same discipline as every other claim in this repo.

## 5. What this change set adds (concrete, in-repo)

- `moe/quant.py` — **NVFP4 backend**: `nvfp4_roundtrip`, `quantized_linear_nvfp4`,
  `nvfp4_memory_reduction`, with error-bound + micro-scale-beats-global invariants in CI.
- `kernels/bench/roofline.py` — **FP4 tensor tier** on `DeviceSpec` and a corrected
  **DGX Spark GB10** profile (273 GB/s confirmed; ~500 TFLOP/s dense FP4).
- Tests in `tests/test_moe.py` and `tests/test_roofline.py` (memory-bound Spark decode +
  FP4-only-on-Blackwell).

## 6. Recommended next milestones

1. **P0** — pre-baked aarch64 CUDA image so a Spark run starts in seconds, not minutes of pip.
2. **P2 data refinery** on the Spark (free local teacher → gate-filtered RFT data) — §3.1.
3. **Fused NVFP4 dequant-GEMM** (Triton first, Mojo pilot second), rooflined on the Spark — §3.2/§3.3.
4. **Gate-as-reward GRPO, abstention-positive** (P3): iterate on Spark, register on RunPod.
5. **One Cargo workspace** (§4) + resolve the `agi-proof/architecture-bets.json` schema collision
   (the one open `HANDOVER.md` decision).

## Sources

- NVIDIA — DGX Spark hardware overview (128 GB LPDDR5x, 273 GB/s, GB10): <https://docs.nvidia.com/dgx/dgx-spark/hardware.html>
- LMSYS — DGX Spark in-depth review (bandwidth-bound inference): <https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/>
- Modular — GTC 2026: MAX on Blackwell, Mojo kernel porting, DGX Spark support: <https://www.modular.com/blog/modular-at-nvidia-gtc-2026-max-on-blackwell-mojo-kernel-porting-and-deepseek-v3-on-b200>
