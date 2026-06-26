# 04 — Production Inference Serving

**Goal:** give `sophia-agi` *real* production-inference-serving experience that
matches frontier-lab bars (Anthropic Cloud Inference / Inference Deployment /
Inference Runtime / Caching; OpenAI Workload). The repo already owns a genuine,
well-tested **Rust paged prefix-sharing KV-cache**. It does **not** yet own a
serving *engine* that does real token generation through that cache with
continuous batching. This plan builds that engine — Rust scheduler + Python model
runner — and measures it honestly against a naive baseline and against vLLM.

**Ethos to preserve (non-negotiable):** every public number is *measured*, on
stated hardware, reproducible, with a baseline and CI-gated offline invariants.
GPU paths are gated and clearly labelled, never silently asserted. We never claim
to beat vLLM in absolute throughput on a tuned A100; we claim a *correct,
measured* engine that recovers the known algorithmic wins (continuous batching,
prefix reuse, speculation) and we report the gap to vLLM as a first-class number.

---

## 1. Thesis & references

Modern LLM serving throughput is governed not by raw FLOPs but by **how the KV
cache and the batch are managed**. Decode is memory-bandwidth-bound and runs at
batch-of-1 arithmetic intensity unless many sequences are batched together; the
KV cache is the scarce resource that caps how many can be. The frontier stack is a
small set of orthogonal ideas, each of which this repo can implement and measure:

- **Continuous / in-flight batching (Orca).** Yu et al., *"Orca: A Distributed
  Serving System for Transformer-Based Generative Models"* (OSDI 2022). Schedule
  at **iteration granularity**, not request granularity: finished sequences leave
  the batch and waiting ones join *every step*, instead of the batch blocking on
  its slowest member. This is the single largest throughput lever (often 2–4×
  over static batching) and is **Milestone 1** here.
- **PagedAttention (vLLM).** Kwon et al., *"Efficient Memory Management for Large
  Language Model Serving with PagedAttention"* (SOSP 2023). KV cache is paged into
  fixed-size **blocks** with a block table per sequence, eliminating
  reservation-based fragmentation (vLLM reports near-zero waste vs. 60–80% in
  static allocators) and enabling copy-on-write **prefix sharing**. *The repo's
  Rust crate already implements the block/prefix half of this* — Milestone 2 wires
  its block-table semantics into the live engine.
- **Prefix / prompt caching (already partly built — connect it).** RadixAttention
  (Zheng et al., *"SGLang: Efficient Execution of Structured Language Model
  Programs"*, NeurIPS 2024) and APC (vLLM automatic prefix caching); productized as
  Anthropic prompt caching and DeepSeek context caching. `sophia-kvcache` already
  does content-addressed block ids + longest-prefix reuse + tiered HBM→DRAM→NVMe
  residency. **The gap is purely that no engine calls it.**
- **Speculative decoding.** Leviathan et al., *"Fast Inference from Transformers
  via Speculative Decoding"* (ICML 2023) and Chen et al. (DeepMind, 2023): a small
  **draft** model proposes k tokens, the target verifies them in one forward pass,
  accepted tokens are free. Self-drafting variants: **Medusa** (Cai et al. 2024,
  extra decoding heads) and **EAGLE / EAGLE-2** (Li et al. 2024, feature-level
  autoregression). 1.5–3× decode speedup at unchanged output distribution
  (lossless) — **Milestone 3**.
- **Chunked prefill.** Agrawal et al., *"Sarathi-Serve: Taming Throughput-Latency
  Tradeoff"* (OSDI 2024). Split long prefills into token-budgeted chunks and
  **piggyback** decode tokens onto them, so a big prompt's prefill stops stalling
  everyone else's decode (kills TPOT spikes). Stretch in Milestone 1/2.
- **Disaggregated prefill/decode.** Zhong et al., *"DistServe"* (OSDI 2024) and
  Patel et al., *"Splitwise"* (ISCA 2024): run compute-bound **prefill** and
  memory-bound **decode** on separate workers so each is tuned independently and
  TTFT/TPOT SLOs are met without interference. The repo's `serving/load_balancer`
  (cache-aware prefix-affinity router) is the natural control-plane seam.
- **KV-cache quantization.** KVQuant (Hooper et al. 2024), KIVI (Liu et al. 2024):
  store K/V at int8/int4 to roughly double the sequences that fit in HBM. Fits the
  crate's per-block payload model directly (quantize the `Vec<u8>`).
- **Tensor-parallel serving.** Megatron-LM (Shoeybi et al. 2019) sharding for
  models too big for one GPU; relevant only at the >13B tier here.
- **SLOs & goodput.** TTFT (time-to-first-token), TPOT (time-per-output-token,
  a.k.a. ITL), end-to-end latency, throughput (tokens/s), and **goodput** —
  throughput counting *only* requests that met their SLO (DistServe's central
  metric). The honest target is **goodput under load**, not peak tokens/s.

**One-line thesis:** *Serving throughput is a KV-cache and batch-scheduling
problem. This repo already has the hard, correct KV-cache half; the work is a
continuous-batching engine on top of it, with speculation, measured end-to-end
against vLLM on rented GPUs under an honest goodput SLO.*

---

## 2. Current repo state (file-level)

### The crown asset — `sophia-storage/crates/sophia-kvcache/` (Rust, tested, real)
A production-shaped paged, prefix-sharing, tiered KV-cache. `#![forbid(unsafe_code)]`,
zero deps, edition 2024. This is genuinely the PagedAttention *memory-management*
half built correctly and unit-tested.
- `src/block.rs` — content-addressed `BlockId` (FNV-1a over *(parent-hash, tokens)*),
  so equal prefixes resolve to the same block id (sharing); position-dependent. Fixed
  `block_len` tokens/block = PagedAttention pages. Opaque `Vec<u8>` payload stands in
  for the device tensor.
- `src/prefix.rs` — `block_chain()` folds ids along the sequence; `shared_prefix_len()`
  returns the contiguous reusable prefix. Returns `Result` on misconfig, doesn't panic.
- `src/eviction.rs` — `LruRefCounted`: pin/unpin refcounting so a **shared prefix can
  never be evicted out from under a live request** (the load-bearing safety invariant),
  LRU among unpinned candidates.
- `src/tier.rs` + `src/store.rs` — HBM→DRAM→NVMe residency over a `BlockStore` trait;
  **NVMe is real disk** (`FileStore`, one file/block, atomic temp+rename, survives
  restart). Per-tier `bytes_in/bytes_out` accounting — the number an RDMA/zero-copy path
  exists to shrink.
- `lib.rs` — `KvCache::admit()` (hit/compute/evict accounting), `pin_prefix`/`unpin_prefix`,
  promote/demote cascade. Tests prove: 2nd identical request = full prefix hit; best-of-N
  shares prefix; pinned prefix survives pressure; NVMe block persists + pages back in.
- **Honest seams (documented):** HBM/DRAM are in-memory stand-ins; `compute(id)` is a
  *callback*, not a real prefill; no GPU, no attention kernel, no tokens generated.

### Secondary serving primitives (Python, policy-only, CI-tested)
- `serving/kv_cache.py` — `TieredKVCache`: the same paging/tiering policy in pure Python
  with `offline_invariants()` (CI-gated): fork-point correctness, budget never exceeded,
  demote-not-drop, cross-tier promotion, byte accounting closes. A *policy twin* of the
  Rust crate.
- `serving/load_balancer.py` — `CacheAwareRouter`: longest-prefix **affinity** routing +
  load-cap hotspot guard + consistent-hash cold fallback. Invariant: cache-aware beats
  round-robin on cluster prefix-hit rate. This is the disaggregation/router seam.

### Network KV store — `storage/kvcache/` (Rust, Tokio, real wire protocol)
- `src/server.rs` / `client.rs` / `protocol.rs` — sharded async in-memory cache over a
  length-prefixed binary TCP protocol; pipelining; per-shard mutex (no lock across await).
- `src/bin/bench.rs` — **honest closed-loop load generator**: real client→TCP→server path
  on loopback, reports throughput + p50/p99/p999, seeded LCG (reproducible). This is the
  template for the serving benchmark harness.
- *Not* a KV-*attention* cache — it's a Redis-shaped string cache. Reuse the
  protocol/bench scaffolding, not the data model.

### Supporting pieces
- `kernels/flash_attention.py` — FlashAttention forward (Dao et al. 2022) in numpy
  (CI ground truth) + gated Triton kernel (deployment artifact). The attention primitive
  a real model-runner needs.
- `services/ann_serving/` — Rust HNSW/NSW ANN core with a Python subprocess bridge +
  recall/latency sweep bench. Template for the **Python↔Rust serving bridge**.
- `models/` — Ollama Modelfile + HF model card for `sophia-v1` (a small fine-tune). The
  thing to actually serve.

### The gap, stated plainly
There is **no engine that generates tokens through the KV-cache under continuous
batching**. The cache is correct but inert (payloads are opaque bytes, `compute` is a
no-op callback). Everything below builds the live serving loop and connects the
existing cache to it.

---

## 3. Top-tier target end-state

A **`sophia-serve`** inference engine, Rust scheduler + Python model runner:

1. **Continuous-batching scheduler (Rust)** — iteration-level scheduling, admission
   control against a real KV block budget, waiting/running/preempted queues, FCFS +
   optional priority, chunked prefill, preemption-by-recompute and by-swap (to the
   crate's DRAM/NVMe tiers).
2. **Paged KV manager (Rust)** — `sophia-kvcache` promoted from a callback toy to the
   engine's block allocator: block tables per sequence, copy-on-write fork, automatic
   prefix caching across requests, ref-counted eviction already done.
3. **Model runner (Python)** — loads `sophia-v1` / Llama-class HF weights, runs paged
   attention (start with HF + a paged-KV adapter; graduate to the repo's FlashAttention
   Triton kernel), exposes a `step(batch) -> logits` FFI the Rust scheduler drives. PyO3
   in-process (like the planned ann_serving cdylib) or a thin shared-memory bridge.
4. **Speculative decoding** — draft-model and self-draft (Medusa-style heads) paths,
   lossless verification, measured acceptance rate + net speedup.
5. **OpenAI-compatible HTTP front end** — `/v1/completions` + `/v1/chat/completions`,
   streaming SSE, so any client (and vLLM's own benchmark harness) drives it unchanged.
6. **Cache-aware multi-worker routing** — `serving/load_balancer.py` promoted to a real
   control plane over N `sophia-serve` workers; optional prefill/decode disaggregation.
7. **Honest benchmark suite** — tokens/s, TTFT, TPOT, goodput-under-load vs (a) naive
   static-batching baseline and (b) vLLM, on rented RunPod GPUs, reproducible, committed
   to `RESULTS.md` with hardware + commit + seed.

End-state bar: *"we built a continuous-batching paged-attention engine that serves a
real model, recovers continuous-batching + prefix-cache + speculation wins as measured
deltas over a naive baseline, and lands within a stated, honest factor of vLLM."*

---

## 4. Phased plan

New top-level crate/package: **`serving-engine/`** (workspace member) with Rust
scheduler + Python runner. Keep `serving/` (policy twins) and `sophia-kvcache` as-is;
depend on the crate.

### Milestone 0 — Harness, baseline, and the honest gap (no GPU)
*Goal: a reproducible benchmark and a naive baseline before any optimization, so every
later number has a denominator.*
- `serving-engine/bench/` (Python) — closed-loop + open-loop (Poisson arrivals) load
  generator modeled on `storage/kvcache/src/bin/bench.rs`. Metrics: tokens/s, TTFT, TPOT,
  p50/p99, goodput @ SLO. Supports ShareGPT-style and synthetic prefix-skewed traces.
- `serving-engine/baseline_static.py` — naive **static batching** server (one batch,
  blocks on slowest seq, full KV reservation). The denominator.
- Wire both `offline_invariants()` suites + new scheduler invariants into CI.
- **Files:** `serving-engine/bench/{harness,traces,metrics}.py`, `baseline_static.py`.
- **Lang:** Python. **Exit:** baseline tokens/s + TTFT/TPOT printed, reproducible, in CI.

### Milestone 1 — Continuous-batching engine (the core; Orca)
*Goal: iteration-level scheduling beats static batching on a measured trace.*
- `serving-engine/crates/sophia-sched/` (Rust) — `Scheduler` with waiting/running queues,
  iteration step loop, admission control vs. a KV block budget, FCFS, **chunked prefill**
  (token-budgeted), preemption (recompute first; swap later). Deterministic, CI-tested
  invariants: no seq starves; block budget never exceeded; finished seqs leave same step.
- `serving-engine/runner.py` (Python) — `step(batch)->logits` over HF model with a
  **paged-KV adapter** (custom KV that indexes the crate's block tables). Greedy + sampling.
- PyO3 bridge `sophia-sched` ↔ `runner.py` (in-process), or shared-memory ring if PyO3
  friction is high (ann_serving subprocess bridge is the fallback template).
- **Files:** `crates/sophia-sched/src/{scheduler,batch,admission,preempt}.rs`,
  `runner.py`, `engine.py` (glue), `crates/sophia-engine-ffi/` (PyO3).
- **Lang:** Rust (scheduler) + Python (runner). **Exit:** continuous batching shows a
  **measured throughput uplift over Milestone-0 static baseline** on the same trace + GPU,
  same outputs.

### Milestone 2 — PagedAttention block management + prefix cache *connected*
*Goal: the existing Rust KV-cache becomes the live block allocator; automatic prefix
caching turns on and is measured.*
- Promote `sophia-kvcache` from callback to engine allocator: per-sequence **block
  tables**, copy-on-write fork on divergence, **automatic prefix caching** across
  requests via existing content-addressed ids + `shared_prefix_len`. Pin during decode
  (already implemented). Preemption-by-swap targets the crate's DRAM/NVMe tiers (already
  real).
- Add **paged attention** in the runner: gather K/V by block table. Start HF-side; then
  swap in `kernels/flash_attention` Triton kernel with a paged variant.
- Optional: **KV quantization** (int8 block payloads) — measure HBM-sequences-fit gain.
- **Files:** `crates/sophia-kvcache/src/block_table.rs` (new), `engine.py` paged-attn path,
  `runner_paged.py`.
- **Lang:** Rust + Python. **Exit:** prefix-cache hit rate > 0 on a prefix-skewed trace;
  **measured TTFT drop on cache hits**; KV fragmentation ≈ 0 (paged) vs. reserved baseline.

### Milestone 3 — Speculative decoding
*Goal: lossless decode speedup, acceptance rate measured.*
- `serving-engine/spec/` — draft-model path (small `sophia`/TinyLlama draft + target
  verify) and a self-draft **Medusa-style** head path. Lossless verification (accepted
  tokens identical to target's greedy/sampled distribution). Integrate into the scheduler
  step (verify is one target forward over k draft tokens).
- **Files:** `crates/sophia-sched/src/speculative.rs`, `spec/{draft,medusa,verify}.py`.
- **Lang:** Rust (accept/rollback in scheduler) + Python (draft+verify forwards).
- **Exit:** **measured net decode speedup** (TPOT) at unchanged output distribution;
  report acceptance rate and the speedup-vs-acceptance curve.

### Milestone 4 — Benchmark vs vLLM + disaggregation + multi-worker
*Goal: the headline honest comparison.*
- Run vLLM and `sophia-serve` on the **same GPU, model, trace** through the same
  OpenAI-compatible benchmark client. Report tokens/s, TTFT, TPOT, **goodput @ SLO** for
  both; publish the gap as a first-class number.
- Promote `serving/load_balancer.py` to a real router over N workers; prototype
  prefill/decode **disaggregation** (DistServe/Splitwise) and measure SLO attainment.
- **Files:** `bench/vllm_compare.py`, `frontend/openai_server.py`, `router/control_plane.py`.
- **Exit:** `RESULTS.md` table: sophia-serve vs naive vs vLLM, hardware/commit/seed stated,
  reproducible.

### Dependency order
M0 → M1 (needs M0 baseline) → M2 (needs M1 loop) → M3 (needs M2 paged cache) → M4 (needs all).
M2's KV-quant and M4's disaggregation are independently droppable stretch items.

---

## 5. Compute / budget tiers (RunPod, live availability checked 2026-06-26)

All numbers are *rented-GPU-hours*; the engine + scheduler + invariants develop CPU-only.

| Tier | GPU (RunPod, in stock now) | What it buys | Rough hrs | Use |
|---|---|---|---|---|
| **T0 — CPU only** | none | scheduler, KV-cache, invariants, baseline harness, traces | 0 GPU-hr | M0, all Rust logic, CI |
| **T1 — single small** | RTX 4090 24GB / L4 24GB / RTX 5090 32GB (High stock) | serve `sophia-v1` / 7B; M1 continuous-batching uplift; M2 prefix cache; M3 spec | 20–40 GPU-hr | core proof |
| **T2 — single big** | A100 80GB SXM (Medium) / H100 80GB (High) | 13B at real batch; the vLLM head-to-head on a "serious" card | 15–30 GPU-hr | M4 headline |
| **T3 — stretch** | 2× H100 / H200 141GB (High) / B200 | tensor-parallel + prefill/decode disaggregation | 10–20 GPU-hr | M4 stretch |

**Discipline:** spin up per benchmark run, capture results, **tear down** (RunPod MCP
create-pod/stop-pod). Every GPU number carries its exact GPU id + commit + seed. Total
honest budget for a credible M0–M4: **~60–90 GPU-hours**, dominated by H100/A100 for the
vLLM comparison; the algorithmic wins are demonstrable on a single 4090.

---

## 6. Honest metrics (reproducible)

Report, always with **hardware id + git commit + seed + trace**:
- **Throughput** — output tokens/s and req/s at saturation.
- **TTFT** — time to first token, p50/p99 (prefill + queue).
- **TPOT / ITL** — time per output token, p50/p99 (the decode-bound number).
- **Goodput @ SLO** — req/s meeting a stated (TTFT ≤ X, TPOT ≤ Y) SLO under Poisson load
  (DistServe's metric; the one that matters under load).
- **Prefix-cache hit rate** + TTFT-on-hit vs TTFT-on-miss.
- **Speculative acceptance rate** + net TPOT speedup + output-distribution-identity check.
- **KV fragmentation / sequences-per-GPU** — paged vs reserved; vs KV-quant.

**Three mandatory comparisons per claim:**
1. vs **Milestone-0 naive static baseline** (same code path, batching off) — isolates each
   optimization's delta.
2. vs **vLLM** (same GPU/model/trace, OpenAI-compatible client) — the frontier reference;
   report the gap honestly, do not hide it.
3. vs **itself ablated** (prefix cache off, spec off) — attributes each win.

CI gate: scheduler + KV invariants run CPU-only on every commit (deterministic). GPU
benchmarks run on-demand, results committed to `RESULTS.md` with a repro command, exactly
as the repo already does for `ann_serving` and `kvcache-bench`.

---

## 7. Risks & overclaim guards

- **"We beat vLLM."** Almost certainly false in absolute throughput on a tuned card; vLLM
  is years of CUDA tuning. **Guard:** never headline a beat; headline *correct engine +
  measured wins over our own baseline + honest gap to vLLM*. State the gap as a number.
- **Toy attention masquerading as a kernel.** The numpy/HF path is correct but slow.
  **Guard:** label runner backends explicitly (HF-eager / FlashAttention-Triton / paged);
  report which produced each number; never quote an HF-eager number as "engine throughput".
- **Lossy speculation.** Easy to "speed up" by silently changing the distribution.
  **Guard:** ship a distribution-identity test (accepted tokens == target's tokens for the
  same RNG); spec is lossless or it doesn't ship.
- **Cherry-picked traces.** Prefix caching looks amazing on a prefix-heavy trace.
  **Guard:** report on both prefix-skewed *and* prefix-flat traces; state the trace.
- **GPU-result irreproducibility.** **Guard:** pin GPU id, driver, commit, seed, trace in
  every `RESULTS.md` row; provide the exact RunPod create + run command.
- **Scope creep into a vLLM clone.** **Guard:** the deliverable is *measured experience*
  with the primitives, not a production engine; tensor-parallel + disaggregation are
  explicitly stretch.
- **Crate regression.** `sophia-kvcache` is already good and tested. **Guard:** the engine
  *depends on* it; block-table additions are new modules with their own invariants, not
  rewrites of the tested core.

---

## 8. Effort

| Milestone | Scope | Effort (focused) |
|---|---|---|
| M0 Harness + naive baseline | Python bench + static server + CI | 0.5–1 wk |
| M1 Continuous batching | Rust scheduler + PyO3 + runner | 2–3 wk |
| M2 Paged KV connected + prefix cache | block tables, paged attn, wire crate | 1.5–2.5 wk |
| M3 Speculative decoding | draft + Medusa + lossless verify | 1.5–2.5 wk |
| M4 vLLM compare + disaggregation | OpenAI front end, router, vLLM bench | 1–2 wk |
| **Total** | M0–M4 | **~7–11 wk**, ~60–90 GPU-hr |

**Highest-signal single deliverable:** Milestone 1 + 2 together — a continuous-batching
engine that serves a real model *through the existing Rust paged prefix-cache*, with a
measured throughput + TTFT uplift over a naive static baseline on a single rented 4090.
That one result converts the repo's correct-but-inert KV-cache into demonstrated
end-to-end serving experience, and everything else (speculation, vLLM head-to-head,
disaggregation) is an extension of it.
