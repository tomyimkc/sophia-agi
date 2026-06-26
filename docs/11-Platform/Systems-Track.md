# Systems Track — train/inference-framework engineering

> **Why this exists.** Sophia's charter is the *trust layer* — provenance,
> verification, calibration, fail-closed reasoning ([VISION.md](../../VISION.md)).
> This track is deliberately a different layer: the **极致工程系统 / extreme
> engineering systems** that sit *below* the model — distributed inference
> serving, RL training systems, attention kernels, MoE, and low-precision. It
> exists so the repo demonstrates the systems-engineering primitives a
> large-model **training/inference framework** role cares about, built to the
> same measurement discipline as everything else here: a deterministic,
> dependency-light reference proven in CI, with the expensive GPU path **gated
> and clearly labelled — never silently asserted**.
>
> Honest framing: these are *reference* implementations. They model the
> **policy/algorithm** exactly (what to cache, where to evict, how to route, how
> the online-softmax recurrence works, how quantization error is bounded), and
> they are unit-tested for correctness on any machine. They do **not** move real
> GPU tensors at scale — that is the deployment artifact each module points to.
> The value is that the hard part (the algorithm + its invariants) is real,
> reproducible, and checkable.

## Mapping to the role description

| JD line (大模型训练/推理框架工程师) | Module | CI-tested claim |
|---|---|---|
| 大规模推理服务的 **KV Cache 磁盘缓存、负载均衡** | [`serving/`](../../serving/) | tiered GPU→CPU→disk paged cache with prefix sharing; cache-aware router beats round-robin on cluster prefix-hit rate |
| RL 训练系统：**异步 RL、Agent RL** | [`provenance_bench/async_rl.py`](../../provenance_bench/async_rl.py), [`tools/run_async_rl.py`](../../tools/run_async_rl.py) | decoupled rollout/trainer loop with GRPO group-advantages + bounded off-policy staleness; async > sync throughput, composes over the repo's real verifier-as-reward seam |
| **CUDA / Triton 算子开发**；长上下文；复现论文 | [`kernels/`](../../kernels/) | FlashAttention (arXiv:2205.14135) online-softmax forward, numpy reference == naive attention while holding one score tile; fused Triton kernel gated |
| **MoE**；**低精度训推** | [`moe/`](../../moe/) | top-k routing + Switch-Transformer load-balancing loss; symmetric INT8 (per-tensor/per-channel) + FP8-E4M3 with proven error bounds |

It also exercises the JD's 加分项: a paper reproduction (FlashAttention), an
operator written for Triton, and RL/MoE/quant system internals.

## What's in each module

### `serving/` — KV-cache serving layer
- **`kv_cache.TieredKVCache`** — block-paged KV cache. A sequence's KV is chunked
  into fixed blocks keyed by a rolling **prefix hash**, so requests sharing a
  prompt prefix share physical blocks (skipping that prefill). Three tiers
  (GPU→CPU→disk) with per-tier byte budgets; overflow **demotes LRU blocks one
  tier down** (cascading GPU→CPU→DISK) instead of dropping them — the "KV Cache
  磁盘缓存". Lookups **promote** blocks back toward GPU. Disk tier is a real
  on-disk store.
- **`load_balancer.CacheAwareRouter`** — routes each request to the worker
  holding the **longest prefix** of its tokens, overridden by a load cap when
  that worker is a hotspot, with consistent-hash placement for cold prefixes.
  Measured: cache-aware routing lifts cluster prefix-hit rate **0.80 → 0.95** vs
  round-robin on a prefix-skewed workload.
- Run: `python -m serving.kv_cache` · `python -m serving.load_balancer`

### `provenance_bench/async_rl.py` — async / off-policy RL
- **`grpo_advantages`** — exact GRPO group-relative advantage `(r-mean)/std`
  (zero-mean, scale-normalized, degenerate-safe).
- **`ReplayBuffer`** — bounded FIFO tagged with the generating policy version;
  enforces a **max-staleness** bound (over-stale trajectories dropped, never
  trained) and a capacity bound.
- **`simulate`** — deterministic discrete-event sim of the **decoupled** loop vs.
  the synchronous **barrier** loop. Measured over 300 ticks: async does **198**
  train steps vs sync's **75** (1.33× throughput) with staleness bounded ≤ 2;
  the sync barrier discards its unconsumed on-policy rollouts.
- Composes over the repo's real reward seam (`provenance_bench.rl_reward`).
- Run: `python tools/run_async_rl.py --compare --reward provenance`

### `kernels/` — FlashAttention reproduction
- **`flash_attention_numpy`** — tiled online-softmax attention. Output is
  `allclose` to naive O(N²) attention (the paper's correctness guarantee) while
  only ever holding **one `Bq×Bk` score tile**. At N=1024 that's 4096 score
  elements vs naive's 1,048,576 — a **256× score-memory reduction**, exact match.
  Causal masking with whole-tile skipping above the diagonal.
- **`flash_attention_triton`** — the same recurrence as a fused Triton program
  (the GPU deployment artifact), gated on Triton + CUDA and skipped in CI.
- Run: `python -m kernels.flash_attention` · `python kernels/bench.py`

### `moe/` — Mixture-of-Experts + low precision
- **`router.MoERouter`** — top-k softmax gating, capacity-bounded dispatch/combine
  (overflow → residual), and the **Switch-Transformer load-balancing aux loss**
  `E·Σ f_e·P_e` (==1.0 balanced, →E under full collapse). Identity experts
  reconstruct the input exactly (dispatch/combine correctness).
- **`quant`** — symmetric INT8 (per-tensor + per-channel) with a proven
  `scale/2` round-trip error bound (per-channel cuts mean error ~14× on
  column-skewed weights), a weight-only INT8 linear (<2% relative error vs fp),
  and an **FP8-E4M3** emulation within the 3-mantissa-bit (6.25%) relative bound.
- Run: `python -m moe.router` · `python -m moe.quant`

## Test + invariant discipline

Every module exposes `offline_invariants() -> (ok, detail)` — a deterministic,
dependency-light proof of its core claims — mirrored by a `tests/test_*.py`:

| Module | Tests |
|---|---|
| `serving/` | [`tests/test_kv_serving.py`](../../tests/test_kv_serving.py) |
| `async_rl` | [`tests/test_async_rl.py`](../../tests/test_async_rl.py) |
| `kernels/` | [`tests/test_flash_attention.py`](../../tests/test_flash_attention.py) |
| `moe/` | [`tests/test_moe.py`](../../tests/test_moe.py) |

```bash
python -m pytest tests/test_kv_serving.py tests/test_async_rl.py \
                 tests/test_flash_attention.py tests/test_moe.py -q
```

The serving and async-RL references are pure-Python (run anywhere, incl. Apple
Silicon); the kernels and MoE references need only numpy (already a CI
dependency). torch / Triton / vLLM / FP8 hardware paths are **gated** — they
refuse rather than silently degrade, exactly like the existing RLVR run
([RLVR-Experiment](../09-Agent/RLVR-Experiment.md)).

## What would make each production-grade (the gated next step)

- **serving**: back the block payload with real device tensors; wire the router
  in front of a vLLM/SGLang fleet via their KV-transfer API; add radix-tree
  prefix matching and a prefill/decode disaggregation split.
- **async_rl**: replace the synthetic policy proxy with a real generator
  (vLLM) + trainer (TRL GRPO) across processes; add importance-sampling
  correction for the off-policy gap; make a rollout a full tool-using **Agent
  RL** episode.
- **kernels**: benchmark the Triton kernel on a GPU vs PyTorch SDPA; add the
  backward pass; extend to paged/sliding-window attention.
- **moe**: fuse the grouped expert GEMM; add an all-to-all dispatch model and
  expert-parallel sharding; calibrate INT8/FP8 against a real model's perplexity.
