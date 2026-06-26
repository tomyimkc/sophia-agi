# Plan 01 — Real Large-Scale Distributed Training for `sophia-agi`

> Goal: give the repo **genuine, measured, reproducible distributed-training experience**
> that reads as "top tier" to a frontier-lab reviewer (Anthropic Pretraining / Science-of-Scaling,
> OpenAI Workload, DeepMind RE) — **without abandoning the repo's honest, fail-closed,
> pre-registered, no-overclaim ethos.** The deliverable is *demonstrated systems competence
> at small-but-real scale with rigorous measurement*, not a frontier model.

Author posture: this is a staff-ML-systems implementation plan. Everything below is designed so
that every headline number is **produced by a run**, checked against a **theoretical reference**
(hardware-peak FLOPs, analytic comms volume, planted scaling law), and **pre-registered** before
it is trusted — exactly the discipline already in `pretraining/PRE-REGISTRATION.md` and `RESULTS.md`.

---

## 1. Thesis — research-backed approach & key references

### 1.1 The core claim a frontier reviewer wants to see

A pretraining/workload engineer is hired to **make a fixed pile of GPUs convert FLOPs into loss
reduction as efficiently and reliably as possible, and to *know* (measure, predict, debug) that
they did.** Concretely the hiring bar is competence across four axes:

1. **Parallelism** — shard a model that does not fit on one device across data / tensor /
   pipeline / sequence(context) dimensions, and reason about the comms cost of each.
2. **Efficiency** — drive and *measure* **MFU/HFU** (Model/Hardware FLOPs Utilization), use
   mixed precision (bf16, fp8), activation checkpointing, and overlap compute with communication.
3. **Scale-out reliability** — multi-node NCCL, sharded checkpoint save/restore, deterministic
   resume, failure handling.
4. **Science of scaling** — fit scaling laws (Chinchilla), transfer hyperparameters across scale
   (µP / µTransfer), and *predict* large-run behavior from small runs, with error bars.

The repo already nails axis-4's *methodology* at toy scale (`pretraining/scaling`,
`optimizer_probe`, `autopilot`) and has the *honesty machinery*. It is essentially absent on
axes 1–3 in any **real** (GPU, torch.distributed) sense. This plan closes that gap.

### 1.2 What "top-tier distributed training" actually requires — the techniques & references

**Sharding / memory strategies**
- **ZeRO** (Rajbhandari et al., 2020, *ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models*) — Stage 1 (optimizer-state), Stage 2 (+gradients), Stage 3 (+parameters)
  partitioning. Implemented by **DeepSpeed**. The conceptual parent of FSDP.
- **FSDP / FSDP2** (Zhao et al., 2023, *PyTorch FSDP*; FSDP2 = per-parameter `DTensor` sharding).
  FSDP ≈ ZeRO-3 in PyTorch-native form. FSDP2 composes cleanly with TP/PP/CP and gives ~7% lower
  per-GPU memory and ~1.5% throughput over FSDP1. **This is the primary backbone for this plan.**
- **Megatron-LM / Megatron-Core** (Shoeybi et al., 2019; Narayanan et al., 2021, *Efficient
  Large-Scale LM Training on GPU Clusters*) — the canonical **tensor-parallel** (column/row-
  parallel linear) and **pipeline-parallel** (1F1B, interleaved 1F1B) reference.

**Parallelism dimensions** (the "4D/5D parallelism" vocabulary)
- **Data parallel (DP/FSDP/HSDP)** — replicate/shard across the batch.
- **Tensor parallel (TP)** — split individual matmuls across devices (Megatron column/row
  parallel). Comms-heavy (all-reduce per layer); keep intra-node over NVLink.
- **Pipeline parallel (PP)** — split layers into stages; schedule micro-batches (GPipe;
  1F1B / interleaved-1F1B) to shrink the pipeline bubble.
- **Sequence / Context parallel (SP/CP)** — shard the sequence dimension for long context;
  **Ring Attention** (Liu et al., 2023) for near-infinite context; Megatron sequence-parallel
  for the LayerNorm/dropout regions TP leaves replicated.
- Composition: **TorchTitan** (Liang et al., 2024, *TorchTitan: One-stop PyTorch-native solution
  for production-ready LLM pretraining*, arXiv:2410.06511) is the modern reference that composes
  FSDP2 + TP + PP + CP + Float8 with `torch.compile`, and reports MFU. **It is the de-facto
  blueprint and a dependency-or-template for this plan.**

**Precision**
- **Mixed precision** bf16 (Micikevicius et al., 2017, *Mixed Precision Training*) — bf16 compute
  with fp32 master weights / optimizer state.
- **fp8** (NVIDIA Transformer Engine; PyTorch `torchao` Float8 rowwise) — E4M3/E5M2, applied
  selectively to linear layers on Hopper+; ~2× activation/grad memory vs bf16, throughput gains,
  with documented convergence parity on Llama-3.1 (PyTorch Float8-rowwise on 2K H200 blog, 2024).

**Memory / throughput tooling**
- **Activation (gradient) checkpointing** (Chen et al., 2016, *Training Deep Nets with Sublinear
  Memory Cost*) — recompute activations in backward to trade compute for memory; selective AC.
- **FlashAttention** (Dao et al., 2022/2023, FA-2; FA-3 for Hopper) — IO-aware exact attention.
  The repo already has a paper-repro of FA forward in `kernels/flash_attention.py`.
- **Comms/compute overlap** — FSDP prefetch, TP+SP overlap, `torch.compile` fusion.

**Multi-node systems**
- **NCCL** collectives (all-reduce / reduce-scatter / all-gather) over NVLink (intra-node) and
  IB/RoCE (inter-node); `torchrun` / rendezvous for elastic multi-node launch. The repo already
  has a **real** `torch.distributed` all-reduce bandwidth bench (`tools/runpod_nccl_bench.py`,
  `tools/bench_nccl_allreduce.py`) — a strong, honest foundation to build on.

**Measurement: MFU / HFU**
- **MFU** introduced in **PaLM** (Chowdhery et al., 2022) and used throughout
  (Megatron, MosaicML LLM-Foundry, TorchTitan). MFU = (model-FLOPs/token × tokens/s) /
  (num_GPUs × peak-FLOPs). The standard per-token transformer FLOP estimate is
  **C ≈ 6·N·D** forward+backward (Kaplan et al., 2020), with the **6·N + 12·L·s·h**
  attention-aware refinement (PaLM appendix). HFU additionally counts recompute from activation
  checkpointing. *MFU is the single most legible competence signal to a workload reviewer.*

**Science of scaling**
- **Kaplan et al., 2020** (*Scaling Laws for Neural LMs*) — power-law loss vs N, D, C.
- **Hoffmann et al., 2022 — Chinchilla** (*Training Compute-Optimal LLMs*) — compute-optimal
  N∝C^0.5, D∝C^0.5, ≈20 tokens/param; the repo's `autopilot --task compute` already gestures at
  this iso-compute frontier.
- **µP / µTransfer** (Yang et al., 2021/2022, *Tensor Programs V: Tuning Large NNs via Zero-Shot
  HP Transfer*) — Maximal Update Parametrization makes optimal LR/init **width-(and depth-)
  invariant**, so HPs tuned on a small proxy transfer to the large model. This is the
  highest-leverage *scaling-science* technique to demonstrate, and it pairs perfectly with the
  repo's existing tiny-proxy ethos.

### 1.3 The approach in one sentence

**Build a small, real, FSDP2(+TP+PP) PyTorch transformer trainer (TorchTitan-aligned), instrument
it with rigorously-checked MFU/HFU and sharded-checkpoint resume, run it on free CPU for
correctness and on RunPod single-node-multi-GPU (and one multi-node burst) for measured numbers,
and tie it to two pre-registered scaling-science results — a measured MFU-vs-parallelism curve and
a µTransfer HP-transfer demonstration — every number reproducible and bounded by a theoretical
reference.**

---

## 2. Current repo state (file-level, honest)

### 2.1 What genuinely exists

**Scaling-science methodology (toy, pure-Python, excellent discipline)**
- `pretraining/nano/{model,data,train}.py` (~391 LOC, pure Python): hand-backpropped 1-hidden-layer
  softmax LM on an order-k **Markov source with closed-form entropy** (the known floor E).
  SGD/momentum/Adam with grad-norm/spike tracking.
- `pretraining/scaling/{fit,run_scaling}.py` (~249 LOC): fits `L(D)=E+A·D^-p`, pre-registered
  extrapolation gate (G3, ~3% err), honest floor-identifiability failure (G4).
- `pretraining/optimizer_probe/run_probe.py` (~118 LOC): optimizer × LR stability frontier.
- `pretraining/data_mixing/run_mixing.py` (~107 LOC): mixture-ratio interior optimum.
- `pretraining/architecture/{moe,run_arch}.py` (~267 LOC): top-1 MoE vs dense, load-balance
  collapse surfaced. `ARCHITECTURE.md` documents the real MLA + fine-grained-MoE design.
- `pretraining/autopilot/` (~462 LOC): **real closed loop** (propose→run→read→iterate) over nano
  experiments; tasks lr / mixture / **compute (iso-compute, Chinchilla-flavored)**; gated RunPod
  escalation (dry-run by default, cost-ceiling, never auto-spends).
- `pretraining/PRE-REGISTRATION.md`: 11 falsification gates (G1–G11), with G4 honestly listed as
  an HONEST-FAIL. `pretraining/README.md`: explicitly labels everything "toy, and labelled toy."

**Reference numerics (numpy, CI-proven, not torch)**
- `moe/router.py` (~240 LOC): top-k gating, capacity drops, Switch-Transformer aux loss + 5
  offline invariants.
- `moe/quant.py` (~183 LOC): INT8 + FP8-E4M3 emulation, weight-only linear + 6 invariants.
- `kernels/flash_attention.py` (~316 LOC): FlashAttention forward paper-repro (numpy ref +
  optional Triton), `kernels/bench.py`.

**Real GPU/distributed touchpoints (the genuine seeds to build on)**
- `tools/runpod_nccl_bench.py` (~200 LOC) + `tools/bench_nccl_allreduce.py` (~150 LOC):
  **the only real `torch.distributed` code** — torchrun ring all-reduce, NCCL **bus-bandwidth**
  across message sizes. Calibrates `clustersim`.
- `tools/train_lora.py` (~974 LOC): real **single-GPU** SFT (manual loop, dynamic padding,
  completion-only loss, **bf16**, **`gradient_checkpointing_enable()`**, early stop via eval_ladder).
- `tools/train_dpo.py`, `tools/run_rlvr.py` (+ `tools/runpod_rlvr.py`): single-GPU DPO and
  GRPO/RLVR; vLLM optional.
- `tools/runpod_train.py` (~483 LOC): single-pod orchestration; "multi-seed parallel" is just
  `CUDA_VISIBLE_DEVICES` fan-out (**embarrassingly parallel, NOT distributed training**).
- `tools/runpod_kernels.py`: kernel micro-benchmarks on RunPod.

**Cluster *simulation* (trace replay, not training)**
- `clustersim/{simulator,topology,scheduler,netcalib,job}.py` (~1050 LOC): discrete-event
  scheduler/placement sim with a **network tax model calibrated from the real NCCL bench**.
  Honest and useful — but **simulated**, no gradients.

**CI / infra**
- `.github/workflows/spark-gpu.yml`: local DGX-Spark (GB10 aarch64) **iteration-only** lane,
  hard `candidateOnly + spark_iteration`, never a registered-results producer (x86 RunPod A100 is
  source-of-record). Strong provenance boundary to preserve.
- RunPod workflows for SFT/RLVR/kernels; `.mcp.json` wires the RunPod MCP server.
- Ethos enforced repo-wide: `RESULTS.md` no-overclaim gate (≥2 independent judges, ≥3 runs, CIs;
  illustrative vs validated), `canClaimAGI:false`.

### 2.2 Honest verdict on the gap

| Capability | Status |
|---|---|
| Real distributed *training* loop (torch.distributed gradients) | ❌ absent |
| FSDP / FSDP2 | ❌ absent |
| DeepSpeed ZeRO | ❌ absent |
| Tensor / Pipeline / Sequence-Context parallel | ❌ absent |
| Multi-node training (only NCCL *bench* exists) | ❌ absent (bench ✅) |
| MFU / HFU measurement | ❌ absent |
| fp8 training | ❌ (FP8 *emulation* in `moe/quant.py` only) |
| Activation checkpointing | ⚠️ single-GPU LoRA only |
| bf16 mixed precision | ⚠️ single-GPU LoRA only |
| Sharded / resumable checkpoints | ❌ absent |
| Scaling-law *methodology* | ✅ strong (toy) |
| Chinchilla iso-compute framing | ⚠️ toy (`autopilot --task compute`) |
| µP / µTransfer | ❌ absent |
| Real NCCL collective measurement | ✅ genuine |
| Honesty / pre-registration machinery | ✅✅ excellent — **must preserve** |

**Summary:** the repo demonstrates *scaling-science taste and honesty* but has **no real
distributed training**. The NCCL bench + RunPod orchestration + clustersim are an unusually good
launchpad: real comms measurement and a simulator that is *waiting to be validated against a real
training run*.

---

## 3. Target end-state ("top tier" to a frontier-lab reviewer)

A new top-level package **`disttrain/`** (PyTorch-native, TorchTitan-aligned) plus a
`pretraining/distributed/` study folder that ties it to the existing scaling-science discipline.
When done, a reviewer sees:

1. **A real, parallelizable transformer trainer** (`disttrain/`): a clean decoder-only GPT with
   FSDP2 wrapping, optional Megatron-style TP and 1F1B PP, optional context-parallel, bf16 default
   + opt-in fp8, activation checkpointing, `torch.compile`. Runs single-process on CPU (tiny
   config) for correctness; scales to 1–8 GPUs single-node and to 2-node on RunPod.
2. **A rigorously-checked MFU/HFU meter** (`disttrain/metrics/mfu.py`): analytic 6N+attention FLOP
   model, hardware peak table, unit-tested against PaLM/Megatron formulas; every run logs
   MFU/HFU + tokens/s + memory.
3. **A pre-registered parallelism efficiency study** (`pretraining/distributed/PRE-REGISTRATION.md`
   + `run_parallelism_sweep.py`): measured **MFU-vs-(DP, TP, PP, AC, bf16↔fp8)** on identical
   model/tokens, with CIs over ≥3 runs and a falsification gate ("TP across NVLink must beat naive
   replication on the OOM config; if not, reported as fail").
4. **A µTransfer demonstration** (`pretraining/distributed/run_mutransfer.py`): optimal LR found on
   a narrow proxy under µP **transfers** to a 4–8× wider model (loss within gate); the standard-
   parametrization baseline's optimum **shifts** (the falsification contrast). This is the
   science-of-scaling headline.
5. **A scaling curve from *real* GPU runs** (`pretraining/distributed/run_gpu_scaling.py`): fit
   `L(D)` / `L(N)` / `L(C)` on real loss curves from the FSDP trainer, extrapolate one held-out
   point, report 10% gate + CIs — the toy `scaling/` study **promoted to real hardware**.
6. **Sharded checkpoint + deterministic resume** (`disttrain/checkpoint.py` via
   `torch.distributed.checkpoint` / DCP): kill mid-run, resume, show bit-comparable continuation.
7. **Simulator validated against reality** (`clustersim` cross-check): predicted vs measured
   step-time / comms from a real multi-GPU run, closing the loop the NCCL bench opened.
8. **A `disttrain` CI lane** (`.github/workflows/disttrain-cpu.yml` always-on tiny CPU correctness
   + `disttrain-runpod.yml` dispatch GPU lane), and a `REPLICATION.md` so any number is rerunnable.
9. **All of it honest:** a `disttrain/PRE-REGISTRATION.md` with gates D1–D8, `canClaimAGI:false`,
   "small-but-real, labelled small," registered numbers only from x86 RunPod, Spark stays
   iteration-only.

**Crucially — the differentiator from a thousand "I ran a GPT on 8 GPUs" repos:** every efficiency
number is checked against an analytic reference (peak FLOPs, comms volume), pre-registered, run ≥3×
with CIs, and the *negative* results (where TP loses, where fp8 doesn't help, where the sim
mispredicts) are reported, not hidden. That is the Anthropic/DeepMind hiring signal.

---

## 4. Phased implementation plan

Each phase ships runnable code + a pre-registered gate + CI. Files are **new** unless noted as
(edit). Nothing outside the new `disttrain/`, `pretraining/distributed/`, CI, and docs is changed.

### Phase 0 — Scaffold & honesty contract (CPU, free) — *foundation*
- Add `disttrain/` package: `__init__.py`, `README.md` (labels: "small-but-real distributed
  training; not a frontier model; `canClaimAGI:false`"), `requirements-disttrain.txt`
  (`torch>=2.5`, `torchdata`, optional `torchao` for fp8, optional `torchtitan` as reference).
- Add `disttrain/PRE-REGISTRATION.md` with gates **D1–D8** (see §6) committed **before** any GPU
  number is trusted — mirrors `pretraining/PRE-REGISTRATION.md`.
- Add `disttrain/config.py`: dataclass configs (model dims, parallel degrees DP/TP/PP/CP,
  precision, AC mode, tokens, seed). Tiny "cpu-smoke" preset.
- Library: PyTorch. **Decision:** FSDP2 (`torch.distributed._composable.fsdp`) + DTensor as the
  backbone; TorchTitan vendored/referenced for TP/PP patterns; DeepSpeed explicitly *out of scope*
  (document why: FSDP2 is PyTorch-native and composes with TP/PP/CP cleanly).

### Phase 1 — Single-process correct transformer + MFU meter (CPU, free) — *correctness*
- `disttrain/model/gpt.py`: clean decoder-only GPT (RMSNorm, RoPE, SwiGLU, optional FlashAttention
  reusing `kernels/`), parameter-counted exactly.
- `disttrain/data/`: streaming token loader (reuse repo corpora / a small public tokenized set;
  deterministic shuffling with seed).
- `disttrain/train.py`: single-process training loop (works on CPU, tiny config), bf16-capable.
- `disttrain/metrics/mfu.py`: analytic FLOPs (`6N + attention` per PaLM/Megatron), peak-FLOP table
  (A100/H100/H200/GB10), MFU & HFU. **Unit test against known formulas** (gate D1).
- CI: `.github/workflows/disttrain-cpu.yml` — always-on, runs cpu-smoke train (a few steps) +
  MFU-formula tests + loss-decreases test (mirrors nano G1).
- Tests: `tests/test_disttrain_mfu.py`, `tests/test_disttrain_smoke.py`.

### Phase 2 — FSDP2 data-parallel + sharded checkpoint (1×GPU, then 2–8×GPU) — *first real distribution*
- `disttrain/parallel/fsdp.py`: FSDP2 wrap (per-param DTensor sharding, mixed-precision policy,
  optional CPU offload), prefetch/overlap.
- `disttrain/checkpoint.py`: `torch.distributed.checkpoint` (DCP) sharded save/load; deterministic
  resume.
- `disttrain/launch/torchrun.sh` + reuse RunPod orchestration: extend `tools/runpod_train.py`
  pattern into `tools/runpod_disttrain.py` (single-pod, multi-GPU, real `torchrun`).
- Gate **D2** (FSDP loss-parity): single-GPU vs FSDP-on-N-GPU reach the same loss (within tol) on a
  fixed seed/token budget. Gate **D3** (resume): kill at step k, resume, continuation matches.
- CI: `disttrain-runpod.yml` (workflow_dispatch) running the 8×GPU smoke + parity on x86 RunPod.

### Phase 3 — MFU/efficiency sweep + precision + activation checkpointing (1×node multi-GPU) — *the workload headline*
- `disttrain/parallel/{tensor.py,pipeline.py}`: Megatron-style column/row-parallel linear (TP) and
  1F1B pipeline schedule (PP), composed with FSDP2 (2D/3D mesh via `DeviceMesh`).
- `disttrain/precision/fp8.py`: opt-in `torchao` Float8 rowwise on linears (Hopper+ only; gated,
  fail-closed on unsupported hardware → falls back to bf16 with a logged note).
- Activation checkpointing: selective AC wrapper (`disttrain/parallel/ac.py`).
- `pretraining/distributed/run_parallelism_sweep.py`: for a fixed model+tokens, sweep
  {DP-only, DP+TP, DP+TP+PP} × {bf16, fp8} × {AC on/off}; log MFU/HFU/tokens-s/peak-mem; ≥3 seeds,
  report **CIs**. Gate **D4** (TP helps the OOM case), **D5** (fp8 ≥ bf16 throughput on H100 *or*
  reported as no-win).
- `pretraining/distributed/PRE-REGISTRATION.md` + a results json (TorchTitan-style table).

### Phase 4 — Scaling science on real GPUs: µTransfer + GPU scaling curve — *the science headline*
- `disttrain/mup.py`: µP parametrization (width-scaled init + per-layer LR), `--mup` flag.
- `pretraining/distributed/run_mutransfer.py`: LR-sweep on narrow proxy under µP and under SP
  (standard); train 4–8× wider model at the proxy-optimal LR; **µP optimum transfers, SP shifts**.
  Gate **D6**.
- `pretraining/distributed/run_gpu_scaling.py`: promote the toy `scaling/` study to the FSDP
  trainer — fit `L(D)`/`L(N)` on real loss, extrapolate one held-out point, 10% gate + CIs
  (reuses `pretraining/scaling/fit.py` directly). Gate **D7**.
- `pretraining/distributed/run_iso_compute.py`: real version of `autopilot --task compute` — a
  small Chinchilla iso-FLOP frontier (N vs D) on GPU, find the interior compute-optimal point.

### Phase 5 — Multi-node + simulator validation (2-node burst) — *scale-out reliability*
- `tools/runpod_disttrain.py --nodes 2`: real 2-node `torchrun` rendezvous over RunPod (IB/RoCE),
  reusing the NCCL-bench network knowledge.
- Cross-check: feed the real per-step comms/throughput into `clustersim/netcalib.py`; compare
  **predicted vs measured** step time. Gate **D8** (sim predicts measured step-time within band, or
  the miss is reported and the model corrected).
- `disttrain/REPLICATION.md`: exact commands, pod specs, seeds, expected numbers + CIs.

### Phase 6 — Write-up & integration — *legibility*
- `pretraining/distributed/README.md`: narrative tying it to the existing pretraining studies and
  the DeepSeek/Anthropic framings already in `pretraining/README.md`.
- Add validated numbers to `RESULTS.md` under a new "distributed-training (small-but-real)" section,
  honoring the no-overclaim gate (illustrative vs validated, CIs, x86-only registered).
- Optional `pretraining/distributed/agent/`: extend the existing reviewer-agent rubric to audit the
  new gates (fail-closed, `cannot_assess` on missing artifacts).

---

## 5. Compute / budget tiers

| Tier | Hardware | ~Cost | What it credibly demonstrates |
|---|---|---|---|
| **T0 — Free / local CPU** | Any CPU (CI, laptop, Spark for iteration) | $0 | Correctness, not performance: tiny-config FSDP runs single-process; MFU **formula** unit tests; FSDP/TP **shape & loss-parity** logic on CPU `gloo`; sharded-checkpoint save/resume determinism; all pre-registration gates that don't need real throughput. Phases 0–1 fully; Phases 2–4 *logic* (not numbers). This is the always-green CI floor. |
| **T1 — Single-node multi-GPU RunPod** | 1× pod, 2–8× A100/H100 (e.g. 8×A100-80GB or 4×H100) | **~$100–500** (a few pods × hours) | The **bulk of the credible story.** Real FSDP2 data-parallel training; measured **MFU/HFU** (target a defensible 30–45% on an 8B-class config, *exactly* TorchTitan's reported 33–42% range — do not claim more); FSDP loss-parity & resume (D2/D3); the **parallelism×precision×AC MFU sweep** (D4/D5); **fp8-vs-bf16** on H100; **µTransfer** (D6); **real scaling curve** (D7); iso-compute frontier. A reviewer reads this as genuine workload competence. |
| **T2 — Multi-node burst** | 2 nodes × 8 GPUs, IB/RoCE | **~$1–5k** (short bursts; minimize via spot + small token budgets) | **Scale-out reliability:** real 2-node `torchrun`/NCCL training; inter-node comms cost measured and compared to the intra-node case; **TP-intra-node + DP-inter-node** mesh; **simulator validated** against a real multi-node run (D8); a genuine (if small) multi-node MFU number. This is the "I have actually debugged a multi-node run" signal. Optional but high-value; T1 alone already clears most bars. |

**Budget discipline (preserve the repo's cost-guard ethos):** every GPU phase keeps the existing
gated-escalation pattern (dry-run plan, explicit `--launch` + `--cost-ceiling` + `RUNPOD_API_KEY`,
spot instances, smallest token budget that makes the measurement statistically meaningful). Start
T1 on the cheapest config that shows the effect; only scale tokens/GPUs when a gate needs it.

---

## 6. Honest success metrics & benchmarks (pre-registered gates D1–D8)

Mirrors `pretraining/PRE-REGISTRATION.md`. Every gate states its falsification condition first;
every headline number is mean ± 95% CI over **≥3 seeds**, reproducible from `REPLICATION.md`.

- **D1 — MFU meter is correct.** Analytic FLOP/MFU/HFU matches the PaLM/Megatron closed form on
  hand-checked configs to <1%. *Falsified if* the meter disagrees with the formula. (CPU, free.)
- **D2 — FSDP is loss-faithful.** Single-GPU and N-GPU-FSDP runs at the same seed/token budget
  reach final loss within a small tolerance. *Falsified if* sharding changes the loss beyond tol
  (a correctness bug).
- **D3 — Deterministic resume.** Kill at step k, restore from sharded checkpoint, continuation loss
  trajectory matches the uninterrupted run within tol. *Falsified if* resume diverges.
- **D4 — Parallelism buys capacity.** A config that OOMs on naive DP **trains** under TP/PP, and
  measured MFU is reported (not asserted). *Falsified if* TP/PP can't fit it or MFU is fabricated.
- **D5 — Precision/AC trade-offs are measured, not assumed.** Report bf16 vs fp8 throughput and
  AC-on vs AC-off memory/throughput with CIs. fp8 is claimed a win **only** where measured ≥ bf16;
  otherwise reported as no-win (honest negative). *Falsified if* a precision win is claimed without
  the measurement.
- **D6 — µTransfer transfers (and SP doesn't).** Proxy-optimal LR under µP applied to the wide model
  lands within the loss gate; the SP baseline's optimum **shifts** by a reported margin.
  *Falsified if* µP fails to transfer or the SP contrast is absent.
- **D7 — Real scaling curve predicts.** Fit `L(D)`/`L(N)` on smaller real runs, extrapolate one
  held-out larger run within **10%** (reusing `scaling/fit.py`), with the floor-identifiability
  caveat reported honestly (as G4 already is). *Falsified if* the gate fails silently.
- **D8 — Simulator matches reality.** `clustersim` predicted step-time/comms vs the real multi-GPU
  run within a stated band; misses are reported and the network model corrected.
  *Falsified if* a sim number is presented as if validated when it wasn't.

**Headline benchmark set a reviewer will scan:** (1) MFU/HFU table across parallel configs + CIs;
(2) tokens/s & peak-mem vs GPU count (strong/weak scaling); (3) bf16 vs fp8; (4) µTransfer plot;
(5) real scaling-law extrapolation pass; (6) resume/fault demonstration; (7) sim-vs-real.

**Honesty bound (carried verbatim in spirit from the repo):** "These are **small-but-real**
distributed-training results — labelled small. They demonstrate systems methodology and
measurement discipline, not frontier capability. `canClaimAGI:false`. Registered numbers come only
from the x86 RunPod path; the Spark lane is iteration-only."

---

## 7. Risks & how to avoid overclaiming

- **MFU inflation** (the #1 reviewer red flag). Mitigate: lock the FLOP formula in D1 with unit
  tests; always state which formula (6N vs 6N+attention), which peak (dense bf16 vs fp8 tensor-core
  peak — *never* mix sparse peak), and report HFU alongside MFU so recompute isn't hidden. Target
  TorchTitan's published 33–42% band; treat anything >55% as suspect and re-derive.
- **"Distributed" theater.** The repo already has the `CUDA_VISIBLE_DEVICES` multi-seed fan-out;
  do **not** let it be read as distributed training. Real gradient-sharing collectives only.
- **Scale overreach.** Resist claiming "trained a 70B." State the actual N/D/tokens; small configs
  with correct method beat big configs with sloppy measurement. The repo's whole brand is honesty.
- **Spark numerics drift.** Keep the existing provenance boundary: Spark = iteration, x86 RunPod =
  registered. Annotate every Spark artifact `candidateOnly + spark_iteration`.
- **fp8 cargo-culting.** Only claim fp8 wins where measured on supported hardware (Hopper+); report
  convergence parity vs bf16; fall back gracefully and log when unsupported.
- **Scaling-law over-extrapolation.** Carry G4's lesson: don't claim floor recovery without runs
  near saturation; report identifiability limits as findings, not defects.
- **Sim presented as measurement.** `clustersim` is a model; never let a predicted number masquerade
  as a measured one (D8 enforces this).
- **Cost runaway.** Keep gated escalation, dry-run default, cost ceilings, spot, smallest-budget-
  that-measures. Never auto-spend.
- **Reproducibility gaps.** `REPLICATION.md` with seeds/pod-specs/commands or a number isn't
  registered. ≥3 runs + CIs for every headline, per `RESULTS.md`.

---

## 8. Effort estimate

Assumes one strong ML-systems engineer; calendar weeks at part-time intensity, plus the bounded
RunPod spend from §5. CPU/logic work is free and front-loaded so most risk retires before money.

| Phase | Scope | Eng effort | Compute |
|---|---|---|---|
| 0 Scaffold + pre-reg | package, configs, gates, CI skeleton | 2–3 days | $0 (T0) |
| 1 Transformer + MFU meter | model, loader, loop, MFU+tests | 4–6 days | $0 (T0) |
| 2 FSDP2 + sharded ckpt | FSDP wrap, DCP, resume, parity | 5–8 days | ~$50–150 (T1) |
| 3 TP/PP + fp8 + AC + sweep | parallelism, precision, MFU sweep | 8–12 days | ~$150–350 (T1) |
| 4 µTransfer + real scaling | µP, transfer, scaling curve, iso-compute | 5–8 days | ~$100–200 (T1) |
| 5 Multi-node + sim validate | 2-node torchrun, netcalib cross-check | 4–7 days | ~$0.5–3k (T2, optional) |
| 6 Write-up + integration | READMEs, RESULTS, reviewer-agent | 3–4 days | $0 |
| **Total** | | **~6–9 weeks part-time** | **~$300–800 (T0+T1)**; **+$1–5k if T2** |

**Minimum credible path to clear most bars:** Phases 0–4 on T0+T1 (~4–6 weeks, ~$300–800). Phase 5
(T2) is the multiplier that turns "single-node competent" into "has debugged multi-node," and is
worth it if budget allows, but is explicitly optional.

---

### Appendix — key references (cite by name in the write-up)
Kaplan 2020 (Scaling Laws); Hoffmann 2022 (Chinchilla); Chowdhery 2022 (PaLM, MFU); Yang 2021/2022
(µP / µTransfer, Tensor Programs V); Rajbhandari 2020 (ZeRO / DeepSpeed); Zhao 2023 (PyTorch FSDP) +
FSDP2/DTensor; Shoeybi 2019 & Narayanan 2021 (Megatron-LM, TP/PP, 1F1B); Liang 2024 (TorchTitan,
arXiv:2410.06511); Liu 2023 (Ring Attention / context parallel); Micikevicius 2017 (Mixed Precision);
Chen 2016 (sublinear-memory activation checkpointing); Dao 2022/2023 (FlashAttention 1/2/3);
NVIDIA Transformer Engine + PyTorch `torchao` Float8 (fp8 training).
