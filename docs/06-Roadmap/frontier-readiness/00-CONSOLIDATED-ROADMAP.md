# Sophia-AGI → Frontier-Lab Repo: Consolidated Roadmap

Goal: develop the repo so it credibly demonstrates the experience behind every **core technical IC family** at Anthropic (and OpenAI/DeepMind reference points), without overclaiming — preserving the repo's measured/fail-closed/pre-registered ethos.

## Decisions locked (2026-06-26)

- **Scope:** run **all 8** items in the recommended wave order (full program).
- **Compute budget:** *serious* tier — plus three **local workbenches** that change the cost model:
  - **DGX Spark** (GB10 Grace-Blackwell, ~128GB unified) → primary **CUDA/Triton/RL/interp** iteration box; already has a `spark-gpu.yml` CI lane.
  - **Mac Studio** (Apple Silicon, MLX) → second **inference/serving + MLX-LoRA** lane (`training/mlx_adapters/`).
  - **MacBook** → dev / offline-CI / mock lane.
  - **RunPod** (MCP-wired) → reserved for the **multi-node distributed-training stamp** and A100/H100 head-to-heads.
- **Base model:** **standardize on Qwen2.5** (`Qwen/Qwen2.5-7B-Instruct`, Apache-2.0) across RLVR / interp / serving. This also removes a licensing wart (GLM-4-9B needed Zhipu registration). The RLVR toolchain (`run_rlvr.py`, `eval_rlvr_adapter.py`, `runpod_rlvr.py`, `run_closed_loop.py`) now defaults to Qwen2.5 with family-aware LoRA target modules (split `gate_proj`/`up_proj`).
- **Execution start:** **Wave 1 #1 — Live RL.** M0 (offline contract) is **DONE & green** (this commit); M1 (first real weight update) is GPU- + judge-key-bound.

### Wave 1 #1 — Live RL: M0 status (offline contract, no GPU)

| Check | Result |
|---|---|
| Gate-reward invariants (violation<abstain<clean, abstain>0, bounded) | ✓ PASS |
| RLVR reward wiring — provenance & math | ✓ VERIFIED |
| Adapter eval (mock) — provenance | ✓ PASS |
| Adapter eval (mock) — math | ✗ not-passed (mock delta below the strict gate — confirm intended) |

**M1 launch blockers (must be provisioned on the GPU box):** `HF_TOKEN`, `RUNPOD_API_KEY`, and ≥2 distinct judge-vendor keys (for the κ≥0.40 / ≥2-family `_is_validated` bar). Without judge keys only the **judge-free rung** is reachable, not the validated headline.



## Key reframe from the deep dive

The agents found the repo is **much further along than the initial gap analysis assumed**. For most items the *hard half is already built and inert* — the work is to "activate" it into a measured result, not greenfield it:

| Item | Already built (the hard half) | What's missing |
|---|---|---|
| 05 Live RL | Full GRPO+RLVR loop (`tools/run_rlvr.py`), gate reward, RunPod orchestration; **math arm already cleared a judge-free rung** | The *validated* provenance arm + a demonstrated reward-hacking defense |
| 04 Inference serving | A real **Rust paged prefix-sharing KV-cache** (PagedAttention-grade) | An engine that generates tokens through it (continuous batching) |
| 06 Frontier evals | RSP-style HALT gate, sandbagging/elicitation gate, probe registry | The *measurement* half: control groups, CIs, calibrated scorers |
| 03 Interpretability | Residual-stream capture/steering hooks (`agent/steering/hooks.py`) | SAEs / dictionary learning / causal feature work |
| 08 Data platform | MinHash dedup, data-passport, contamination guard, quality scorer | Streaming WARC, real table format (Iceberg), distributed engine |
| 01 Distributed training | NCCL bench, cluster *simulator*, scaling methodology | **Genuine gap:** zero real FSDP/TP/PP, no MFU |
| 02 Kernels | First-class roofline harness, RunPod plumbing | **Genuine gap:** zero measured GPU kernels (FA is numpy) |
| 07 Multimodal | Honest multimodal eval + GUI-gate harness | **Furthest gap:** no trainable multimodal params at all |

So two items are true greenfield gaps (distributed training, kernels), one is far (multimodal modeling), and five are "finish the inert half."

## Master table — effort, compute, signal

| # | Item | Effort (focused) | Min compute / cost | Readiness | Highest-signal deliverable |
|---|---|---|---|---|---|
| 05 | **Live RL** | 3–5 wks (M1: 3–5 days) | 1 GPU · **$15–30** for M1 | ★★★★★ closest | First *validated* gated GRPO weight update (CI delta) — closes a self-identified OPEN ledger item |
| 03 | **Interpretability** | MVP 1.5–2.5 wks; headline 4–6 wks | 1×24GB · **$30–80** | ★★★★ | Clamp one honesty/deception SAE feature, CI-bounded behavioral shift |
| 06 | **Frontier evals** | 16–24 eng-days | mostly **$0** | ★★★★ | Calibrated `monitor_subversion` probe → ASL-style capability report |
| 02 | **Kernels** | 3–4 wks | **<$30** | ★★★ greenfield | Triton FlashAttention-v2 fwd, `ncu`-attributed, vs SDPA roofline |
| 08 | **Data platform** | crit path 10–14 days | mostly **$0**; M3 $50–300 | ★★★ | Trusted CommonCrawl corpus, Iceberg-versioned, per-row passport |
| 07 | **Multimodal** | M0+M1 2–3 wks | **$80–250** (A100) | ★★ furthest | Trained LLaVA-style VLM scored by the existing honest harness |
| 04 | **Inference serving** | 7–11 wks | 60–90 GPU-hr | ★★★ | Continuous-batching engine over the Rust paged KV-cache; vs naive + vLLM |
| 01 | **Distributed training** | 6–9 wks part-time | **$300–800** (T0+T1); +$1–5k T2 | ★★ biggest gap | Pre-registered MFU sweep (parallelism×precision) with CIs + µTransfer |

## Recommended sequencing (signal-per-dollar-per-week)

**Wave 1 — Activate the inert half (weeks 1–4, < ~$150 total).** Cheap, fast, and each converts existing scaffolding into a real result. Reinforces the repo's distinctive **safety/honesty identity** (Alignment, Interpretability, Frontier Red Team — Anthropic's signature teams):
- **05 Live RL M1** — the single best signal-per-effort; closes the repo's own OPEN item.
- **03 Interpretability MVP (M0–M2)** — mostly offline + one cheap GPU run.
- **06 Frontier evals Milestone A** — mostly $0, pure rigor.

**Wave 2 — Systems credibility (weeks 4–10).** Targets the numerous Infra/Inference/Performance roles; the two pair naturally into one inference story:
- **02 Kernels** (Triton FlashAttention) · **04 Inference serving M0–M2** (engine over the Rust KV-cache).

**Wave 3 — The headline lift + breadth (weeks 8–16):**
- **01 Distributed training** — biggest prestige + biggest gap; benefits from Wave 2's shared GPU harness and MFU intuition.
- **08 Data platform** crit path (mostly $0) · **07 Multimodal** (optional breadth).

## Cross-cutting infra to build once
- **Shared rented-GPU lane**: RunPod MCP plumbing already exists (`tools/runpod_*.py`); standardize one "rent → honest-bench → teardown" path all GPU tracks reuse.
- **Base-model choice** (currently inconsistent: Qwen2.5-7B interp, Qwen2.5-3B multimodal, GLM-4-9B RLVR, sophia-v1 serving) — standardizing improves coherence + shared activation/weight reuse.
- **No-overclaim gate** (≥2 judges, ≥3 runs, CIs, pre-registration) — every plan already honors it.

## Budget / timeline envelopes
- **Wave 1 only:** ~4–6 weeks part-time, **< $150** — reshapes the repo's safety-research positioning fast.
- **Waves 1–2:** ~3 months part-time, **~$300–600**.
- **Full program (all 8), single-node:** ~4–6 months part-time, **~$500–1,500**.
- **+ multi-node distributed-training stamp:** **+$1–5k** (optional T2).
