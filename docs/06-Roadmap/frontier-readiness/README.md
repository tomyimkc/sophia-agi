# Frontier-Readiness Roadmap

A program to develop this repo so it credibly demonstrates the experience behind every **core
technical IC family** at frontier labs (Anthropic, with OpenAI/DeepMind reference points) — built
to the repo's own measured / fail-closed / pre-registered / no-overclaim standard.

Origin: a gap analysis of Anthropic's open technical roles vs this repo, then a per-item deep dive
(one research agent per item) producing the implementation plans below.

## Start here
- **[00-CONSOLIDATED-ROADMAP.md](00-CONSOLIDATED-ROADMAP.md)** — master table, wave sequencing,
  budget envelopes, locked decisions, and current execution status.

## Per-item plans (thesis + phased implementation)

| # | Plan | Bucket | Highest-signal deliverable |
|---|------|--------|----------------------------|
| 01 | [Distributed training](01-distributed-training.md) | high-value (greenfield) | Pre-registered MFU sweep + µTransfer |
| 02 | [GPU/TPU kernels](02-kernels.md) | high-value (greenfield) | Triton FlashAttention-v2, `ncu`-attributed |
| 03 | [Interpretability](03-interpretability.md) | high-value | Clamped honesty/deception SAE feature |
| 04 | [Inference serving](04-inference-serving.md) | high-value | Continuous-batching engine over the Rust KV-cache |
| 05 | [Live RL](05-live-rl.md) · [M1 pre-reg](05-live-rl-M1-prereg.md) | high-value (**executing**) | First *validated* gated GRPO weight update |
| 06 | [Frontier evals](06-frontier-evals.md) | adjacent | Calibrated probe → ASL-style capability report |
| 07 | [Multimodal](07-multimodal.md) | adjacent | Trained LLaVA-style VLM scored by the honest harness |
| 08 | [Data platform](08-data-platform.md) | adjacent | Trusted CommonCrawl corpus, Iceberg-versioned, per-row passport |

## Status
- **Wave 1 #1 (Live RL):** M0 offline contract **DONE & green**; M1 pre-registered, GPU-/key-bound.
- **Wave 1 #2 (Interpretability):** M0 offline core **DONE & green** — pure-stdlib TopK SAE
  (`interp/sae/`), metrics (L0/FVU/CE-recovered/dead-%), Qwen2.5 hookpoint adapter, offline CLI
  (`tools/run_interp.py --mode mock`), 12 CI tests. Reference SAE reconstructs a planted signal to
  FVU 0.088 (91% explained variance). M1 (real Qwen2.5-7B harvest) is GPU-bound.
- **Wave 1 #3 (Frontier evals):** M0 (Milestone-A core) **DONE & green** — `eval/frontier/` paired
  control-vs-treatment harness, held-out split + content hash, gold-calibrated scorer (published
  FP/FN), bootstrap CIs, fail-closed unmeasured path; proven end-to-end on the lowest-hazard probe
  (`monitor_subversion`) via `tools/run_frontier_eval.py --mode mock`; 7 CI tests. Defensive-only;
  the live G8 registry stays fail-closed (no mock registration).
- **GPU runs:** see [DGX-SPARK-RUNBOOK.md](DGX-SPARK-RUNBOOK.md) — turnkey recipes for Live-RL M1
  and Interpretability M1/M2 (run on the Spark; paste reports back for gating).
- Base model standardized on **Qwen2.5-7B-Instruct (Apache-2.0)** across RLVR + interp tracks.
