# Continual Provenance QA (CPQA) — benchmark spec

A small, reproducible **continual-learning** benchmark for provenance-gated, non-parametric
knowledge: it measures whether a system can learn a *stream* of facts, correct/forget on
demand, and answer **without fabricating** — the sequential-retention protocol most
pipelines never run. `candidateOnly` / `validated:false` (see `docs/06-Roadmap/cpqa-results-candidate.md`).

## Protocol
A run is a **stream of episodes**; each episode may `learn` OKF pages, `retract` sources, and
ask `queries`. Two systems answer the *same* questions:
- **graph_backed** — knowledge in the OKF belief graph: learns by page write, revises
  conflicts, unlearns on demand, answers only from the grounded belief state.
- **parametric_baseline** — knowledge frozen at episode 0 (a weight model without retraining).

Each query is `assert` (a fact should be answerable) or `abstain` (a retracted/never-taught
trap that must be refused).

## Datasets
- `episodes_v1.jsonl` — 15 synthetic queries (recall / retention / conflict / unlearning / control).
- `episodes_v2_wiki.jsonl` — 92 queries auto-generated from the real `wiki/` corpus
  (`tools/gen_continual_qa_from_wiki.py`).

## Metrics
- **forgetting**: `unintendedForgetting` (catastrophic) vs `deliberateUnlearning` (retraction/revision).
- **retention matrix** + backward transfer; bootstrap 95% CIs (rule-of-three when 0 errors).
- **fabrication rate** and **abstention correctness** (traps).
- **control-flow gap**: oracle vs real-router accuracy (`run_continual_qa_llm.py`).
- **judged answers**: grounded vs raw, multi-/cross-family judge panel, mean pairwise Cohen's κ
  (`run_continual_qa_judged.py`).

## Reproduce
```bash
python tools/run_continual_qa_benchmark.py                                  # deterministic substrate (offline)
python tools/gen_continual_qa_from_wiki.py                                  # regenerate the 92-query set
python tools/run_continual_qa_validation.py                                 # bootstrap CIs + control-flow sweep
python tools/audit_cpqa_recall.py                                           # source-sufficiency audit
# live (needs API keys; cross-gateway judges):
python tools/run_continual_qa_judged.py --limit 92 --runs 3 \
  --answer llmhub:gpt-4o-mini \
  --judge openrouter:deepseek/deepseek-chat \
  --judge openrouter:meta-llama/llama-3.3-70b-instruct \
  --judge openrouter:qwen/qwen-2.5-72b-instruct
```

## Headline (candidate)
Graph-backed substrate: **0 unintended forgetting**, accuracy CI **[1.0, 1.0]** on retention.
Judged answers (full-92, cross-gateway): on **traps** grounded **1.0** vs raw **0.0**
(fail-closed abstention); on **recall** grounding is corpus-bound (enrichment lifts it).
Honest framing and limits in `docs/06-Roadmap/continual-learning-limitations.md`.

## Citation
> Sophia AGI — Continual Provenance QA (CPQA) benchmark, 2026. https://github.com/tomyimkc/sophia-agi
