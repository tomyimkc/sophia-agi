---
license: apache-2.0
task_categories:
  - text-classification
  - question-answering
language:
  - en
  - zh
tags:
  - provenance
  - hallucination
  - calibration
  - abstention
  - benchmark
pretty_name: Sophia Provenance & Calibration Benchmark
---

# Sophia Provenance & Calibration Benchmark

**The open dataset for source discipline in AI.**

Bilingual (EN + 中文) training examples, abstain packs, legal citation/holding faithfulness tests, and misattribution gold labels — all under a strict no-overclaim gate.

Used by the Sophia gate to drive hallucinated attributions **36% → 23%** (validated) with 0% false positives on 8B models.

- 528 training examples across philosophy, psychology, history, religion
- Multiple held-out calibration and verifier packs
- Legal citation existence + holding faithfulness benchmarks
- OKF provenance wiki + belief graph

For researchers, agent builders, and anyone who needs AI that knows *who wrote what*.

## Contents
- **Misattributions / gold attributions** (`data/misattributions.json`,
  `data/wikidata_snapshot.json`) — false vs. true author/work pairs; the basis of the
  cross-entity grounding result (false-positive 100% → 0%).
- **Abstain pack** (`agi-proof/baseline-ablation/abstain-pack-2026-06-22.json`) — 18 cases
  (12 genuinely-unknown-author / unverifiable-quote / unsolved-identity + 6 definite
  controls); the basis of the fabrication/calibration result (sophia-full 0% vs raw 17–25%).
- **Hard ablation pack** (`...hard-pack-2026-06-22.json`) — 17 mixed-domain cases.
- **OKF provenance wiki** (`wiki/`, `docs/04-Disputes/`) — the belief graph corpus.

## Scoring
- **Deterministic:** calibration scorer (abstention vs fabrication), grounding gate,
  cross-entity invariants — no LLM judge.
- **LLM-judged:** multi-family corroboration (Cohen's κ reported) via the no-overclaim gate.

## How to use
See `agi-proof/REPLICATION.md` for the exact commands and `docs/11-Platform/Methodology.md`
for the method. Hidden-eval prompts are not published — only aggregates (see `SECURITY.md`).

## Honest scope
All public numbers are gated (≥2 judge families + CIs). Packs are currently maintainer-authored; third-party replication tracked in `agi-proof/`. 

**Not an AGI claim.** This is the measurement and training substrate for provenance-aware reasoning.

GitHub: https://github.com/tomyimkc/sophia-agi
Thesis: https://tomyimkc.github.io/sophia-agi/
