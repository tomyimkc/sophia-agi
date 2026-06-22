---
license: mit
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

Datasets and packs behind Sophia's provenance-aware, verifier-gated reasoning — for
independent replication of the published results.

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
Packs/labels are currently self-authored; a fully independent claim needs a third-party
pack + human review (tracked in `agi-proof/failure-ledger.md`). Not an AGI claim.
