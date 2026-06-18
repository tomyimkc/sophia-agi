# Sophia AGI Roadmap

Step-by-step plan to grow this repository into a general **provenance gate** for AGI-shaped systems.

## Phase 0 — Corpus bootstrap ✅ (this release)

- [x] Public repo: **Sophia AGI**
- [x] Attribution schema
- [x] First dispute note + training example
- [x] Validation tooling

## Phase 1 — Expand corpus (Weeks 2–4)

- [ ] 30+ attribution records
- [ ] 10+ dispute notes
- [ ] Tradition boundary docs for Platonist / Stoic / Buddhist entries

## Phase 2 — Teacher loop (Weeks 3–6)

- [ ] Scale `grok-cli-teacher` generation to 500+ reviewed pairs
- [ ] `training/corpus.jsonl` export pipeline
- [ ] Human review checklist in CONTRIBUTING.md

## Phase 3 — Runtime gate (Weeks 6–10)

- [ ] RAG retrieval over `data/` + `docs/04-Disputes/`
- [ ] Post-generation misattribution checker API
- [ ] Benchmark suite pass rate ≥ 95%

## Phase 4 — Multi-agent integration (Weeks 10–16)

- [ ] Pluggable epistemic gate for any LLM orchestrator
- [ ] Correction loop: failed eval → new training example
- [ ] Cross-domain attribution schema (history, law, science)

## Phase 5 — AGI-shaped milestones

| Milestone | Description |
|-----------|-------------|
| M1 | Expert philosophy teacher with provenance |
| M2 | Runtime gate blocks lineage-merge errors |
| M3 | Gate generalizes beyond philosophy |
| M4 | Self-correcting loop from eval failures |
| M5 | Planner + tools + memory + Sophia epistemic gate |

See `tests/attribution_bench.json` for measurable pass criteria.