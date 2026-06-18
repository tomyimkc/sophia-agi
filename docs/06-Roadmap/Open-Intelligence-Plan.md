# Sophia AGI Roadmap

Step-by-step plan to grow this repository into a general **provenance gate** for AGI-shaped systems.

## Phase 0 — Corpus bootstrap ✅ (this release)

- [x] Public repo: **Sophia AGI**
- [x] Attribution schema
- [x] First dispute note + training example
- [x] Validation tooling

## Phase 1 — Expand corpus (Weeks 2–4)

- [x] 30+ philosophy attribution records (v0.5.0: 30 in `data/attributions.json`)
- [x] 10+ dispute notes (v0.5.0: 10 in `docs/04-Disputes/`)
- [ ] Tradition boundary docs for Platonist / Stoic / Buddhist entries
- [x] Psychology domain: scope locked via [Expansion-Questionnaire](../08-Domains/Expansion-Questionnaire.md)
- [x] History domain: starter events + myth traps + dated events with `primarySource`
- [x] Religion domain: scripture attribution + sect boundaries
- [x] 50+ training examples (`training/examples/`, `corpus.jsonl`)

## Phase 2 — Teacher loop (Weeks 3–6)

- [x] Scale Claude teacher (`tools/claude_teacher.py`) to **500** pairs (v0.5.1)
- [x] `training/corpus.jsonl` export pipeline (auto on teacher run)
- [x] Human review checklist in CONTRIBUTING.md

## Phase 3 — Runtime gate (Weeks 6–10)

- [x] RAG retrieval over `data/` + `docs/04-Disputes/` (agent retrieval)
- [x] Curated online RAG index + Gemini / Vertex path (v0.6.0 — [Online-RAG.md](../09-Agent/Online-RAG.md))
- [x] LoRA sophia-v1 local eval **20/23 (87%)**; HF adapter `tomyimkc/sophia-agi-lora-v1`
- [x] Post-generation misattribution checker (`agent/gate.py` + `agent/benchmark_checks.py`)
- [x] Philosophy benchmark reference 100% at gate (`tests/test_gate.py`)
- [x] Claude Sonnet **100%** on all four domain benchmarks (v0.5.1)
- [ ] Benchmark suite pass rate ≥ 95% on GPT/Grok (keys or Monica gateway pending)

## Phase 4 — Multi-agent integration (Weeks 10–16)

- [ ] Pluggable epistemic gate for any LLM orchestrator
- [x] Correction loop tooling (`agent/correction_loop.py`, `tools/run_correction_loop.py`) — no failures to promote yet
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