# Changelog

All notable changes to Sophia AGI are documented here.

## [0.5.1] - 2026-06-18

### Added

- Phase 2 teacher: `tools/claude_teacher.py` — **450** Claude-generated examples (multi-round paraphrase) → **500** total
- Phase 4 correction: `agent/correction_loop.py`, `tools/run_correction_loop.py`
- `CONTRIBUTING.md` Phase 2 human-review checklist and Phase 4 correction workflow

### Changed

- Claude Sonnet external benchmarks re-run: **100%** on philosophy, psychology, history, religion
- Leaderboards and `web/data/manifest.json` refreshed
- `training/corpus.jsonl` regenerated (**500** lines)

## [0.5.0] - 2026-06-18

### Added

- Phase 1 corpus expansion: **30** philosophy attributions, **10** dispute notes, **50** training examples
- `tools/expand_phase1_corpus.py` — idempotent corpus growth script
- Phase 3 runtime gate: `agent/benchmark_checks.py`, upgraded `agent/gate.py` (attribution traps)
- `tests/test_gate.py` — reference teacher 100% on philosophy traps; bad-answer rejection
- History dated events with `primarySource` (GF-20); myth records tagged

### Changed

- `tools/score_benchmark.py` shares trap logic with runtime gate
- Agent CLI + `POST /api/ask` pass `question`/`sources` into gate; web UI shows gate status
- `training/corpus.jsonl` regenerated (50 lines)

## [0.4.2] - 2026-06-18

### Added

- GF-01–05 complete: Mencius, Zhuangzi, Symposium attributions + dispute notes
- Training example `020-socrates-plato-mencius-zhuangzi.json`
- 5 new philosophy benchmark traps (9 cases total)
- Launch drafts: Show HN, Reddit, GitHub Pages setup (`docs/07-Growth/launch/`)
- GitHub issue templates + `tools/create_github_issues.py`

### Changed

- Claude Sonnet re-scored 100% on expanded philosophy benchmark (9/9)
- Corpus export: 20 training examples

## [0.4.1] - 2026-06-18

### Added

- **Thesis web UI** — `web/` scholarly monograph site (Abstract → Agent chapters)
- **UI Council** — design decisions in `docs/10-Web/UI-Council-Decisions.md`; council panel in Chapter IV
- `tools/build_web_data.py` — bundle leaderboards into `web/data/manifest.json`
- `tools/serve_web.py` — static serve + `POST /api/ask` for advisor | repo | life agent

## [0.4.0] - 2026-06-18

### Added

- **Sophia Agent** — three paths: `advisor`, `repo`, `life` (`tools/sophia_agent.py`)
- `agent/` package: RAG retrieval, LLM client, epistemic gate, memory log, repo tools
- Docs: `docs/09-Agent/Sophia-Agent.md`
- Repo tools with `--execute --approve` gate: validate, export, benchmark, HF upload

## [0.3.1] - 2026-06-18

### Added

- Psychology hub example 018 + religion hub example 019 (philosophy-style source discipline)
- `docs/08-Domains/Source-Discipline-Methodology.md`
- Expanded `psychology_concepts.json` and `religion_concepts.json` source records
- LLMHub / custom `ANTHROPIC_BASE_URL` support in `run_external_models.py`

### Changed

- Psychology and Religion domain docs marked **Active** with data-center layout
- Benchmark SYSTEM prompt: explicit subfield/tradition/myth vocabulary
- Scorer: author aliases (Freud, Festinger) and tradition aliases (Christian, Buddhist)
- Claude Sonnet benchmark: **100%** on all four domains via LLMHub

## [0.3.0] - 2026-06-18

### Added

- Training examples 005–017 (psychology myths, history traps, religion council cases)
- Dedicated Dao De Jing philosophy/religion council example (014)
- Reference response pipeline: `benchmark/reference/case_map.json`, `tools/build_reference_responses.py`
- External model runner: `tools/run_external_models.py` (GPT-4o, Claude, Grok — requires API keys)
- Hugging Face upload script: `tools/upload_huggingface.py` + `docs/07-Growth/HuggingFace-Upload.md`
- Leaderboard refresh: `tools/update_leaderboards.py`
- `.env.example` for HF and model API tokens

### Changed

- Religion reference mapping: `dao_de_jing_religion_philosophy` → example 014 (was 004)
- README: 17 training examples, domains marked Active

## [0.2.0] - 2026-06-18

### Added

- Per-domain benchmarks: philosophy (4), psychology (4), history (5), religion (5)
- Training examples 002–004 (psychology, history, religion council panel)
- Council panel mode: all voices on one panel; sensitive traps in scope
- Confucian split-when-appropriate guide (philosophy vs 禮教)
- Response templates per domain under `benchmark/templates/`
- Expanded psychology/history/tradition data records

### Changed

- `score_benchmark.py` supports `--domain` flag
- `run_benchmark.py baseline all` runs every domain

## [0.1.0] - 2026-06-18

### Added

- Initial public corpus: philosophy attributions, traditions, dispute notes
- Bilingual seed training example (Dao De Jing / Confucius trap)
- `tools/validate_attribution.py`, `tools/export_training_jsonl.py`
- `tools/corpus_stats.py`, `tools/score_benchmark.py`, `tools/run_benchmark.py`
- Sophia Attribution Benchmark (`tests/attribution_bench.json`)
- Domain schema for future psychology, history, and religion expansion
- Hugging Face dataset card template (`huggingface/README.md`)
- GitHub Actions CI workflow
- 90-day launch playbook and good-first-issue guide

### Changed

- Project branding: Sophia AGI (wisdom before intelligence)