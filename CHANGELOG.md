# Changelog

All notable changes to Sophia AGI are documented here.

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