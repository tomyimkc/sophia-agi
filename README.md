# Sophia AGI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml/badge.svg)](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Training examples](https://img.shields.io/badge/training_examples-4-green)
![Domains](https://img.shields.io/badge/domains-philosophy%20%7C%20psychology%20%7C%20history%20%7C%20religion-purple)

**Wisdom before intelligence.** Open-source corpus and benchmark for **provenance-aware** reasoning — knowing *who wrote what*, *what happened when*, and *which tradition owns which idea* — before AGI-scale belief propagation.

> *Sophia* (σοφία) = wisdom. Philosophy is live today; **psychology**, **history**, and **religion** are on the roadmap.

## Why it matters

LLMs merge lineages: Confucius → 《道德經》, Socrates → *Republic*, pop psych → clinical science. **Source discipline** is our fix — evidence first, then reasoning.

## Quick start

```bash
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
python tools/validate_attribution.py
python tools/export_training_jsonl.py
python tools/run_benchmark.py template    # create model response template
python tools/run_benchmark.py score benchmark/responses.template.json
```

## Benchmarks (per-domain leaderboards)

| Domain | Cases | Leaderboard | Seed reference |
|--------|-------|-------------|----------------|
| Philosophy | 4 | [leaderboard-philosophy.json](benchmark/results/leaderboard-philosophy.json) | example 001 |
| Psychology | 4 | [leaderboard-psychology.json](benchmark/results/leaderboard-psychology.json) | example 002 |
| History | 5 | [leaderboard-history.json](benchmark/results/leaderboard-history.json) | example 003 |
| Religion | 5 | [leaderboard-religion.json](benchmark/results/leaderboard-religion.json) | example 004 (council panel) |

```bash
python tools/run_benchmark.py templates              # per-domain response templates
python tools/run_benchmark.py score FILE --domain psychology
```

Templates: `benchmark/templates/responses-{domain}.template.json`

## Repository layout

```text
sophia-agi/
├── data/              # attributions, domains, schema (multi-domain)
├── docs/              # disputes, growth playbook, domain expansion
├── training/          # JSONL-ready examples
├── benchmark/         # responses template + leaderboard
├── tools/             # validate, export, score, stats
├── tests/             # attribution benchmark cases
└── huggingface/       # HF dataset card (upload corpus.jsonl)
```

## Domains

| Domain | Status | Data file |
|--------|--------|-----------|
| Philosophy | Active | `data/attributions.json` |
| Psychology | Planned | `data/psychology_concepts.json` |
| History | Planned | `data/history_events.json` |
| Religion | Planned | `data/religion_concepts.json` |

See [docs/08-Domains/Overview.md](docs/08-Domains/Overview.md) and answer [Expansion-Questionnaire.md](docs/08-Domains/Expansion-Questionnaire.md) to shape the next domains.

## Roadmap & growth

- [Open Intelligence Plan](docs/06-Roadmap/Open-Intelligence-Plan.md)
- [90-Day Launch Playbook](docs/07-Growth/90-Day-Launch.md)
- [Good first issues](GOOD_FIRST_ISSUES.md)

## Hugging Face

Dataset card: `huggingface/README.md` — upload `training/corpus.jsonl` as `tomyimkc/sophia-agi-corpus`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changelog: [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).