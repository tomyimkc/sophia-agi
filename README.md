# Sophia AGI

**Wisdom before intelligence.** Open-source corpus and tooling to train AGI-shaped systems in **provenance-aware philosophy** — knowing *who wrote what* before reasoning with ideas.

> *Sophia* (σοφία) — wisdom in the Greek philosophical tradition. This project treats rigorous authorship and tradition boundaries as a foundation layer for trustworthy general intelligence.

## Why Sophia AGI

Language models confuse lineages: they attribute 《道德經》 to Confucius, treat compilers as authors, and flatten 儒家 and 道家 into one voice. **Source discipline** — our core method — fixes that by enforcing evidence-based attribution before belief propagation.

This repository provides:

- Structured **attribution data** (machine-readable)
- **Dispute notes** on contested authorship (bilingual)
- **Training examples** for fine-tuning and evaluation
- **Validation tools** that catch misattribution
- A **teacher agent** (`grok-cli-teacher`) for scaling the corpus

## Quick start

```bash
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
python tools/validate_attribution.py
python tools/export_training_jsonl.py --out training/corpus.jsonl
```

## Repository layout

```text
sophia-agi/
├── docs/           # Documentation and dispute notes
├── data/           # attributions.json, traditions.json
├── training/       # JSONL-ready examples
├── tools/          # Validators and exporters
├── tests/          # Evaluation benchmarks
└── .grok/agents/   # grok-cli-teacher agent spec
```

## Roadmap

See [docs/06-Roadmap/Open-Intelligence-Plan.md](docs/06-Roadmap/Open-Intelligence-Plan.md) for the step-by-step path from philosophy teacher → epistemic gate → AGI-shaped stack.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).