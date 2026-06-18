# Source Discipline

Open-source corpus, training data, and tooling for **provenance-aware philosophy education** — teaching AI (and humans) to distinguish real authorship from legendary attribution across Eastern and Western intellectual traditions.

## Why this project

Philosophy models fail in predictable ways: they merge lineages (e.g. attributing 《道德經》 to Confucius), treat compilers as authors, and ignore uncertainty (Laozi historicity). **Source discipline** is the practice of following evidence about who wrote what, when, and in which tradition — before reasoning with the ideas.

This repository holds:

- Structured **attribution data** (machine-readable)
- **Dispute notes** (human-readable, bilingual)
- **Training examples** for fine-tuning or evaluation
- **Validation tools** to catch misattribution
- A **teacher agent spec** for generating new training pairs

## Quick start

```bash
git clone https://github.com/tomyimkc/source-discipline.git
cd source-discipline
python tools/validate_attribution.py
python tools/export_training_jsonl.py --out training/corpus.jsonl
```

## Repository layout

```text
source-discipline/
├── docs/           # Documentation and dispute notes
├── data/           # attributions.json, traditions.json
├── training/       # JSONL-ready examples
├── tools/          # Validators and exporters
├── tests/          # Evaluation benchmarks
└── .grok/agents/   # grok-cli-teacher agent spec
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every new text must include an attribution record in `data/attributions.json` and, when contested, a dispute note in `docs/04-Disputes/`.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this corpus in research or training, please cite the repository URL and note which commit or release you used.