# Sophia AGI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml/badge.svg)](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.6.3-blue)
![Training examples](https://img.shields.io/badge/training_examples-518-green)
![Domains](https://img.shields.io/badge/domains-philosophy%20%7C%20psychology%20%7C%20history%20%7C%20religion-purple)

**Wisdom before intelligence.** Open-source corpus, benchmark, RAG/local-model baseline, and **AGI-candidate proof package** for **provenance-aware** reasoning — knowing *who wrote what*, *what happened when*, and *which tradition owns which idea* — before AGI-scale belief propagation.

**Thesis site:** https://tomyimkc.github.io/sophia-agi/

> *Sophia* (σοφία) = wisdom. Four domains active — philosophy, psychology, history, religion — plus a **three-path agent** (advisor, repo, life).

## AGI-candidate proof package

Sophia is **not claimed as proven AGI**. The stronger and more defensible public claim is:

> Sophia AGI is an AGI-candidate proof package for provenance-aware reasoning.

The proof package defines the operational AGI definition, pre-registered thresholds, current benchmark evidence, external benchmark gaps, ablation plan, hidden-reviewer protocol, long-horizon autonomy logs, learning-under-shift protocol, failure ledger, and third-party replication checklist.

- Evidence package: [agi-proof/README.md](agi-proof/README.md)
- Machine-readable manifest: [agi-proof/evidence-manifest.json](agi-proof/evidence-manifest.json)
- Religion figure council: [docs/08-Domains/Religion-Figure-Council.md](docs/08-Domains/Religion-Figure-Council.md)
- Public thesis chapter: https://tomyimkc.github.io/sophia-agi/#agi-proof

```bash
python tools/build_agi_proof_package.py
python tools/build_web_data.py
```

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

## Sophia Agent (3 paths)

```bash
pip install anthropic
python tools/sophia_agent.py advisor "Should I launch on HN this week?"
python tools/sophia_agent.py repo "What should I do next?" --execute --approve
python tools/sophia_agent.py life "Should I prioritize corpus or marketing?"
```

See [docs/09-Agent/Sophia-Agent.md](docs/09-Agent/Sophia-Agent.md).

## Online RAG (Gemini / Vertex)

Curated corpus retrieval (no open-web grounding) + Gemini generation + epistemic gate:

```bash
pip install -r requirements-rag.txt
python tools/build_rag_index.py
python tools/sophia_rag.py "Did Confucius write the Dao De Jing?"
```

Cloud Run API: `services/rag_api/` — see [docs/09-Agent/Online-RAG.md](docs/09-Agent/Online-RAG.md).

## Thesis web UI (council-decided)

Scholarly single-page site with chapter nav, UI council panel, live leaderboards, and optional agent API.

```bash
python tools/build_web_data.py   # refresh web/data/manifest.json
python tools/serve_web.py        # http://127.0.0.1:8765
```

- **Live site:** https://tomyimkc.github.io/sophia-agi/ (leaderboards + thesis; agent panel falls back to CLI hints).
- **Live agent:** `POST /api/ask` with `{ "mode": "advisor|repo|life", "question": "..." }` when `serve_web.py` runs (requires `ANTHROPIC_API_KEY` in `.env`).
- **Design record:** [docs/10-Web/UI-Council-Decisions.md](docs/10-Web/UI-Council-Decisions.md)

## Benchmarks (per-domain leaderboards)

| Domain | Cases | Leaderboard | Seed reference |
|--------|-------|-------------|----------------|
| Philosophy | 9 | [leaderboard-philosophy.json](benchmark/results/leaderboard-philosophy.json) | examples 001 + reference |
| Psychology | 4 | [leaderboard-psychology.json](benchmark/results/leaderboard-psychology.json) | examples 002, 005–007 + reference |
| History | 5 | [leaderboard-history.json](benchmark/results/leaderboard-history.json) | examples 003, 008, 012–013 + reference |
| Religion | 5 | [leaderboard-religion.json](benchmark/results/leaderboard-religion.json) | examples 004, 009–011, 014 (council panel) |

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
├── agi-proof/         # AGI-candidate proof package and evidence manifest
├── tools/             # validate, export, score, stats, serve_web
├── web/               # thesis UI (council-decided; GitHub Pages)
├── tests/             # attribution benchmark cases
└── huggingface/       # HF dataset card (upload corpus.jsonl)
```

## Domains

| Domain | Status | Data file |
|--------|--------|-----------|
| Philosophy | Active | `data/attributions.json` |
| Psychology | Active | `data/psychology_concepts.json` |
| History | Active | `data/history_events.json` |
| Religion | Active | `data/religion_concepts.json` |

See [docs/08-Domains/Overview.md](docs/08-Domains/Overview.md) and answer [Expansion-Questionnaire.md](docs/08-Domains/Expansion-Questionnaire.md) to shape the next domains.

## Roadmap & growth

- [**2026 Year-Top Roadmap**](docs/07-Growth/2026-Year-Top-Roadmap.md) — stars, authority, category ownership
- [Open Intelligence Plan](docs/06-Roadmap/Open-Intelligence-Plan.md)
- [90-Day Launch Playbook](docs/07-Growth/90-Day-Launch.md)
- [Good first issues](GOOD_FIRST_ISSUES.md)

## AI skills + MCP

| Layer | Command / path |
|-------|----------------|
| **Project skill** | `/sophia-agi` — `.grok/skills/sophia-agi/SKILL.md` |
| **Portable skill** | `/sophia-source-discipline` — `python tools/install_skills.py --all` |
| **MCP server** | `sophia_mcp/server.py` — 10 tools (validate, gate, benchmark, lookup) |

```bash
pip install -r requirements-mcp.txt
python tools/install_skills.py --all --cursor
```

See [Skills-Install.md](docs/09-Agent/Skills-Install.md) and [MCP-Server.md](docs/09-Agent/MCP-Server.md).

## Build your local LLM (Claude + LoRA)

Claude API builds data and packaging; **open weights** run offline:

```bash
python tools/claude_model_lab.py run-all          # review, distill, Ollama Modelfile
pip install -r requirements-lora.txt
python tools/train_lora.py --4bit --epochs 3      # Qwen2.5-7B QLoRA
python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1 --with-gate
ollama create sophia-7b -f models/ollama/Modelfile
```

See [Model-Lab.md](docs/09-Agent/Model-Lab.md) and [LoRA-Experiment.md](docs/09-Agent/LoRA-Experiment.md).

## Hugging Face

Dataset: [tomyimkc/sophia-agi-corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus) (**518** examples, synced from `training/corpus.jsonl`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changelog: [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
