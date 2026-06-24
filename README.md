# Sophia AGI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml/badge.svg)](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.7.42-blue)
![Training examples](https://img.shields.io/badge/training_examples-528-green)
![Domains](https://img.shields.io/badge/domains-philosophy%20%7C%20psychology%20%7C%20history%20%7C%20religion-purple)

**Wisdom before intelligence.** An open corpus, benchmark, and **verifier-gated reasoning loop** for **provenance-aware** reasoning — knowing *who wrote what*, *what happened when*, and *which tradition owns which idea*, and **refusing to assert what it cannot machine-check**.

> **Scope, stated plainly.** This is a research program for *grounded, machine-checked* reasoning — **not a claim of AGI**. It pre-registers AGI thresholds and measures itself against them honestly; **those thresholds are not met**. The deliverable is the honest machinery — verifiers, an abstaining loop, and a no-overclaim measurement gate — not the label. Every public number must clear that gate ([SECURITY.md](SECURITY.md), [RESULTS.md](RESULTS.md)). The mission and design commitments are stated in [VISION.md](VISION.md).

**Thesis site:** https://tomyimkc.github.io/sophia-agi/

⭐ **Star if you want LLMs that actually know who wrote what** — the foundation for the world's first wise AGI.

**Live links:** [Thesis + leaderboards + Ask Sophia](https://tomyimkc.github.io/sophia-agi/) • [HF Dataset (528 examples)](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus) • Try gate: `python scripts/demo_gate.py`

**Proof (validated):** Sophia gate fabricates **0%** on unknown-answer questions where raw models fabricate **17–25%** (DeepSeek subject, 3 runs, 2-judge families κ=0.74). Teacher/Grok-CLI: 100% on domain leaderboards. See [RESULTS.md](RESULTS.md).

> **10-sec demo GIF idea:** Run `python scripts/demo_gate.py` (or `python tools/serve_web.py` + agent query on "Did Confucius write the Dao De Jing?") and record the abstain + provenance verdict. Place in README / social preview assets.

> *Sophia* (σοφία) = wisdom. Four humanities domains active — philosophy, psychology, history, religion — plus a **three-path agent** (advisor, repo, life). The same verifier-gated core extends to **sector councils** (law · finance · economy) and to **disciplining small local LLMs** — see [Applied verticals](#applied-verticals--the-same-gate-beyond-the-humanities-corpus).

## What it does (main usage)

**Sophia is the provenance gate that makes AI safe to ship:** it verifies every AI claim against its sources, **abstains instead of fabricating**, and only lets *accepted* output through — so a solo operator can run AI services without babysitting.

> **Validated proof point:** the one result that clears the no-overclaim gate — on a local 8B model (`dolphin-llama3`) Sophia cuts hallucinated attributions **36.1% → 23.6%** (Δ **12.5%**, 95% CI [5.6%, 19.4%], N=24) at **0% false-positive cost**, judged by **two independent families** (DeepSeek + Llama-3.3-70B) across 3 runs ([RESULTS.md](RESULTS.md)). A separate **self-authored** 18-case calibration pack shows 0% fabrication vs 17–25% raw (corroborated by GPT-4o + Claude, κ=0.74) — reported as *calibration evidence, not the headline*, until a third-party pack lands.

**Use it three ways:**

- **As a governance gate for any AI pipeline** — a versioned MCP/Python contract your code pins against: `record_claim → verify_claim` returns `accepted | rejected | superseded | held`; **only `accepted` may be published** (fail-closed), with Bell-LaPadula classification, budget caps, and a kill switch. Drop it into LangGraph, the Claude Agent SDK, or n8n. → [CONTRACT.md](CONTRACT.md). Roadmap: the **super-MCP / super-skills** gateway that gates *any* tool → [docs/11-Platform/Sophia-Gateway.md](docs/11-Platform/Sophia-Gateway.md)
- **As the spine of a one-person AI company** — 9 least-privilege role pipelines, an Obsidian vault that only publishes *accepted* notes, a durable task queue, an approve-by-exception review queue, and Langfuse traces. → `sophia_contract/`
- **As an honest reasoning corpus + benchmark** — provenance-aware QA across philosophy, psychology, history, religion, under a no-overclaim measurement gate (≥2 judge families, κ, ≥3 runs, CI). → [RESULTS.md](RESULTS.md)

```bash
python scripts/demo_gate.py     # Sophia in 30s: verify → classify → abstain → publish-only-if-accepted (offline, no key)
```

## Moral + epistemic Conscience Kernel (seven paths)

Sophia now exposes a **candidate Conscience Kernel**: a deterministic, fail-closed control layer that decides when an AI output/tool/memory action should `allow | revise | retrieve | clarify | escalate | abstain | block`. It is **not a claim of AGI**; it is the moral/epistemic guardrail layer around AGI-shaped autonomy.

Seven implemented paths:

1. **Unified conscience gate** — `agent/conscience.py`, `tools/run_conscience_demo.py`.
2. **Metacognition** — uncertainty typing, self-consistency, semantic-entropy proxy, P(True)/P(IK) hooks.
3. **Constitution + deontic rules** — via-negativa prohibitions and hard action rules for AGI overclaim, reward/verifier tampering, hidden-eval leakage, and unverified trusted-memory writes.
4. **Moral parliament** — bounded moral-uncertainty aggregation for gray zones.
5. **Constitutional classifier** — fast input/output screen derived from the constitution.
6. **Deception signals** — confidence/evidence mismatch, source laundering, gate tampering, and internal-vs-stated contradiction hook.
7. **MCP conscience surface** — `sophia_conscience_check`, `sophia_uncertainty_score`, `sophia_constitution_check`, `sophia_deontic_check`, `sophia_deception_check`, `sophia_moral_parliament`, `sophia_conscience_benchmark`.

```bash
python tools/run_conscience_demo.py        # deterministic seven-path conscience benchmark
python tools/build_conscience_proof_package.py  # aggregate seven-priority conscience evidence
python tools/run_agi_missing_pillars.py    # program induction, active inference, MCTS, world model, plasticity, layered memory
```

Docs: [Conscience Kernel](docs/11-Platform/Conscience-Kernel.md) · [AGI Missing Pillars](docs/11-Platform/AGI-Missing-Pillars.md). Artifacts: `agi-proof/conscience/`, `agi-proof/agi-kernel/`.

## Self-extending verification flywheel (the path-to-AGI engine)

`selfextend/` connects Sophia's static pieces into a loop that **grows its own competence**: *abstain → localize the gap → synthesize a verifier → validate it on held-out data → promote only if it clears the bar (else stay abstained) → coverage rises → repeat* — no human writing the new checks, no reward-hacking. Plus the components that loop needs: calibrated uncertainty (ECE/Brier), a competence self-model, a **causal world model** (do-operator, beyond provenance), cross-domain transfer, environment-as-verifier (verify by executing), the verified-reward signal with an anti-gaming held-out check, and a long-horizon runner with recovery. Deterministic, offline, falsifiable.

```bash
python tools/run_selfextend.py        # coverage 0%→100% (0% held-out false-accept), transfer, causal vs correlational, long-horizon
python tools/run_selfextend_loop.py   # the loop CLOSED on a held-out domain: abstain→synthesize→validate→improve→answer
```

**The loop closes (offline, deterministic):** on a held-out domain the system abstains, synthesizes + validates its own verifier, uses it as verified reward to lift policy accuracy **0.5 → 1.0** on an independent eval split, and flips competence abstain→answer — no human writing the check, fail-closed on unlearnable data ([agi-proof/self-extension](agi-proof/self-extension/README.md)). The remaining rung is a live-RL weight update (GPU) on a third-party domain.

> Honest scope: this is the **machinery and its falsifiable metrics**, not an AGI claim. Live self-improvement (RLVR, needs GPU) and live grounding (needs network) consume these interfaces but are out of scope to *run* here. The defensible AGI signature is the full loop closing on a **held-out domain** with the no-overclaim gate clearing.

## AGI-candidate proof package

Sophia is **not claimed as proven AGI**. The stronger and more defensible public claim is:

> Sophia AGI is an AGI-candidate proof package for provenance-aware reasoning.

The proof package defines the operational AGI definition, pre-registered thresholds, current benchmark evidence, external benchmark gaps, ablation plan, hidden-reviewer protocol, long-horizon autonomy logs, learning-under-shift protocol, failure ledger, and third-party replication checklist.

- Evidence package: [agi-proof/README.md](agi-proof/README.md)
- **Conscience Kernel:** [docs/11-Platform/Conscience-Kernel.md](docs/11-Platform/Conscience-Kernel.md), `agi-proof/conscience/` — seven-path moral + epistemic gate; candidate-only control infrastructure, not AGI proof.
- **Missing-pillars mechanisms:** [docs/11-Platform/AGI-Missing-Pillars.md](docs/11-Platform/AGI-Missing-Pillars.md), `agi-proof/agi-kernel/` — program induction, active inference, MCTS planning, predictive world model, safe plasticity, and layered memory.
- **Generality track + verifier synthesis:** [docs/11-Platform/Generality.md](docs/11-Platform/Generality.md), [docs/11-Platform/Verifier-Synthesis.md](docs/11-Platform/Verifier-Synthesis.md) — the verifier-gated loop reused beyond provenance, plus a loop that **writes and trust-tests its own checks** and **abstains** when it cannot (the honest direction *toward* generality, with falsifiable metrics)
- **Public results (honest, gated):** [RESULTS.md](RESULTS.md) — only multi-judge-validated numbers headline; transparency boundary in [SECURITY.md](SECURITY.md)
- Machine-readable manifest: [agi-proof/evidence-manifest.json](agi-proof/evidence-manifest.json)
- Religion figure council: [docs/08-Domains/Religion-Figure-Council.md](docs/08-Domains/Religion-Figure-Council.md)
- Public thesis chapter: https://tomyimkc.github.io/sophia-agi/#agi-proof

```bash
python tools/build_agi_proof_package.py
python tools/build_web_data.py
```

## OKF provenance wiki (new in 0.7.0)

An **Open Knowledge Format / LLM-Wiki** layer that turns Sophia's scattered provenance
(`data/*.json` + `docs/04-Disputes/*.md`) into **one version-controlled, machine-checkable
belief graph** — because Sophia's differentiator, *source discipline*, literally **is** the
frontmatter (`authorConfidence`, `doNotAttributeTo`, `doNotMergeWith`, `tradition`).

- **`okf/`** — dependency-free package: frontmatter codec, schema, wikilinks, a belief
  **graph** with contradiction detection + min-over-chain confidence propagation, plus
  **counterfactual queries** (*"what would I conclude if this source were removed?"*) and
  first-class, auditable **retraction** (`okf/counterfactual.py`,
  `tools/belief_counterfactual.py`).
- **Provenance gate** (`agent/verifiers.py:provenance_faithful`) — encodes "never merge
  lineages" as a hard, machine-checked verifier (catches "Confucius wrote the Dao De Jing"
  across many phrasings; passes the dispute pages that *correctly debunk* such merges).
- **58 OKF pages** generated from `data/*.json` (`tools/wiki_sync.py`, data stays source of
  truth, CI fails on drift); the 10 dispute pages gained OKF frontmatter.
- **Librarian + memory** — `agent/wiki_librarian.py` ingests raw sources into gated drafts;
  `agent/memory_consolidation.py` folds verified runs into provenance-gated memory the
  planner recalls (continual learning without retraining).
- **Flywheel + proof** — `tools/wiki_to_training.py` (provenance SFT/DPO),
  `tools/wiki_health.py`, `tools/run_compounding_curve.py`, audited `sophia_wiki_*` MCP tools.

```bash
python tools/wiki_sync.py emit          # data/*.json -> 58 OKF pages
python tools/wiki_validate.py           # schema + links + contradictions + drift
python tools/lint_wiki_provenance.py    # provenance falsifier: 0 forbidden attributions
```

See [docs/11-Platform/OKF-Wiki.md](docs/11-Platform/OKF-Wiki.md).

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
├── data/              # attributions, domains, schema + sector-council figures
├── docs/              # disputes, growth playbook, domains, platform/verticals
├── agent/             # verifier-gated core, council deliberate, gate, model
│   └── legal_sources/ # federated HK/UK/US live citator (HKLII, e-Leg, TNA, CL)
├── benchmark/         # responses template + leaderboard + gated harnesses
├── training/          # JSONL-ready examples + gate-filtered council traces
├── agi-proof/         # AGI-candidate proof package and evidence manifest
├── tools/             # validate, export, score, council + uplift + distill
├── scripts/           # ops helpers (e.g. safe one-way iCloud backup)
├── web/               # thesis UI (council-decided; GitHub Pages)
├── tests/             # attribution + verifier + council + legal cases
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

## Applied verticals — the same gate, beyond the humanities corpus

The verifier-gated, abstaining core is domain-general, so the same machinery extends
past the philosophy/psychology/history/religion corpus. These are **applications of
the core gate**, not a separate project — each reuses the no-overclaim rule (a number
is "validated" only with multi-judge consensus + CIs; see [RESULTS.md](RESULTS.md)).

- **Sector councils (law · finance · economy)** — hard, contested questions are
  modelled as constrained, source-inspired *seats* with always-on guardians
  (citation auditor, jurisdiction/freshness, ethics, human-review gate).
  `data/{law,financial,economy}_council_figures.json` ·
  [Sector-Councils.md](docs/08-Domains/Sector-Councils.md).
- **Council deliberation for small LLMs** — `agent/council_deliberate.py` runs each
  seat as one focused pass, **gates each**, then synthesises (map-reduce): a weak
  model becomes a disciplined, tool-checked reasoner. The uplift is *measured*, not
  assumed (`tools/run_council_uplift.py`) ·
  [Council-For-Small-LLMs.md](docs/11-Platform/Council-For-Small-LLMs.md).
- **Legal-AI application** — the gate aimed at the legal industry's defining risk
  (hallucinated / misstated citations): `legal_citation_exists` (existence,
  fail-closed) + a federated **HK/UK/US live citator** (`agent/legal_sources/`) +
  a semantic **holding-faithfulness** tier. Sophia's **first gate-validated number**
  lives here (RESULTS.md → *Semantic evals*), with honest small-N bounds. Not legal
  advice · [Legal-Industry-Fit.md](docs/08-Domains/Legal-Industry-Fit.md).
- **Council distillation** — teach a small student (Qwen2.5-7B) the discipline from
  **gate-filtered** teacher traces, so it stays disciplined without the scaffold ·
  [Council-Distillation.md](docs/11-Platform/Council-Distillation.md).
- **Cantonese (粵語)** — written-Cantonese detection + output (`agent/cantonese.py`),
  the Hong Kong access-to-justice niche.

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
| **MCP server** | `sophia_mcp/server.py` — 32 tools (validate, gate, check_claim, belief, counterfactual, retract, revise, benchmark, lookup, sector councils, council deliberate, OKF wiki, OpenClaw infer, governance contract: record_claim/verify_claim/explain_verdict/describe/health/enqueue_task/next_task) |
| **Gateway MCP** | `gateway/server.py` — super-MCP front door (`gateway_call_tool`, `gateway_verify`, downstream HTTP-MCP federation, verified consensus, Skill Forge) |

```bash
pip install -r requirements-mcp.txt
python tools/install_skills.py --all --cursor
python gateway/server.py  # standalone fail-closed Sophia Gateway MCP
```

See [Skills-Install.md](docs/09-Agent/Skills-Install.md) and [MCP-Server.md](docs/09-Agent/MCP-Server.md).

**Model providers.** The unified adapter (`agent/model.py`) speaks Anthropic, any OpenAI-compatible
server (GLM / vLLM / SGLang / Ollama / llama.cpp / DeepSeek), `grok`, **`openclaw`** (the local
[OpenClaw](https://github.com/openclaw/openclaw) gateway), and an offline `mock`. OpenClaw is
opt-in (`--provider openclaw`) and shells out to the `openclaw` CLI behind a stubbable adapter —
it adds no knowledge-write path and never bypasses the provenance gate. See
[docs/11-Platform/OpenClaw.md](docs/11-Platform/OpenClaw.md).

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

Dataset: [tomyimkc/sophia-agi-corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus) (**527** examples, synced from `training/corpus.jsonl`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changelog: [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
