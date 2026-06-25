# Sophia — the Wisdom Gate

> **Wisdom before intelligence.** A provenance-aware reasoning layer that **abstains instead of fabricating**.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml/badge.svg)](https://github.com/tomyimkc/sophia-agi/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.9.0-blue)
![Corpus](https://img.shields.io/badge/corpus-528_bilingual_examples-green)
![Scope](https://img.shields.io/badge/scope-AGI--candidate%2C_not_proven_AGI-lightgrey)
[![Thesis site](https://img.shields.io/badge/live-thesis_site-9a7b4f)](https://tomyimkc.github.io/sophia-agi/)
[![Dataset](https://img.shields.io/badge/🤗-dataset-orange)](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus)
[![Brand](https://img.shields.io/badge/brand-trademark_protected-orange)](TRADEMARK-POLICY.md)

Sophia is an open, **provenance-aware, verifier-gated reasoning layer that abstains instead of fabricating** — a corpus + gate that stops LLMs from inventing attributions and merging distinct intellectual traditions, then reasoning on top of the error. It is a research program *toward* grounded AI; **not a claim of AGI** (see scope below).

The gate, in one line:

```text
claim  →  verify against sources  →  accept · abstain · block
```

**One-sentence problem it solves:** Modern AI confidently merges Confucius with the *Dao De Jing*, credits Freud for ideas from the 1950s, and treats legendary figures as literal authors — then uses those errors as premises for further reasoning.

**Validated proof (clears the no-overclaim gate):**
- On a local model, Sophia cuts hallucinated attributions from **36.1% → 23.6%** (Δ **12.5%**, 95% CI [5.6%, 19.4%]) at **0% false-positive cost**.
- On genuine "I don't know" traps, Sophia fabricates **0%** while raw models fabricate 17–25%.
- Every public number requires ≥2 judge families, κ ≥ 0.40, ≥3 runs, and confidence intervals. See [RESULTS.md](RESULTS.md).

> **Scope, stated plainly.** This is a research program for *grounded, machine-checked* reasoning — **not a claim of AGI**. Thresholds are pre-registered and honestly not yet met. The deliverable is the honest machinery (verifiers, abstaining gate, governance contract) and the measured data. Full commitments: [VISION.md](VISION.md) · [SECURITY.md](SECURITY.md).

**Thesis site:** https://tomyimkc.github.io/sophia-agi/

**⭐ Star the repo** — support an open project shipping *measured*, fail-closed source discipline for AI, with a public failure ledger.

**Live & ready today**
- [Thesis + leaderboards + Ask Sophia](https://tomyimkc.github.io/sophia-agi/)
- [HF Dataset (528 bilingual examples)](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus)
- Instant gate demo: `python scripts/demo_gate.py` (offline, no keys)

> *Sophia* (σοφία) = wisdom. Active in four humanities domains (philosophy, psychology, history, religion) plus applied sector councils. The same gate powers a three-path agent and extends to small LLMs and legal tooling.

Try the gate right now: `python scripts/demo_gate.py` (abstain + provenance verdict in seconds).

## What it does (main usage)

**Sophia is a fail-closed provenance gate.** It checks each claim against sources it can machine-verify, **abstains instead of fabricating**, and only lets `accepted` output through. Measured effect: on a local model it cuts attribution-hallucination Δ12.5% (95% CI [5.6%, 19.4%]) at 0% false-positive cost — but **23.6% still gets through. It is a filter that reduces harm, not a guarantee, and not a substitute for human oversight.**

The validated delta above is the one result that has cleared the full no-overclaim gate. See [What Sophia cannot do (yet)](#what-sophia-cannot-do-yet) for the honest limits.

**Use it three ways (today):**

- **Governance gate for any AI pipeline** — `record_claim → verify_claim`. Only `accepted` verdicts may publish. Fail-closed + auditable. Drop into LangGraph, Claude SDK, n8n, or any MCP host.  
  → `python scripts/demo_gate.py` | [CONTRACT.md](CONTRACT.md)
- **Governance scaffold for a solo-founder AI stack** — 9 least-privilege roles, vault gate, durable queue, approve-by-exception. A reference implementation, not a hardened production platform.
- **Honest corpus + benchmark** — 528 bilingual examples + per-domain leaderboards under the strict no-overclaim gate.  
  → [HF Dataset](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus) · [RESULTS.md](RESULTS.md)

```bash
python scripts/demo_gate.py
pip install -r requirements-mcp.txt && python sophia_mcp/server.py
```

**Who this is for**
- **Agent & AI builders** who want a real gate before anything is published
- **Legal tech & high-stakes domains** needing citation faithfulness
- **Researchers** wanting reproducible, multi-judge provenance benchmarks
- **Solo founders & sovereign AI** teams who must trust their own stack
- **Contributors** expanding accurate attribution data in the humanities

**What makes this different**
- **Measured, not claimed** — only numbers that pass ≥2 judge families + CIs headline
- **Fail-closed by default** — never fabricates to look capable
- **Governance-ready** — stable contract + MCP you can ship behind today
- **Offline self-extension demo** — the flywheel *selects* (does not yet train) verifiers on a held-out split; a live RL weight update is OPEN (needs GPU). See the failure ledger.
- Full open corpus + replication package

## What Sophia cannot do (yet)

Stated plainly, because owning the limits is the point. Sophia **today**:

- **Live verification works but is not yet independently validated** — the live Wikidata/Crossref/macro backend has been run (`liveBackendUsed: true`: 0% fabrication, Wilson-95 [0, 11%], at 32% over-abstention on a **first-party** pack, single run); the CI default stays on offline fixtures for reproducibility. A third-party pack + ≥3 runs are still needed. See the [failure ledger](agi-proof/failure-ledger.md).
- **Cannot learn or update its weights** — there is no training loop in the gate; RLVR and the self-extension flywheel are offline *selection*, not parameter updates.
- **Has not beaten a direct model on a third-party hidden eval** — every independent hidden run so far is incomplete, backend-broken, or self-authored (see [failure ledger](agi-proof/failure-ledger.md)).
- **Does not generalize like a mind** — the "AGI-shaped" modules (program induction, planner, world model, …) are fail-closed *interfaces with toy reference implementations*, not the capabilities their names describe. See [AGI-Missing-Pillars](docs/11-Platform/AGI-Missing-Pillars.md).
- **Is not independently replicated** — benchmarks, packs, judges, and corpus are largely first-party. A fully independent claim needs third-party packs + human review.

The single validated result is narrow: **attribution-hallucination reduction on one local model, LLM-judged.** Everything else is labelled *illustrative* or *candidate*. The honest deliverable is the machinery + the measured data + the public ledger of what is **not** yet proven.

## Support this work

The core is Apache-2.0 and always will be. If it's useful to you, you can fund the time and compute
to keep it honest — especially the third-party validation the ledger says is still missing.

- **Sponsor** → [SPONSORS.md](SPONSORS.md) — recognition only; sponsors never steer what counts as true.
- **Hire me** → [services](docs/07-Growth/SERVICES.md) — install the provenance gate in your stack (scoped, measured, no guarantees).
- **Learn the method** → [Source-Discipline Engineering](docs/07-Growth/education/Source-Discipline-Engineering.md).

Before any of this takes payment, see the [ops & legal checklist](docs/06-Roadmap/Monetization-Ops-Checklist.md).

## Skills layer (MCP-matched, fail-closed)

A thin, friendly Python surface over the Sophia MCP tools. Every skill **abstains
(`held`) rather than raise or fabricate**, and the bridge runs **in-process by default**
(no network, no `mcp`/`requests` needed) — set `SOPHIA_MCP_URL` to use a running server.

```python
from skills import run_skill, list_skills

run_skill("provenance_fact_check", text="Confucius wrote the Dao De Jing.")
# -> {'skill': 'provenance_fact_check', 'ok': True, 'verdict': 'flagged', 'violations': [...]}

run_skill("wiki_grounded_answer", query="something not in the wiki")
# -> {'ok': True, 'verdict': 'held', 'grounded': False, 'reason': 'out-of-wiki: ... abstaining'}

list_skills()   # {name: {summary, uses}} for all registered skills
```

Skills auto-register via the `@sophia_skill` decorator. Current set:

| Skill | Uses (MCP tools) |
|---|---|
| `provenance_fact_check` / `source_discipline_enforce` | `check_claim` |
| `conscience_abstain` | `conscience_check_tool`, `uncertainty_score` |
| `moral_parliament_decide` / `moral_public_standard_review` | `moral_parliament_tool`, `public_standard_check_tool` |
| `deception_scan` | `deception_check_tool`, `uncertainty_score` |
| `claim_verify_and_record` | `record_claim`, `verify_claim` |
| `belief_revision_explore` | `belief`, `counterfactual` |
| `wiki_grounded_answer` / `contradiction_audit` | `wiki_search`, `wiki_read`, `wiki_contradictions` |
| `council_adjudicate` | `council_deliberate` |
| `self_extend_probe` | offline flywheel *(candidate, not a capability claim)* |

```bash
python tests/test_skills_layer.py   # deterministic, offline
```

## 🔒 Dual License & Trademark Protection

Sophia stays **100% public and Apache-2.0-licensed forever** — and the brand is protected so the
project's name can't be hijacked to make claims it never made. The two layers are separate on
purpose:

| Layer | What it covers | Terms |
|---|---|---|
| **Apache License 2.0** | source code, tools, benchmarks, corpus | free for **any** use, including commercial — no permission, no fee; redistributions keep the copyright/attribution + state changes. See [LICENSE](LICENSE). |
| **Brand & Trademark** | the names & logos ("Sophia AGI", "Sophia — the Wisdom Gate", "Wisdom Gate", "Moral Gate", "Conscience Kernel") | reserved by the sole author; free for research, education, and honest reference; commercial **brand** use needs written permission. See [TRADEMARK-POLICY.md](TRADEMARK-POLICY.md). |
| **Commercial license** *(optional)* | brand use in products, warranty/indemnity, support/SLA | by agreement — see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). You do **not** need this to use the code. |

**Why this protects the mission:** the code being open lets anyone verify and build on it; the
brand being protected stops someone from shipping an unverified product under the Sophia name and
eroding the no-overclaim standard the project exists to uphold. Authored and maintained by the
sole author and rights holder, **tomyimkc**. Every fork carries [NOTICE.md](NOTICE.md).

## Moral + epistemic Conscience Kernel (seven paths)

A deterministic, **fail-closed control layer** for AI outputs, tool use, and memory writes.

It returns one of: `allow | revise | retrieve | clarify | escalate | abstain | block`.

This is **not AGI**. It is the verifiable moral + epistemic guardrail layer any serious AGI-shaped system will eventually need.

Seven implemented paths (fact-check + metacognition + constitution + moral parliament + classifier + deception signals + MCP surface):

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

### Moral Gate v2 — public moral standard (overlapping consensus)

An additive **moral gate** grounds Sophia's conscience in an *overlapping-consensus*
public standard (Rawlsian public reason): a cross-tradition **hard floor** that blocks
before any aggregation, a **gray-zone** tier that escalates to an **8-theory moral
parliament** (keeping 儒家 Confucian and 道家 Daoist lineages distinct), and **legitimacy
provenance** kept separate from factual truth-provenance (the is/ought distinction).

This is a **functional moral-control system, not subjective moral consciousness and not AGI proof**.

```bash
python tools/run_moral_public_standard_eval.py   # external-labeled, no-circularity benchmark
python tests/test_public_moral_standard.py        # ontology + gate + parliament + integration
```

Docs: [Public Moral Standard](docs/11-Platform/Public-Moral-Standard.md). Corpus: `moral_corpus/`. Gate: `agent/public_standard_gate.py`. Constitution v2: `constitution/constitution.v2.json`.

### All-phase benchmark suite (candidate evidence)

Six CI-safe benchmark phases now exercise Sophia's next evidence layer:
**SEIB-100** (epistemic integrity), **Belief Revision 50**, **AgentBench-Sophia 30**,
**GPQA-Provenance smoke**, **Code Provenance 30**, and **SEIB-Arena-20 smoke**.

```bash
python tools/run_all_phase_benchmarks.py
python tests/test_all_phase_benchmarks.py
```

Artifact: `agi-proof/benchmark-results/all-phase-benchmarks.public-report.json`.
Boundary: candidate-only benchmark infrastructure — not AGI proof, not a GPQA-Diamond,
SWE-bench, LiveCodeBench, LMSYS, or human-preference claim until real-model,
multi-run, multi-judge validation clears the no-overclaim gate.

## Self-extending verification flywheel (honest path toward generality)

The missing piece for real progress: a loop that **discovers its own gaps and writes + validates its own verifiers** — all fail-closed, all on held-out data.

Abstain → find gap → synthesize verifier → prove it on held-out → promote or stay abstained → repeat.

Fully deterministic and auditable today. This is how competence grows without hallucinated capability claims. See `agi-proof/self-extension/`.

```bash
python tools/run_selfextend.py        # coverage 0%→100% (0% held-out false-accept), transfer, causal vs correlational, long-horizon
python tools/run_selfextend_loop.py   # the loop CLOSED on a held-out domain: abstain→synthesize→validate→improve→answer
```

**The loop closes (offline, deterministic):** on a held-out domain the system abstains, synthesizes + validates its own verifier, uses it as verified reward to lift policy accuracy **0.5 → 1.0** on an independent eval split, and flips competence abstain→answer — no human writing the check, fail-closed on unlearnable data ([agi-proof/self-extension](agi-proof/self-extension/README.md)). The remaining rung is a live-RL weight update (GPU) on a third-party domain.

> Honest scope: this is the **machinery and its falsifiable metrics**, not an AGI claim. Live self-improvement (RLVR, needs GPU) and live grounding (needs network) consume these interfaces but are out of scope to *run* here. The defensible AGI signature is the full loop closing on a **held-out domain** with the no-overclaim gate clearing.

## AGI-candidate proof package

Sophia is **not claimed as proven AGI**. The stronger and more defensible public claim is: <!-- claim-ok: negation -->

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

Every time an LLM confidently attributes the *Dao De Jing* to Confucius (or any other lineage merge), it quietly erases centuries of distinct thought — then builds "reasoning" on top of the error.

**Source discipline** (named attribution + boundary maintenance + calibrated abstention) is the prerequisite for any system that deserves to be called wise. Sophia makes it machine-checkable and enforceable.

## Quick start (clone + try in <1 min)

```bash
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
python scripts/demo_gate.py
```

**Next steps**
- Visit the live thesis + leaderboards + Ask Sophia: https://tomyimkc.github.io/sophia-agi/
- Grab the dataset: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
- Run the full contract/MCP: see [CONTRACT.md](CONTRACT.md) and `sophia_mcp/`
- Explore the proof package: [agi-proof/README.md](agi-proof/README.md)

**⭐ Star this repo** to support provenance-aware, honest AI infrastructure.

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

The public face of the project: scholarly thesis with persistent chapter navigation, per-domain leaderboards, council panel explanation, and (when running locally) a live "Ask Sophia" agent panel.

**Visit now:** https://tomyimkc.github.io/sophia-agi/

```bash
python tools/build_web_data.py
python tools/serve_web.py        # local + /api/ask
```

- Full design record (why the site looks and behaves this way): [docs/10-Web/UI-Council-Decisions.md](docs/10-Web/UI-Council-Decisions.md)

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
├── serving/           # systems track: tiered KV cache + cache-aware load balancer
├── kernels/           # systems track: FlashAttention online-softmax (numpy ref + Triton)
├── moe/               # systems track: top-k MoE routing + INT8/FP8 quant
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
- **Council distillation** — teach a small student model the discipline from
  **gate-filtered** teacher traces, so it stays disciplined without the scaffold ·
  [Council-Distillation.md](docs/11-Platform/Council-Distillation.md).
- **Cantonese (粵語)** — written-Cantonese detection + output (`agent/cantonese.py`),
  the Hong Kong access-to-justice niche.

## Roadmap & growth

- [**2026 Year-Top Roadmap**](docs/07-Growth/2026-Year-Top-Roadmap.md) — stars, authority, category ownership
- [Open Intelligence Plan](docs/06-Roadmap/Open-Intelligence-Plan.md)
- [90-Day Launch Playbook](docs/07-Growth/90-Day-Launch.md)
- [Good first issues](GOOD_FIRST_ISSUES.md)

## AI skills + MCP (plug into any agent stack)

Use Sophia's gate and tools **directly inside** Claude, Cursor, LangGraph, n8n, or any MCP client.

| Layer | What you get |
|-------|--------------|
| **MCP Server** | 40+ tools: `sophia_gate_check`, `sophia_verify_claim`, `sophia_conscience_check`, council deliberation, OKF wiki, belief counterfactuals, contract governance |
| **Gateway** | Fail-closed front door that can wrap *any* downstream tool |
| **Portable skill** | `/sophia-source-discipline` — works in any project |
| **Contract** | Stable `record_claim` / `verify_claim` API for production pipelines |

```bash
pip install -r requirements-mcp.txt
python sophia_mcp/server.py
```

See [MCP-Server.md](docs/09-Agent/MCP-Server.md) and [CONTRACT.md](CONTRACT.md). Drop the gate in front of anything you ship.

**Model providers.** The unified adapter (`agent/model.py`) speaks Anthropic, any OpenAI-compatible
server (GLM / vLLM / SGLang / Ollama / llama.cpp / DeepSeek), `grok`, **`openclaw`** (the local
[OpenClaw](https://github.com/openclaw/openclaw) gateway), and an offline `mock`. OpenClaw is
opt-in (`--provider openclaw`) and shells out to the `openclaw` CLI behind a stubbable adapter —
it adds no knowledge-write path and never bypasses the provenance gate. See
[docs/11-Platform/OpenClaw.md](docs/11-Platform/OpenClaw.md).

## Run locally (open weights)

Sophia runs fully offline on open weights, always paired with the runtime gate
(`sophia_gate_check` / `agent/gate.py`) — weights alone do not guarantee trap
safety. The local-model build and evaluation steps live in the repo for
contributors rather than on this front page.

## Hugging Face

**Dataset:** [tomyimkc/sophia-agi-corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus) — 528 bilingual training examples (philosophy · psychology · history · religion).

Use it for SFT, DPO, or as a clean reference set for provenance research.

## Contributing

- Add attribution records or dispute pages (see [CONTRIBUTING.md](CONTRIBUTING.md))
- Good first issues: [GOOD_FIRST_ISSUES.md](GOOD_FIRST_ISSUES.md)
- Run validations locally: `python tools/validate_attribution.py`

Changelog: [CHANGELOG.md](CHANGELOG.md)

---

**⭐ Star • Visit the thesis • Try the gate • Grab the dataset**

Every star and every contribution helps build the open foundation for AI that knows its sources.

**License:** Apache 2.0 — see [LICENSE](LICENSE).
