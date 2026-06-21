# Changelog

All notable changes to Sophia AGI are documented here.

## [0.7.4] - 2026-06-21

### Added — The Provenance Delta benchmark (external, non-circular evidence)

The first measurement of what Sophia's provenance gate buys *against the outside
world*: how often a model asserts a false authorship lineage when used **alone**
vs **behind the gate**, scored on ground truth that is independent of the gate.
Targets claim-ladder items 6–7 (external evaluation, replication).

- **External ground truth** — `provenance_bench/data/misattributions.json`
  (cited FALSE lineage-merges) + `provenance_bench/data/wikidata_snapshot.json`
  (TRUE attributions, Wikipedia/Wikidata-sourced). Labels live in files
  physically separate from the gate's `doNotAttributeTo` corpus — the
  non-circularity guarantee.
- **Independent judge** — `provenance_bench/judge.py` shares **no code** with the
  gate (`agent/verifiers.py`); the gate is the runtime treatment, the judge is
  the referee. Default lexical screen + an optional independent-LLM-judge hook
  (`provenance_bench/llm_judge.py`).
- **Alone-vs-gated runner** — `provenance_bench/runner.py` produces a plain model
  answer and the same model behind `agent/guarded.py`, judging both.
- **Three honest metrics** — `provenance_bench/score.py`: hallucinated-attribution
  rate (alone vs gated; the **delta**), false-positive cost (does the gate break
  correct answers?), coverage/recall (does it name the gate's narrowness?).
- **Report + CLI** — `provenance_bench/report.py` and `tools/run_provenance_delta.py`
  (`--models`, `--llm-judge`, `--on-fail`, `--emit-dataset`). Optional Wikidata
  QID verification via `tools/fetch_wikidata_authors.py`.
- **Hard / obscure cases + gate-rule derivation** — added 7 verified
  spurious-attribution probes (Rhetoric to Alexander→Anaximenes, Epinomis→Philip
  of Opus, the Old Oligarch, Gallic War bk 8→Hirtius, De Mundo, the I Ching Ten
  Wings, the Donation of Constantine) plus hard true controls.
  `dataset.build_gate_records()` derives the gate's do-not-attribute rules from
  the cited misattributions (the realistic `SOPHIA_DISCIPLINE_RECORDS` path) so
  the gate can fire on the benchmark's works; the judge now handles
  scholarly-hedge / pseudonymity language and excludes claimed-author tokens when
  crediting gold (fixes a "Pseudo-Aristotle" name collision).
- **First real delta** — illustrative single run (lexical judge, 19 false cases):
  `deepseek` 0% alone (frontier models already faithful here) vs a weak local
  `dolphin-llama3:8b` 15.8% → **5.3%** behind the gate (Δ≈10.5), 0% false-positive
  cost, 67% coverage. See `docs/11-Platform/Provenance-Delta.md`.
- **Tests (TDD, offline)** — `tests/test_provenance_bench.py` (dataset, derived
  gate records, judge incl. scholarly hedges, runner alone-vs-gated, scoring,
  report) + a `--models mock` smoke run, both wired into CI.
- **Docs** — design spec
  (`docs/superpowers/specs/2026-06-21-provenance-delta-design.md`), platform doc
  (`docs/11-Platform/Provenance-Delta.md`), and a deliberately staged
  **what-to-do-next checklist**
  (`agi-proof/external-benchmarks/PROVENANCE-DELTA-CHECKLIST.md`).

### Notes

- Reuses the gate and guarded loop unchanged; no new runtime dependencies.
  Generated reports/datasets are git-ignored (regenerable) — only numbers from
  real, judged, multi-run passes should ever be published.

## [0.7.3] - 2026-06-21

### Added — Discipline layer (small-model source discipline, CPU-only)

A layer that lets any local/small model inherit Sophia's "never merge lineages"
discipline at run time, plus the data to train it in. All offline, no GPU for the
runtime paths (only the DPO *training* step needs one).

- **User-supplied records (Phase 0)** — `agent.verifiers._load_provenance_records`
  now also merges JSON records from the `SOPHIA_DISCIPLINE_RECORDS` env var
  (directory / glob / single file), so a user can enforce their OWN attribution
  rules (legal/corporate/code provenance) through the same machine-checked gate,
  beyond the seeded domains. Validation warnings on malformed/skipped records.
- **Guarded completion loop (Phase 1)** — `agent/guarded.py`: `guarded_complete()`
  wraps a model as retrieve → generate → judge (`provenance_faithful`) and, on a
  violation, branches by `SOPHIA_ON_FAIL` = `repair` (one bounded re-generation,
  else cited abstention) | `abstain` | `hedge` | `passthrough`. The cited
  abstention itself passes the gate. `check_claim()` is the mode-free verifier
  surface, exposed as the `sophia_check_claim` MCP tool.
- **Best-of-N reranker + belief graph + confidence injector (Phase 2)** —
  `agent/best_of.py` samples N candidates and ranks by the gate (early-exit on the
  first passing one); `okf.belief(entity)` exposes `effectiveConfidenceRank`
  (min-over-derivesFrom chain) with a `confidenceLaundered` flag, via the
  `sophia_belief` MCP tool; `harness._memory_recall` now annotates recalled pages
  with that effective (laundering-aware) confidence instead of face value.
- **Hard-negative DPO miner (Phase 3)** — `tools/mine_hard_negatives.py` mines
  every `doNotAttributeTo` edge into direct / sibling / alias / laundering
  negatives, each SELF-VALIDATED through `provenance_faithful` (rejected must trip
  the gate, chosen must pass), emitting the `wiki_to_training` DPO schema. CPU-only
  data gen; DPO training needs a GPU.
- **sophia-guard CLI (Phase 4)** — `tools/sophia_guard.py` runs any local model
  (ollama, llama.cpp, grok, openclaw, …) behind the guarded loop from the command
  line (`--on-fail`, `--provider`, `--json`).
- Tests (TDD, offline): `test_discipline_records`, `test_guarded`,
  `test_mcp_check_claim`, `test_okf_belief`, `test_best_of`,
  `test_memory_recall_confidence`, `test_mcp_belief`, `test_mine_hard_negatives`,
  `test_sophia_guard_cli` — all wired into CI.

### Notes

- Builds on the source-discipline gate (v0.7.2) and reuses the existing
  `doNotAttributeTo` corpus; the runtime paths add no new dependencies and stay
  3.9-safe in `okf/`. Generated DPO `.jsonl` is regenerable output and not committed.

## [0.7.2] - 2026-06-20

### Added

- **Source-discipline gate (Sophia → OpenClaw)** — `tools/source_discipline_cli.py`, a
  dependency-free, offline CLI that runs Sophia's `provenance_faithful` /
  `source_discipline` verifier (a ~2 ms local-regex check, no model call) over text on
  stdin and prints `{passed, reasons, violations}`. It is the bridge an OpenClaw
  `before_agent_finalize` plugin spawns to block agent replies that assert a forbidden
  lineage merge / hallucinated attribution — Sophia's "never merge lineages" rule now
  governs an external gateway's output.
- `tests/test_source_discipline_cli.py` (offline) — proves the forbidden-attribution case
  fails and the negation/debunk case passes across the CLI boundary; wired into CI.
- Design note: `docs/superpowers/specs/2026-06-20-source-discipline-gate-design.md` and
  `docs/11-Platform/Source-Discipline-Gate.md`.

### Notes

- Output-gating only; reuses the existing ~31-record `doNotAttributeTo` corpus (high
  precision, honestly narrow). Writes no knowledge and does not touch Sophia's provenance
  gate. The OpenClaw plugin lives outside this repo (`~/.openclaw/plugins/`). Independent of
  the OpenClaw model-provider work (v0.7.1, PR #9).

## [0.7.1] - 2026-06-20

### Added

- **OpenClaw model provider** — integrates the local [OpenClaw](https://github.com/openclaw/openclaw)
  multi-channel AI gateway as a Sophia model backend, behind a clean adapter that mirrors the
  existing `grok` CLI transport. New `openclaw` preset (default route `xai/grok-4.3`) +
  `_call_openclaw` transport in `agent/model.py`, shelling to `openclaw infer model run --json`;
  the `<provider>/<model>` route flows through as data (`openclaw:anthropic/claude-sonnet-4-6`).
- Read-only audited MCP tool `sophia_openclaw_infer` (`risk="low"`, no approval) in `sophia_mcp/`.
- `tests/test_model_openclaw.py` + `tests/test_mcp_openclaw.py` — fully offline (the `openclaw`
  binary is never invoked; `subprocess.run` is stubbed). Wired both, plus the previously-unwired
  `tests/test_model_adapter.py`, into CI.
- `docs/11-Platform/OpenClaw.md` design note; `SOPHIA_OPENCLAW_BIN` env override.

### Notes

- Inference plumbing only: stdlib-only, no new dependency, `okf/` untouched; degrades to `ok=False`
  when OpenClaw is absent so the stack stays offline-testable via the `mock` fallback. OpenClaw is
  never auto-selected — strictly opt-in. **No** knowledge-write path is added: any OpenClaw output
  destined for the wiki still passes the source-discipline (provenance) gate unchanged. OpenClaw's
  side-effecting `agent`/`message send` are deliberately **not** wired. Adds nothing to and makes
  no claim about the AGI-candidate proof package.

## [0.7.0] - 2026-06-20

### Added

- **OKF provenance wiki** — an Open Knowledge Format / LLM-Wiki layer that unifies
  `data/*.json` and the dispute pages into one machine-checkable belief graph.
- `okf/` package (dependency-free, 3.9+): frontmatter codec, schema, wikilinks, belief
  graph with contradiction detection + min-over-chain confidence propagation, linker.
- Provenance verifiers in `agent/verifiers.py` (`provenance_faithful` / `source_discipline`,
  `frontmatter_schema_valid`, `no_broken_wikilink`, `wiki_consistent`) — "never merge
  lineages" as a hard gate; zero false positives on the corpus, robust to phrasing bypasses.
- `tools/wiki_sync.py` (data → 58 OKF pages + CI drift gate), `tools/wiki_validate.py`,
  `tools/lint_wiki_provenance.py` (provenance falsifier), `tools/wiki_health.py`,
  `tools/wiki_ingest.py`, `tools/consolidate_runs.py`, `tools/wiki_to_training.py`,
  `tools/run_compounding_curve.py`.
- Librarian (`agent/wiki_librarian.py` + `wiki-maintenance` skill), gated
  `agent/wiki_store.py`, `agent/memory_consolidation.py`, and plan-time recall in the harness.
- Audited `sophia_wiki_*` MCP tools (read surface + permission-gated `wiki_upsert`).
- OKF frontmatter on the 10 `docs/04-Disputes/*.md`; `docs/11-Platform/OKF-Wiki.md`.
- Test suites: `test_okf`, `test_wiki_tools`, `test_wiki_librarian`, `test_wiki_mcp`,
  `test_memory_consolidation`, `test_wiki_proof`, plus CI wiring.

### Changed

- `agent/retrieval.py` carries provenance on `SourceChunk` and surfaces `doNotAttributeTo`
  at generation time; markdown readers strip OKF frontmatter.
- `data/schema.json` reconciled with corpus (`authorConfidence: layered` added) — a real
  pre-existing schema/data drift caught by the new OKF validator.

## [0.6.3] - 2026-06-19

### Added

- Religion figure council seats for Jesus traditions and Buddhist dharma
  traditions, grounded as source witnesses rather than impersonation.
- `data/religion_council_figures.json` plus Christianity/Buddhism source-seat
  docs and the Religion Figure Council guide.
- Hidden-reviewer pack schema, operating protocol, commitment generator, and
  hidden-eval scoring/template helper.
- `agi-proof/benchmark-results/` for visible and hidden evaluation artifacts.

### Changed

- Sophia prompts now route religion founder/scripture questions through a
  source-grounded council with no sacred-figure impersonation.

## [0.6.2] - 2026-06-19

### Added

- `agi-proof/` — AGI-candidate proof package with operational definition,
  pre-registered thresholds, external benchmark plan, ablation protocol,
  hidden-reviewer protocol, long-horizon autonomy plan, learning-under-shift
  protocol, failure ledger, and third-party replication checklist.
- `tools/build_agi_proof_package.py` — writes
  `agi-proof/evidence-manifest.json` from current repo evidence.
- GitHub Pages thesis chapter for the AGI-candidate proof package.

### Changed

- README and repo-about copy now describe Sophia as an AGI-candidate proof
  package while explicitly avoiding a proven-AGI claim.

## [0.6.1] - 2026-06-18

### Added

- LoRA v2 pipeline: paraphrase train examples `516–518`, `--resume-adapter` in `train_lora.py`, `tools/run_v2_pipeline.ps1`
- Correction loop proof: `training/corrections_pending/`, `tests/test_correction_loop.py`
- `tools/eval_rag_benchmark.py` — score curated RAG path on all 23 cases
- Gemini provider hook in `run_external_models.py` (requires `GOOGLE_API_KEY`)
- RAG benchmark runs: `rag-claude` leaderboards; `rag-auto` 3/3 on former LoRA gaps

### Changed

- `update_leaderboards.py` computes `score_pct` when missing
- Launch docs updated for v0.6.0 Reddit + GitHub release
- RAG index rebuilt (541 chunks)

## [0.6.0] - 2026-06-18

### Added

- **Online RAG** — curated corpus retrieval + Gemini / Vertex generation + epistemic gate
  - `agent/rag_sources.py`, `agent/vector_store.py`, `agent/rag_pipeline.py`
  - `agent/google_genai_client.py`, `agent/gemini_llm.py`, `agent/rag_embed.py`
  - `tools/build_rag_index.py`, `tools/sophia_rag.py`, `tools/deploy_rag_api.ps1`
  - `services/rag_api/` — FastAPI `POST /ask` for Cloud Run
  - `rag/index/chunks.jsonl` — **538** curated chunks (benchmark holdouts excluded)
  - [Online-RAG.md](docs/09-Agent/Online-RAG.md), `requirements-rag.txt`, `tests/test_rag_index.py`

### Changed

- `agent/retrieval.py` prefers `rag/index` when present (agent + web API)
- LoRA **sophia-v1** benchmark: **20/23 (87%)** after scorer fix; v2 train seeds `511–515`
- `training/lora/manifest.json` — 515 examples, 79 holdouts
- `models/ollama/Modelfile` — base `Qwen/Qwen2.5-3B-Instruct` (matches trained adapter)
- Thesis web UI — v0.6.0 stats, LoRA row, online RAG section

## [0.5.4] - 2026-06-18

### Added

- **Claude Model Lab:** `tools/claude_model_lab.py` + `tools/model_lab_lib.py`
  - `review-batch` — Claude QA on teacher examples
  - `distill` — gold answers for new attribution questions
  - `judge` — Claude judge on failed local benchmark runs
  - `write-modelfile` — Ollama Modelfile + HF adapter model card
  - `run-all` — orchestrated pipeline
- [Model-Lab.md](docs/09-Agent/Model-Lab.md), `tests/test_model_lab.py`

## [0.5.3] - 2026-06-18

### Added

- `tools/create_github_release.py` — publish release from CHANGELOG
- HF corpus sync (500 examples) + launch doc updates
- Portable user skill: `skills/portable/sophia-source-discipline/` (`/sophia-source-discipline`)
- `tools/install_skills.py` — install to `~/.grok/skills/` (+ optional `~/.cursor/skills/`)
- MCP expanded: attribution lookup, domain records, disputes, export corpus (10 tools total)
- `sophia_mcp/` package (renamed from `mcp/` to avoid pip clash), `tests/test_mcp_tools.py`
- [Skills-Install.md](docs/09-Agent/Skills-Install.md)

## [0.5.2] - 2026-06-18

### Added

- Grok project skill: `.grok/skills/sophia-agi/SKILL.md` (`/sophia-agi`)
- Sophia MCP server: `mcp/server.py` — validate, gate, benchmark list/score, corpus stats
- `docs/09-Agent/MCP-Server.md`, `requirements-mcp.txt`, `.cursor/mcp.json.example`

### Changed

- `tools/validate_attribution.py` exposes `run_validation()` for MCP

## [0.5.1] - 2026-06-18

### Added

- LoRA experiment pipeline: `prepare_lora_dataset.py`, `train_lora.py`, `eval_local_model.py`, `requirements-lora.txt`
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
