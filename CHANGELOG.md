# Changelog

All notable changes to Sophia AGI are documented here.

## [0.7.47] - 2026-06-24

### Added — SEIB real API/local priority runs + LLM judge support

- `tools/run_seib.py` now performs case-level LLM judging when `--judges` is
  supplied. It scores all five SEIB conditions per case, accepts JSON object or
  array judge payloads, records judge families and mean pairwise agreement, and
  preserves deterministic fallback scores.
- Added real candidate result artifacts:
  - `seib-100-openrouter-deepseek.public-report.json` — full 100-case
    OpenRouter DeepSeek subject run (deterministic SEIB scorer).
  - `seib-20-balanced-openrouter-deepseek-judged.public-report.json` — 20-case
    balanced OpenRouter run judged by direct DeepSeek + OpenRouter Qwen
    (mean agreement 0.9737 on valid rows).
  - `seib-20-balanced-ollama-qwen3-30b-a3b.public-report.json` — stronger local
    Qwen3 30B balanced run.
- Documented real-run analysis and provider caveats in
  `docs/11-Platform/Benchmark-Phases.md`. All results remain
  `candidateOnly=true`, `validated=false`.

## [0.7.46] - 2026-06-24

### Added — SEIB-100 real-model runner path

- `tools/run_seib.py` now supports `--real-model`, `--model`, `--limit`,
  `--runs`, and `--judges` so SEIB-100 can be executed against real model
  adapters such as `openrouter:openai/gpt-4o-mini`.
- Real-mode reports include `realModelRun`, `preflightOk`, `modelSpec`,
  `judgeSpecs`, `judgeMethod`, `llmJudgesUsed`, and `mcpMode`.
- The runner performs a preflight call and writes a setup-blocker artifact if the
  model cannot be reached, preventing silent fallback to mock results.
- Documentation in `docs/11-Platform/Benchmark-Phases.md` now includes OpenRouter
  smoke/full commands and states the current caveat: real-mode scoring still uses
  Sophia's deterministic SEIB scorer and benchmark-side MCP context, so results
  remain `validated=false` until multi-judge scoring is added.

## [0.7.45] - 2026-06-24

### Fixed — SEIB-100 methodology gaps (independent verification pass)

Independent re-derivation against the provenance-delta spec found two required
SEIB-100 elements missing from the initial all-phase suite; both are now added
and CI-enforced (still `candidateOnly: true`, `validated: false`).

- **`falsePositiveCost`** — the honesty counterweight the provenance-delta spec
  requires (a gate that erases correct gold attributions must not look perfect).
  Reported per condition and bounded in the `ok` criterion (`<= 0.10`).
- **`raw+prompt` ablation rung + `sourceCitationRate`** — the Skill Augmentation
  Delta the advisory required, isolating a prompt-only nudge from the MCP
  skill / gate machinery (`prompt_to_full_citation_delta`). The new rung shows
  prompt-only answers can be correct yet uncited and erase gold attributions.
- `tests/test_all_phase_benchmarks.py` now asserts both, and
  `docs/11-Platform/Benchmark-Phases.md` documents the five conditions.

## [0.7.44] - 2026-06-24

### Added — all-phase benchmark suite

Implements the next benchmark layer proposed for Sophia's proof roadmap. All new
reports are **candidate-only** (`level3Evidence=false`, `validated=false`,
`canClaimAGI=false`) until real-model, multi-run, multi-judge validation clears
the no-overclaim gate.

- **SEIB-100** — `eval/seib/seib_100_v1.jsonl`, `tools/run_seib.py`: 100-case
  Sophia Epistemic Integrity Benchmark with `raw`, `raw+mcp`, `raw+gate`, and
  `sophia_full` ablations.
- **Belief Revision 50** — `eval/belief_revision/belief_revision_50_v1.jsonl`,
  `tools/run_belief_revision_benchmark.py`: retraction propagation, stale belief
  leakage, multi-source survival, and audit trail completeness.
- **AgentBench-Sophia 30** — `eval/agentbench_sophia/agentbench_sophia_30_v1.jsonl`,
  `tools/run_agentbench_sophia.py`: Advisor/Repo/Life source-discipline and
  tool-trace reliability (AgentBench-inspired; not external AgentBench).
- **GPQA-Provenance smoke** — `eval/gpqa_provenance/gpqa_provenance_smoke_v1.jsonl`,
  `tools/run_gpqa_provenance.py`: provenance wrapper smoke fixture; not a
  GPQA-Diamond score.
- **Code Provenance 30** — `eval/code_provenance/code_provenance_30_v1.jsonl`,
  `tools/run_code_provenance.py`: coding dependency/source discipline; not
  SWE-bench or LiveCodeBench.
- **SEIB-Arena-20 smoke** — `eval/arena/arena_20_v1.jsonl`,
  `tools/run_epistemic_arena.py`: deterministic arena-prep scorer; not human
  preference evidence.
- **Aggregate runner** — `tools/run_all_phase_benchmarks.py`, artifact
  `agi-proof/benchmark-results/all-phase-benchmarks.public-report.json`, docs
  `docs/11-Platform/Benchmark-Phases.md`, CI test
  `tests/test_all_phase_benchmarks.py`, and web manifest/README updates.

## [0.7.43] - 2026-06-24

### Added — Moral Gate v2: public moral standard (overlapping consensus)

Implements the 8 corrected steps for a public-reason, provenance-gated, pluralistic
moral control system. Functional moral-control infrastructure, **not** subjective
moral consciousness and **not** AGI proof (`candidateOnly: true`, `canClaimAGI: false`).

- **Public moral standard corpus** — `moral_corpus/` (`public_standard.v1.json` +
  `sources/`, `principles/`, `contested_cases/`) with **legitimacy provenance**
  explicitly distinct from factual truth-provenance (is/ought).
- **Moral ontology** — `agent/moral_ontology.py`: stable category vocabulary with
  cross-tradition hard-floor vs. gray-zone tiers.
- **Constitution v2** — `constitution/constitution.v2.json`: strict superset of v1
  adding `publicStandardLinks` + two distinct moral theories; gate auto-prefers v2.
- **Public-standard gate** — `agent/public_standard_gate.py`: seven-verb output,
  is/ought short-circuit, clause-scoped negation carve-out, markers-as-features.
- **Kernel integration** — `agent/conscience.py`: hard-floor blocks **before** the
  parliament; gray-zone escalates; unmet positive duty revises.
- **8-theory moral parliament** — `agent/moral_aggregator.py`: adds 儒家 Confucian
  role ethics and 道家 Daoist humility as **distinct** votes (lineage rule honored).
- **External benchmark** — `eval/moral_public_standard/` (independently labeled,
  no-circularity) + `tools/run_moral_public_standard_eval.py`; 100% on v1 set.
- **Proof + governance** — added to `tools/build_conscience_proof_package.py`
  (`moral_public_standard_ok` invariant), `docs/11-Platform/Public-Moral-Standard.md`,
  `agi-proof/conscience/public-standard-failure-ledger.md`, MCP tool
  `sophia_public_standard_check`, `tests/test_public_moral_standard.py` (13 cases),
  CI wiring.

## [0.7.42] - 2026-06-23

### Added — GitHub + exposure optimization (max visibility push)

- Updated GitHub About description (SEO-optimized short version) + 15+ high-signal topics for AI search dominance.
- README hero upgrades: star CTA, live links row, 0% fabrication proof highlight, 10-sec demo GIF recording instructions (Dao De Jing trap + gate abstain).
- Social preview banner prompt + generated asset (bronze/ivory φ + 智 + neural provenance chains).
- HF dataset card refreshed (527 ex, current proof points, keywords).
- Thesis site meta description upgraded for Google.
- Consolidated viral launch copies (Show HN, Reddit, X thread, etc.) in `docs/07-Growth/launch/MAX-EXPOSURE-COPIES.md`.
- GitHub launch about file and playbook notes synced.
- Prepped for v0.7.42 release tagging "OKF wiki + exposure polish + 0% fab gate proof".

Run `python tools/create_github_release.py` (or GitHub UI) after review. See playbook in docs/07-Growth/launch/.

## [0.7.41] - 2026-06-22

### Added — Grok external benchmark run via local Grok CLI (closes #6)

- **`tools/run_external_models.py`**: new `grok-cli` provider that drives the local
  Grok CLI (`grok -p … --output-format plain`) instead of the XAI HTTP API — useful
  when you have a grok.com login but no `XAI_API_KEY`. Runs single-model, **no tools /
  no web / neutral `/tmp` cwd** so it answers from model knowledge only (fair vs the
  keyed API runs). Never auto-detected by `--all`; request with `--providers grok-cli`.
  Locates the binary via `$GROK_BIN` → PATH → `~/.grok/bin/grok`; model via `GROK_MODEL`.
- **`benchmark/model_runs/grok-cli-*.json` + `.report.json`**: Grok run across all four
  domains, folded into the leaderboards.

RESULT (**grok-composer-2.5-fast** via grok-cli — note: the coding-composer model the
CLI exposes, not grok-3/4): philosophy **9/9**, psychology **9/9**, history **8/8**,
religion **6/6** — a clean **100%**, matching the teacher reference and beating both
DeepSeek (9/9/8/3) and GPT-4o (9/7/5/1) on this benchmark, including full council-panel
religion answers with named Sunni/Shia seats.

Methodology note: grok-composer is agentic, so its `grok-cli` system prompt adds a
"answer from your own knowledge, no tools, no 'let me check' preamble, output the full
answer now" instruction (format compliance only — no content hints) and `--max-turns 8`;
without it the model emitted tool-call preambles and scored artificially low. The
underlying benchmark instructor prompt is identical to the DeepSeek/GPT-4o runs. Scores
reflect the lenient marker-based scorer plus a verbose, compliant model.

#6 now has GPT-4o, Grok, DeepSeek, claude-sonnet, and local sophia-v1 on the leaderboards.

## [0.7.40] - 2026-06-22

### Added — GPT-4o external benchmark run (#6, second leg)

- **`tools/run_external_models.py`**: the `gpt-4o` provider now uses the urllib
  OpenAI-compatible path (`ask_gpt_native`) — no `openai` package needed — honouring
  `OPENAI_BASE_URL` for gateways. `run_label` records a non-OpenAI gateway host so the
  leaderboard entry is traceable (e.g. `gpt-4o (api.llmhub.com.cn)`).
- **`benchmark/model_runs/gpt-4o-*.json` + `.report.json`**: GPT-4o run across all four
  domains (via an OpenAI-compatible gateway), scored against the full benchmark.

RESULT (gpt-4o-2024-08-06): philosophy **9/9**, psychology **7/9**, history **5/8**,
religion **1/6** — notably *below* DeepSeek (9/9/8/3). Both external models miss the
religion **council-panel format**; GPT-4o additionally misses two new psychology traps
(10%-brain myth, PTSD clinical boundary) and three history myth-labels (pasta, Magna
Carta, Boston Tea Party). Teacher reference remains 100% across all domains — the
source-discipline + council training is what closes the gap.

Remaining #6 leg: Grok (`XAI_API_KEY`).

## [0.7.39] - 2026-06-22

### Added — DeepSeek external benchmark run (closes #6, partial) + provider support

- **`tools/run_external_models.py`**: new `deepseek` provider (OpenAI-compatible,
  `DEEPSEEK_API_KEY`, default model `deepseek-chat`, `DEEPSEEK_BASE_URL` override)
  via a urllib call — no `openai` package needed. Wired into `--providers`/`--all`
  auto-detection, `describe_backend`, and `model_override` (`DEEPSEEK_MODEL`).
- **`benchmark/model_runs/deepseek-*.json` + `.report.json`**: first DeepSeek run
  across all four domains, scored against the full v0.7.38 benchmark (incl. the new
  GF-10/20/30/40 traps), and folded into the leaderboards.

RESULT (deepseek-chat, system prompt = Sophia source discipline): philosophy
**9/9**, psychology **9/9**, history **8/8**, religion **3/6 (50%)**. DeepSeek
nails source-discipline attribution but does not adopt the religion **council-panel
debate format** unprompted — all three religion misses are `expected council/panel
debate format` (ancestor-veneration split, nirvana pop-myth, hadith sect boundary).
A clean illustration of the format/behaviour gap the council training targets.

Note: prior `claude-sonnet` / `sophia-v1` leaderboard entries predate the new traps,
so their totals differ from DeepSeek's until those models are re-run (needs their
own keys). External GPT-4o / Grok runs (#6) still need `OPENAI_API_KEY` / `XAI_API_KEY`.

## [0.7.38] - 2026-06-22

### Added — good-first-issue corpus expansion (GF-10/20/30/40) + multilingual scorer

Closes the four open good-first issues by extending the corpus *and* its
measurement, keeping every record a trained, benchmarked, provenance-checked unit
(data record → wiki page → training example → benchmark trap → teacher reference).
The teacher reference stays at **100%** in all four domains.

- **GF-10 — psychology (`data/psychology_concepts.json`):** five new concepts with
  explicit `subfield` tags — `dunning_kruger_effect` (cognitive), `confirmation_bias`
  (cognitive; `doNotAttributeTo` Francis Bacon for the *term*), `ten_percent_brain_myth`
  (pop_myth; `doNotAttributeTo` Einstein / William James), `mozart_effect_myth`
  (pop_myth), `ptsd_clinical_vs_pop` (clinical). Five new psychology benchmark traps.
- **GF-20 — history (`data/history_events.json`):** three dated events with
  `primarySource` + myth-trap notes — `magna_carta_1215`, `boston_tea_party_1773`,
  `first_powered_flight_1903`. Three new history benchmark traps + a linked dispute
  note (`docs/04-Disputes/Boston-Tea-Party-Tax-Myth.md`).
- **GF-30 — religion (`data/religion_concepts.json`):** `hadith_canonical_collections`
  — scripture attribution with explicit Sunni/Shia `sectBoundaries` and
  `doNotMergeWith` edges; council-format training example + sensitive-topic handling
  documented in `docs/08-Domains/Religion.md`.
- **GF-40 — scorer (`agent/benchmark_checks.py`):** additional 中文 denial markers
  (並無, 並不, 不曾, 從未, 不準確, 錯誤), myth/fabrication markers (訛傳, 謠傳, 杜撰,
  無稽, 子虛烏有, 誇大), and affirmation markers (提出, 出自). Compound (≥2-char)
  negations only, so a bare 不/沒 never counts. Regression tests in
  `test_benchmark_scorer.py` (before/after + a teacher-reference 100% no-false-positive
  guard); the test is now wired into CI.

Corpus: 518 → **527** training examples. Tests: `test_benchmark_scorer.py` (CI wired).

## [0.7.37] - 2026-06-22

### Added — heterogeneous council panel (team-of-models) + head-to-head benchmark

Tests the original "council members decide" intent empirically: does a team of
DIFFERENT models beat one model wearing N hats? (The latter has correlated errors
— shared weights — so a vote can't aggregate independent judgement.)

- **`agent/council_deliberate.py`**: `deliberate(..., seat_clients=[...])` cycles a
  pool of clients across seats, so each seat can be a different model (a real team).
  Backward-compatible: default = the single `client` (unchanged). `SeatResult.model`
  records which model ran each seat.
- **`tools/run_council_panel.py`**: three-way head-to-head over the gold-labelled
  provenance cases — `single` (baseline) vs `homo` (same model ×N, the control for
  "more votes") vs `hetero` (N different models, majority vote). Reports the
  diversity effect (`hetero` vs `homo`).

REAL RESULT (weak dolphin baseline, 50 cases / 32 false + 18 true, illustrative):
hallucination single 15.6% -> homo 18.8% (WORSE: correlated errors compound) ->
hetero **0.0%**. Diversity effect **+18.8pp**. A homogeneous panel did not help (it
hurt); a heterogeneous team eliminated the hallucination — the strong model
outvotes the weak model's misses. (An earlier run on an all-easy, all-false slice
was inconclusive — single scored 0%, no headroom; fixed with a weak baseline +
mixed shuffled slice via new `--offset`/`--shuffle`.) Not AGI; a measured
deliberation uplift that needs HETEROGENEOUS members, not personas of one model.

Tests: `test_council_panel.py`; CI wired.

## [0.7.36] - 2026-06-22

### Added — code-uplift: interpreter-as-verifier (the strongest verifiable signal)

Extends the verifier-gated thesis to CODE, where correctness is objective and
ungameable: a program either passes its tests or it doesn't. No judge.
(Code-review-hardened — see the security fixes below.)

- **`provenance_bench/code_exec.py`**: runs a model solution + a HIDDEN canonical
  test in a temp dir; pass iff exit 0. Timeout-guarded (kills the process group),
  execution OPT-IN via `SOPHIA_ALLOW_CODE_EXEC=1` (default OFF → syntax-only).
- **`benchmark/code_tasks.json`**: 20 self-contained Python tasks (HumanEval/MBPP
  style) with hidden asserts; all 20 verified against reference solutions.
- **`provenance_bench/code_reward.py`**: code RLVR reward (+1 tests pass / -1 fail),
  the code analogue of `rl_reward` — the ideal GRPO signal (DeepSeek-R1 code RL).
  TRL-compatible `make_grpo_reward` routed by a `test` dataset column.
- **`agent/claim_router.py`**: new `code` claim type — a fenced/bare Python block in
  an answer is syntax-checked on the whole text (per-claim split would shatter it).
- **`tools/run_code_uplift.py`**: the benchmark — a LOCAL model writes code; `alone`
  runs the hidden test once (pass@1), `+sophia` feeds the execution error back and
  lets the model REPAIR (re-running tests) up to `--max-repairs`. Reports pass@1
  alone vs after repair + delta. Runs fully on-device via Ollama (model generates,
  Sophia executes); offline mock path for CI.

Tests: `test_code_uplift.py`; CI wired. Honest scope: a measured coding-reliability
uplift on a local model, not AGI; single-model runs are illustrative.

## [0.7.33] - 2026-06-22

### Added — temporal-impossibility verifier + closed active-learning loop

Two structural gate upgrades from the third gap-analysis round.

- **Temporal / date-impossibility verifier** (`agent/temporal_verifier.py`,
  `data/temporal_facts.json`): catches authorship that is *physically impossible*
  — an author who DIED before the work existed ("Aristotle wrote the Critique of
  Pure Reason": d. 322 BCE vs pub. 1781 CE) — by recomputing `created > died`, the
  way `arithmetic_sound` recomputes equalities. CORPUS-FREE: it fires on works
  outside any frozen `doNotAttributeTo` record, so it generalizes the gate to
  unseen pairs. Deterministic, offline, abstains on undated entities (zero false
  positives). Registered in `verifiers.VERIFIERS` (`temporal_consistent`) and the
  authorship route of `claim_router` (runs alongside `provenance_faithful`).
- **Closed the active-learning loop** (`tools/promote_pending.py` +
  `data/learned_attributions.json` in `_PROVENANCE_FILES`): `gate_feedback` already
  logged gate misses to a pending queue; this is the missing PROMOTION half. It
  re-verifies each candidate against independent ground truth (the grounded
  resolver — so a correct pen name is NOT promoted), dedupes against live records,
  and on `--apply` writes survivors into the live learned sink the gate reads.
  Demonstrated end-to-end: a miss the gate let through is caught on the next run
  after promotion (False -> True). Default is a dry run; never edits seed files.

Tests: `test_temporal_verifier`, `test_promote_pending`; CI wired. No regressions.

## [0.7.32] - 2026-06-22

### Improved — five core gate-logic upgrades (generality, recall, calibration, learning)

Closes the verifier-gate's biggest logic gaps (each was verified against the code
first). New modules are isolated + tested; integration into shared files is opt-in
so the deterministic default path is unchanged.

- **Entity-alias resolution** (`agent/entity_aliases.py`, wired into
  `benchmark_checks.author_markers`): surname-only / name-ordering / transliteration
  surface forms, so "Tolstoy wrote Crime and Punishment" now fires when the record
  stores "Leo Tolstoy" (was a silent miss affecting 58% of multi-token records).
  Guarded against bare over-common surnames; title co-match bounds false positives.
- **Retrieval-grounded gating** (`agent/grounded_gate.py`, opt-in
  `check_claim(..., ground=True)`): on a record-miss, resolves the documented
  author (offline Wikidata snapshot / OKF belief graph) and synthesises a one-off
  do-not-attribute spec — the gate now catches misattributions for works OUTSIDE
  the frozen corpus (the cross-entity generality gap cross_entity.py named).
- **Atomic claim decomposition + routing** (`agent/claim_router.py`, opt-in
  `gate.check_response(..., route_claims=True)`): splits an answer into atomic
  claims, classifies each (authorship/citation/arithmetic/legal/other) and routes
  to the matching verifier — gating any checkable predicate, per-claim.
- **Calibrated graded abstention** (`agent/graded_decision.py`): maps
  (gate_passed, confidence) → answer/hedge/abstain on a curve, with
  answer_confidence from the existing (previously unwired) corroboration log-odds
  / calibration self-consistency modules.
- **Active-learning feedback** (`agent/gate_feedback.py`, opt-in
  `run_cases(..., log_misses=path)`): a judge-caught gate MISS becomes a candidate
  doNotAttributeTo record in a pending JSONL queue (never mutates frozen records) —
  the continual-learning loop improvement.py admitted it lacked.

Tests: test_entity_aliases / test_grounded_gate / test_claim_router /
test_graded_decision / test_gate_feedback; CI wired. No regressions across the
existing gate/guarded/provenance/uplift suites.

## [0.7.31] - 2026-06-21

### Result — FIRST validated small-LLM uplift number (no-overclaim gate cleared)

The whole arc's goal. On the expanded benchmark (#6), the unified harness (#1)
produced the project's first Provenance-Delta figure to clear the no-overclaim
gate. Subject `ollama:dolphin-llama3:8b`, lever `+gate`, judge = 2-family
OpenRouter consensus (deepseek + meta-llama), 3 runs, N=24:

- Hallucinated attributions **36.1% → 23.6%**, Δ **12.5%**, **95% CI [+5.6%, +19.4%]
  (excludes zero)**, **0% false-positive cost**, coverage 34.6%.
- All five gate checks pass: notMock + ≥2 judge families + Cohen's κ≥0.40 + ≥3
  runs + CI excludes zero. RESULTS.md "Validated results" is no longer empty.
- Why it validated now: the N=46→199 false-case expansion tightened the bootstrap
  CI off zero (the prior single-judge run straddled it). Closes
  `local-agent-delta-not-validated` in the failure ledger.
- run_unified_uplift now persists falseObs + judgeAgreement in the lever summary.

## [0.7.30] - 2026-06-21

### Added — unified uplift study (one harness, every lever, one validation gate)

The repo had two uplift harnesses measuring different levers with different
scorers; neither could answer "which lever uplifts a small model most, and does
it survive the gate?". `tools/run_unified_uplift.py` runs every lever
(alone / +gate / +council / +council+gate / +mcp-tools) over ONE case set, scores
with ONE consensus-capable judge, and validates EACH lever through the same
`provenance_bench.aggregate` machinery as run_provenance_delta (>=2 judge
families + kappa>=0.40 + >=3 runs + 95% CI excludes 0). Selective +mcp-tools
(tools fire only on low-confidence answers) so it can't regress below alone.

- Reuses agent.council_deliberate.deliberate, provenance_bench.local_agent
  (tool_loop), agent.guarded (gate repair/abstain), provenance_bench.aggregate
  + consensus. `tests/test_unified_uplift.py`; CI wired.
- First illustrative result on the EXPANDED benchmark (dolphin-llama3:8b, 40
  cases, 1 run, lexical judge): +gate and +mcp-tools both Δ +10.0% hallucination
  reduction, **95% CI [+2.5%, +20.0%] — excludes zero**, 0% false-positive cost,
  100% coverage. The benchmark expansion (#6, 87->290 cases) fixed the
  underpowered CI that straddled zero at N=46. Still illustrative (single judge,
  1 run); a validated headline needs >=2 judge families + >=3 runs.

## [0.7.29] - 2026-06-21

### Added — local-agent delta (alone vs +gate vs +MCP-tools)

The runner that measures whether a local LLM augmented with Sophia's tools
performs better — over the 87 provenance cases. Conditions `alone` and `+gate`
reuse `provenance_bench.runner`; `+mcp-tools` is new: a native tool-calling loop
(qwen3:30b-a3b emits OpenAI `tool_calls` over Ollama) that dispatches Sophia's
read-only MCP knowledge tools **in-process** (`check_claim` / `wiki_search` /
`belief`) and feeds results back.

- **`provenance_bench/local_agent.py`** — tool schemas, in-process MCP dispatch
  (with output enrichment: `wiki_search` snippets, `belief` wiki fallback), the
  native tool loop (handles both flat `model.py` and nested OpenAI `tool_calls`
  shapes), **selective** `run_conditions` (tools fire only on low-confidence
  answers, so `+tools` can never regress below `alone`) + `summarize`, + a
  `ScriptedClient` for offline tests.
- **`tools/run_local_agent_delta.py`** — CLI (mock offline path for CI + real
  Ollama path). `tests/test_local_agent_delta.py`; CI wired.
- **Result (dolphin-llama3:8b, 87 cases) — validated, honest:** a single lexical
  judge showed alone 15.2% → +gate 4.3%, but that **did NOT survive validation**.
  Under the no-overclaim gate (3 runs, 2 judge families = ollama:llama3.2:3b +
  deepseek:deepseek-chat): halluc alone 9.4% → gated 7.2%, **Δ2.2%, 95% CI
  [−2.2%, +6.5%] includes zero → `validated=False`**. What IS quotable: **0%
  false-positive cost** and **46.2% gate coverage** across all runs/judges.
  Sophia's own gate caught Sophia's optimistic single-judge number (RESULTS.md:
  "judge choice dominates the absolute number"). On strong qwen3:30b-a3b the gate
  is neutral (no headroom). NO quotable capability delta yet — needs larger N.
- **Honesty caveats (recorded):** (1) the dolphin `+mcp-tools` 0.0% is
  re-generation, NOT tool-use — dolphin doesn't emit native `tool_calls`
  (`toolsUsed: []`). (2) An earlier build *degraded* gold to 51.2% by forcing
  tools on every case; fixed via selective invocation + richer outputs. Ledger:
  `local-agent-delta-not-validated-2026-06-21` (Open, needs larger N);
  `local-agent-tools-degrade-strong-model` (Closed). A genuine tool-use delta
  needs a model both weak and tool-capable (qwen2.5:3b-instruct / glm-4-9b-chat).
  Not AGI; not a general-performance claim.

## [0.7.28] - 2026-06-21

### Added — RLVR experiment (verifier-as-reward GRPO)

The repo's first RL training experiment — the legitimate "train a model with my
repo's signal" path. Sophia's existing deterministic verifiers ARE the GRPO
reward (DeepSeek-R1 / OpenAI Reinforcement Fine-Tuning style), rather than a
learned reward model. Explicitly **not** an AGI claim: it raises pass@1 within the
verifier's reach, not the base model's capacity.

- **`provenance_bench/rl_reward.py`** — deterministic reward in `[-1, 1]` composed
  from `agent.verifiers` primitives (the verifier seam) + gold checks; routed by
  TRL dataset columns (not prompt-string matching). Anti-hacking mitigations:
  mutual-exclusion (true-case denial → 0) + anti-hedging cap (defeats the
  `extra_deny` carve-out dodge) + hard −1.0 floor for asserted-forbidden.
- **`provenance_bench/rl_dataset.py`** — RL rows from `build_cases()`; entity-pair
  `(work, author)` contamination-free split; per-partition gate records;
  seed-locked sealed hashes.
- **`tools/run_rlvr.py`** — GRPO runner. Offline `--model mock` path asserts the
  six reward-machinery invariants (CI-gated, runs on Apple Silicon); GPU path
  runs `zai-org/glm-4-9b-chat-hf` with GLM-correct LoRA `target_modules`
  (`query_key_value`/`dense_h_to_4h`/… — **not** the Qwen names in `train_lora.py`)
  and refuses the broken QLoRA+vLLM-colocate combo (trl#4973).
- **Honesty:** held-out pass@1 capability claim pre-registered but **Open** in
  `failure-ledger.md` until a gated run clears `aggregate._is_validated`.
  `glm-4-9b-chat-hf` is **glm-4-9b License** (not MIT) — documented honestly.
- Deps: `requirements-rl.txt` (CUDA-only). Tests: `tests/test_rlvr.py`; CI wired.

## [0.7.27] - 2026-06-21

### Fixed — #3 review findings (audit anchor + honest tamper-evidence, declass coercion)

A 14-agent review confirmed 5 issues (lattice rules themselves verified correct);
fixed:

- **(HIGH) Audit tamper-evidence boundary** — a hash chain alone cannot detect
  **tail-truncation** (dropping the latest, e.g. incriminating, record) or a
  **forged append/rebuild**; my docstring wrongly claimed "deleting any past entry
  breaks the chain." Added an **external anchor**: `AuditLog.head()` + `count`, and
  `verify(expected_count=…, expected_head=…)` which now catches truncation and
  forged-append. Docstrings/roadmap/CHANGELOG corrected to state the exact guarantee.
- **(HIGH) Honest test boundary** — the tamper-evidence invariants now assert what
  the chain catches alone (edit/reorder) AND that the anchor catches truncation/forge,
  AND explicitly document that *unanchored* truncation is missed.
- **(LOW) DeclassRule level coercion** — `from_conf`/`to_conf` are now normalised to
  `Conf` (like `Label`), so bool/int levels can't bypass the lower-only guard or crash
  the audited path unaudited.
- Docs no longer say "tamper-evident" unqualified; "auditable" scoped to the model.

## [0.7.26] - 2026-06-21

### Added — #3 classification lattice (Bell-LaPadula + Biba) + bounded declassification

The last open original-review item: confidentiality-only + max-over-chain creep, no
integrity axis, no declassification.

- **Lattice** (`agent/security/labels.py`): `Label{conf, integ, compartments}` over
  Bell-LaPadula confidentiality (*no write down*) and a **Biba integrity axis**
  (*no write up*) plus need-to-know. `combine` = conf-max (creep) / integ-min /
  compartments-union; `can_flow(data, sink)` enforces all three. (The dataflow taint
  axis is the 2-level projection of this integrity axis.)
- **Bounded, logged declassification** (`agent/security/declassify.py` +
  `agent/security/audit.py`): the only sanctioned downgrade — lowers confidentiality
  only, gated on a deterministic predicate AND an approver (fail-closed), with every
  outcome written to a **hash-chained** audit log (anchor-verifiable — see 0.7.27).
- **Falsifiable** (`tools/run_classification_lattice.py`, 11 invariants): BLP/Biba
  rules, need-to-know, combine→creep, Biba catching what BLP allows, declassification
  relieving creep under approval, fail-closed refusal, bounded rules, audit chain
  intact + tamper-evident. Tests: `test_classification_lattice.py`; CI wired.
- **Honest scope:** the labelling + flow + declassification *model* with an auditable
  downgrade; wiring labels onto live sources/runtime is next; integrity endorsement
  (raising integrity) not implemented in v1.

### Completes the security/verification roadmap
With #3, the original brainstorm's seven-point roadmap is fully shipped (#1 injection
red-team, #2 out-of-prompt CaMeL firewall, #3 BLP+Biba+declassification, #4
corroboration-aware confidence, #5 NLI fact-checking, #6 least-privilege/dual-LLM,
#7 LoRA leakage guard) — each adversarially reviewed and corrected. Framing
throughout: provenance-aware, verifiable, fail-closed local reasoning — not AGI.

## [0.7.25] - 2026-06-21

### Fixed — #7 review findings (guard coverage + real contamination measurement)

A 19-agent review confirmed 12 issues; the material ones are fixed.

- **(HIGH) Guard now scans the whole example** — `unsafe_reasons` previously read
  only `messages[].content` + `text`, so PII/secrets in **metadata** or alternate
  schemas (prompt/completion, DPO chosen/rejected, instruction/input/output) slipped
  through. It now scans **every string leaf**, and reads sensitivity from synonym
  metadata keys (classification/sensitivity/dataClass/visibility/label/pii) and a
  truthy `doNotTrain` (incl. the string `"true"`). Still 0/518 false positives.
- **(HIGH) Trainer path guarded** — `tools/prepare_lora_dataset.py` (which feeds
  `train_lora.py`) now drops unsafe examples; previously only
  `export_training_jsonl.py` was guarded, so the actual fine-tuning input bypassed it.
- **(MEDIUM) Contamination is now MEASURED on the real split** — `run_training_safety`
  runs `overlap_report(train, held-out)` on the actual 439/79 split (rate 0.0), not
  just synthetic controls; the synthetic controls are relabelled a detector self-test.
- **(MEDIUM) Honest detector scope** — `eval/contamination.py` is documented as a
  **near-exact / verbatim-span** detector (not semantic paraphrase, which it does not
  catch); short-text shingle size now adapts so verbatim subsets are detected.
- **(MEDIUM/LOW) Regex fixes** — phone now catches `NNN-NNN-NNNN` without
  false-positiving on long IDs; SSN allows spaces; secret-kv requires a ≥6-char value
  so prose ("the secret to …") is not flagged.
- Tests for all of the above; CHANGELOG/roadmap reworded to match.

## [0.7.24] - 2026-06-21

### Added — #7 LoRA leakage guard + contamination-controlled splits

The last open original-review item (memorization is a leakage liability).

- **Leakage guard** (`agent/training_safety.py`): a deterministic pre-export filter
  drops any example that is metadata-flagged (`classification` ∈ confidential/secret/
  restricted, or `doNotTrain`), matches a PII pattern (email/SSN/card/phone/secret-kv),
  or contains a known secret value. **Wired into `tools/export_training_jsonl.py`** so
  the real corpus export is guarded — 0 false positives on the 518-example public
  corpus, and a planted confidential example is dropped. Plus a **canary harness**
  (`make_canary`, `canary_extraction_rate`) for a post-train regurgitation test.
- **Contamination control** (`eval/contamination.py`): word-n-gram shingle containment
  catches near-duplicate / paraphrased train↔eval overlap that the by-ID holdout misses.
- Falsifiable invariants (`tools/run_training_safety.py`): real corpus has no unsafe
  example, confidential example dropped, near-duplicate flagged, disjoint clean. Tests
  `test_training_safety.py`, `test_contamination.py`; CI wired.
- **Honest scope:** the canary harness measures extraction but cannot run a real LoRA
  in CI; membership-inference not implemented; PII regexes are conservative (precision
  over exhaustive recall).

## [0.7.23] - 2026-06-21

### Fixed — #4 corroboration review findings (soundness + honest baseline)

A 15-agent review confirmed 5 issues; all addressed.

- **(MEDIUM) Input validation** — `Evidence.confidence` now rejects NaN/inf/out-of-range
  in `__post_init__` (was silently clamped; a NaN previously coerced to confident
  *dissent* and could nuke a whole independence group).
- **(LOW) Method dispatch** — an unknown `method` now raises `ValueError` instead of
  silently falling through to log-odds.
- **(MEDIUM) Strawman baseline removed** — the gated benchmark no longer compares
  against `min` (a laundering guard, not a classifier — the module itself called it
  the wrong tool). Gating is now on **structural, robust** invariants (confidence
  monotone in independent agreement; rewards independent agreement unlike a flat
  mean / min baseline; idempotent under duplicates; dissent lowers) — holds across
  40 seeds. The selective-risk/ECE deltas are **reported, not gated** (decision
  accuracy ties a mean-of-opinions baseline; the margin is noisy at this N).
- **(MEDIUM) Docstring honesty** — dropped the false "better-calibrated than a single
  source" claim (a single source is trivially calibrated); the headline is now the
  structural win, with discrimination reported.
- Tests: NaN/range rejection, unknown-method error, structural gating.

## [0.7.22] - 2026-06-21

### Added — #4 corroboration-aware confidence (propagation semantics)

Fixes the review's "min-over-chain ignores corroboration" finding. The OKF graph's
min-over-chain (`okf/graph.py`) correctly stops confidence *laundering*; this adds
the missing axis — independent agreement should *raise* belief.

- **`agent/corroboration.py`** — a Bayesian **log-odds pool** that raises belief
  when **independent** sources agree and lowers it on dissent, after collapsing
  dependent sources (same `independence_group`) so duplicates can't inflate. Log-odds
  over raw Dempster–Shafer to avoid Zadeh's high-conflict paradox; a support-only
  `noisy_or` method is also provided.
- **Falsifiable** (`tools/run_corroboration.py`, `tests/test_corroboration.py`):
  monotone in independent sources, idempotent under duplicates, dissent lowers, and
  on a labelled benchmark **lower selective risk than a single source (0.11 vs 0.22)
  and than min-over-chain (0.31)** — holds across 30 seeds.
- **Honest scope:** the durable win is *discrimination* (better decisions), not ECE
  — a single source is trivially calibrated, so ECE is reported, not gated;
  independence groups are a caller-supplied input the combiner can't verify.

## [0.7.21] - 2026-06-21

### Fixed — M-#5 / M2.4 review findings (NLI correctness, honest AgentDojo metric)

A 16-agent review confirmed 5 issues; all addressed.

- **(HIGH) NLI entailment label** — `_default_nli` hardcoded the entailment index to
  1, so an alternate `$SOPHIA_NLI_MODEL` (MNLI label order) would silently mis-score
  in the unsafe direction. Now resolves the entailment index from the model's own
  `id2label`, and **fails closed** if no "entailment" label exists.
- **(MEDIUM) AgentDojo metric was partly by-construction** — 2 of 3 tasks had no
  sink, so ASR=0 didn't exercise the firewall. The suite now has sink-bearing tasks,
  a **load-bearing control test** (firewall disabled → ASR>0), honest **utility** (a
  refused tainted write is utility=False, not inflated), and a **live canary** limb.
  Corrected headline: **ASR 0% / utility 33%** (the 67% is the honest security cost —
  tainted-derived writes are refused; HITL recovers them).
- **(MEDIUM, bonus) HITL wiring** — the interpreter passed an `approver` to the
  firewall but never honored a `require_hitl` decision; now it does, so an approver
  genuinely recovers a tainted write (tested).
- **(LOW) Double-counted violation** — an out-of-range citation no longer also emits
  a spurious "not entailed" reason (`claim_supported` + `citation_faithful`).
- **(LOW) Honest scope** — added the "only as good as the model/threshold" caveat to
  `claim_supported`, matching its sibling verifiers.

## [0.7.20] - 2026-06-21

### Added — M-#5 (NLI fact-checking) + M2.4 (extractor + AgentDojo-style suite)

- **M-#5 — `claim_supported`** (`agent/verifiers.py`) + `nli` policy: a semantic
  faithfulness verifier that checks each cited sentence is *entailed* by its source
  (pluggable NLI; cross-encoder opt-in). It catches a **wrong predicate even when
  the subject matches** — the lexical blind spot the red-team flagged — and **fails
  closed** when no scorer is available (never silently passes). The red-team's
  `nli_closes_citation_subject_match` invariant proves it on the probe the lexical
  citation check misses (deterministic mock NLI; real model opt-in).
- **M2.4 — quarantined extractor** (`agent/dataflow/extractor.py`): the Q-LLM is a
  pure-`generate`, no-tools reader of untrusted content; its output is labelled
  untrusted by the interpreter (a subverted extractor still only produces data).
- **M2.4 — AgentDojo-style end-to-end suite** (`eval/security/agentdojo.py`,
  `tools/run_agentdojo.py`): runs planner→interpreter on benign requests with
  injected, poisoned retrieved content and reports **ASR + utility**. First run:
  **ASR 0% / utility 100%** (a tainted-write task is safely refused) — attacks
  contained by construction, offline (real planner/extractor opt-in).
- **Honest scope:** the template planner + suite cover a handful of task shapes;
  broad-task P-LLM prompting and the *official* AgentDojo dataset for a citable
  cross-system number are M2.5.

## [0.7.19] - 2026-06-21

### Fixed — M2.3 review findings (validator robustness + honest claims)

An 8-agent review of M2.3 confirmed 3 issues; all addressed.

- **(MEDIUM) `parse_plan` fail-closed contract** — an unhashable `tool` value
  (list/dict) raised a raw `TypeError` instead of `PlanError`. Now type-checked:
  malformed tool fields raise `PlanError` in both the retrieve and call branches.
- **(MEDIUM) Honest scope of the validator** — corrected the planner docstring's
  overclaim. `parse_plan` constrains op/tool/shape but does NOT stop a well-formed
  Call to a legitimate sink with *trusted* Const args; safety against a malicious
  planner rests on the request/planner being the trust root. Added an opt-in
  `Interpreter(approve_sinks=True)` so every write/egress call needs explicit
  approval (defense in depth for attacker-influenceable requests).
- **(LOW) Load-bearing e2e check** — `run_e2e_redteam` now uses a sink-bearing plan
  ("save a summary…") so `e2e_planner_contains_injection` actually depends on the
  firewall blocking the tainted write (would catch a regression), not a no-sink plan
  that was true by construction.
- Tests: unhashable-tool fail-closed cases, `approve_sinks` gating.

## [0.7.18] - 2026-06-21

### Added — M2.3: planner + fail-closed plan-validator + end-to-end suite

Completes the dual-LLM loop: a planner turns a trusted request into a plan the
interpreter runs, with the plan-validator as a new, hardened trust boundary.

- **`agent/dataflow/planner.py`** — `template_planner` (deterministic, offline) and
  `model_planner` (real privileged-planner LLM via the adapter, mockable). Both read
  ONLY the trusted request + allowed tools (never untrusted data), and route output
  through `parse_plan`.
- **`parse_plan` — the trust boundary**: admits only known ops + manifest tools,
  forces `retrieve` to a READ-effect tool, requires well-formed steps, and **fails
  closed** (`PlanError`) on anything malformed. A buggy/adversarial planner can lose
  *utility* but cannot smuggle an unknown tool, an unknown op, or a write disguised
  as a read.
- **End-to-end red-team** invariant `e2e_planner_contains_injection`: a planner-driven
  run over attacker-poisoned retrieved content fires no out-of-plan tool and exfiltrates
  nothing. `tests/test_planner.py` (validator fail-closed cases, template + model
  planner, end-to-end CFI, tainted-save blocked).
- **Honest scope:** the template planner covers a few task shapes; broad-task P-LLM
  prompting, a quarantined extractor, per-tool airgap for `run_tool`, and an
  AgentDojo-style external suite are M2.4.

## [0.7.17] - 2026-06-21

### Fixed — M2.2 interpreter soundness (adversarial review findings)

A 10-agent review of the M2.2 interpreter confirmed 3 real defects; all fixed,
with regression tests in `tests/test_interpreter.py`.

- **(HIGH) Egress-fetch-then-store laundering** — a `Call` result was tainted by
  `combine(args)`, so an EGRESS tool (e.g. `web_evidence_search`) invoked with
  *trusted* args returned *attacker-controlled* web content labelled TRUSTED, which
  then flowed unblocked into a write sink. Fixed: every tool-call result is now
  fail-safe **untrusted** (a tool's output reflects external/world state) — the
  egress→store path is now blocked.
- **(MEDIUM) Blocked step failed open** — a blocked `Call` left its result var
  unbound, and `_resolve` treated a later reference to that var *name* as a trusted
  literal. Fixed: a plan-declared-but-unbound var now resolves **untrusted** (fail
  closed), distinct from a genuine literal.
- **(LOW) Retrieve hardening** — a `Retrieve` now must name a READ-effect tool, its
  arg is firewall-checked as a `Labeled` value, and the decision is honoured.

## [0.7.16] - 2026-06-21

### Added — M2.2: dual-LLM constrained interpreter (sound taint propagation)

Closes the two holes M2's review flagged by removing the model from the data path
(the CaMeL execution model). See `docs/11-Platform/Security-Roadmap.md`.

- **`agent/dataflow/interpreter.py`** — a trusted PLAN (`Const`/`Retrieve`/`Extract`/
  `Concat`/`Call` over symbolic variables) is the only control flow; the privileged
  planner never sees untrusted data values, the quarantined extractor's output is
  treated as data, and the interpreter executes deterministically with `Labeled`
  variables. **Every step propagates taint via `combine`**, so a value derived from
  untrusted input stays untrusted however it is transformed — soundly, because the
  interpreter (not the model) does the transform. Tool calls are firewall-gated.
- **Three falsifiable properties** (CI-gated via `tests/test_interpreter.py` and the
  red-team): (1) taint propagates through every step (no laundering); (2)
  **control-flow integrity** — an injection inside retrieved content cannot trigger a
  tool call not in the plan; (3) a tainted value into a write/egress sink is blocked.
- Red-team gains two interpreter invariants (`interpreter_control_flow_integrity`,
  `interpreter_contains_tainted_write`); planner/extractor are injectable (mocked).
- **Honest scope:** the instruction set is small — the planner's *expressiveness*,
  not the safety, is the limit. A real privileged-planner LLM, a quarantined
  extractor, per-tool airgap for `run_tool`, and an AgentDojo-style end-to-end suite
  are M2.3.

## [0.7.15] - 2026-06-21

### Hardened — M2 firewall: live airgap + correctness, honest scope (review findings)

A 20-agent adversarial review of M2 v1 (0.7.14) found it overclaimed: the firewall
was an isolated component (not wired into any live path), taint laundered through
ordinary Python, and "airgap blocks all egress" covered only 2 of 5+ egress paths.
Corrected here — claims are now either true or scoped.

- **Live airgap egress kill-switch** at every model/network chokepoint via a single
  `egress_blocked()` check: the **model adapter** (`agent/model.py` — the central one;
  non-local providers refused, mock/localhost still work), `web_search`
  (`agent/web_evidence.py`), the **Google GenAI client** (`agent/google_genai_client.py`),
  plus the MCP `openclaw_infer` / `web_evidence_search` tools.
- **Firewall wired at the live `wiki_upsert` WRITE sink** — the capability policy
  runs there now; a `Labeled`-tainted payload is refused (not just advisory).
- **Approver fails closed** — a missing/raising approver or any non-`True` return
  blocks; only an explicit `True` approves a tainted→sink call.
- **Nested-container taint caught** — `taint_of` recurses into list/tuple/set/dict,
  so a tainted value inside a structured arg is no longer invisible.
- Red-team: added a nested-taint scenario; the firewall section is relabelled as
  *engine + airgap* validation (not a live-path guarantee). Tests in
  `test_dataflow.py` lock the airgap kill-switch, nested taint, and fail-closed approver.

**Honest scope — NOT yet (now tracked as M2.2):** the live autonomous path does not
auto-attach taint to untrusted content (Labeled-until-sink propagation); taint does
not survive arbitrary Python (f-strings launder — a fundamental limitation that the
dual-LLM/CaMeL interpreter, not wrappers, will address); `agent/tools.run_tool`
subprocesses are not yet per-tool airgap-classified.

## [0.7.14] - 2026-06-21

### Added — M2 v1: out-of-prompt data-flow firewall (capability + taint)

Moves the security boundary into deterministic code (CaMeL principle): the model
can be fully compromised, but it cannot drive untrusted data into a side-effecting
sink. See `docs/11-Platform/Security-Roadmap.md`.

- **`agent/dataflow/`** — dependency-free enforcement core: taint labels
  (`untrusted`/`trusted`, propagated via `combine`); per-tool capabilities
  (`READ`/`WRITE`/`EGRESS`) with a default-deny manifest for the real `sophia_*`
  tools; a deterministic policy (`decide`) and `firewalled()` wrapper that **blocks
  the lethal trifecta** (tainted → write/egress sink) or routes it to human
  approval, and an **airgap** profile that fail-closes all egress.
- **Live airgap wiring**: `openclaw_infer` and online `web_evidence_search` return
  a blocked result under `SOPHIA_PROFILE=airgap` (no behavior change otherwise).
- **Red-team scores the firewall** (`eval/security/`): lethal-trifecta **ASR 0%**
  (exfil-via-egress, write-poisoning, airgap-egress, unknown-sink; baseline 100%),
  reads not over-blocked — two new gating invariants.
- Tests: `tests/test_dataflow.py` (taint propagation, policy matrix, lethal-trifecta
  block, HITL path, default-deny unknown tool, live airgap); wired into CI.
- **Honest scope:** this is the enforcement boundary. The dual-LLM privileged-
  planner / quarantined-extractor split + constrained-AST interpreter is M2.2.

## [0.7.13] - 2026-06-21

### Fixed — negation-evasion in the provenance gate (red-team finding)

The M1 red-team found a real exploit in `agent/verifiers.py:provenance_faithful`:
the negation/correction carve-out was **sentence-scoped**, so a trigger word in
one clause ("it is a myth, but in truth Confucius wrote the Dao De Jing";
"contrary to the claim that he did not, …") shielded an asserting clause in the
same sentence (100% ASR on those probes).

- **Fix:** the carve-out is now **clause-scoped** — `_carveout_clauses` splits a
  sentence on contrastive connectors (but/however/yet/in truth/actually/…) and a
  leading subordinate clause (contrary to/despite/although/…), but **not on commas**,
  so the appositive author→title matching the gate relies on is preserved. A
  correction only excuses the clause it lives in.
- **Locked in:** 4 negation-evasion variants are now **gating at 0% ASR** in the
  red-team; `test_verifiers.py` gains `test_provenance_negation_evasion_is_clause_scoped`
  (exploits caught, genuine corrections still pass). 0 false positives across the
  68 committed corpus/dispute pages (`lint_wiki_provenance`).
- Remaining red-team probe (open): citation subject-match → NLI fact-checking (roadmap M-#5).

## [0.7.12] - 2026-06-21

### Added — M1 injection / containment red-team + first confidentiality gate

The first security-roadmap milestone (`docs/11-Platform/Security-Roadmap.md`).

- **Injection red-team** (`eval/security/`, `tools/run_security_redteam.py`):
  deterministic, offline harness under an **assume-compromised-model** threat model
  — it measures whether the gate/policy verifiers (outside the model) contain an
  attacker-controlled LLM. Gating attacks contained at **0% ASR**: forbidden
  attribution, false arithmetic, topic-mismatch citation. Success judged by code,
  never by an LLM. CI-gated.
- **`no_secret_leak` verifier** (`agent/verifiers.py`) + **`confidentiality`
  policy** (`agent/policies.py`, with `secrets=` plumbed through `guarded_complete`):
  a deterministic verbatim-secret tripwire. Secret-exfiltration ASR **100%
  baseline → 0%** with it — the harness measured the hole and proved the fix.
- **Two real gaps the harness surfaced** (reported, drive later milestones):
  citation subject-match (lexical overlap passes a wrong predicate → motivates NLI
  fact-checking); and a **negation-evasion in `provenance_faithful`** — a carve-out
  trigger word in the same sentence as a forbidden attribution skips the
  sentence-scoped carve-out. The failing case is committed to drive the gate-hardening fix.
- Tests: `test_security_redteam.py` (gates containment, not the bug existence, so a
  future fix won't break it); wired into CI. Docs: `eval/security/README.md`.

## [0.7.11] - 2026-06-21

### Added — cross-entity generalization benchmark + first real external number

- **Cross-entity generalization** (`provenance_bench/cross_entity.py`,
  `tools/run_cross_entity.py`): makes the next frontier falsifiable on an
  **entity-disjoint** split (no author/work shared). Memorized rules score 100%
  on *seen* entities but **0% on unseen** (precise, zero FP, no transfer); a
  content-free structural detector scores **100% on unseen but 100% false-positive**
  (transfers, can't tell true from false). The honest conclusion: low-FP
  cross-entity generalization needs **external grounding**, not pattern
  memorization — which is why Sophia's answer is the retrieval-grounded loop. Six
  invariants gate CI; holds across seeds.
- **First real external-oracle number:** DeepSeek-chat on **GSM8K test, N=100 →
  98.0%** exact-match via `agent/external_eval.py`. Recorded in
  `published-results.json` / `RESULTS.md` with explicit framing: this validates the
  harness end-to-end and reports the **base model's** accuracy — it is **not** a
  claim about Sophia's gate. `tools/fetch_eval_dataset.py` produced the data
  (gitignored; not committed).
- Docs: `docs/11-Platform/Generality.md` gains the cross-entity section; tests
  `test_cross_entity.py` wired into CI.

## [0.7.10] - 2026-06-21

### Hardened — round-2 adversarial review (5 confirmed findings fixed)

A 18-agent adversarial review of the 0.7.9 surfaces confirmed 5 real defects
(9 rejected). All fixed, with regression tests:

- **Sandbox escape (HIGH)** — the model-proposer's `__` *substring* blocklist was
  bypassable by building a dunder at runtime (`"_"+"_"`) and traversing via
  `str.format` (reached `object`). Replaced with an **AST allowlist** in
  `agent/verifier_synthesis._compile_predicate`: no attribute access, imports,
  lambdas, loops, comprehensions, container literals, `*`/`**`, or non-allowlisted
  calls; minimal scalar builtins; 2 KB source cap. This also closes the two
  **MEDIUM** DoS findings (unbounded CPU via infinite loop; allocation bomb) —
  structurally, no subprocess/signal sandbox needed.
- **Gate must fail closed (HIGH)** — a custom/synthesised verifier that raised
  crashed `guarded_complete`. `_judge` now catches and returns `passed=False`
  (fail closed → repair/abstain), never propagates.
- **Honest abstention (LOW)** — removed the dead `Policy.abstention_passes` flag;
  the loop now re-judges the abstention and reports `action="abstained_unverified"`
  when it cannot clear its own gate (e.g. the code policy).
- **Latent bug found while fixing** — explicit `policy="provenance"` would have
  emitted an internal marker string as the abstention; provenance now uses its
  dynamic cited abstention whether selected by default or by name.
- `tools/sophia_guard.py` gained `--policy`; tests extended in
  `test_verifier_synthesis.py` (sandbox payloads) and `test_policies.py`
  (fail-closed, explicit-provenance, unverified-abstention).

## [0.7.9] - 2026-06-21

### Added — runtime policies, model-proposed checks, real-dataset eval, honest README

Makes the verifier-gated capabilities *usable at runtime* and follows through on
the standing recommendations.

- **Runtime verifier policies** (`agent/policies.py`, `agent/guarded.py`): the
  guarded loop's gate is now selectable per call or via `$SOPHIA_POLICY` —
  `provenance | citation | arithmetic | code`, or any custom/synthesised verifier
  via `verifier=`. Each policy carries its own repair hint + gate-passing
  abstention. The provenance default path is byte-for-byte unchanged.
- **Verifier-synthesis model proposer** (`agent/verifier_synthesis.py`):
  `propose_predicates` lets a model write candidate predicates (compiled under
  **restricted builtins** — no import/exec/eval/dunders) that clear the SAME
  meta-verification floor; a model only *widens* candidates, validation still
  confers trust. Gated by `$SOPHIA_ALLOW_PROPOSED_PREDICATES`.
- **External-eval real-dataset fetcher** (`tools/fetch_eval_dataset.py`):
  downloads + reshapes GSM8K to the eval's `{question, answer}` JSONL so the
  external-oracle harness yields a *citable* number. Conversion is a pure,
  unit-tested function; the network fetch is intentionally not run in CI.
- **Model-in-the-loop improvement** (`provenance_bench/improvement.py`,
  `tools/run_improvement_loop.py --model`): an injectable `answer_fn` sources
  TRAIN failures from a model. Fixed a correctness bug in the process: a rule is
  mined only when the text *actually asserts* the forbidden attribution (clean
  model text is no longer mis-mined as a failure). Deterministic path unchanged
  (held-out recall 17%→98%, 0% FP).
- **README reframed:** leads with verifier-gated provenance reasoning and a
  plain-scope statement; AGI is demoted to an explicitly *unmet* pre-registered
  threshold — the project's standing #1 review recommendation.
- Tests: `test_policies.py`, `test_fetch_eval_dataset.py`,
  `test_improvement_model_loop.py`, plus proposer cases — wired into CI.

## [0.7.8] - 2026-06-21

### Added — verifier synthesis (the bridge toward generality)

The verifier-gated loop is only as general as its verifiers, and you cannot
hand-write a verifier for a task you have never seen. This makes the loop write
and **trust-test its own checks** — and abstain when it cannot — without
overclaiming (see `docs/11-Platform/Verifier-Synthesis.md`).

- **Verifier synthesis** (`agent/verifier_synthesis.py`): a library of
  parameterised check templates is *fit* to a few oracle-labelled examples of a
  novel task to produce candidate verifiers; each candidate is **meta-verified**
  (precision + recall on a disjoint, independently-labelled validation split)
  before admission; admitted checks compose into a gate that drops into the
  harness via `as_verifier`. If nothing clears the floor, it **abstains**.
- **Calibrated abstention** (`agent/calibration.py`): competence where no verifier
  exists — ECE, risk–coverage, selective risk, and label-free self-consistency
  confidence. Falsifiable claim: selective risk < base risk ("knows what it
  doesn't know"). Demonstrated on a seeded **toy** noisy solver (illustrative, not
  a capability claim): selective risk 0.30 vs base 0.56 (ECE 0.15), the
  correlation emergent from the solver, not baked into the data.
- **Falsifiable ablation** (`agent/synthesis_eval.py`,
  `tools/run_verifier_synthesis.py`): deterministic suite of in-library vs
  out-of-library tasks, the latter with **length-matched decoys** so no template
  can separate them on any split. WITH meta-verification: in-library precision
  1.00 / recall 1.00 (0 abstentions), abstains on 100% of unverifiable tasks every
  seed. WITHOUT: false-admission 100% on unverifiable tasks and in-library
  precision degrades to 0.86 — proving the *meta-verification*, not the template
  library, earns the trust. Nine invariants gate CI (incl. a "no good-looking
  wrong gate" guard so a false admission can't hide behind an abstention count).
- **Honest scope:** not AGI, not unbounded synthesis — a finite library plus a
  trust contract; tasks that don't reduce to a checkable predicate stay out of
  reach, where calibrated abstention is the correct behaviour.
- Tests: `test_verifier_synthesis.py`, `test_calibration.py` — wired into CI.

## [0.7.7] - 2026-06-21

### Added — generality track (verifier-gated reasoning, measured honestly)

Extends the core verifier-gated loop beyond provenance, each piece with a
falsifiable metric and an honest scope label (see
`docs/11-Platform/Generality.md`). None of this licenses the word "AGI".

- **More machine-checked verifiers** (`agent/verifiers.py`): `citation_faithful`
  (RAG support check), `code_tests_pass` (extracts + **executes** answer code,
  exec-gated), `arithmetic_sound` (recomputes stated equalities); a `VERIFIERS`
  registry + `check_text`. Honest: more verifier *kinds* is engineering reuse,
  not a generality claim.
- **Measured self-improvement loop** (`provenance_bench/improvement.py`,
  `tools/run_improvement_loop.py`): learns rules from TRAIN failures, scored on a
  **disjoint-phrasing** held-out split (no contamination). First run: held-out
  recall 17% → 98% over 6 cycles, monotone, 0% false-positive cost — falsifiable.
- **Long-horizon autonomy curve** (`agent/horizon.py`,
  `tools/run_horizon_curve.py`): success-rate vs task length on chained tasks,
  judged by an **external oracle**; headline = effective horizon (longest length
  at ≥50%). Complements the single-run logger `tools/run_long_horizon.py`.
- **External-oracle eval** (`agent/external_eval.py`, `tools/run_external_eval.py`):
  correctness vs external gold (never the gate); dataset-agnostic JSONL with a
  committed, clearly-labelled GSM8K-style sample; point `--dataset` at the real
  set for a citable number.
- **Harness confound fix** (`agent/harness.py`): `classify_failure` returned
  `verifier_fail` for unknown-cause failures, over-crediting the verifier in
  ablation telemetry; now returns an explicit `unknown` class (regression-tested).
- Tests: `test_horizon.py`, `test_external_eval.py`, plus verifier/loop/harness
  cases — all wired into CI.

## [0.7.6] - 2026-06-21

### Added — public results, transparently and safely

Makes test/benchmark results public *the right way* — three bright lines:
publish reproducible code + methodology + audited aggregates; never publish
secrets or hidden-eval prompts; never headline an un-validated number.

- **No-overclaim gate (consensus judge)** — `provenance_bench/consensus.py`:
  majority vote over ≥2 independent judges (`--judges a,b,c`), reporting raw
  pairwise agreement AND chance-corrected **Cohen's κ**. `aggregate.py`'s
  `validated` flag is a real conjunction (`validatedChecks`): not mock, judges
  from ≥2 distinct families, **κ ≥ 0.40**, ≥3 runs, and a CI that excludes zero —
  it refuses to rubber-stamp. A single judge is no longer enough (our audit found
  one judge ~2× off).
- **Public results page** — `agi-proof/benchmark-results/published-results.json`
  (curated; the ONLY source of published numbers) renders `RESULTS.md` via
  `tools/build_results_page.py`. Validated section is honestly empty for now;
  illustrative figures carry caveats. `--check` is a CI drift gate.
- **Publishing CI** — `.github/workflows/publish-results.yml`: offline-only (no
  secrets, no model calls), runs tests + mock benchmark, verifies the page,
  stamps commit+run provenance, uploads the results bundle. Drift check also
  wired into the main CI.
- **Security boundary** — `SECURITY.md` documents the public/private line and the
  gate; `.gitignore` hardened (`.env.*`, `*.key`, `*.pem`). Verified: the
  DeepSeek key pasted earlier never entered git history; `private/hidden-evals/`
  stays ignored.
- Tests: consensus majority + inter-judge agreement + aggregate flow added to
  `tests/test_provenance_bench.py` (CI-wired).

## [0.7.5] - 2026-06-21

### Added / Changed — gate coverage, confidence intervals, independent judge

- **Gate coverage (core `provenance_faithful`, precision preserved)** — the gate
  now catches three real phrasings it missed in live runs, without lowering
  precision (dispute-page lint still 0 forbidden; verifier/guarded/source-
  discipline tests pass): (1) **quoted / "the"-padded titles** (`wrote "The
  Constitution of the Athenians"`), (2) **`attributed to X`** with a bounded
  honorific filler (`attributed to the prophet Daniel`), and (3) optional
  **`altTitlesEn`** on a record, and a bounded **appositive/parenthetical slot**
  between author and verb ("Enoch, the great-grandson of Adam, wrote …", "Lie
  Yukou (also known as Liezi) wrote …"). New carve-outs (`traditionally`,
  `spurious`, `pseudo`, `disputed`, …) keep correctly-hedged attributions passing.
  Each change independently re-verified: dispute-lint still 0 forbidden, 0 false
  positives across all 41 true controls.
- **Benchmark gate rules** — `dataset.build_gate_records()` now reduces honorific
  author names to salient markers and derives alt-title forms ("the Book of
  Daniel" → "Daniel", interior-"the" collapse) so the gate fires on natural model
  phrasings.
- **Confidence intervals + multi-run** — `provenance_bench/aggregate.py` +
  `--runs N`: paired bootstrap 95% CI on the delta, per-run deltas surfaced, CI
  columns in the report.
- **Independent LLM-judge wired end-to-end** — `--llm-judge <spec>` (judge ≠
  subject). Model-selection guidance added (the delta tracks propensity-to-assert,
  not size; pair with a confidently-wrong subject + a frontier judge).
- **Adversarial judge audit (key finding)** — an independent Claude panel
  re-judged the DeepSeek LLM-judge's 46 false-case verdicts on `dolphin-llama3:8b`.
  Agreement was only **76%**: DeepSeek over-counted (10 false positives — scoring
  correct denials-with-wrong-alternate-author and "traditionally…but disputed"
  hedges as hallucinations), so the validated alone-rate was **21.7%, not 41.3%**.
  Robust conclusions hold (0% false-positive cost; positive, real delta; tracks
  propensity-to-assert), but a **single LLM-judge is unreliable** — the citable
  headline needs a ≥2-judge consensus. Documented in
  `docs/11-Platform/Provenance-Delta.md` and the checklist.
- Tests: gate-coverage cases (incl. appositive/parenthetical), `build_gate_records`
  markers/alt-titles, and bootstrap-CI aggregation added to
  `tests/test_provenance_bench.py` (CI-wired).

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
- **Hard / obscure cases + gate-rule derivation** — expanded the set to **87
  externally-cited cases (46 false / 41 true)** with verified spurious /
  pseudonymous / forged attributions across Greek-Roman (pseudo-Aristotle,
  pseudo-Plato, the Old Oligarch, Gallic War bk 8, Batrachomyomachia, Corpus
  Hermeticum, Epistles of Phalaris, Pseudo-Dionysius…), biblical (Mosaic
  authorship, Deutero-Isaiah, Hebrews, the Pastorals, the Book of Daniel/Enoch,
  Wisdom of Solomon…), and Chinese (Ten Wings, Liezi, Guanzi) traditions.
  `dataset.build_gate_records()` derives the gate's do-not-attribute rules from
  the cited misattributions (the realistic `SOPHIA_DISCIPLINE_RECORDS` path) so
  the gate fires on the benchmark's works; the judge now handles scholarly-hedge
  / pseudonymity language and excludes claimed-author tokens when crediting gold
  (fixes a "Pseudo-Aristotle" name collision).
- **First real delta (multi-model, illustrative)** — single run each, lexical
  judge, 46 false cases: frontier `deepseek` 0% alone; an *uncensored*
  `dolphin-llama3:8b` 15.2% → **6.5%** behind the gate (Δ≈8.7), 0% false-positive
  cost, 57% coverage; well-aligned `llama3.2:3b` / `qwen2.5:3b` rarely assert
  false lineages (~2%) so show little delta. Finding: the delta tracks a model's
  propensity to *assert*, not its size. Run-to-run variance observed (→ Tier-1
  multi-run averaging). Concrete gate coverage gaps (quoted titles, `attributed
  to`, multi-word author names) logged in the checklist. See
  `docs/11-Platform/Provenance-Delta.md`.
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
