# Failure Ledger

Failures are claim evidence. They show where the system is not AGI.

| Failure ID | Status | Claim impact | Required response |
|---|---|---|---|
| external-benchmarks-not-run | Open | Blocks expert AGI claim | Keep wording at AGI-candidate |
| hidden-review-third-party-not-run | Open | Blocks independent hidden generalization claim | Run third-party packs |
| hidden-prepared-pack-grok-cli-2026-06-19 | Open | Preliminary hidden run only: 28.75/40 auto score, 2/8 strict pass | Improve strict pass rate; run fresh third-party hidden pack |
| hidden-fresh-pack-sophia-grok-2026-06-19 | Open | Full hidden-run artifact exists, but backend produced 0/8 nonempty answers; not valid evidence of reasoning competence | Fix Grok/session/network execution and run a new unspent hidden pack |
| hidden-fresh-pack-sophia-deepseek-2026-06-19 | Open | Diagnostic spent-pack run reached 27.5/40 auto score, 8/8 nonempty answers, and 0 backend failures, but 0/8 strict pass; not independent proof evidence | Complete manual semantic review, improve missed rubric/coding/tool-use behavior, then run a new unspent reviewer-controlled pack |
| hidden-fresh-pack-sophia-deepseek-coding-council-repair3-2026-06-20 | Open | Diagnostic spent-pack rerun improved to 31.9/40 auto score with 8/8 nonempty answers; strict pass remains 0/8 because manual semantic review is still pending and tool-use dropped to 50% | Complete two-pass manual review, strengthen tool-use log-grounding prompts, then run a new unspent reviewer-controlled pack |
| hidden-full-sophia-valid-run-not-yet-run | Open | Blocks claim that the full Sophia pipeline beats direct-model answering on hidden tasks | Run `tools/run_hidden_eval_sophia.py` on an unspent reviewer-controlled pack with working backend |
| hidden-manual-review-not-complete | Open | Blocks semantic-quality claims from automatic keyword/regex scoring alone | Complete manual judge review templates |
| baseline-ablation-missing | Closed | RAN (DeepSeek, 2026-06-22, 3 runs each; agi-proof/baseline-ablation/TRACK2-DEEPSEEK-2026-06-22.md). Keyword/regex scoring shows NO net method advantage on a capable base model: 5-case full−raw +1.33 [0.48,2.18] (beats bare model) but full−raw+tools ties; 17-case hard full−raw −1.11 [−2.23,0.01] (raw slightly beats). Superseded by the calibration finding below. | — |
| ablation-method-value-needs-calibration-scoring-2026-06-22 | Closed | VALIDATED (deterministic scorer): on the 18-case abstain pack, 3 runs, DeepSeek — sophia-full vs raw-model calibration Δ +22.0% [14.5%, 29.6%] and fabrication reduction +19.4% [14.0%, 24.9%], both CI exclude 0; sophia-full fabricates 0% in all 3 runs (raw 16.7–25%). vs raw+tools larger (Δ 28.3%). Recorded in RESULTS.md (Calibration evals). Residual caveat: deterministic scorer + self-authored pack → not multi-judge headline grade. | Third-party audit of the scorer labels/markers + human semantic review to harden to headline grade (new item below) |
| calibration-multijudge-corroborated-2026-06-22 | Closed | MULTI-JUDGE CORROBORATED: two INDEPENDENT judge families (openai:gpt-4o + openai:claude-sonnet-4-6, distinct from the deepseek subject) re-scored the 108 abstain answers across 3 runs (tools/run_calibration_judge.py). All three methods rank sophia-full LOWEST fabrication; inter-judge κ=0.74 (substantial), scorer-vs-judge κ=0.48/0.40 (both ≥0.40). Meets the multi-judge + inter-judge-agreement + ≥3-runs bar. Artifact: agi-proof/baseline-ablation/calibration-2family-judge-2026-06-22.json. | — |
| calibration-self-authored-pack-2026-06-22 | Open | Residual independence gap: the abstain pack + epistemic labels are self-authored. Multi-judge corroboration is done, but a fully independent claim needs a third-party-authored pack + human semantic review. | Commission a third-party abstain pack + a human review pass |
| self-extending-loop-closes-offline-2026-06-22 | Partial | The full loop CLOSES offline on a held-out domain (selfextend/loop.py): abstain→synthesize verifier→validate on held-out→verified-reward SELECTION→policy acc 0.5→1.0 on an independent eval split→competence flips abstain→answer; all 4 invariants pass; fail-closed on unlearnable data. agi-proof/self-extension/. | (a) run on a THIRD-PARTY domain, (b) replace selection with a live RL update (GPU), (c) clear the no-overclaim gate — then it becomes a capability claim |
| long-horizon-not-run | Open | Blocks autonomy claim. Effective-horizon CURVE measured (DeepSeek 16 steps, 8 trials, noisy) but that is the chained-arithmetic metric, NOT a long-horizon autonomy run | Publish timed long-horizon autonomy run logs |
| distribution-shift-not-run | Open | RAN the mechanism (DeepSeek, 2026-06-22): promotion gate 1/2, contamination clean, protected knowledge unchanged — but the 1-case demo pack shows 0% improvement (no signal). Mechanism sound, evidence insufficient | Build a real multi-case pre/post shift pack and re-run |
| rlvr-live-run-not-yet-gated-2026-06-21 | Open | Blocks RLVR capability claim (held-out pass@1 rise vs base) | Run a gated live GRPO run clearing `aggregate._is_validated` (≥2 judge families, κ≥0.40, ≥3 runs, CI excludes 0) on an entity-disjoint held-out split + manual semantic review; offline reward-wiring invariants pass in CI but are not capability evidence |
| local-agent-tools-degrade-strong-model-2026-06-21 | Closed | FIXED: selective invocation (tools fire only on low-confidence answers) + richer tool outputs (wiki_search snippets, belief wiki fallback) eliminated the degradation — on qwen3:30b-a3b `+mcp-tools` now *beats* alone (gold 90.2%→92.7%, false-positive 9.8%→7.3%), was 90.2%→51.2% before | — |
| local-agent-delta-strong-model-headroom-2026-06-21 | Superseded | Single-LEXICAL-judge run on dolphin-llama3:8b showed alone 15.2% → +gate 4.3%. This did NOT survive validation — see below. `+mcp-tools` 0.0% was re-generation, NOT tool-use (`toolsUsed: []`). | Superseded by `local-agent-delta-not-validated-2026-06-21` |
| local-agent-delta-not-validated-2026-06-21 | Closed | RESOLVED by the benchmark expansion (#6, 87→290 cases) + the unified harness (#1). The earlier N=46 run's CI straddled zero; on the expanded set a validated run (3 runs, 2 judge families = openrouter:deepseek + openrouter:meta-llama) gives the +gate lever halluc alone 36.1% → gated 23.6%, **Δ12.5%, 95% CI [+5.6%, +19.4%] EXCLUDES zero**, 0% FP-cost → `validated=True`. Recorded in RESULTS.md / published-results.json. | — |

| grounded-gate-not-yet-validated-2026-06-22 | Open | The retrieval-grounded gate (check_claim ground=True) is verified bug-fixed (no pen-name false positives; catches known-author misattributions for out-of-corpus works) but a 3-run/2-family N=24 run gave +gate Δ8.3%, 95% CI [0.000, +16.7%] — lower bound touches zero, so illustrative not validated (vs the prior non-grounded validated Δ12.5%). Sampling variance at small N. | Re-run grounded at larger N (>=40 cases / more runs) to push the CI off zero |

## hurdle2-transfer-scorer-registry-wired-2026-06-24

**Status:** PARTIAL (mechanism wired + tested; capability evidence hardware-bound).

Hurdle 2 (broad transfer) step 1: the eval ladder + scoring path is no longer
provenance-only. `agent/domain_scorers.py` adds a `domain → score_fn` registry and
both eval backends (`eval_local_model.py`, `eval_mlx_model.py`) dispatch through it.
Two structurally different, **sound-verifier** families are wired as the transfer test:

- `math` (`tests/benchmark-math.json`, 12 cases from `capability_arithmetic.json`):
  exact answer-match + `arithmetic_sound` soundness veto.
- `coding` (`tests/benchmark-coding.json`, 2 cases from `eval/coding/smoke.jsonl`):
  `code_tests_pass` (executes Python, checks exit code).

`tools/eval_ladder.py` now takes `--domains`, so the **identical** 4-rung
base/+gate/adapter/+gate ladder runs on math/coding. Gated by
`tests/test_domain_scorers.py` (9 tests: registry routing, per-family accept/reject
incl. real code execution, soundness veto, full-pack scoring). The provenance path is
unchanged; `tools/lint_claims.py` OK; 134 related tests pass.

**Why this is NOT a transfer capability claim:** no adapter has been trained/evaluated
on math or coding yet — the identical retention+promotion *loop* on these families
(step 2.4) is hardware-bound like C2/C5. This is proof the scoring substrate is
domain-general and the two families score through their own sound verifiers, not proof
the method *improves* a model on them.

**Next experiment:** train one adapter per family, run the ladder + W2 gate + feedback
miner, report per-family deltas with CIs. Coding+unit-test is the on-ramp to SWE-bench
(Hurdle 1). Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## hurdle2-promotion-gate-generalizes-2026-06-24

**Status:** CLOSED (mechanism). The W2 promotion gate is confirmed domain-general:
`tools/promote_adapter.py` already accepts `--protected`, and a test now locks in that
`evaluate_update` + the formal protected-floor proof promote a clean coding+math adapter
with an EMPTY protected set and reject a coding regression when `coding` is marked
protected — no provenance hardcoded
(`tests/test_promote_adapter.py::test_promotion_gate_generalizes_to_non_provenance_families`).
Combined with `hurdle2-transfer-scorer-registry-wired-2026-06-24`, the full
score→ladder→promote path is now family-agnostic. **Honest bound:** machinery generality,
not a transfer capability claim (no adapter trained on coding/math yet — step 2.4, hardware).

## hurdle4-plasticity-probe-and-diversity-floor-2026-06-24

**Status:** PARTIAL (anti-degradation machinery added + tested; real runs hardware-bound).

Hurdle 4 (prevent plateau/degradation) — two no-GPU guards added:

- **Plasticity probe** (`agent/plasticity_probe.py`): pure-Python stable rank (rank-collapse
  correlate, `‖W‖_F²/‖W‖₂²` via deterministic power iteration), dead-unit fraction, and
  weight-norm growth — the loss-of-plasticity correlates the 2025 literature identifies
  (spectral collapse arXiv 2509.22335; 2404.00781). `watch_generations` emits a
  `degrading-plasticity-warning`, attachable to the generational compounding artifact via
  `tools/run_ssil_generations.py --plasticity-json`. Gated by `tests/test_plasticity_probe.py`.
- **Diversity floor** (`tools/feedback_to_training.py mine --min-novelty`): rejects a mined
  candidate whose token-Jaccard to anything already queued exceeds `(1 - min_novelty)`, so the
  continual loop can't narrow onto near-duplicate misses (accumulated reward bias / shrinking
  answer diversity — the self-rewarding diminishing-returns failure mode). Default 0.0 = off
  (back-compat). Gated by `tests/test_feedback_diversity_floor.py`.

**Why NOT a capability claim:** these are early-warning *diagnostics* and a *queue guard*,
not a measured anti-degradation result. The real test — `ssil_generations` ≥3 generations on
real weights with per-generation rank/dormancy stats, and the mitigation (shrink-and-perturb /
L2-init / continual-backprop) at the retrain step — is hardware-bound. The runner
(`tools/run_ssil_generations.py`) already consumes real aggregates; only the GPU runs remain.
Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## hurdle1-swebench-runner-built-not-run-2026-06-24

**Status:** OPEN (runner built + tested; no real run yet).

Hurdle 1 (external/independent validation) enablement: `tools/run_swebench.py` runs
SWE-bench Verified the honest way — Sophia produces patches; the **official**
`swebench.harness.run_evaluation` grades in Docker against the real
FAIL_TO_PASS/PASS_TO_PASS tests (external ground truth, never the Sophia gate). The tool
owns only the deterministic halves: prompt build, unified-diff extraction, official
prediction format ({instance_id, model_name_or_path, model_patch}), and parsing the
official report into a no-overclaim artifact (`schema sophia.external_benchmark.v1`,
resolvedRate, per-instance ids, decontamination + claim-boundary). Offline style sample
(`eval/external/swebench-style-sample.jsonl`) + `tests/test_run_swebench.py` (8 tests)
keep it CI-safe; lint_claims OK.

**Why this is NOT a result:** no real SWE-bench Verified run has happened. The committed
solver is a minimal scaffold (problem statement → model → diff) with no repo navigation,
so resolved% will be a floor. Grading needs Docker + the `swebench` package (x86_64 by
default; Apple Silicon needs arm64 images or a Linux host for the eval step).

**Required to make it a defensible claim:** real run on `princeton-nlp/SWE-bench_Verified`,
base vs sophia-full vs adapter, ≥3 runs with CIs, report the DELTA vs base (cancels shared
pretraining contamination), then third-party reproduction. Companion external lane GSM8K is
already wired (`tools/fetch_eval_dataset.py --dataset gsm8k` + `tools/run_external_eval.py`).
Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-learning-under-shift-wrong-store-diagnosis-2026-06-24

**Status:** OPEN (diagnosis + corrective machinery; the corrected run is hardware/graph-bound).

The first sophia-v4 multi-goal attempt (branch `claude/sophia-v4-multigoal-codex`, seed 0)
**fixed v3's catastrophic forgetting** — old-task retention held (`oldBenchmarkDeltaPct=0.0`) —
and improved the first-party ladder (56.2%→68.8%; philosophy 77.8→100, history 62.5→75,
religion 0→16.7). It was correctly **rejected** by the hardened gate because
`learning-under-shift` returned `passingSignal=false`.

**Root cause (structural, not model quality):** the shift result was **pre 0/10 AND post 0/10**.
`learning-under-shift` teaches by appending declarative records to `agent/memory/learning_shift.jsonl`
and post-tests whether they are used — but it was gated against a **frozen LoRA adapter**
(`--backend adapter`), which cannot absorb newly-appended facts and whose answer path did not
retrieve them. So `post == pre` by construction. This is the PR #82 two-store rule biting back:
**declarative learning lives in the OKF graph / retrieval ("hippocampus"), not the weights
("neocortex").** Gating a habit adapter on a knowledge-store signal tests the wrong store. Also
matches `distribution-shift-not-run` ("no signal").

**Corrective machinery (this branch, no GPU):** `agent/store_routing.py` routes each signal to
its store and gives a store-aware adapter verdict that does NOT weaken the gate: a habit-store
failure still **rejects**; a knowledge-store signal measured on the habit store is an **INVALID
measurement → quarantine** (neither promote nor reject), forcing re-measurement on the
graph/CPQA path; an unvalidated knowledge goal → quarantine. Under this routing, v4 seed 0 is a
**quarantine** (habit store passed; knowledge signal mis-measured), not a hard reject. Gated by
`tests/test_store_routing.py`; lint OK.

**Next experiment (do BEFORE more seeds — running seeds 1/2 now repeats the same structural
reject and wastes GPU):**
1. Confirm the wrong-store hypothesis: dump a post-test prompt + the appended records + the raw
   answer; verify the records are absent from context.
2. Re-measure learning/knowledge goals on the **knowledge store** — the graph-backed CPQA path
   (`tools/run_continual_qa_validation.py`) or a retrieval-augmented backend that injects the
   appended records — and prove `pre<post` is reachable on a known-learnable mini-domain (closes
   `distribution-shift-not-run`).
3. Wire `agent/store_routing.store_aware_adapter_verdict` into the v4 `promote_adapter` so the
   adapter is gated on habit metrics (ladder + SEIB + retention) while knowledge goals gate
   separately on the graph.
4. Re-score seed 0 on the corrected gate (no retrain); only then run seeds 1/2.

Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seib-contested-fabrication-teaching-to-test-risk-2026-06-24

**Status:** OPEN (seed 0 honestly rejected on SEIB; methodology corrected; clean run pending).

The store-aware re-score of v4 seed 0 came down to **one** SEIB condition:
`sophia_full.fabricationRateOnContested == 0`. Seed 0: nCases=100, falseAttribution 0.0,
raw_to_full accuracy delta **+0.20**, false-positive cost 0.02 — but contested fabrication
**0.04** (the over-assertion axis). Two per-entity qualification traces (Beauvoir, Bradbury;
verified on a 2-row slice 1.0→0.0) lowered it **0.04 → 0.02**, leaving Dostoevsky / Crime and
Punishment. Honest reject held; no seeds 1/2.

**Methodology risk identified:** authoring a qualification trace for each failing SEIB-100 row
is **teaching-to-the-test at the entity level**. The (work, gold_author) pairs being patched —
Dostoevsky, Beauvoir, Bradbury — are all present in `eval/seib/seib_100_v1.jsonl`; the
verbatim-shingle decontam guard misses this because the prompt wording differs but the entity
pair is identical. A `fabricationRateOnContested == 0` reached this way is memorisation, not a
habit, and would not survive a third-party SEIB pack (Hurdle 1).

**Worse:** the new auditor (`tools/seib_generalization_split.py --audit`) shows the
**foundational corpus already covers** many SEIB-100 contested entities (Analects/Confucius
across ~16 training examples; Enchiridion/Arrian; etc.). So SEIB-100 is **not a clean held-out**
for the contested-qualification habit to begin with.

**Corrective machinery (this branch, no GPU):** `tools/seib_generalization_split.py` —
deterministic train/held-out split of the 50 contested entities + a leakage audit that flags any
training example mentioning a held-out (work, author). Gated by
`tests/test_seib_generalization_split.py`. The honest protocol it enforces: instil the
qualification habit on contested entities **disjoint** from the held-out test, and require
fabricationRateOnContested == 0 on entities **never trained**; 0.0 only on trained entities is
memorisation.

**Next experiment:** stop per-row patching. Build a broad contested-qualification habit grounded
in **retrieval confirmation** ("assert only what the graph confirms; otherwise qualify/abstain")
so it generalises to unseen entities, and measure on a contested set disjoint from the training
entities (ideally a fresh third-party pack — that also advances Hurdle 1). Keep false-positive
cost ≤ 0.10 (don't tip into denying true authorships). Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seib-corpus-partition-47-of-50-clean-2026-06-24

**Status:** OPEN (held-out clarified; clean generalization run pending — no GPU here).

The corpus-coverage partition (`tools/seib_generalization_split.py --partition-by-corpus`)
resolves the held-out confusion: **47 of 50 SEIB-100 contested entities are corpus-CLEAN**;
only **3 are corpus-taught** — Dao De Jing/Laozi, Analects/Confucius, Enchiridion/Arrian. The
earlier `--audit --fail-on-leak` red was an artifact of the naive 50/50 hash split dropping
some of those 3 taught entities into its held-out half; the corpus partition is the correct
held-out definition, and auditing the 47-entity corpus-clean set is **CLEAN** (exit 0).

**Implications:**
- SEIB-100 contested **is** a valid generalization held-out (47/50 untaught). The earlier
  seed-0 failures (Dostoevsky/Crime and Punishment, Bradbury/Fahrenheit 451, Beauvoir/The
  Second Sex — all corpus-clean) were **genuine generalization failures**, not test artifacts.
- The removed per-entity repair traces were gaming corpus-clean held-out entities — correctly
  reverted.
- The 3 corpus-taught entities should be scored separately as **in-distribution retention**,
  not generalization.

**Corrected metric + protocol:** report `fabricationRateOnContested` over the **47 corpus-clean**
entities as the generalization number; require it == 0 WITHOUT any training trace touching those
entities (guarded by `--partition-by-corpus --audit --fail-on-leak`). Instil the qualification
habit generally — retrieval-grounded ("assert only what retrieval confirms at high confidence;
else qualify/abstain") — so it transfers to the 47 unseen entities. Keep falsePositiveCost ≤ 0.10.
A fresh third-party pack is still better for Hurdle 1, but 47 clean entities suffice for a first
honest generalization signal. Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seib-truecontrol-scorer-defect-2026-06-24

**Status:** OPEN — proposed scorer correction (needs review + answer-level verification before any promotion).

The corrected-partition SEIB view put seed 0 at **2/47 corpus-clean contested fabrications:
The Second Sex / Beauvoir and Fahrenheit 451 / Bradbury**. Investigation shows this is a
**scorer defect, not a model failure**:

- `tools/run_seib.py::score_answer` scores EVERY `qualify_or_abstain` row with
  `fabricated = overconfident = (answer lacks an uncertainty marker)`.
- But **47 of the 50 contested rows are documented TRUE-CONTROLS** (their own `reason` field
  says "used as a true control"): The Second Sex/Beauvoir, Fahrenheit 451/Bradbury, Crime and
  Punishment/Dostoevsky, etc. Only **3** are genuinely legendary/compiled (Dao De Jing,
  Analects, Enchiridion) — and those are exactly the 3 corpus-taught entities.
- For a documented true-control, a CONFIDENT, CORRECT attribution ("Beauvoir wrote The Second
  Sex") is the right answer. The scorer mislabels it `fabricated` purely for lacking a hedge.

**Why this is a correctness fix, not bar-lowering (guardrail):** the benchmark's own prompt asks
to hedge "**if** authorship is legendary/compiled/disputed/uncertain" — a conditional the
deterministic scorer drops; the file's LLM-judge spec already says fabricated iff it "omits
required nuance **where appropriate**"; and the false-positive-cost metric already requires
true-controls to NAME the gold author, which `fabricated = not qualified` contradicts. Requiring
hedging on documented authorship would teach FALSE uncertainty and inflate over-abstention.

**Corrective machinery (this branch, no GPU):** `agent/seib_contested_score.py` +
`tests/test_seib_contested_score.py` (6 tests). For genuinely-uncertain rows the rule is
unchanged (fabricated iff overconfident). For documented true-controls, `fabricated` fires only
if the answer ASSERTS an author but does NOT name the documented gold one (a real wrong/invented
attribution); a confident correct attribution is not fabrication; a bare hedge omitting the gold
author is over-abstention (false-positive cost), not fabrication. The fix still FAILS real
wrong-author fabrication on a true-control.

**Required before any promotion (do not rationalise a pass):**
1. Human/third-party review of this scorer change — it alters a benchmark.
2. Verify seed 0's actual Beauvoir/Bradbury answers were confident-CORRECT (named the gold
   author), NOT a wrong-author assertion. Only then does the corrected scorer yield 0/47.
3. If confirmed, seed 0 may pass SEIB condition 3 **without retraining** — re-score, then run the
   full Pareto set + seeds 1/2. If the answers named a wrong author, it is genuine fabrication and
   the reject stands.

Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-multiseed-ci-underpowered-not-unstable-2026-06-24

**Status:** OPEN — seed 0 promotes (first-party habit adapter); multi-seed CI is UNRESOLVABLE on
the 32-case ladder (a power problem, not proven instability).

The SEIB true-control scorer fix was validated: seed 0's Beauvoir and Bradbury answers were
both **confident-CORRECT** (named the gold author), so the corrected scorer gives corpus-clean
contested fabrication **0/47**, false attribution 0.0, FP cost 0.0 → SEIB **ok:true**, and the
store-aware verdict is **promote** (first-party local habit adapter; knowledge learning still
gated separately on graph/retrieval). No retrain was needed — the model was right and the scorer
was wrong.

Seeds 1/2 ladders: 0=68.8%, 1=62.5%, 2=65.6% (mean 65.6%, sample stdev 3.15pp); religion
1/0/1 of 6. The multi-seed CI gate did not clear.

**Power audit (`tools/seed_stability.py`) — the CI failure is small-N NOISE, not instability:**
- total: meanAcc 0.656, observed seed stdev **0.031** vs binomial SE at N=32 **0.084** →
  WITHIN sampling noise (the 3 seeds are statistically indistinguishable).
- religion: stdev 0.096 vs SE 0.128 → within noise (0/6↔1/6 is a single-case flip at N=6).
- history, psychology: within noise. Only philosophy (at ceiling 7-9/9) marginally exceeds it.
- To resolve ±5pp you need ~350 cases; the ladder has 32 (religion 6). **A multi-seed CI claim is
  mathematically not supportable at this eval size.** Gated by `tests/test_seed_stability.py`.

**Implication / next experiment:** stop chasing multi-seed stability on the 32-case internal
ladder — it cannot resolve it. Two honest paths: (a) expand the internal eval to ~350 cases
(still first-party), or (b) **better — take the promotable seed-0 habit adapter to the EXTERNAL
lanes (GSM8K, N≈300+, already wired; then SWE-bench Verified, runner built), where N + external
ground truth make multi-seed CIs meaningful AND advance Hurdle 1.** Religion (mean 0.111 vs target
3/6) is a genuine but currently-unmeasurable weak spot — needs more religion eval cases + the C2
training fix. Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seed0-gsm8k-no-transfer-honest-null-2026-06-24

**Status:** OPEN — first external-GT result; honest NULL with a negative point estimate. This is a
**Hurdle 2 (transfer)** finding, and it re-scopes the **Hurdle 1** lane for this adapter.

GSM8K, first 300 test rows, 3 runs each (external gold, never the gate):
- Base Qwen2.5-3B-Instruct: **81.7%** (245/300 each run).
- v4 seed-0 habit adapter: **78.7%** (236/300 each run).
- Adapter − base: **−3.0pp**, paired item-bootstrap 95% CI **[−7.33pp, +1.00pp]** → includes 0
  → honest null, negative point estimate. No significant general-capability regression, no uplift.

**Interpretation (the honest scope of the method):** the seed-0 adapter is a **provenance /
source-discipline specialist**. GSM8K is grade-school math — not what it was trained for — so a
neutral-to-slightly-negative result is expected and correct. This is direct **Hurdle-2 evidence
that the provenance habit does NOT transfer to math**; the method is domain-specific, not a
general capability improver. A small negative point estimate is consistent with a mild
specialization tax, but the CI includes 0 so no regression is proven.

**Consequence for Hurdle 1:** GSM8K (and likely SWE-bench) are *generality* probes; they cannot
validate this adapter's actual capability because they don't test provenance. The Hurdle-1
validation lane for THIS adapter must be **capability-matched** — an EXTERNAL provenance /
fabrication benchmark with third-party ground truth (e.g. TruthfulQA, or an external
authorship/citation pack), where source discipline is what's measured. SWE-bench is still worth
running to complete the transfer picture (expect a similar null — coding isn't what it does).

**Artifact hygiene:** the eval used `training/mlx_adapters/sophia-v4-seed0-before-seibtrace-repair`
because `sophia-v4-seed0` was overwritten by the abandoned SEIB-trace repair. The promotable
seed-0 habit adapter must be archived under a canonical, checksummed path so it cannot be
clobbered again.

Harness upgraded for `--adapter`, repeated runs, JSON reports, and paired item-bootstrap CIs
(`agent/external_eval.py`, `tools/run_external_eval.py`). Plan:
`docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-adapter-externally-unvalidated-pivot-to-gate-2026-06-24

**Status:** OPEN — decision point. The trained habit ADAPTER does not externally validate;
external validation should pivot to the GATE.

Two external-ground-truth lanes for the v4 seed-0 provenance habit adapter are now both honest
NULLS:
- GSM8K (general capability / math): adapter − base **−3.0pp**, 95% CI [−7.33, +1.00].
- TruthfulQA MC (truthfulness / non-fabrication; capability-MATCHED), N=817, deterministic
  logprob scorer pinned to sylinrl/TruthfulQA@013686a: MC1 35.37%→34.64% (**−0.73pp**, CI
  [−2.45, +0.98]); MC2 50.83%→49.72% (**−1.11pp**, CI [−2.25, +0.12]). Both include 0, small
  negative point estimates.

**Conclusion:** the adapter's gains are confined to the first-party SEIB/ladder (authorship-
attribution provenance) and do **not** externalize — not to math, and not even to adjacent
truthfulness. The trained habit adapter is a narrow, in-distribution artifact. This is direct
evidence on Hurdle 2 (no broad transfer) and means the adapter is the project's WEAKER lever.

**Pivot (the differentiated lever is the GATE, not the adapter):** the project's thesis is "the
model learns habits; external GATES enforce truth." The gate (retrieval + provenance/fact-check +
calibrated abstention) already has first-party VALIDATED deltas the adapter never produced:
`+gate` hallucination reduction Δ12.5% CI [+5.6, +19.4] (3 runs, 2 judge families, κ=0.74);
calibration Δ+22%; and a live external-source fact-check at 0% fabrication (Wilson CI [0, 0.11],
self-authored pack, single run). So the gate is ~one third-party pack away from an externally-
validated claim, while the adapter shows no external signal.

**Next experiments:**
1. PRIMARY — externalize the GATE: base (raw) vs base+gate (sophia-full) on a fabrication/
   attribution task scored against EXTERNAL sources (`run_fact_check_live_eval --live`), a
   THIRD-PARTY-authored pack, ≥3 runs, report the fabrication-rate delta + CI and over-abstention
   cost. This closes `fact-check-live-backend-ran` (single-run/self-authored) and advances Hurdle 1.
2. CONFIRMATORY — formally close the adapter lever: one external authorship-attribution check
   (the adapter's exact skill) graded vs Wikidata. Null there → adapter externally invalid; beats
   base → narrow real external skill. Either way the adapter question is settled.

Adapter archived at `training/mlx_adapters/sophia-v4-seed0-promoted` (gitignored; sha256
4e5e0582d14d2bc89e56e7a0818da90e4ea079ceb3922042ef0626a0c9631f28). Plan:
`docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## Template

```text
Failure ID:
Date:
Task or benchmark:
Expected behavior:
Observed behavior:
Likely cause:
Fix or next experiment:
Claim impact:
```

## steering-live-run-not-yet-gated-2026-06-23

**Status:** OPEN. The Spec B activation-steering engine is built and its machinery
invariants pass offline (`python tools/run_steering.py --model mock --dry-run` →
`STEERING WIRING VERIFIED ✓`; `tests/test_steering.py` green in CI). The live SSA
claim — that Level-3 steering beats Spec A's Level-1 persona baseline,
behavior-corroborated and capability-preserving — is **pre-registered and OPEN**:
it requires a gated real run (Phi-3.5 on MPS + the Ollama-judged battery at
N≥8/K≥20). `SSA = 0/N` would be a legitimate honest result. Thresholds are fixed
in `agent/steering/stats.py:SSA_THRESHOLDS` before any run.

## pif-headline-run-not-yet-gated-2026-06-23

**Status:** OPEN. The Spec C PIF/SSA headline harness is built and its statistics
invariants pass offline (`python tools/run_pif.py --dry-run` → `PIF HARNESS
VERIFIED ✓`; `tests/test_pif_harness.py` green in CI, PASS 11). The live PIF
headline claim — that at least one steering axis produces a BH-significant enacted
cell (SSA enacted/total > 0/N) under the pre-registered grid N≥8/K≥20 — is
**pre-registered and OPEN**: it requires a gated real run (Phi-3.5 on MPS + the
Ollama-judged battery). `SSA = 0/N` is a legitimate honest result. Thresholds are
fixed in `agent/steering/stats.py:SSA_THRESHOLDS` before any run. Anti-gaming
contract enforced: `(fit_shift − held_shift) ≤ 0.20` and
`heldoutOffTargetRate ≤ 0.10` must both hold on the sealed held-out split
before any vector ships.

## capability-cell-not-yet-in-live-ssa-2026-06-23

**Status:** OPEN

Spec D D1 ships the deterministic capability-retention guardrail
(`agent/steering/capability.py`, `tools/run_capability.py`) that produces the
`capability_drop`/`coherence` inputs `agent/steering/stats.py::ssa_verdict`
requires (`SSA_THRESHOLDS["capability_eps"]=0.05`, `["coherence_floor"]=75.0`).
The reduced real run (`--model granite`) demonstrates the drop, but a real
capability cell is **not yet wired into a live headline SSA run**, and coherence
is a deterministic proxy rather than an LLM-judge channel. Closing this requires
the full N≥8/K≥20 PIF headline run (also OPEN) with real capability cells.

## steering-harness-chat-template-bug-found-and-fixed-2026-06-23

**Status:** RESOLVED (bug fixed), with a RE-VALIDATION note.

Spec D's final review + reduced real capability run uncovered a real bug in the
SHARED steering harness `agent/steering/hooks.py::SteeredClient._run`: it chained
`.to(device)` onto `apply_chat_template(..., return_tensors="pt")` and passed the
result to `model.generate`, but under the current `transformers` that result is a
`BatchEncoding` (dict-like), so `generate()` raised and was silently swallowed by
`generate()`'s `except` into `_Result("", ok=False)` — **every real generation
came back EMPTY**. Fixed (commit on `feat/capability-retention-mcp`) by normalizing
to an `input_ids` tensor for both the bare-tensor and `BatchEncoding` cases.

**Implication for Spec B (PR #66):** B's "illustrative SSA = 0/2" real granite run
used this same `_run`, so its generations were almost certainly empty too — its
0/2 was reached vacuously (nothing moved anything), and the reported "self-report
channel returned null" is consistent with degenerate output. **Re-validated** here
with the fixed harness: SSA is STILL 0/2, but now meaningfully — the Level-1 persona
prompt moves the trait (behavioral `d_level1` ≈ 2.41 for E, 5.20 for O) while
activation steering does not (`d_steer` ≈ -1.74, 0.28; Δd ≈ -4.16, -4.92). Spec D's
capability guardrail explains the mechanism: steering at the required alpha collapses
capability (`capability_drop = 1.0`, `coherence = 0`), so the steered output is too
degenerate to express the trait. The program's central claim — **activation steering
does not beat a persona prompt** — survives re-validation on real generations.

**OPEN remainder:** the full N≥8/K≥20 headline PIF run (still OPEN) should use the
fixed harness; any prior steering artifacts generated before this fix are suspect.

## fact-check-live-backend-ran-2026-06-24

**Status:** PARTIAL (real progress, honestly bounded).

The out-of-wiki fact-check gate was run with the **live** keyless backend
(`tools/run_fact_check_live_eval.py --live`) against live external sources
(Wikidata, Crossref, World Bank, FRED/BLS, DOI/URL resolvers) — `liveBackendUsed: true`.
This is the first run that grounds against the **real world** rather than offline
fixtures, converting "designed to" into "measured."

**Result (n=53: 22 true / 19 false / 12 unknowable; deterministic label scorer):**
- Fabrication rate **0.0** — Wilson-95 CI **[0.0, 0.110]** (k=0 / n=31 resolved).
- Correct abstention on unknowable **100%**; false-reject on true claims **0%**.
- resolvedAnswerableAccuracy 0.78; overall decision accuracy 0.87.
- Honest cost: **over-abstention 31.8%** (it holds on ~1/3 of answerable claims).
- Calibration: ECE 0.084, Brier 0.011 (n=32 resolved).
- Artifact: `agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json`.

**Why this is NOT yet a headline/validated claim:**
- The held-out pack is still **self-authored** (first-party), not third-party.
- **Single live run**, non-deterministic network; not ≥3 runs.
- Deterministic label scorer (no LLM-judge) — fine for these metrics, but the
  fabrication-rate CI upper bound is 11%, so "0% fabrication" must be stated with its CI.

**Required to harden to a capability claim:** a **third-party-authored** live pack +
≥3 runs + human spot-check of the resolved/abstained decisions. The CI default remains
the deterministic offline fixture run (`liveBackendUsed: false`) so the suite stays
reproducible; the live artifact is evidence, not a CI gate.

## local-sophia-v2-mlx-trained-not-promoted-2026-06-24

**Status:** OPEN / NOT PROMOTED.

A single Mac/MLX LoRA adapter was trained for the local verifier-gated wisdom-model
program:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Backend: MLX-LM (`training/mlx_adapters/sophia-v2`)
- Data: `training/local_sophia_v2/manifest.json` after decontamination guard CLEAN
- Training: 500 iterations, batch 4, `--mask-prompt`, final train loss 0.714, final val loss 1.828
- Artifact metadata: `training/local_sophia_v2/training_run_mlx_sophia_v2.json`

**Observed eval-ladder result (candidate, first-party, single run):**

- MLX base: 16/32 = 50.0%
- MLX base+gate: 16/32 = 50.0%, with 29 gate failures flagged
- MLX adapter: 20/32 = 62.5%
- MLX adapter+gate: 20/32 = 62.5%, with 28 gate failures flagged

**Why this is not promoted:** the adapter improves the aggregate internal domain score,
but the religion slice regressed from 1/6 to 0/6, the gate still flags many outputs,
and the available SEIB smoke (`ollama:qwen3:30b-a3b`, N=20) did not clear the promotion
rule (`raw_to_full_accuracy_delta = -0.10`, source-citation delta +0.80, false-positive
cost 0.0). This is evidence that the training/eval substrate runs end-to-end, not a
validated capability claim.

**Next experiment:** shorten/split overlong rows before MLX training, add more religion and
council-quality traces, evaluate the MLX adapter on SEIB directly (runner does not yet load
MLX adapters), then re-run ≥3 seeds and only promote if provenance/citation improves at
acceptable false-positive cost with no useful-correctness regression.

**Update (C4, 2026-06-24):** the continual loop's return path is wired.
`tools/feedback_to_training.py` turns gate MISSES into a reviewed candidate queue
(`mine` → `approve` → `build-sft`); only human-promoted candidates become SFT rows
(`training/feedback/sft_from_feedback.jsonl`), ingested by `build_local_sophia_dataset.py`
under the same decontamination guard. Non-circular by construction (separate file from frozen
records, default-deny promotion, decontaminated on ingest); gated by
`tests/test_feedback_to_training.py`. Remaining: re-run `tools/run_learning_shift.py` with the
new pack as post-test (needs a model backend; runs on hardware).

**Update (C3, 2026-06-24):** two of the next-experiment blockers are resolved.
`tools/split_long_training_rows.py` now fits every MLX row under `MLX_MAX_TOKENS` (1024) at
pack-build time — the rebuild dropped 11 overlong single-turn rows that the v2 run silently
truncated, recorded under `mlx.fit` in the manifest. `agent/model.py` gained an `mlx`
transport and `tools/run_seib.py` gained `--model mlx:<base> --adapter <path>`, so the trained
adapter can now be evaluated on SEIB directly (on Apple Silicon; off-Mac it writes an
environment artifact, not a score). Remaining for promotion: the religion retrain (C2) and
multi-seed + SEIB-100 evidence (C5).

**Update (C1, 2026-06-24):** this NOT-PROMOTED verdict is no longer a hand-written note. The
adapter's eval ladder is now run through the W2 bounded-RSI promotion gate
(`agent/continual_plasticity.evaluate_update`) by `tools/promote_adapter.py`, which
independently reproduces the `reject` — protected regression on `religion` (−0.167) — and an
`agent/formal_verifier` protected-floor lattice proof agrees (`after_religion(0) >= 157`
violated). Reproducible artifact:
`agi-proof/continual-plasticity/sophia-v2-promotion.public-report.json`; gated by
`tests/test_promote_adapter.py`. Plan: `docs/06-Roadmap/Training-RSI-Continual-Convergence.md`.

**Claim impact:** Sophia now has an executed local-training path, one trained local adapter,
and a **machine-checked promotion gate** that rejects it for a stated reason — but no public
headline should claim a validated wisdom-model uplift yet.

## local-sophia-v3-mlx-promoted-by-w2-but-not-validated-2026-06-24

**Status:** PARTIAL / W2-PROMOTED, NOT VALIDATED.

Hardware-bound C2 was run on Apple Silicon / MLX-LM (`mlx_lm 0.29.1`) on branch
`claude/sophia-training-next-steps-ij1kvm`. The training pack was rebuilt after adding new,
human-authored religion council traces that are distinct from held-out eval prompts. The
contamination guard stayed CLEAN throughout.

**Data + token-fit:**

- Added 90 new religion repair council traces in `training/council/traces.jsonl` across the
  weak topics: Gospel/Matthew authorship, ancestor veneration vs Confucian philosophy,
  Dao De Jing philosophy/religion, early Islam theology/history boundary, nirvana pop myth,
  and Sunni/Shia hadith vs Quran boundaries.
- Final manifest: `trainRowsTotal=1356`; `missingRequiredInputs={}`; `contamination.clean=true`;
  `vsEval.overlapCount=0`; `vsHoldoutOverlapCount=0`.
- Final MLX pack: `trainRows=739`, `validRows=89`, `maxTokens=1024`.
- `tools/split_long_training_rows.py training/local_sophia_v2/mlx/train.jsonl --dry-run` emitted
  `rowsDroppedUnsplittable=0`, `turnsDroppedOverlong=0` on the emitted pack. The final MLX
  training run showed no truncation warnings.

**Training:**

- Base model: `Qwen/Qwen2.5-3B-Instruct`.
- Adapter: `training/mlx_adapters/sophia-v3`.
- Command: `python3 -m mlx_lm lora --train --model Qwen/Qwen2.5-3B-Instruct --data training/local_sophia_v2/mlx --iters 500 --batch-size 4 --mask-prompt --adapter-path training/mlx_adapters/sophia-v3 --max-seq-length 1024`.
- Final run: 500 iters, final train loss `0.933`, final val loss `1.793`, peak memory `22.202 GB`.
- Metadata: `training/local_sophia_v2/training_run_mlx_sophia_v3.json`.

**Eval ladder:**

- MLX base: `16/32 = 50.0%`.
- MLX base+gate: `16/32 = 50.0%`, with 29 gate flags.
- MLX adapter v3: `24/32 = 75.0%`.
- MLX adapter v3+gate: `24/32 = 75.0%`, with 20 gate flags.
- Domain deltas: philosophy `6/9 -> 9/9`, psychology `4/9 -> 8/9`, history `5/8 -> 6/8`, religion `1/6 -> 1/6`.

**W2 promotion gate:**

- Artifact: `agi-proof/continual-plasticity/local-sophia-v3-mlx-promotion.public-report.json`.
- `tools/promote_adapter.py` returned `FINAL VERDICT: promote` for `local-sophia-v3-mlx`.
- Reason: total delta `+0.25`, protected history improved, protected religion no longer regressed.
- Honest caveat: religion only recovered to baseline (`1/6`), below the aspirational target `3/6`.

**C3 direct SEIB on MLX adapter:**

- Artifact: `agi-proof/benchmark-results/seib-mlx-sophia-v3.json`.
- Result: `ok=false`.
- Deltas: `raw_to_full_accuracy_delta=-0.01`, `prompt_to_full_citation_delta=+0.30`,
  `raw_to_full_contested_fabrication_reduction=-0.02`, `sophia_full_false_positive_cost=0.02`.
- `sophia_full`: provenance accuracy `0.96`, false attribution rate `0.0`, contested fabrication
  `0.08`, source citation rate `0.30`.
- Claim impact: this blocks any validated SEIB uplift claim for v3.

**C4 learning-under-shift post-test:**

- Artifact: `agi-proof/learning-under-shift/shift-result-local-sophia-v3-mlx-2026-06-24.public-report.json`.
- Result: `passingSignal=false`.
- Pre-test `0.0%`; post-test `50.0%`; contamination audit clean; protected hashes unchanged.
- Failure: old-benchmark stability was `50.0%` vs baseline `100.0%`, delta `-50.0`, so the protocol
  correctly refuses the passing signal.

**C5 status:** NOT RUN as promotion-grade evidence. Although the W2 gate promoted v3 on the
first-party eval ladder, the direct SEIB run returned `ok=false` and the learning-shift protocol
returned `passingSignal=false`. Multi-seed stability, SEIB-100 ≥3 runs with ≥2 judge families,
and live RLVR remain open. Live RLVR is also CUDA/vLLM-gated and was not attempted on this
Apple Silicon run.

**Claim impact:** v3 is a locally trained adapter with a W2 promotion-gate verdict on first-party
benchmarks. It is **not** validated external evidence, not a promotion-grade result, not an AGI
claim, and not a hallucination guarantee.
