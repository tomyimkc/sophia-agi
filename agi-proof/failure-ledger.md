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
| rlvr-math-live-run-not-yet-run-2026-06-24 | Cleared (rung) | First (2026-06-24) run was WITHIN NOISE (N=8, mean Δ +0.083, CI incl. 0). RE-RUN 2026-06-25 on the larger non-gameable N=60 fixed-held-out pack via the fast vLLM-colocate stack (trl 0.19.1 + vllm 0.9.1, `accelerate launch`) CLEARS the rung gate: 3 seeds, base **0/60 every seed** → adapter 7/60, 6/60, 5/60, all Δ>0 (mean **+0.10**), **95% across-seed CI [0.059, 0.141] excludes 0**, contamination-free, no regression, judge-free deterministic verifier, family-disjoint held-out. Evidence: `agi-proof/self-extension/math-rlvr-3seed-n60/`. HONEST SCOPE: modest/narrow (~10% where base floors at 0%) — clears THIS rung, NOT an AGI claim; `canClaimAGI` stays False | Optional: scale pack/epochs for a larger effect; the loop + fast harness are proven |
| local-agent-tools-degrade-strong-model-2026-06-21 | Closed | FIXED: selective invocation (tools fire only on low-confidence answers) + richer tool outputs (wiki_search snippets, belief wiki fallback) eliminated the degradation — on qwen3:30b-a3b `+mcp-tools` now *beats* alone (gold 90.2%→92.7%, false-positive 9.8%→7.3%), was 90.2%→51.2% before | — |
| local-agent-delta-strong-model-headroom-2026-06-21 | Superseded | Single-LEXICAL-judge run on dolphin-llama3:8b showed alone 15.2% → +gate 4.3%. This did NOT survive validation — see below. `+mcp-tools` 0.0% was re-generation, NOT tool-use (`toolsUsed: []`). | Superseded by `local-agent-delta-not-validated-2026-06-21` |
| local-agent-delta-not-validated-2026-06-21 | Closed | RESOLVED by the benchmark expansion (#6, 87→290 cases) + the unified harness (#1). The earlier N=46 run's CI straddled zero; on the expanded set a validated run (3 runs, 2 judge families = openrouter:deepseek + openrouter:meta-llama) gives the +gate lever halluc alone 36.1% → gated 23.6%, **Δ12.5%, 95% CI [+5.6%, +19.4%] EXCLUDES zero**, 0% FP-cost → `validated=True`. Recorded in RESULTS.md / published-results.json. | — |

| grounded-gate-not-yet-validated-2026-06-22 | Open | The retrieval-grounded gate (check_claim ground=True) is verified bug-fixed (no pen-name false positives; catches known-author misattributions for out-of-corpus works) but a 3-run/2-family N=24 run gave +gate Δ8.3%, 95% CI [0.000, +16.7%] — lower bound touches zero, so illustrative not validated (vs the prior non-grounded validated Δ12.5%). Sampling variance at small N. | Re-run grounded at larger N (>=40 cases / more runs) to push the CI off zero |

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

## agi-shaped-architecture-pilots-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE ONLY (`candidateOnly: true`, `level3Evidence: false`).

Implemented three verifier-gated research harnesses from the feasibility plan. **Not AGI claims.**

**Phase 1 — Shadow OKF bulk-boundary lattice:**

- Modules: `okf/bulk_graph.py`, `okf/projection.py`, `tools/run_shadow_lattice.py`
- Artifact: `agi-proof/shadow-okf-lattice/shadow-lattice.public-report.json`
- Demo: promoted `2/3` bulk nodes; lineage trap abstained (`bulk_lineage_trap`)
- Honest bound: quarantined cross-tradition exploration; bulk never shipped raw

**Phase 2 — REM dream collective + symbiosis network:**

- Modules: `agent/dream_collective.py`, `skills/symbiosis_network.py`, `tools/run_dream_collective.py`
- Artifact: `agi-proof/dream-collective/dream-collective.public-report.json`
- Demo: REM blocked `1/2` eval-leak dreams; symbiosis held bad nutrient packets
- Honest bound: offline consolidation seam; wake uses conscience + `memory_consolidation`

**Phase 3 — Embryogenesis crucible arena:**

- Modules: `agent/embryogenesis/`, `tools/run_crucible_arena.py`
- Artifact: `agi-proof/embryogenesis-crucible/crucible-arena.public-report.json`
- Demo: population `8`, generations `2`, top fitness `0.8286` (trap-weighted stub scoring)
- Honest bound: verifier config search only; `weightsFrozen: true` — no LoRA reproduction

**C2 religion repair data (prerequisite for adapter retrain):**

- Added `training/council/religion_repair_c4.jsonl` (12 council-panel traces)
- Builder: `tools/build_religion_repair_traces.py` — `evalOverlapCount=0`, contamination CLEAN
- Wired into `tools/build_local_sophia_dataset.py` as `sft_religion_repair_c4.jsonl`
- **Not yet proven:** religion ladder uplift requires hardware retrain + eval; v3 remains `1/6`

**Claim impact:** Strengthens provenance-aware exploration substrate. Does not prove AGI, validated
uplift, or zero hallucination. `python tools/lint_claims.py` passes.

## agi-shaped-next-steps-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE + LOCAL TRAIN/EVAL (`candidateOnly: true`, `level3Evidence: false`).

Closed the feasibility-plan next steps: conscience in projection, promotion loop, REM wake proof,
and C2 religion-repair retrain on Apple Silicon. **Not AGI claims; not validated external uplift.**

**Phase 1 — Conscience-wired projection:**

- `okf/projection.py` calls `conscience_check` on bulk body text during `project_node`
- Fail-closed: `block`/`abstain`/`escalate`/`retrieve`/`clarify` abstain; `allow`/`revise` proceed
- `skip_conscience` for offline CI (mirrors `skip_provenance`)
- Shadow lattice demo bodies include candidate-only boundary wording so conscience passes offline

**Phase 2 — Promotion loop (projection → boundary queue):**

- `okf/promotion_loop.py`: `submit_projection_candidates()` → `training/feedback/pending_projection_candidates.jsonl` (`promoted: false`)
- `commit_approved_candidate()` for human-gated `wiki_store.upsert`
- `tools/run_shadow_lattice.py` logs promotion artifact after projection

**Phase 3 — REM wake consolidation proof:**

- `agent/dream_collective.py` wake uses memory-mode conscience + `trustUpstreamVerdict`
- Demo dream passes conscience and consolidates: wake `1/1` consolidated, eval-leak `1/2` blocked
- Artifact: `agi-proof/dream-collective/dream-collective.public-report.json`
- `tools/run_dream_collective.py --cron` docstring for scheduled runs

**C2 religion repair retrain + eval (Apple Silicon, mlx_lm):**

- `tools/build_religion_repair_traces.py`: 12 rows, `evalOverlapCount=0`, contamination CLEAN
- `tools/build_local_sophia_dataset.py`: 751 MLX train / 89 valid rows, contamination CLEAN
- Short MLX LoRA: 50 iters → `training/mlx_adapters/sophia-v4-religion-repair` (gitignored weights)
- `tools/eval_ladder.py --backend mlx --adapter …/sophia-v4-religion-repair`:
  - base `16/32 = 50.0%`, adapter `23/32 = 71.9%` (delta `+21.9%`)
  - religion `1/6 → 2/6` (`16.7% → 33.3%`); history `5/8 → 7/8`; philosophy `6/9 → 8/9`
- `tools/promote_adapter.py`: `FINAL VERDICT: promote` for `local-sophia-v4-religion-repair-mlx`
- Honest bounds: 50-iter smoke train only; religion still `2/6` (below aspirational `3/6`);
  no SEIB / learning-shift re-run; adapter weights not in repo

**Skipped / not proven:**

- Human-approved wiki boundary commit (queue only; default-deny)
- Full-length train, multi-seed stability, SEIB-100, learning-under-shift post-test
- CUDA/vLLM live RLVR

**Claim impact:** Wires conscience + promotion seam + REM wake demo; v4 shows first-party ladder
gain on religion repair data with W2 promote verdict. Not validated external evidence or AGI.
`python tools/lint_claims.py` passes.

### Format-fitting caveat (reviewer defect #2, 2026-06-25)

The v4 religion combined uplift **1/6 → 2/6** may reflect **council-panel FORMAT learning**
rather than substantive content repair:

- On re-score with split channels (`tools/rescore_religion_channels.py`), v4 vs Qwen2.5-3B baseline:
  - **FORMAT:** 1/6 → 2/6 (+1 case)
  - **CONTENT:** 5/6 → 5/6 (**no change**)
  - **Combined:** 1/6 → 2/6 (+1 case)
- Of 6 religion cases, **4/6** combined failures on v4 are format-graded (`mustUseCouncilPanel`);
  the 12 `religion_repair_c4.jsonl` traces teach that panel structure explicitly.
- Entity-level decontamination remains CLEAN (`evalOverlapCount=0`), but the eval conflated
  format compliance with content correctness until Task 4 split the scorer.
- **Confidence in religion uplift is lowered** until a retrain targets CONTENT channel ≥3/6
  without format-only Goodhart. Artifact: `agi-proof/religion-channel-rescore/religion-channels.public-report.json`.

## agi-pilots-feasibility-review-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Task 1 — Crucible determinism:**

- Added `--seed` (default 0) to `tools/run_crucible_arena.py`; generality stub uses `zlib.crc32`
  instead of salted `hash()`; `seed` recorded in arena report.
- Two consecutive `--seed 0` runs produce **byte-identical** JSON; `topFitness: 0.8286` reproducible.
- Regenerated `agi-proof/embryogenesis-crucible/crucible-arena.public-report.json`.

**Task 3 — Promotion loop default-deny:**

- `approve_projection_candidate()` + idempotent `commit_approved_candidate()`; full body stored at submit.
- Demo: `tools/run_promotion_loop_demo.py` → `agi-proof/promotion-loop/promotion-loop.public-report.json`
- Tests: default-deny + approve-once idempotent commit.

**Task 4 — Religion FORMAT vs CONTENT channels:**

- `score_case_format` / `score_case_content` / `score_case_channels` in `agent/benchmark_checks.py`
- `tools/rescore_religion_channels.py` on committed v4 artifacts:
  - baseline FORMAT 1/6, CONTENT 5/6; v4 FORMAT 2/6, CONTENT 5/6
  - **Honest finding:** v4 combined +1 is **format-only**; content channel flat.

**Task 5 — Full retrain:** deferred pending Task 4 completion (now unblocked); requires GPU session.

## agi-pilots-task5-v5-full-retrain-2026-06-25

**Status:** HONEST NEGATIVE on religion CONTENT target (`candidateOnly: true`, `level3Evidence: false`).

**Train (Apple Silicon, mlx_lm, 500 iters):**

- Pack: `training/local_sophia_v2/mlx` (751 train / 89 valid, contamination CLEAN)
- Adapter: `training/mlx_adapters/sophia-v5-full-religion-repair` (gitignored weights)
- Final train loss `0.695`, final val loss `1.838`, peak mem `22.203 GB`

**Eval ladder (combined channel, legacy):**

- Base `16/32 = 50.0%` → adapter `21/32 = 65.6%` (delta `+15.6%`)
- Religion **1/6 → 1/6** (no combined uplift; below aspirational **3/6**)
- History `5/8 → 5/8` (protected, no regression)

**Split-channel rescore** (`agi-proof/religion-channel-rescore/religion-channels-v5.public-report.json`):

- vs Qwen2.5-3B baseline: FORMAT **1→2**, CONTENT **5→4** (regression), combined **1→1**
- **Target not met:** CONTENT channel **4/6 < 3/6** goal was wrong direction — content regressed
- v4 smoke (50 iters) had CONTENT 5/6; full train **hurt** content while adding format cases

**W2 promotion gate:**

- `tools/promote_adapter.py`: **promote** (no protected regression vs baseline ladder)
- Artifact: `agi-proof/continual-plasticity/sophia-v5-full-religion-repair-promotion.public-report.json`
- Honest caveat: promote reflects baseline-relative protected floors, not religion CONTENT success

**Claim impact:** Full retrain does not validate religion repair; format/content split exposed
content regression. Not AGI. Do not retrain-to-fit without new content-focused traces.

## religion-repair-lora-path-falsified-2026-06-25

**Status:** Closed / Falsified (`candidateOnly: true`, `level3Evidence: false`).

**Three-channel religion ladder (Qwen2.5-3B baseline → v4 50-iter → v5 500-iter):**

| Run | FORMAT | CONTENT | COMBINED |
|---|---|---|---|
| Baseline | 1/6 | 5/6 | 1/6 |
| v4 smoke (50 iter) | 2/6 (+1) | 5/6 (0) | 2/6 (+1) |
| v5 full (500 iter) | 2/6 (+1) | 4/6 (−1) | 1/6 (0) |

N=6 ⇒ ±1/6 is within noise; the v5 CONTENT regression is still a protected-floor breach
under the corrected CONTENT-channel gate.

**Corrected W2 gate (CONTENT + invariant oracle):**

- Legacy COMBINED-only gate: **promote** (false positive — blind to CONTENT regression)
- Corrected gate: **reject** — `protected_floor_content` breached (religion CONTENT 5/6 → 4/6)
- Proof bundle: `agi-proof/self-gate/invariant-suite.local-sophia-v5-full-religion-repair-mlx.public-report.json`

**Conclusion:** Weight-training on 12 council-panel religion repair traces is the **wrong
lever** for this benchmark. FORMAT uplift is inference-time structure (council prompt /
gate), not LoRA weights. CONTENT channel did not improve and regressed under full train.
v4/v5 artifacts retained for audit. **Do not retrain religion repair LoRA.**

**Claim impact:** Falsifies the religion-repair LoRA hypothesis. `canClaimAGI: false`.

## okf-local-global-consistency-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Scope:** Syntactic local-global consistency over OKF pages — finds undeclared
cross-context disagreements (epistemic holes) when the same entity carries
different asserted claims in different tradition partitions. Declared ``contradicts``
edges defer to ``contradiction_ledger`` (no double-report). Does **not** decide
truth or auto-generate training facts.

**Wiki run** (`python3 tools/run_consistency_check.py`):

| Metric | Count |
|---|---|
| Contexts (tradition partition) | 17 |
| Entities spanning >1 context | 0 |
| Undeclared epistemic holes | 0 |
| Declared contradictions deferred | 0 |

**Synthetic unit-test gate** (`tests/test_consistency_check.py`):

| Metric | Count |
|---|---|
| Holes in fixture graph | 1 |
| Patch candidates passed provenance gate | 1 |
| Rejected (no source citation) | 1 |

**Artifacts:** `okf/consistency_check.py`, `tools/run_consistency_check.py`,
`agi-proof/okf-consistency/consistency.public-report.json`,
`training/feedback/epistemic_holes.jsonl` (queue; empty on wiki until holes appear).

**Claim impact:** Consistency escalation only; `canClaimAGI: false`. No adapter
promotion or protected-suite change.

## okf-referent-attribution-consistency-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Prior false negative:** Tradition partition (`partitionKey: tradition`) reported
0 entities spanning >1 context and 0 holes — entities are tradition-bound by design.
Real conflict axis is **attribution** via `links`, `attributedAuthor`, and
`doNotAttributeTo` on shared referents (works/figures).

**Re-key:** Default checker mode is referent-attribution (`partitionKey: referent`).
Undeclared hole when >=2 pages assert conflicting attribution on the same referent:
(a) different non-empty `attributedAuthor`, or (b) one page attributes author X while
another lists X in `doNotAttributeTo`. Declared ``contradicts`` / ledger tradition-merge
rows defer (counted, not re-emitted). Consistency checks **not** truth.

**Wiki run** (`python3 tools/run_consistency_check.py`):

| Metric | Count |
|---|---|
| Shared referents (>=2 pages) | 1 |
| Undeclared epistemic holes | 0 |
| Declared contradictions deferred | 0 |
| Gate pass | yes |

**Synthetic unit-test gate** (`tests/test_consistency_check.py`):

| Metric | Count |
|---|---|
| Referent dnm-violation hole (type b) | 1 |
| Declared contradicts deferred | 1 |
| Patch rejected (no source) | 1 |
| Patch accepted (grounded source) | 1 |

**Artifacts:** `okf/consistency_check.py`, `tools/run_consistency_check.py`,
`agi-proof/okf-consistency/consistency.public-report.json`,
`training/feedback/epistemic_holes.jsonl` (empty — no undeclared holes on wiki).

**Claim impact:** Consistency escalation only; `canClaimAGI: false`.

## gate-z3-backend-deadcode-found-and-fixed-2026-06-25

**Status:** RESOLVED (bug fixed) + accept-path validated.

Installing `z3-solver` and running a POSITIVE CONTROL through the invariant
oracle (`tools/run_positive_control.py`) exposed that the z3 backend of
`agent/formal_verifier.py::_z3_lattice` had **never executed**. The accept-path
unit tests (`test_godel_oracle`, `test_invariant_suite`) mock `require_z3`, but
z3 was not installed in any environment, so `check_lattice_consistency` always
took the pure-Python fallback. The z3 branch contained a latent crash:
`vs.get(lhs, z3.IntVal(_as_int(lhs)))` evaluates the default eagerly, so a NAMED
variable on the LHS (every production invariant, e.g.
`content_after_religion >= floor`) was passed to `z3.IntVal(None)` →
`Z3Exception: parser error`. The whole gate was fallback-only by accident.

**Fix:** lazy operand resolution in `_z3_lattice` mirroring the fallback's
unbound-variable handling (held/error instead of crash). Added a real-z3
regression test (`test_lattice_named_variable_runs_on_z3_backend`) that pins
accept+reject verdicts and `backend == "z3"`; generalized the unbound-variable
test to both backends; added `z3-solver` to the CI test install so the path is
exercised going forward.

**Accept-path validation:** with z3 installed, a synthetic known-good candidate
(`positive-control-synthetic`) now promotes with all five invariants `accepted`
on the **z3** backend — the gate is proven to ACCEPT a good candidate, not only
to REJECT (v5 still correctly rejects on `protected_floor_content`). Artifact:
`agi-proof/self-gate/invariant-suite.positive-control-synthetic.public-report.json`.

**Claim impact:** the self-gate's formal proofs are now actually solver-checked,
not silently fallback-only. Decidable numeric invariants only; not alignment,
not AGI, not a Gödel machine. canClaimAGI stays False.

## agi-pilots-way-forward-gate-rollup-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Task 1 — z3 hard promotion requirement (`solverChecked`):**

- `build_proof_bundle` / promotion reports stamp top-level `solverChecked` (= every invariant `backend == "z3"`).
- Default: promotion blocked when `solver_attestation` is `held` (no z3) or any invariant uses fallback.
- `--allow-fallback-proof` (OFF by default): may promote with `solverChecked: false` + note
  `fallback proof — not solver-checked`.
- Fixed z3 `dict.get` eager-default bug in `agent/formal_verifier.py`.
- Positive control (`tools/run_positive_control.py`): **promote True**, all five invariants **z3**, `solverChecked: true`.

**Task 2 — CONTENT is pass gate:**

- `tools/eval_ladder.py`: headline `passed` / `score_pct` = **CONTENT** channel; FORMAT + COMBINED reported only.
- Regenerated `training/local_sophia_v2/eval_ladder_{baseline,adapter}.json` with `passGate: content`.
- Baseline religion CONTENT **5/6** unchanged; no training improved it — measurement artifact fix, **not** capability uplift.

**Task 3 — Council-panel format at inference (no weights):**

- `agent/council_format.py` + `--religion-council-panel` on `eval_local_model.py` / `eval_mlx_model.py`.
- Qwen2.5-3B base, same model, religion N=6 (`benchmark/model_runs/local-qwen-qwen2.5-3b-instruct-council-panel-religion.report.json`):

| Template | FORMAT | CONTENT | COMBINED |
|---|---|---|---|
| WITHOUT | 1/6 | 5/6 | 1/6 |
| WITH council panel | 6/6 | 5/6 | 5/6 |

- **Honest finding:** FORMAT uplift is inference-time structure, not LoRA weights. CONTENT flat (no regression); ship template.
- N=6 ⇒ ±1/6 within noise; **no religion uplift claim**.

**Task 4 — `provenance_complete` on 12 religion repair traces:**

| | lackingCount | verdict |
|---|---|---|
| Before (`religion_repair_c4.jsonl` without `metadata.sourceCitation`) | 12 | rejected |
| After (citations → `data/attributions.json` / `data/traditions.json` keys) | 0 | accepted (z3) |

- `tools/build_local_sophia_dataset.py --check`: contamination **CLEAN**.

**Task 5 — End-to-end gate re-confirmation (z3 installed):**

| Candidate | oracle promote | solverChecked | FINAL |
|---|---|---|---|
| Positive control | True | True | accept |
| v5 full religion repair | False (`protected_floor_content`) | True | **reject** |
| Known-good cited fixture | True | True | **promote** |

**Claim impact:** z3 solver-checked promotion is now explicit policy; CONTENT channel is the ladder pass gate;
religion FORMAT is prompt-structurable at inference; religion-repair LoRA path remains falsified.
`canClaimAGI: false`.

## sophia-7b-train-verify-data-flywheel-2026-06-25

**Status:** STAGE 0–1 COMPLETE; STAGE 2+ BLOCKED (SSH egress timeout to RunPod mapped pod ports from local Mac / Cursor agent shell).

**Pre-registration (Stage 0):** `agi-proof/sophia-7b-train-verify/preregistration.json`,
oracle split `oracle-split.md`, holdout seal `heldout-seal.manifest.json`
(`contentHash: 84d00bdc36205abdb5a162530d8fc972ee27075c053bfd0615de59d3ed9aeb97`, 89 rows).
Commit `9f00733`.

**Data flywheel (Stage 1) — `training/local_sophia_7b/`, base `Qwen/Qwen2.5-7B-Instruct`:**

| Pack | Rows | Decontam dropped |
|---|---:|---:|
| sft_source_discipline | 439 | 89 |
| sft_wiki_provenance | 34 | 0 |
| sft_council_traces | 125 | 0 |
| sft_religion_repair_c4 | 12 | 0 |
| sft_moral_gate | 35 | 0 |
| general_instruct | 120 | 0 |
| dpo_hard_negatives | 590 | 25 |
| dpo_wiki_provenance | 34 | 0 |
| **train total** | **1389** | **114** |
| holdout (sealed) | 89 | — |
| MLX SFT train / valid | 754 / 89 | 11 turns dropped overlong |

`build_local_sophia_dataset.py --check`: contamination **CLEAN** (0 eval overlap, 0 holdout overlap).
Hard-negative miner: 615 pairs (gate-validated). Moral Gate SFT: 35 rows.

**Stage 2 blocker (2026-06-25, updated @ `8975744`):**

1. **This session (2026-06-25T11:10Z):** `RUNPOD_API_KEY` **present**. Stage 0 gates re-verified
   (contamination CLEAN, holdout seal `84d00bdc…`, lint_claims OK). SSH smoke probe
   (`sophia-7b-ssh-smoke`, interruptible) created pods `g6de2tbp9jzge1`
   (`213.173.109.78:15792`) and `6l4go54e2n4f54` (`213.173.107.230:12881`); RunPod API
   reported SSH mapping but **outbound SSH login timed out** (300s wait, 2 attempts each).
   Pods deleted. Log: `agi-proof/benchmark-results/runpod-train/ssh-smoke-20260625-111011.log`.
   Local Mac outbound TCP/22 to `github.com` and `ssh.runpod.io` **passes**; mapped pod
   high-ports **do not**.
2. **Prior session (same day):** pods `crdl0788rpc98m` / `8k7vqe3m5nbynv` — same
   `ssh_login_timeout` pattern. Logs: `sft-3seed-20260625-095511.log`, `sft-seed0-retry.log`.
3. **Earlier session:** `RUNPOD_API_KEY` unset — dry-run only (`868fa31`).

**Next step:** run `runpod-sft-3seed.sh` from a host with outbound egress to RunPod mapped pod
ports (not Cursor agent shell): e.g. local Terminal outside agent, GitHub Actions
`speedup-runpod` workflow pattern, or RunPod web terminal. Do **not** cite 0/3 seeds as a
training verdict.

**Stage 3 prep (no GPU):** `tools/train_dpo.py` (TRL `DPOTrainer`), `runpod-dpo-3seed.sh`
(DPO on `dpo_hard_negatives.jsonl`, 590 pairs in pack / 615 mined), pre/post internal
`eval_ladder` on pod. Blocked on Stage-2 SFT adapter tarballs per seed.

**Stages 5–6:** NOT RUN (no adapter). Local wiring OK: `run_positive_control.py` ✓,
`promote_adapter` + invariant oracle policy unchanged (release gate — NOT evidence).

**Stage 7 blockers:** `VECTARA_*` credentials absent; hidden reviewer pack needs served model +
backend credentials. Do **not** substitute internal gate for third-party evidence.

**Headline (falsifiable, OPEN):** Qwen2.5-7B QLoRA SFT (≥3 seeds) has **not** started — blocked
on SSH egress timeout from Cursor agent host to RunPod mapped pod ports (API key OK; smoke
reproduced 2026-06-25); **0/3 seeds**, not a training verdict.

**Tradeoff (pre-registered, not yet measured):** abstention/MMLU-Pro regression ≤2.0 points vs base;
honest reporting required even if internal release gate passes.

**canClaimAGI:** False.
