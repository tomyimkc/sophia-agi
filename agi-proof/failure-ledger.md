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
