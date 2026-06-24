# Plan — Hurdles 2–5 toward a defensible capability claim

**Status:** working plan. Nothing here is an AGI claim. Every item names its honest
bound and ships under `tools/lint_claims.py`. The binding prerequisite under *all*
of it is Hurdle 1 (external/third-party validation, `hidden-review-third-party-not-run`):
until an independent party reproduces a result on external benchmarks, every number
below stays first-party.

## Framing — the substrate is more built than it looks

The hurdle report reads as if 2–5 are unstarted. They are not. The repo already
contains, **offline / first-party**:

- verifier *synthesis* with held-out precision/recall gating (`selfextend/verifier_synthesis.py`),
- conformal abstention with a finite-sample coverage guarantee (`agent/conformal_gate.py`),
- environment-as-verifier execution (`selfextend/env_verifier.py`),
- reward-hacking detection (`selfextend/verified_reward.py::reward_is_hackable`),
- generational compounding with CI-separated promotion and convergence detection
  (`agent/ssil_generations.py`, `agent/ssil_compound.py`),
- a long-horizon logging + autonomy-classification harness (`tools/run_long_horizon.py`).

So for each hurdle the gap is narrow and specific: the machinery exists but has only
been exercised on **synthetic / first-party** data, and the genuinely-hard research
boundary is smaller than the report implies.

**Key dependency:** Hurdles 2 and 3 are the same hurdle from two angles. Transfer (2)
is bounded by verifier reach (3): the loop transfers exactly as far as a cheap, *sound*
verifier can be built. Treat them together.

---

## Hurdle 2 — Broad transfer  *(STARTING HERE)*

**Claim under test:** "run the identical retention + promotion loop on ≥2 structurally
different task families and show it helps in each." If it only helps where provenance
verifiers were hand-built, it is a provenance method, not a general one.

**Where the repo stands.** The transfer-blocking coupling is thin:

- *Domain-agnostic already:* `tools/eval_ladder.py` (4-rung structure), `agent/verifiers.py`
  (`Verifier` interface), `agent/continual_plasticity.py::evaluate_update`,
  `tools/promote_adapter.py` (lattice proof).
- *Provenance-specific:* `agent/benchmark_checks.py::score_case` (attribution traps),
  `provenance_faithful()`, and 100% of `training/examples/`.
- *Present but unwired:* `code_tests_pass()` (executes Python, checks exit code) and
  `arithmetic_sound()` (recomputes `a OP b = c`) are real, tested verifiers sitting unused.
  `tools/hidden_eval_protocol.py` already declares 8 domains incl. `logic, coding, planning`.

**Why coding + math first.** They are structurally maximal-distance from provenance
(execution-grounded / numeric-grounded vs corpus-grounded), and their verifiers are
*sound* and *already in-repo*. They are also the on-ramp to Hurdle 1: coding + a unit-test
verifier is SWE-bench in miniature; math + answer-match is GSM8K in miniature.

**Steps**

- **2.1 — Domain-scorer registry (no GPU).** Introduce `agent/domain_scorers.py`: a
  `domain → score_fn(case, response, ctx) -> (ok, reasons)` registry. Register
  `provenance` (wraps the existing `score_case`), `math` (extract final answer +
  `arithmetic_sound` soundness veto), `coding` (wraps `code_tests_pass`). Both eval
  backends (`eval_local_model.py`, `eval_mlx_model.py`) dispatch through it. **Status: in this PR.**
- **2.2 — Normalized benchmarks (no GPU).** `tests/benchmark-math.json` and
  `tests/benchmark-coding.json` in the same `{"cases":[...]}` schema the pipeline expects,
  built from the existing `data/capability_arithmetic.json` and `eval/coding/smoke.jsonl`.
  **Status: in this PR.**
- **2.3 — Ladder/promotion parameterized by family (no GPU). ✅ done.** `tools/eval_ladder.py`
  takes `--domains`; `tools/promote_adapter.py` already took `--protected`, and a test now
  locks in that the W2 gate promotes/rejects a coding+math adapter with a configurable
  (or empty) protected set — no provenance hardcoded
  (`tests/test_promote_adapter.py::test_promotion_gate_generalizes_to_non_provenance_families`).
- **2.4 — Run the identical loop on each family (hardware).** Train one adapter per family,
  run the 4-rung ladder + W2 gate + feedback miner, report per-family deltas. **Status: hardware-bound (like C2/C5).**

**Acceptance.** The identical scorer/ladder/gate path produces a gated result on coding
AND math AND provenance, each with its own sound verifier. **Honest bound:** demonstrates
the *method* generalizes across verifier-backed families — not that it reaches families
where no sound verifier exists (that is Hurdle 3).

---

## Hurdle 3 — How far sound verification reaches  *(the deep one)*

**Where the repo stands.** A three-tier soundness taxonomy already exists in code:
Tier 1 sound (formal/Z3, arithmetic, code-exec, temporal, secret-leak, deontic);
Tier 2 sound-vs-frozen-table (provenance, temporal facts); Tier 3 measured-not-sound
(LLM-judge / NLI / lexical). Plus `verifier_synthesis`, `conformal_gate`, `reward_is_hackable`.

**The sharpened question:** a *verification–generation asymmetry map* — for which families
is checking cheaper/more reliable than producing? Sound verifiers exist where the
asymmetry is favorable; where verification is as hard as generation, no sound verifier
can exist *by construction*, and the correct behavior is calibrated abstention.

**Steps**

- **3.1** Publish the verifiability map as a first-class artifact: classify each candidate
  domain as (a) sound verifier exists, (b) reducible to sound sub-claims, (c) conformal-bounded
  only, (d) unverifiable → abstain. Promote `fact_check_gate.classify_claim` into this map.
- **3.2** Reduction as the main reach-extender: decompose open-ended outputs into the
  Tier-1 fragment (citations, arithmetic, dates, code) + abstain on the residue. Lean on
  `fact_check_gate.decompose_and_type`.
- **3.3** Make calibrated abstention the headline product beyond soundness: report the
  coverage guarantee + over-abstention cost (cf. the live fact-check run: 0% fabrication,
  31.8% over-abstention). Frame over-abstention as the *measured price of soundness*.
- **3.4** Extend `verifier_synthesis` template classes beyond decision stumps for new families.
- **3.5** Keep Tier-3 generative/LLM judges bounded (multi-judge family, κ≥0.40, CI excludes 0,
  `reward_is_hackable`): measured claims, never sound ones.

**Honest conclusion to publish.** Sound verification reaches as far as the decidable,
computable, or externally-grounded — and no further. That boundary is the project's
organizing theorem; the contribution is mapping it, extending it by decomposition, and
making calibrated abstention the principled behavior beyond it.

---

## Hurdle 4 — Preventing plateau / slow degradation

**Where the repo stands.** Over-built, under-run. `ssil_compound` (compounds vs canonical
best, `stop_after_dry`), `ssil_generations` (N-gen compounding on real weights, CI-separated
promotion, `convergedAt`, gated-vs-ungated negative control), `continual_plasticity`
(protected floors), `improvement.py` (monotone held-out recall), `learning-under-shift`.
**Never run: multi-generation compounding on real trained weights.**

**Distinction that matters.** *Plateau* (gains stop stacking) is acceptable and honest for
bounded RSI — measure where it is, don't prevent it. *Degradation* (old capability rots) is
catastrophic forgetting / loss of plasticity — must be prevented; the protected-floor +
append-only + fresh-held-out machinery is built to catch it.

**Steps**

- **4.1 (hardware)** Run `ssil_generations` for ≥3 generations on real weights; publish the
  `compoundingCurve` with CIs. Plateau is a fine outcome — report it. *(Runner already exists:
  `tools/run_ssil_generations.py`; consumes real per-generation 3-seed aggregates.)*
- **4.2 — plasticity probe ✅ done (no GPU); mitigation hardware.** `agent/plasticity_probe.py`
  computes the loss-of-plasticity correlates the 2025 literature identifies — stable rank
  (collapse), dead-unit fraction, weight-norm growth — in pure Python; `watch_generations`
  emits a `degrading-plasticity-warning` early-warning, now attachable to the generational
  artifact via `tools/run_ssil_generations.py --plasticity-json`
  (`tests/test_plasticity_probe.py`). The *mitigation* (shrink-and-perturb / L2-init /
  continual-backprop at the retrain step) and the real per-generation weight stats are hardware.
- **4.3 — diversity/novelty floor ✅ done (no GPU).** `tools/feedback_to_training.py mine`
  takes `--min-novelty`: a candidate is rejected if its token-Jaccard to anything already
  queued exceeds `(1 - min_novelty)`, so the queue can't narrow onto near-duplicate misses
  (accumulated reward bias / shrinking diversity). Default 0.0 = off, back-compat
  (`tests/test_feedback_diversity_floor.py`).
- **4.4** Mandate fresh held-out per generation in the generational harness (best detector of
  contamination-driven fake gains).
- **4.5** Keep the protected-floor gate exactly as-is — it is the anti-degradation mechanism.

---

## Hurdle 5 — Long-horizon + autonomy  *(split capability from alignment)*

**Where the repo stands.** `agent/horizon.py` measures effective horizon on *synthetic*
chained arithmetic; the real agent loop (`agent/harness.py`) is short-horizon by design
(`max_steps=4`, stops at first unrecoverable step); `tools/run_long_horizon.py` is a built
logging + autonomy-classification harness that **has never been run on a real task.** The
human is load-bearing in exactly two places: `feedback_to_training.py::approve`
(non-circularity) and `conscience_enforcement.py` escalate/abstain on high-impact actions.

**5a — Long-horizon capability (do now; measurement, low risk)**

- **5a.1** Run `run_long_horizon.py` on a real multi-step repo task; publish the log. Closes
  `long-horizon-not-run`.
- **5a.2** Report METR-style: longest task duration at which it succeeds 50% of the time, not
  "did it finish." First honest result is likely "partial-autonomy with N interventions."
- **5a.3** Raise `max_steps` + add checkpoint/resume to span the 30-min "short" tier the
  harness already defines. Every sound per-step verifier extends the horizon by cutting the
  per-step failure rate.

**5b — Removing the human from the gate (do last; alignment problem)**

- **5b.1** Convert human-*blocking* into async human-*audit*: auto-promote only candidates
  clearing a validated quality score; everything logged to a review queue. Non-circularity
  survives via decontamination + audit trail.
- **5b.2** Replace "escalate→block" with "escalate→fallback-to-prior-known-good" (registry
  already supports counterfactual revert).
- **5b.3** Two-version verifier consensus (validate a candidate with the *previous*
  generation's frozen verifier) for routine cases; humans reserved for genuine novelty.
- **5b.4** Do not run a human-free loop. Deliverable now is the *pre-registered design*, not
  a live run. Going slow here is the project's identity, not a limitation.

---

## Recommended sequencing

| Order | Hurdle | Why | First concrete action |
|---|---|---|---|
| 1 | **2 transfer** | Mostly wiring; on-ramp to Hurdle 1 (SWE-bench/GSM8K) | domain→verifier registry + coding/math benchmarks |
| 2 | **4 compounding** | One experiment from real evidence | `ssil_generations` ≥3 gens on real weights + rank probe |
| 3 | **5a long-horizon capability** | Pure measurement, low risk | run `run_long_horizon.py` on a real task; METR 50%-horizon |
| 4 | **3 verifier reach** | Deep frontier; partly a mapping/writing task | publish verifiability map + decomposition + abstention |
| 5 | **5b autonomy alignment** | Genuinely unsolved; rushing is the one real danger | pre-register audit-queue/fallback/consensus design |

## References (current literature)

- RLVR coverage ceiling: arXiv 2503.23829; "verifier problem" survey (subhadipmitra.com, 2026); RLPR 2506.18254; "An Imperfect Verifier Is Good Enough" 2604.07666.
- Loss of plasticity: arXiv 2509.22335 (spectral collapse), 2404.00781, 2410.20098 (self-normalized resets).
- Self-improvement returns: arXiv 2401.10020 (self-rewarding), CREAM (OpenReview Vf6RDObyEF), 2509.09677 (illusion of diminishing returns).
- Long-horizon: METR time-horizons (metr.org/time-horizons), arXiv 2503.14499, 2505.05115 (success half-life), 2603.29231 (reliability science beyond pass@1).
