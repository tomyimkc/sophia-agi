# PEFT + Test-Time Compute vs Raw Scale (P2 — design)

> Status: design. No code in this pass. This doc fixes the head-to-head, the
> harnesses, the falsifiable claim, and the statistical gate.

## The thesis

The advisor's point ③: *"Parameter-efficient + test-time scaling beats raw
pretraining at your hardware size. Your council deliberation, self-extension
flywheel, and gate already provide powerful test-time compute. A 30–70B model +
strong Sophia layer can outperform a much larger raw model on grounded tasks."*

This is the architectural bet the whole repo is built on. Today it is an
*assumption*. The point of this doc is to make it a **falsifiable, benchmarked
claim** — and the capability panel (P0) now provides the axes to measure it on.

## What "the Sophia layer" actually is (the test-time compute stack)

This is not hand-waving — every component exists and is measured:

- **Council deliberation** (`agent/council_deliberate.py:83-131`) — map-gate-reduce
  over constrained seats: map → N narrow substantive seats (small-model
  friendly); gate → each seat output gate-checked via `agent.gate.check_response`,
  flagged seats quarantined (`:75-80`); reduce → synthesize under guardian seats
  + decision contract (`:134-151`). Heterogeneous panels (`seat_clients`) give
  independent voters vs a homogeneous one-model-in-N-hats panel (`:86-92`).
- **The runtime fail-closed gate** (`agent/guarded.py`, `agent/gate.py`) —
  `check_claim` (`guarded.py:77-107`) is the mode-free verifier; `guarded_complete`
  (`:161-342`) is the full repair/abstain/hedge loop; `check_response`
  (`gate.py:96-188`) is the post-generation epistemic gate (strict attribution,
  legal, numeric).
- **The self-extension flywheel** (`docs/11-Platform/Safe-Self-Improvement-Loop.md`)
  — Layer 0 (skills/verifiers/rules via SSIL) now, Layer 1 (LoRA/RLVR weight
  delta through the *identical* gate via `agent/continual_plasticity.py`) later
  (`Safe-Self-Improvement-Loop.md:110-116`).
- **Deliberation roofline** — `docs/06-Roadmap/Reasoning-As-Compute.md:114-162`
  measured `N*=8` seats; the ceiling is set by the **verifier**, not compute.

The advisor's claim is that this stack, riding on a modest PEFT-tuned model,
beats a much larger raw model on the grounded tasks the stack is built for.

## The head-to-head

Four arms, identical prompts, identical harnesses, identical gold:

| Arm | Model | Sophia layer |
|---|---|---|
| **A. Raw small** | base 7–9B (e.g. GLM-4-9B) | none |
| **B. Raw large** | much larger raw model (API or local) | none |
| **C. PEFT small** | base 7–9B + trained LoRA/RLVR adapter | none at inference |
| **D. Sophia-full** | base 7–9B + trained adapter **+** gate + council | full |

The falsifiable comparison the advisor predicts: **D ≳ B on grounded axes, at a
fraction of the parameters.** That is the claim. The falsifier is **B ≫ D**
(the raw large model dominates despite the Sophia layer) — in which case the
thesis is wrong and the architecture should be reconsidered.

## The axes (from P0 — already built)

The capability panel (`tools/eval_capability_panel.py`) scores every arm on:

- **attribution accuracy** — `verdictAccuracy` (affirm gold on true cases, via
  `provenance_bench.judge`, independent of the gate).
- **hallucination rate** — fraction asserting a forbidden attribution.
- **integrity recall** — of FALSE cases, the fraction correctly not-certified
  (abstain/correct). Higher is better; the capability number.
- **abstention calibration** — `calibrationScore`, `fabricationRate`,
  `overAbstentionRate` (via `provenance_bench/calibration_score` over the
  SEIB-100 pack).

Plus the existing harnesses the advisor names:

- **SEIB-100** (`tools/run_seib.py:453`) — the only harness that already supports
  `--real-model --model --adapter`; scores `provenanceAccuracy`,
  `falseAttributionRate`, `fabricationRateOnContested`, `traditionMergeRate`,
  `falsePositiveCost`, `sourceCitationRate`.
- **Provenance Delta** (`provenance_bench/score.py`) — alone vs gated hallucination
  delta + FP cost + coverage, with paired bootstrap CIs (`aggregate.py:54`).
- The other phase benchmarks (`tools/run_all_phase_benchmarks.py`) — belief
  revision, code provenance, GPQA-provenance, moral public standard — once they
  grow a `--model`/`--adapter` path (today most are fixture/mechanism-only).

## The statistical gate (no-overclaim, the repo's standard)

A claim does not land from one run. It lands through the existing discipline:

- **≥3 seeds**, entity-disjoint splits, contamination-guarded
  (`provenance_bench/heldout_split.py`, `rl_dataset.py`).
- **Paired bootstrap 95% CIs** on the headline delta
  (`provenance_bench/aggregate.py:54`, `n_boot=2000`).
- **The `validated` flag** (`aggregate.py:131`): not-mock, ≥2 distinct judge
  families, κ≥0.40 (`KAPPA_FLOOR=0.40`), ≥3 runs, CI excludes 0.
- **Promotion gate** (`tools/promote_adapter.py`) — the adapter's promotion
  verdict (`promote_adapter.py:373-377`) and, for multi-goal claims, the Pareto
  gate (`agent/continual_plasticity.py:137-213`: REJECT on any cross-goal
  regression, PROMOTE only when every goal clears its floor).

The head-to-head claim "D ≳ B" is accepted only when D's CI on the grounded axes
excludes B's, across ≥3 seeds, with no protected-suite regression. Otherwise it
stays `candidateOnly: true, validated: false` — exactly like every other claim.

## Honest scope / what could falsify this

- **On raw capability axes** (general knowledge, pure math without provenance)
  the large raw model will likely win — and *should*. The claim is specifically
  about **grounded** tasks (attribution, calibration, integrity), where the gate
  and the fidelity of the training data are the load-bearing advantage.
- **The deliberation roofline** (`Reasoning-As-Compute.md:114-162`) already shows
  the verifier, not compute, sets the ceiling — so there is prior evidence the
  Sophia layer converts test-time compute into grounded capability rather than
  just more tokens. This experiment sharpens that into a model-size comparison.
- If B ≫ D even on grounded axes, the lesson is that the Sophia layer needs a
  stronger base than 7–9B to ride on — itself a useful, falsifiable result.

## Non-goals (this pass)

- No code. No live run. This doc fixes the protocol; the run is a later,
  budgeted experiment (it needs ≥2 model sizes × ≥3 seeds × the full harness
  set — real GPU/API cost).
- Does not change the council, the gate, or the panel — it *uses* them.

## Verification target (when run)

A single `sophia.peft_vs_raw.v1` report: per-arm panel deltas (P0) + SEIB +
Provenance Delta, with CIs, across ≥3 seeds, plus the promotion/Pareto verdicts
for the PEFT and Sophia-full arms. `candidateOnly` until the no-overclaim gate
passes; `canClaimAGI: false` regardless (this is a relative-capability claim,
not an AGI claim).
