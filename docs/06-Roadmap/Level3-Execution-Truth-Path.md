# The Execution-Truth Path to Level-3

**Status:** candidate scaffolding + rehearsal wiring. `candidateOnly`, `level3Evidence: false`
throughout — these are the seams and measurements around which real Level-3 evidence is later
produced, not that evidence itself.

## The thesis, grounded in the repo's own result

`reasoning/deliberation_roofline.py` proves a closed-form ceiling: **the verifier sets the
quality roof, not the compute.** Leaky verifier → 0.777 plateau; oracle → 1.000; 8× compute
buys +0.017. Two corollaries decide everything:

1. A competence amplifier bounded by its verifier *cannot exceed its verifier.* To grow
   capability you must grow the verifier.
2. Citation verification verifies "is this claim true *about a text*." It cannot verify "will
   this *action* achieve the goal." For that you need **execution ground truth** — the outcome
   of actually doing the thing.

**The wedge:** for the agent domains that define Level-3 (code, tools, repo repair, planning),
execution ground truth is *machine-checkable*. Tests pass or they don't. A build succeeds or
fails. A tool returns a value or an error. **Execution verification is the highest-fidelity,
most κ-stable verifier family for agent tasks — and raising it to first-class raises the
verifier ceiling the roofline bounds capability by.** This is the AlphaGo recipe (verifiable
ground truth + learned model + search + self-play loop) gated by Sophia's fail-closed overlay.

## The four-phase path (Phase A was the engine; this doc covers B–E)

### Phase A — Close the co-evolution loop (the engine) ✅ shipped

`agent/closed_loop.py` + `tools/run_closed_loop.py`. Composes `uplift → trace_distill →
continual_plasticity → re-measure` into one gated, multi-cycle loop with two load-bearing
invariants: **NON-DEGENERACY** (a promoted model's uplift never goes negative — halts loud if
it does) and **SATURATION IS SUCCESS** (uplift → 0 after a promotion means the harness was
distilled into the model; the next gain needs a harder harness). See `docs/09-Agent/Harness-Roadmap.md`
Build 4b. Train step injected (no-op in CI, live `run_rlvr.py` on GPU).

### Phase B — Execution verification as a first-class family (the wedge) ✅ shipped

**B1 — `agent/execution_verifiers.py`.** The execution/outcome ground-truth family registry:
`code_tests_pass`, `unit_test`, `tool_result_validity` (new), `plan_completion` (new), plus
`verifier_for_task(TaskSpec)` — one call routes a coding/tool/plan case to its execution
verifier instead of every caller hand-wiring one. Wired into the closed loop via
`task_spec_for`; a coding case is now graded by `code_tests_pass` (execution truth), not the
epistemic gate (citation truth). `tests/test_execution_verifiers.py` (10 tests).

**B2 — long-horizon objective gate.** `tools/run_long_horizon.py` gained `run_objective_gate`
+ spec `objectiveGate`: a machine-checked command whose exit code becomes `objectivePassed` in
the report — execution truth *distinct from* the semantic `autonomy.substantive` classification.
Demonstrated end-to-end: a short self-test reports `objectivePassed: true` while
`substantive: false` — the stronger signal. `tests/test_long_horizon.py` (+3 tests).

### Phase C — Replace the toy pillars with learned, verified ones (the research) ✅ scaffolded

**C1 — verified world-model trainer.** `agent/verified_world_model.py`. The scaffold that turns
the lookup-table `predictive_world_model.py` into a held-out-validated, **shift-checked** learned
predictor. The canary (mirror of `selfextend/evolve.py`): promote only when held-out accuracy
clears a bar AND shift-degradation is bounded. The core research risk made *visible*: a
predictor that aces held-out but collapses under shift is flagged `shift-degenerate` and held —
it memorized, it did not generalize, and it must never mislead the planner. Default predictor is
a dependency-free feature-logistic model; the live torch seam is injected.
`tests/test_verified_world_model.py` (6 tests).

**C2 — learned planner simulator.** `agent/planner_learned_sim.py`. `LearnedSimulator`
subclasses `VerificationSimulator` so the planner, actions, states, reward are unchanged — only
`outcome()` is overridden to consult an injected predictor. **Fail-closed:** when the predictor
is OOD/uncertain (`|p − 0.5| < min_confidence`) or errors, it falls back to the verified
scripted rule rather than guessing. `run_mcts_with_model` returns predicted-vs-fallback stats so
a caller sees how much of the plan the learned model actually drove.
`tests/test_planner_learned_sim.py` (7 tests).

### Phase D — Level-3 lane rehearsal surfaces execution truth ✅ shipped

`tools/run_level3_candidate_benchmark.py` long-horizon candidate lane now surfaces
`objectivePassed`, so the rehearsal demonstrates the execution-truth signal flowing through the
full Level-3 workflow. Confirmed: candidate rehearsal runs end-to-end, all lanes `ok=True`,
`candidateOnly=True`.

### Phase E — The honesty gate ✅ held

```
python tools/run_agi_verification_gate.py --target level3
→ targetPassed: False
→ highestMachineVerifiedLevel: below-level2
→ canClaimAGI: False
```

The gate did not weaken. `canClaimAGI` stays not-true because the artifact lanes
(`hidden_full_comparison`, `distribution_shift`, `long_horizon_30m`, `rlvr_live_training`)
require **real** hidden data, an independent pack owner, and a gated GPU training run — none of
which this scaffolding produces, by design.

## What this path does NOT claim

- It does **not** claim the world model generalizes. C1 makes generalization *measurable*
  (the shift-degeneracy check); solving it is the deep research.
- It does **not** claim Level-3. The gate is held at `below-level2`; real Level-3 needs a
  private hidden suite (see `AGI-Level3-Execution-Protocol.md`) and a gated run.
- It does **not** weaken the no-overclaim discipline. Every artifact carries
  `candidateOnly: true, level3Evidence: false`; the claims linter passes.

## The honest ceiling

Sophia is a strong candidate for the **safest, most-verifiable Level-3 agent substrate**: a
fail-closed, provenance-aware, verifier-gated loop where execution truth (not citation truth)
sets the ceiling, the world model is held-out- and shift-validated before it can steer the
planner, and a promotion is earned only on a proven held-out gain. It is **not**, and is not
claimed to be, a path to Level-4 generality — that requires the world model to generalize to
situations with no proximal training trace, which no one has solved. State that explicitly; it
is the strongest defensible position and it is true.

## Files added/changed in this arc

| File | Phase | Role |
|---|---|---|
| `agent/closed_loop.py` | A | co-evolution loop orchestrator + non-degeneracy/saturation invariants |
| `tools/run_closed_loop.py` | A | `--mock` (CI) / `--live` (GPU) driver |
| `agent/execution_verifiers.py` | B1 | execution/outcome verifier family + `verifier_for_task` |
| `agent/closed_loop.py` (`task_spec_for`) | B1 | wedge wiring into the loop |
| `tools/run_long_horizon.py` | B2 | `run_objective_gate` + `objectivePassed` |
| `agent/verified_world_model.py` | C1 | verified, shift-checked world-model trainer |
| `agent/planner_learned_sim.py` | C2 | learned-model planner simulator (fail-closed) |
| `tools/run_level3_candidate_benchmark.py` | D | surfaces `objectivePassed` in rehearsal |
| `tests/test_{closed_loop,execution_verifiers,verified_world_model,planner_learned_sim}.py` | B,C | 29 new tests |
| `tests/test_long_horizon.py` | B2 | +3 objective-gate tests |

**Verification:** 9/9 touched-area suites pass · 11 files compile · claims linter clean ·
AGI gate held (`canClaimAGI: false`).
