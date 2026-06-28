# Agent Harness Roadmap — the "trustworthy harness"

**Model + Harness = Agent.** Sophia already owns the *trustworthy* half of a harness
(provenance, verification, calibration, fail-closed gating). This roadmap adds the
*more-capable* half — context management, delegation, long-horizon execution, and a
measured model↔harness co-evolution loop — **in Sophia's idiom**: every addition is
deterministic, offline-testable, fail-closed, and auditable. The goal is a harness
that makes a *fixed* model both **more capable** and **more verifiable**, not a
re-implementation of what frontier labs already do well.

The four builds map to the harness frontiers the field is actively pushing:
*context engineering / KV-cache*, *subagent & multi-agent*, *ultra-long-horizon
tasks*, and *harness↔model co-evolution*.

## Where the loop lives

`agent/harness.py` is the spine: `plan → execute → critic → reflect/retry`, with
append-only decision logs (`RunStore`), checkpoint/resume, and failure
classification. Everything below plugs into that spine as a component rather than a
rewrite.

## Build 1 — Context management (KV-cache-aware) ✅ shipped

`agent/context_manager.py`. Replaces the old `"\n\n".join(prior_outputs)[-4000:]`
char chop with an explicit, auditable context-window policy:

- **Bilingual token estimation** — CJK billed heavier than Latin; a real tokenizer
  is injectable via `token_counter`.
- **Segment model** with provenance tags + two structural flags:
  - `stable` → the **cache-stable prefix**: emitted first, never compressed, so the
    provider's KV cache survives across turns. `PackResult.cache_key` signatures the
    prefix so a caller can detect a cache-invalidating change.
  - `pinned` → never dropped.
- **Fail-closed keep / compress / drop** — pinned content is never silently dropped
  (overflow is *flagged*); every drop/compression is reported by provenance tag; the
  default compressor is deterministic head+tail elision that preserves the trailing
  `Decision` / `中文摘要` block. An injected summarizer is distrusted if it overshoots.
- `compact_history()` is recency-aware (pins the latest output) and wires into the
  harness loop, which now logs a `context_compact` audit event on any elision.

Tests: `tests/test_context_manager.py` (offline, deterministic).

## Build 2 — Subagent delegation ✅ shipped

`agent/subagent.py`. The councils are *deliberation* (one model, many personas, one
shared context); this is genuine **delegation**: a parent decomposes a goal into
child `SubagentSpec`s, each running the full `run_agent` loop in its **own** run
store, seeing only the context it is handed, scoped to only the tools it is allowed,
and bounded by its own step/retry/cost budget. Successful children are reduced
through one calibrated synthesis step.

- **Least privilege** — `allowed_tools` is enforced *inside the harness*: an
  out-of-scope tool request fails the step fail-closed (`tool_scope_block`). `None`
  inherits the parent scope; `set()` is a pure-reasoning child.
- **Isolation** — distinct `task_id` + `RunStore` per child; traces never interleave.
- **Bounded** — `max_steps`/`max_retries` is the hard bound; `cost_budget_usd` is an
  additional soft post-hoc ceiling (honest about what is/ isn't pre-emptive).
- **Fail-closed synthesis** — zero successful children ⇒ ABSTAIN, never an answer
  synthesised from failures. `synthesize=False` hands the raw children to a caller
  that wants `team_agents.deliberate_team`'s divergence-aware reduce instead.

Tests: `tests/test_subagent.py`. (Sequential/deterministic today; parallel fan-out is
a later optimisation that must preserve trace isolation and determinism in tests.)

## Build 3 — Long-horizon execution engine ✅ shipped

`agent/long_horizon.py`. `agent/horizon.py` *measures* an effective-horizon curve
(METR-style, oracle-judged); this is the *engine* that survives long, dependent task
chains:

- **Durable task tree** — `TaskLedger` of `SubtaskNode`s, persisted to JSON after
  every node transition, so a resumed run skips `done` nodes and re-attempts only
  `pending`/`failed` ones (multi-task checkpoint beyond the harness's per-task one).
- **Dependency ordering, fail-closed** — a node runs only once its `deps` are all
  `done`; a node whose dependency failed is left `blocked` and never executed on an
  unmet prerequisite.
- **Recovery memory in the loop** — `RecoveryMemory` (dependency-free, append-only)
  records a hint keyed by the node's *failure signature*; before re-attempting a
  similar node the engine recalls the hint and injects it into the child's context.
  Complementary to the provenance-specific `agent/failure_memory.py`.
- **Execution via delegation** — each node runs through `subagent.run_subagent`,
  inheriting isolation, least-privilege tool scope, and per-node budgets.

Tests: `tests/test_long_horizon_engine.py` — including a deterministic demonstration
that a recalled recovery hint flips a failing sibling node to success.

Remaining (measurement): wire the engine into `horizon.py`'s oracle-judged curve and
report the effective horizon *with the recovery loop on vs off*, under the
no-overclaim gate. Recovery today is within-run hint injection, not weight learning.

## Build 4 — Harness↔model co-evolution loop ✅ shipped

The two halves of the flywheel that turns harness behaviour into a better model:

**Eval half — `agent/uplift.py`** (`tools/harness_uplift.py`). Measures, holding the
model fixed, **uplift = passRate(model+harness) − passRate(model alone)**. Both
conditions are graded by the SAME external verifier (the harness never grades
itself), the per-case results are *paired*, and the headline carries a **bootstrap
95% CI**: a positive point estimate is reported `demonstrated=False` unless the CI
lower bound clears 0 (no-overclaim). The bare baseline is one single-shot
generation; the harness condition is the full plan→execute→critic→reflect/retry
loop, so the delta isolates the harness as the only changing variable.

**Data half — `agent/trace_distill.py`** (`tools/collect_preferences.py`). Turns the
append-only run logs into preference pairs: a *fail-then-fixed* step is its own
training signal — the passing attempt is `chosen`, an earlier failing attempt is
`rejected`, for the same prompt. Fail-closed (a pair is emitted only when chosen
cleared the critic and rejected did not; blank rejecteds are skipped), deterministic
(pure JSONL parsing, no model call), and provenance-preserving (each pair records
its task/step and the rejected attempt's failure class). Produces the dataset only —
DPO/SFT training is a separate, separately-gated job that consumes `to_jsonl`.

Tests: `tests/test_uplift.py`, `tests/test_trace_distill.py` (offline, deterministic),
including a stub that demonstrates positive uplift and a live harness trace that
distills to a usable preference pair.

Remaining (out of scope here, gated separately): wire the distilled pairs into an
actual training run, and run the uplift benchmark on a real-repo / SWE-bench-style
suite with a real model under the multi-judge consensus gate in `RESULTS.md`. The
loop is closed in code; the model-training arc is the next, heavier program.

### Build 4b — Closed-loop engine (the training arc) — candidate, not Level-3

`agent/closed_loop.py` + `tools/run_closed_loop.py`. Composes the two Build 4 halves
into the one cycle that turns a *designed* loop into a *closing* one:

    measure uplift (paired, bootstrapped) → the harness run writes fail-then-fix traces
      → distill traces into preference pairs → a train step produces a candidate
      → gate the candidate through continual_plasticity (hard reject on regression /
        contamination / catastrophic forgetting; promote only on a clean gain)
      → re-measure uplift with the promoted model → repeat

Two invariants make this the honest signature of a *closing* loop:

- **NON-DEGENERACY** — a promoted model's uplift never goes negative. The plasticity
  gate enforces pass-rate non-regression per cycle; this module additionally asserts
  the harness-vs-bare relationship the DeepSeek thesis depends on, and **halts loud**
  (rolls back the spec, sets `nonDegenerate=false`) if a promoted model's uplift goes
  negative — that means reward hacking / regression slipped past the gate, a loop
  failure, not a data point to publish.
- **SATURATION IS SUCCESS** — if uplift converges to ~0 *after a promotion*, the
  harness's rescue behaviour was distilled into the model (the model now does first-try
  what the harness used to fix). That is a valid terminal state; the next competence
  gain needs a *harder* harness (a learned world model / planner — see
  `AGI-Missing-Pillars.md`), not more of the same distillation.

The **train step is injected**: `noop_train_step` in CI (every cycle reports
`no-candidate` — proves the plumbing, not a model advance), and a live step that
writes the distilled pairs to JSONL and shells out to `tools/run_rlvr.py` on a CUDA
pod (DPO/GRPO over the traces). The orchestrator trains nothing itself and changes
no weights.

Tests: `tests/test_closed_loop.py` (offline, deterministic) — plumbing + RUNS_DIR
restoration, promotion on a clean gain, gate-blocks-pass-rate-regression, the
non-degeneracy wall halting on negative post-uplift, saturation-as-success, and the
no-overclaim payload fields.

> **Claim boundary.** Closing the loop offline (or with the mock trainer) is
> **rehearsal, not Level-3 evidence.** Real Level-3 evidence needs a private hidden
> suite (see `AGI-Level3-Execution-Protocol.md`) and a gated run; this artifact
> carries `candidateOnly=true, level3Evidence=false` until then. The crux that bounds
> this whole arc is `reasoning/deliberation_roofline.py`: the *verifier* sets the
> quality ceiling, not the compute — so the next real competence gain runs through
> strengthening the verifier (Phase B: execution/outcome verification as a first-class
> family), not through more distillation cycles on a fixed gate.

## Design invariants (all builds)

- **Deterministic / offline-testable** via the mock model client — every build ships
  with CI tests that need no network.
- **Fail-closed** — ambiguity, over-budget, out-of-scope, or zero-evidence ⇒ abstain
  or refuse, never a silent best-guess.
- **Auditable** — every non-trivial decision (compaction, scope block, delegation)
  emits a structured `RunStore` event so the whole agent tree is replayable.
- **No overclaim** — public numbers clear the measurement gate in `RESULTS.md`.
