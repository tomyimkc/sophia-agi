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

## Build 4 — Harness↔model co-evolution loop (next)

The harness already logs full traces for SFT/DPO (`harness.py` `step_output`) and has
reward seeds (`agent/gate_reward.py`, `provenance_bench/rl_reward.py`). The missing
piece is the **closed flywheel**:

1. Harness runs on real tasks → traces.
2. Traces → preference pairs (pass vs failed-then-fixed attempts are already in the
   log) → training data, gated by the existing provenance/no-overclaim checks.
3. A **harness-conditioned eval** that measures *model + harness* jointly, reporting
   **uplift = (model+harness) − (model alone)** — the number the team actually cares
   about ("is the Agent helping more people in more scenarios?").

Acceptance: an uplift benchmark (SWE-/terminal-bench-style or real-repo tasks) with
confidence intervals through the same multi-judge consensus gate as `RESULTS.md`.

## Design invariants (all builds)

- **Deterministic / offline-testable** via the mock model client — every build ships
  with CI tests that need no network.
- **Fail-closed** — ambiguity, over-budget, out-of-scope, or zero-evidence ⇒ abstain
  or refuse, never a silent best-guess.
- **Auditable** — every non-trivial decision (compaction, scope block, delegation)
  emits a structured `RunStore` event so the whole agent tree is replayable.
- **No overclaim** — public numbers clear the measurement gate in `RESULTS.md`.
