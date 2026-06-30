# OKF — Ordered Knowledge File: traceable memory-augmented *training*

**Status:** research/design analysis. No capability claim; `canClaimAGI` stays **false**. This is a
design + measurement plan, not a result. It pairs with `docs/06-Roadmap/MegaTrain-Memory-Centric-Training.md`
(the memory-centric *training-systems* mirror) and with the council / verifier-gating docs
(`docs/11-Platform/Verifier-Gated-Trust-Boundary.md`, `docs/11-Platform/Gated-Memory.md`,
`docs/06-Roadmap/Coding-Council.md`, `docs/06-Roadmap/Council-and-PIF-Experiment.md`). Where MegaTrain
asks *how to fit a larger model on owned hardware*, OKF asks *how to make whatever we train auditable,
locatable, and editable at the step level*. The two compose: MegaTrain is the substrate, OKF is the
provenance skeleton laid over it.

> **Abstract.** An OKF (Ordered Knowledge File) is an external, content-addressed, provenance-bearing
> markdown memory node. A graph of OKF nodes forms a decision DAG, and an append-only retrace log
> records the path a system actually took. The thesis under analysis is that an OKF layer makes a
> trained system's reasoning **traceable** (you can name the path), **locatable** (you can name the
> wrong step), and **editable** (you can correct that step cheaply) — properties that pure weight
> training does not expose. This doc maps the thesis to five prior lines of work, states the honest
> wall (you cannot store training *in OKF instead of weights*), specifies a two-arm benchmark, and
> gives a phased plan. Nothing here is claimed to improve raw accuracy; the predicted, defensible
> finding is an editability / error-location advantage that may come at a tie or loss on raw quality.

---

## 1. The 5-component thesis mapping

The OKF design is a synthesis of five existing lines of work. Each component below cites the prior art
it leans on; OKF's contribution is the *composition*, not any single component.

1. **External traceable markdown memory** — the OKF node and its retrace log are an external memory
   stream the system reads, writes, and reflects over, rather than holding everything in activations.
   This follows the **Generative Agents** memory-stream + reflection design (Park et al., 2023) and the
   **MemGPT / Letta** memory-hierarchy idea of treating an external store ("filesystem as memory") as a
   tier the model pages in and out of. The OKF markdown file *is* that external tier, with provenance
   attached.
   - Generative Agents, Park et al., 2023, arXiv:2304.03442.
   - MemGPT (Letta), Packer et al., 2023, arXiv:2310.08560.

2. **Locate-the-wrong-step** — given a multi-step trace, identify the exact step that first goes wrong,
   instead of only scoring the final answer. This is **process supervision** as framed by *Let's Verify
   Step by Step* (Lightman et al., 2023): a process reward model that judges each step. In OKF this is
   the `locate_wrong_step` primitive (first node in trace order whose `verdict == "fail"`).
   - Let's Verify Step by Step, Lightman et al., 2023, arXiv:2305.20050.

3. **Edit-the-located-point** — once the wrong step is a named node, a correction can target it
   directly. This is the **model-editing** line, **ROME / MEMIT** (Meng et al.): locating and editing a
   specific factual association rather than retraining the whole model. In OKF the editable point is the
   addressable node; an *optional* ROME/MEMIT edit can mirror the node-level correction into weights.
   - Locating and Editing Factual Associations in GPT (ROME), Meng et al., 2022, arXiv:2202.05262.
   - MEMIT (mass editing), Meng et al., 2022, arXiv:2210.07229.

4. **Tool / skill / MCP / swarm / mode triggering** — the decision to call a tool, invoke a skill,
   route to a swarm team, or switch mode is itself a recordable decision node. This follows
   **ReAct** (reason+act interleaving) and **Toolformer** (learning when to call a tool), plus the
   **Voyager** skill-library idea of accumulating reusable, named skills. In OKF, a skill is a `skill`
   node and a tool/route decision is a `decision` node, so the act of triggering is part of the trace.
   - ReAct, Yao et al., 2022, arXiv:2210.03629.
   - Toolformer, Schick et al., 2023, arXiv:2302.04761.
   - Voyager (skill library), Wang et al., 2023, arXiv:2305.16291.

5. **The path "instinct"** — the policy that decides which skill / tool / route to reach for is a
   **learned router over the skill space**, trained with a **verifier-gated reward** so that only paths
   the verifier accepts earn signal. This is not a separate citation so much as the union of (4)'s skill
   space with the repo's verifier-gating discipline: the router learns a *habit* (which node to reach for
   next) under a gate, rather than a memorized format. In the repo this router maps onto
   `agent/swarm_router.py`'s action space.

> Read together: (1) gives the *store*, (2) gives the *where-it-broke*, (3) gives the *fix*, (4) gives
> the *vocabulary of moves*, and (5) gives the *learned policy* over those moves under a verifier gate.

---

## 2. What an OKF concretely *is*

An OKF node is a content-addressed, provenance-bearing markdown node. It **extends the existing
`wiki/` frontmatter** (which already carries `id`, `sources`, `links`) with the fields needed for
traceable training:

- `id` — content id, deterministic from `(node_type, title, body)` (sha256, form `"<type>:<hex12>"`).
- `sources` — provenance pointers (as in the wiki frontmatter today).
- `links` — outgoing edges; the union of all `links` forms a **decision DAG**.
- `type` (`node_type`) — one of `fact | step | skill | decision`.
- `verifier` — the verifier that judged this node (name / id), if any.
- `verdict` — `pass | fail | none` — the judged outcome of this node.
- `moral_standard` — an optional constraint label (e.g. a PROTECTED domain marker; see §6).

The node serializes to and from markdown with a no-dependency frontmatter roundtrip (`to_markdown` /
`from_markdown`), so an OKF is a plain, diffable, version-controllable markdown file — the same shape as
the existing wiki pages.

**The DAG.** `build_dag` turns a node set into an adjacency map over `links`; `has_cycle` enforces the
*acyclic* invariant so a trace cannot loop. The DAG is the decision structure; an edge says "this step /
skill / decision led to that one."

**The retrace log.** `okf/trace/*.jsonl` is an **append-only** log: each row is
`{node_id, step, payload, ts}` with `ts` injectable (no wall-clock in tests). `retrace` reconstructs the
path the system actually walked; `locate_wrong_step` walks that path and returns the first
`verdict == "fail"` node — the exact error location.

**The correction loop.** A human or verifier correction is *not* an in-place mutation of history. It is:
(a) a **new node** (corrections are content-addressed and append, preserving the audit trail);
(b) a **process / DPO signal** keyed by the corrected node (so the router learns from the fix); and
(c) an **optional ROME/MEMIT edit** to mirror the correction into weights when a factual association
must change. The log stays append-only; corrections accrete rather than overwrite.

---

## 3. The honest wall

You **cannot** store training "in the OKF instead of in the weights." Gradient descent updates weights;
a markdown file does not. Any framing that says the OKF *replaces* the weights is wrong, and this doc
does not make that claim.

The feasible design is **dual-write**: every training step *also* emits an OKF trace node. The OKF is
then the **audit layer over the weights**, not a substitute for them. The weights remain the learned
function; the OKF records, for each step, which decision was taken, which verifier judged it, and what
the verdict was — so the trained behaviour becomes traceable and locatable *without* claiming the
learning lives outside the parameters.

A full non-transformer neuro-symbolic "brain" — where structured memory genuinely carries the learned
function — is a research-frontier direction and is **not built here**. The feasible synthesis this doc
commits to keeps the **transformer as the neural substrate** and lays an **OKF + verifier auditable
skeleton** over it. That skeleton buys traceability and editability; it does not change what
gradient descent is doing underneath.

---

## 4. The two-benchmark spec

A two-arm, offline, deterministic harness measures the thesis. It does not train a real model; it
operates over fixture traces so the comparison is reproducible and instrument-faithful.

- **Arm A — OKF-integrated + process reward.** Training steps dual-write OKF nodes; the reward is
  verifier-gated at the step level; a correction is a single node edit.
- **Arm B — pure-weight, outcome-only control.** No OKF layer; reward is outcome-only; a correction
  means a full retrain pass.

**Five metrics** (each a pure function over fixture traces):

1. **Traceability** — fraction of traces where `locate_wrong_step` matches the seeded wrong step.
2. **Editability / correction-cost** — cost of one correction. Arm A is an O(1) node edit; Arm B is a
   fixed retrain pass. Reported as a ratio (Arm A << 1, Arm B == 1 baseline).
3. **Forgetting** — degradation of earlier-learned items after a correction. The OKF arm should not be
   *worse* than the control here (append-only correction does not clobber prior nodes).
4. **Path-efficiency** — addressable actions per trace (steps + tools); monotonic in trace length.
5. **Kappa-gated quality** — raw quality, reported only when inter-rater / verifier agreement (kappa)
   clears the gate, so a quality number is never read off an unreliable judge.

**Predicted, honest result (for the record, not a capability claim).** Arm A is expected to win
**traceability**, **editability**, and **correction-sample-efficiency**: a wrong step is an exact,
addressable node, and a fix is an O(1) edit. Arm A may **tie or lose** on raw accuracy / kappa-gated
quality, which the weight-only control can match or exceed. "OKF improves where-it-broke and
cost-to-fix, possibly at no accuracy gain" is a defensible, falsifiable finding — and is the actual
thesis under measurement, not raw accuracy.

---

## 5. Five creative directions

1. **Verifier-gated PROCESS memory.** Gate *writes to memory* on the verifier: a step only becomes a
   persisted OKF node if its verdict passes. The memory then contains verified process, not raw
   chatter, and the router in §1.5 trains over a cleaner skill space.

2. **Seeded wrong-step localization benchmark.** Construct traces with a *known* injected wrong step,
   then score `locate_wrong_step` against the seed. This turns process supervision into a measurable,
   decontaminated benchmark instead of a qualitative claim.

3. **Skill-library-as-OKF-nodes.** Represent each Voyager-style skill as a `skill` node, and do
   retrieval over **verified** skills (verdict == pass). Reuse becomes retrieval over an audited
   library rather than over unverified snippets.

4. **First-class `moral_standard` edges.** Make a PROTECTED domain (e.g. religion / history) a
   **graph constraint**, not a prompt suffix: a `moral_standard` edge constrains which corrections and
   which routes are admissible, so the protection is structural and checkable in the DAG.

5. **Editable belief WITH a measurement receipt.** A belief change is a ROME/MEMIT edit *plus* a
   re-certification: the edit is only accepted if a fresh measurement receipt (the repo's claim-gate
   discipline) clears. An edit without a receipt is not a belief change, just an unverified mutation.

---

## 6. Phased plan

- **P0 — substrate (shipped).** `agent/okf_trace.py` (trace log, DAG, `has_cycle`, `retrace`,
  `locate_wrong_step`, `offline_invariants`) and `agent/okf_schema.py` (`OKFNode`, `content_id`,
  markdown roundtrip, `validate`). Pure, offline, deterministic, stdlib-only, with self-tests.

- **P0b — loop-engineering incident logger (shipped).** `agent/okf_loop.py` (`LoopLog`): records an
  agentic incident-response loop (the ReAct→Reflexion *observe → reason → decide → act → verify →
  resolve* cycle) as a chain of linked OKF nodes — the **decision DAG** — written into
  `okf/incidents/<id>/` (one node markdown per step + an append-only `trace.jsonl`). Every error event
  and every step taken in response (sub-agent spawn, reasoning, tool/coding-agent call, gate run) is
  a node; `summary()` reuses `locate_wrong_step` to surface the first failing step. The real
  `2026-06-30-bench-a-04-stall` incident is recorded as the worked example
  (`tools/record_bench_a04_incident.py`, deterministic). This is the runtime mirror of the
  *training-time* dual-write below — same OKF substrate, recording the agent's own loop.

- **P1 — process verifier + dual-write (PARKED).** Extend `tools/gen_reasoning_distill.py` and
  `tools/gen_verifier_dpo.py` to emit a **step-level** signal keyed by OKF `node_id`, so each training
  step dual-writes a trace node and earns process-level (not just outcome-level) reward.

- **P2 — the A/B harness (shipped).** `tools/eval_okf_vs_pureweight.py`: the two-arm, five-metric,
  offline harness from §4, consuming the P0 `locate_wrong_step` contract.

- **P3 — router "instinct" (PARKED).** Train a policy over the `agent/swarm_router.py` action space
  (the learned router of §1.5), with verifier-gated reward over the skill / route space.

- **P4 — editable belief (PARKED).** ROME/MEMIT edit + re-certification receipt (§5.5), so a belief
  change carries a measurement receipt.

> P1 / P3 / P4 are **parked behind the live bench queue** — they touch the training pipeline and the
> GPU, and are deferred until the live benchmark queue clears. P0 and P2 are offline and already
> landed.

---

## 7. Shipped offline (this iteration)

Two pure, offline, deterministic, stdlib-only modules landed, each with a passing self-test:

- **`agent/okf_trace.py`** — the OKF traceable-memory substrate. Implements the shared OKF API contract:
  `append_trace`, `read_trace`, `build_dag`, `has_cycle`, `retrace`, `locate_wrong_step`,
  `offline_invariants`, plus a `__main__` CLI. The `ts` field is injectable so tests never touch
  wall-clock; the DAG enforces the acyclic invariant; `locate_wrong_step` returns the first
  `verdict == "fail"` node in trace order (the Lightman exact-error-location primitive).
  Self-test (`--self-test`): `PASS cycle_detect`, `PASS locate_wrong_step`,
  `PASS append_read_roundtrip`, `PASS ALL`.

- **`tools/eval_okf_vs_pureweight.py`** — the offline two-arm, five-metric A/B harness (§4). Defines
  `traceability_score`, `correction_cost_ratio`, `path_efficiency`, forgetting, and kappa-gated quality
  as pure functions over fixture traces, and produces an A/B table. It consumes
  `agent/okf_trace.locate_wrong_step` per the shared contract, with a contract-faithful local fallback
  so the harness is self-testable in isolation. Self-test (`--self-test`):
  `PASS compare_has_all_5_metrics`, `PASS determinism`, `PASS forgetting_arm_a_le_arm_b`, `PASS ALL`.

Both modules carry the header `PLANNING/SUBSTRATE ONLY` / `PLANNING/HARNESS ONLY — no capability claim;
canClaimAGI stays false`, and neither trains a model, touches the network, or claims an accuracy result.
