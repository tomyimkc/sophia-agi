# MBTI Vector Agents — Spec D: Capability-Retention Guardrail + Full FastMCP Packaging

**Date:** 2026-06-23
**Status:** Approved → ready for implementation plan
**Program:** "MBTI Vector Agents" — Spec **D** of 4 (the last; stacks on Spec C PR #67)
**Branch:** `feat/capability-retention-mcp` (off `feat/personality-council-pif` / Spec C)

## Problem

Specs A–C built the measurement gate (A), the activation-steering engine + SSA
(B), and the personality council + held-out anti-gaming + PIF harness (C). Two
gaps remain before the program is a coherent, shippable whole:

1. **B's SSA verdict names `capability_drop` and `coherence` but nothing ever
   computes them.** `agent/steering/stats.py::ssa_verdict` requires
   `cell["capability_drop"] < capability_eps (0.05)` and
   `cell["coherence"] >= coherence_floor (75)` — the guarantee that steering
   "didn't lobotomize the model." Today those are unmeasured inputs. A headline
   SSA cell is not honest until they are real.
2. **The program's surface is only half-exposed.** Spec A shipped a thin MCP
   surface (`sophia_personality_target/faithful` + the `mbti://types` resource);
   B's measurement and C's council/PIF have no portable, tool-callable surface,
   and the portable skill still describes only A.

Spec D closes both, as **one cohesive PR, two pieces (D1 then D2)**, on the same
two-tier discipline as B/C: a **pure-stdlib, deterministic CI core** plus
**opt-in reduced real runs**.

- **D1 — Capability-retention guardrail:** a small bundled arithmetic-word-problem
  slice, scored **deterministically** (answer-correctness vs a gold number +
  the existing `arithmetic_sound` soundness check + a deterministic coherence
  proxy), run **steered vs unsteered** to produce the `capability_drop` /
  `coherence` inputs B's SSA needs.
- **D2 — Full MCP/skill packaging:** the program's read-only surface (OCEAN
  measure, capability-retention, council-diversity, PIF-dryrun) added as
  `@mcp.tool()`s reusing the existing `@audited` gating; a program-status
  resource; the portable `SKILL.md` updated to the whole program; and a
  **cross-platform import/frontmatter validation** so CI proves the surface
  loads without the `mcp` package.

## The central constraints

1. **Two-tier execution, mirroring B/C / `tools/run_rlvr.py`.** The **shipped,
   CI-verified contribution** is the deterministic core: the capability scorer
   (answer-correctness + soundness + coherence proxy), the retention math on
   per-item scores, and the MCP `tools_impl` functions + skill validation — all
   **pure stdlib, no torch/numpy, no `mcp` package, no network**. **Opt-in real
   runs** are cheap (D1 a reduced steered-vs-unsteered slice on granite-2b) — no
   deferred-only piece in D.

2. **A capability *drop* is a legitimate, expected result.** Steering at B's
   alphas likely **degrades** arithmetic — and that is the point: it *explains*
   B's null (steering strong enough to move a trait also breaks capability, so
   it cannot enact the trait while staying coherent). The contribution is the
   honest, deterministic measurement, never a "steering is free" claim.

3. **Deterministic capability scoring — no LLM judge in the loop.** Accuracy is
   `extract_final_number(response) == gold` (within tol); soundness is the
   existing `arithmetic_sound` verifier; coherence is a **deterministic
   degeneracy proxy** (repetition / type-token / length sanity → 0–100), which
   is exactly the failure mode high-alpha steering produces. An LLM-judge
   coherence channel is a documented richer option, **deferred** (kept out so
   the guardrail stays reproducible and CI-gateable).

4. **MCP logic lives in `tools_impl.py`, not `server.py`.** Every new tool's
   body is a pure function in `sophia_mcp/tools_impl.py` (importable **without**
   FastMCP); `server.py` adds only the thin `@mcp.tool()` wrapper that calls
   `dumps(impl(...))`. This is what lets CI validate the surface with no `mcp`
   install. New tools are **read-only** → no `@audited` (matching the existing
   read-only tools); the model-touching real runs stay CLI-only (`tools/`), not
   auto-callable MCP tools.

5. **Reuse, not fork.** `arithmetic_sound` (A's deterministic verifier),
   `measure_ocean`/`score_items` (A), B's `SteeredClient`/`hooks`/`stats`, C's
   `council_diversity`/`build_cells_from_scores`, the existing `sophia_mcp`
   server + `audit.py` gating, and the existing portable skill.

6. **MBTI veneer-invariance inherited (A+B+C):** no capability/coherence path,
   MCP tool, or skill instruction reads an MBTI string for any decision.

## Locked decisions (owner)

- **One Spec D, one PR**, D1 → D2 ordered (D1 CI-green before D2).
- **Capability slice = bundled arithmetic word problems, deterministic**
  (answer-correctness + `arithmetic_sound` + coherence proxy). Not GSM8K-download,
  not coherence-only.
- **Build everything CI-green + a reduced real run**: D1 runs a reduced real
  steered-vs-unsteered slice on granite-2b to demonstrate the drop; D2 validates
  the MCP surface imports + lists the new tools. No deferred-only piece.
- **Coherence = deterministic degeneracy proxy** in the shipped core; LLM-judge
  coherence is deferred.
- **New MCP tools are read-only** (no `@audited`); real steering/runs stay CLI.

## Components

### D1 — Capability-retention guardrail

| Module (new) | Responsibility | Interface | Reuse |
|---|---|---|---|
| `data/capability_arithmetic.json` | A small bundled set (~12) of arithmetic word problems, each `{id, prompt, answer}` with a known numeric gold; trait-name-free, single-number answers. | `json.load` | — |
| `agent/steering/capability.py` | Pure-stdlib deterministic scorer + retention math. | functions below | `agent/verifiers.py::arithmetic_sound` |
| `tools/run_capability.py` | Two-tier CLI: `--dry-run`/`--mock` runs the deterministic core on bundled fixtures (prints `CAPABILITY RETENTION VERIFIED ✓`); `--model granite` runs a reduced real steered-vs-unsteered slice (reuses B's `SteeredClient`) and writes a public report. | `python tools/run_capability.py [--dry-run] [--model …] [--alpha …]` | `agent/steering/hooks.py::SteeredClient`, `agent/personality_map.py` |

`agent/steering/capability.py` functions (all pure, deterministic):

- `extract_final_number(text: str) -> float | None` — the answer the response
  commits to: the number after the last `=` / "answer is" / the final standalone
  number; `None` if none parseable.
- `answer_correct(text: str, gold: float, *, tol: float = 1e-6) -> bool` —
  `extract_final_number` within `tol` of `gold`.
- `coherence_proxy(text: str) -> float` — deterministic 0–100 coherence: starts
  at 100, penalizes degeneracy (immediate token/3-gram repetition, empty/near-empty,
  pathological length); the failure mode high-alpha steering produces.
- `score_response(text: str, gold: float) -> dict` →
  `{correct: bool, sound: bool, coherence: float}` (`sound` from `arithmetic_sound`).
- `capability_cell(base_scored: list[dict], steered_scored: list[dict]) -> dict` →
  `{n, base_accuracy, steered_accuracy, capability_drop, coherence, base_coherence, retains}`
  where `capability_drop` is the **relative** drop
  `max(0.0, (base_accuracy − steered_accuracy) / base_accuracy)` when
  `base_accuracy > 0` else `0.0` (no capability to lose; `base_accuracy` is
  carried in the cell so a degenerate base is visible) — matching
  `SSA_THRESHOLDS["capability_eps"]`'s "≤5% **relative** capability drop" intent;
  `coherence = mean steered coherence`; and
  `retains = capability_drop < 0.05 and coherence >= 75` — **the exact predicate
  `ssa_verdict` applies to `cell["capability_drop"]`/`cell["coherence"]`**, so a
  D1 cell drops straight into B's SSA.

This is the missing producer for B's SSA capability gate. `tools/run_steering.py`
is **not** modified in D (the wiring of a real capability cell into a live SSA
run is the deferred headline); D1 ships the producer + a standalone reduced demo.

### D2 — Full MCP / skill packaging

| Module | Change | Interface | Reuse |
|---|---|---|---|
| `sophia_mcp/tools_impl.py` | Add 4 pure read-only impls: `ocean_measure(answers: dict) -> dict` (A's IPIP scorer over a `{item_id: 1..5}` map); `capability_retention_demo() -> dict` (runs the D1 deterministic core on the bundled fixtures, returns the cell); `council_diversity_summary() -> dict` (reads the committed `council-diversity.public-report.json`); `pif_dryrun_summary() -> dict` (the `build_cells_from_scores` invariants on synthetic fixtures). All deterministic, no model, no network. | functions | A `score_items`/`measure_ocean`, D1 `capability_cell`, C report + `pif_harness` |
| `sophia_mcp/server.py` | Add 4 thin `@mcp.tool()` wrappers (`sophia_ocean_measure`, `sophia_capability_retention`, `sophia_council_diversity`, `sophia_pif_dryrun`) each `return dumps(impl(...))`; add a `sophia://program/status` resource summarizing A–D + the OPEN ledgers. **Read-only ⇒ no `@audited`.** | `@mcp.tool()` / `@mcp.resource()` | existing `dumps`, `audit.py` posture |
| `skills/portable/sophia-personality-faithful/SKILL.md` | Expand the open-standard skill to cover the whole program (measure → steer → council → capability → PIF), the no-overclaim posture, and the new tools; keep the existing frontmatter contract. | SKILL.md | existing skill |
| `tests/test_capability.py` | The D1 deterministic core (extract/answer/coherence/cell) + the D2 `tools_impl` functions + a **cross-platform validation**: the skill frontmatter is valid YAML with the required keys, and (only if `mcp` importable) `server.py` imports and exposes the 4 new tool names; otherwise the `tools_impl` functions are exercised directly. No pytest — plain `main()`. | `python tests/test_capability.py` | repo test idiom |

## Anti-gaming / honesty posture

- The capability slice gold answers are **in-repo and deterministic** — there is
  no model to game; the only claim is "accuracy vs a fixed gold," reproducible by
  anyone. The reduced real run reports the raw `base_accuracy`/`steered_accuracy`
  and the drop; a drop is reported as a drop.
- New MCP tools cannot mutate corpus/state (read-only, no `@audited`) so they
  add no new write-risk surface to the server.
- The full headline (a real capability cell wired into a live SSA run, and an
  LLM-judge coherence channel) stays OPEN — a Spec D ledger entry records it.

## Testing

- **CI core (shipped):** `python tests/test_capability.py` — deterministic
  scorer (a correct vs a wrong vs a degenerate response → expected
  `correct/sound/coherence`), the retention cell math (base 1.0 / steered 0.5 →
  `capability_drop=0.5, retains=False`; identical → `drop=0, retains=True`), the
  4 `tools_impl` functions, and the skill-frontmatter + MCP-surface validation.
  `python tools/run_capability.py --dry-run` → `CAPABILITY RETENTION VERIFIED ✓`.
  Spec A/B/C suites stay green (add-only).
- **Reduced real (opt-in, run once this session):** `python tools/run_capability.py
  --model granite` — granite-2b on the bundled arithmetic, unsteered vs steered
  at a B alpha; writes `agi-proof/benchmark-results/capability-retention.public-report.json`
  with the honest drop. Expected: a measurable drop / coherence hit at steering
  strength, explaining B's null.

## Deferred beyond D (the program's honest frontier — recorded, not hidden)

The full N≥8/K≥20 real headline PIF run; a real capability cell wired into a live
SSA headline; an LLM-judge coherence channel; validated Level-3 steered council
seats; true third-party sealing; model×trait crossover; live GRPO; calibration
tracking. These stay OPEN in `agi-proof/failure-ledger.md`. **The program ships
its machinery and its honest nulls — not an AGI claim.**

## File summary

- **New:** `data/capability_arithmetic.json`, `agent/steering/capability.py`,
  `tools/run_capability.py`, `tests/test_capability.py`,
  `docs/09-Agent/Capability-and-Packaging.md`, a Spec D ledger entry.
- **Modified (add-only):** `sophia_mcp/tools_impl.py`, `sophia_mcp/server.py`,
  `skills/portable/sophia-personality-faithful/SKILL.md`, `.github/workflows/ci.yml`
  (run `tests/test_capability.py`).
- **Unchanged:** `agent/steering/stats.py` (D1 produces its inputs; no edit),
  `tools/run_steering.py`, all A/B/C logic.
