# Corpus, Math/Code, and Capability Roadmap

**Status:** planning doc (no capability claims). Sequenced next steps for three
distinct workstreams that are often conflated. Grounded in a repo audit of
`wiki/`, `okf/`, `data/`, `provenance_bench/`, `agent/verifiers.py`, `eval/`,
`benchmark/`, `training/`, `selfextend/`, and `agi-proof/`.

> **Scope discipline.** Per [VISION.md](../../VISION.md) and
> [agi-proof/definition.md](../../agi-proof/definition.md), "the benchmark is
> trustworthiness and verifiability, **not breadth of knowledge**." Expanding the
> corpus is product value, **not** progress toward an AGI-candidate claim, and must
> never be framed as such (the `tools/lint_wiki_provenance.py` falsifier and the
> conscience kernel exist to catch that conflation). This doc keeps the three
> goals separate so each is measured by its own honest yardstick.

## The three workstreams and their real leverage

| Workstream | What it actually buys | AGI-candidate leverage |
|---|---|---|
| **A. Corpus coverage + provenance hardening** | Broader, harder provenance coverage; stronger leaderboards | **Low** (curatorial) — real product value, not a capability claim |
| **B. Math + coding hardening** | Stronger machine-checked verifiers + an RLVR substrate | **Medium–High** — verifiers are the moat and the fuel |
| **C. Turn a toy pillar real** | A genuine learning/generalization result | **Highest** — the actual frontier (per the failure ledger) |

**Connective thesis.** Math and code are not a side quest. They give
*machine-checkable ground truth with no LLM judge* — exactly what a real RLVR /
self-extension loop needs. So **B is the on-ramp to C.** Build the math/code
verifier + RLVR pack (B), and the live GPU run (C) has a rigorous,
contamination-safe, judge-free domain to train on.

---

## Current state (audited)

**Corpus (A).** 67 wiki pages across 6 tiers (text 32, concept 13, event 11,
tradition 8, figure_source_seat 2, figure 1); ~150 structured records in
`data/*.json` across 4 domains (philosophy, psychology, history, religion);
provenance benchmark of 290 cases (`provenance_bench/data/misattributions.json`
199 false + `wikidata_snapshot.json` 91 true). Schema frozen in `okf/schema.py`
and `data/schema.json` (drift = test failure). `dispute/` is a declared page type
with **zero pages on disk**. Machinery (graph, counterfactual, revision, linker,
validators) is strong; the corpus is thin and the graph is sparse.

**Math/code (B).** Real and working: sandboxed `code_tests_pass`
(`agent/verifiers.py`, opt-in `SOPHIA_ALLOW_CODE_EXEC`), `arithmetic_sound`
(binary equalities only), `benchmark/code_tasks.json` (24 tasks), verifier-gated
code RLVR reward (`provenance_bench/code_reward.py`), the repair-loop benchmark
(`tools/run_code_uplift.py`), entity-disjoint splits
(`provenance_bench/rl_dataset.py`), and a GSM8K external-eval harness (98% base
model, N=100, honestly scoped in [RESULTS.md](../../RESULTS.md)). **Missing:** any
symbolic math (no sympy/CAS), MATH/HumanEval/MBPP packs, math RLVR data, and
coverage/mutation feedback in the repair loop.

**Capability loop (C).** `selfextend/loop.py` already closes the loop on a
**self-authored toy domain** (policy 0.5→1.0, all 4 invariants) via
**verifier-guided selection (rejection sampling), not a gradient update**.
`tools/run_rlvr.py` is fully wired for live GRPO (trl + vLLM + QLoRA) and refuses
on non-CUDA. The W2 promotion gate (`tools/promote_adapter.py` +
`agent/continual_plasticity.py`) is real and has really rejected an adapter
(religion regression −0.167). Open, pre-registered gaps in
[agi-proof/failure-ledger.md](../../agi-proof/failure-ledger.md):
`rlvr-live-run-not-yet-gated`, `self-extending-loop-closes-offline` (PARTIAL),
`hidden-review-third-party-not-run`, `long-horizon-not-run`,
`distribution-shift-not-run`.

---

## Workstream A — Corpus coverage + provenance hardening

Do **hardening before breadth**. Gate credibility scales with adversarial
difficulty, not page count.

### A1. Provenance hardening (do first)
- **Harder misattribution traps** in `provenance_bench/data/misattributions.json`:
  near-miss contemporaries (same school/century), partial-authorship cases that
  require the `compiled`/`layered` ranks to discriminate, and
  translation/edition confusions. The current set skews to easy cross-tradition
  merges (Confucius ≠ Laozi).
- **First `dispute/` pages** — the page type exists in `PAGE_TYPES` but has zero
  pages. Disputes exercise `okf/graph.py`'s contradiction ledger and
  `okf/counterfactual.py`, the most differentiated code in the repo.
- **Densify edges** (`derivesFrom`, `contradicts`, `supersedes`). Confidence-
  laundering detection and counterfactual retraction only fire on a connected
  graph.

### A2. Coverage breadth (do second; frame as coverage, never "AGI")
- **Science/medicine** — retracted papers, misattributed discoveries; high
  trap density, externally verifiable via the live Crossref/Wikidata backend.
- **Law** — `legal_citation_exists` / `legal_holding_faithful` verifiers and the
  Mata v. Avianca case already exist; a `data/legal.json` domain turns them into
  a full leaderboard tier.

### A — Mechanics
`data/<domain>.json` → register in `data/domains.json` → extend `DOMAINS` in
**both** `okf/schema.py` and `data/schema.json` → `tools/wiki_sync.py emit` →
`tools/wiki_validate.py` + `tools/wiki_health.py`. Frontmatter read-only from
data; benchmark labels external-only (non-circularity). Commit as
**"corpus coverage + provenance hardening,"** never "AGI expansion."

### A — Acceptance
`wiki_health.py` reports all hard defects 0 (`coherent: true`); new traps raise
benchmark difficulty without raising false-positive cost on the 91 true controls.

---

## Workstream B — Math + coding hardening

Ordered by leverage.

### B1. Symbolic-math verifier (highest value/effort, low risk)
Add a `sympy`-backed verifier to `agent/verifiers.py` (the math analogue of
`code_tests_pass`): parse a final answer + claimed equality / simplification /
derivative / integral, recompute symbolically, accept/reject deterministically,
fail-closed if sympy is unavailable. Wire into `_numeric_gate` in `agent/gate.py`
to upgrade it from "2+2=5" detection to real algebra. Tests follow the patterns
in `tests/test_verifiers.py`.

### B2. Real eval packs (unlocks measurement)
Add **MATH** and **MBPP/HumanEval** subsets behind the existing external-eval
harness (the one that scored GSM8K). Honest framing: these measure the **base
model** through the harness *until* wired to the gate/repair loop — then the
headline becomes the **Sophia delta** (pass@1 baseline vs. after verifier-gated
repair via `tools/run_code_uplift.py`), which is a legitimate Sophia-specific
claim once it clears the no-overclaim gate.

### B3. Math RLVR pack (feeds Workstream C)
Build a math RLVR pack with the B1 sympy verifier as the deterministic reward
seam (mirror `provenance_bench/code_reward.py`), entity-disjoint via
`provenance_bench/rl_dataset.py`. Optional: partial credit for verified
intermediate steps. Validate the offline invariants in `tools/run_rlvr.py`
(deterministic, monotone, bounded, contamination-free).

### B4. Harden the code path
Add coverage/mutation feedback to the repair loop so a passing-but-weak test
can't launder a wrong solution; expand `benchmark/code_tasks.json` beyond 24
tasks (MBPP import).

### B — Acceptance
B1: sympy verifier passes a constructed pack with 0 false accepts, fail-closed
when sympy absent. B2/B3: external-oracle accuracy reported in `RESULTS.md` under
the external-eval caveat; RLVR offline invariants green in CI.

---

## Workstream C — The real (non-toy) next pillar

**Do not** try to make `predictive_world_model`, `active_inference`, or
`layered_memory` real first — those need open research. The achievable real
pillar, and the one the failure ledger calls "the remaining rung":

**→ Close the self-extension / RLVR loop with a *live weight update* on a
*third-party* domain.**

Three pre-registered, open gaps:
1. **Live RLVR run** — `tools/run_rlvr.py` is wired (trl + vLLM + QLoRA), refuses
   on non-CUDA. Needs one cloud GPU run (~40 GPU-hrs estimated). Single highest-
   leverage action in the repo. Closes `rlvr-live-run-not-yet-gated`.
2. **Third-party held-out domain** — replaces the self-authored domain; unlocks
   three stuck claims at once (self-extension, RLVR, full-Sophia hidden-eval).
   **Math/code is the cleanest such domain** (gold tests, no LLM judge) → this is
   why B3 is the on-ramp.
3. **Clear the no-overclaim gate** — ≥2 judge families, Cohen's κ ≥ 0.40, ≥3
   runs, 95% CI excludes 0 (for any model-judged surface; the math/code reward is
   deterministic and needs no judge).

### C — Acceptance
The self-extension loop closes on a **third-party / math-code held-out domain**
with a **live gradient update** (not selection), the W2 promotion gate +
formal protected-floor proof both pass, and the result clears the no-overclaim
gate. Update the failure ledger entries; flip nothing to a headline before the
gate clears.

---

## Recommended sequence

Each step is independently shippable; early steps unblock later ones.

| # | Step | Workstream | Where it runs | Unblocks |
|---|------|-----------|---------------|----------|
| 1 | Sympy math verifier + gate wiring + tests | B1 | here (CPU) | B2, B3 |
| 2 | Provenance hardening pack + first `dispute/` pages + denser edges | A1 | here (CPU) | A2 |
| 3 | MATH + MBPP/HumanEval eval packs | B2 | here (CPU/local model) | measurement |
| 4 | Math RLVR pack (sympy reward seam, entity-disjoint) | B3 | here (CPU offline-invariants) | C |
| 5 | One new corpus domain (science/medicine **or** law) | A2 | here (CPU) | leaderboard tier |
| 6 | Live RLVR GPU run on the math pack, gated | C1–C3 | **needs CUDA GPU** | the real pillar |

Steps 1–5 run offline/CPU in this environment. Step 6 needs CUDA hardware
(out of scope to *run* here); everything up to the launch command can be prepared
here.

## Non-goals / guardrails

- No corpus expansion framed as "AGI progress" in commits, docs, or PRs.
- No headline number before the no-overclaim gate clears (≥2 judge families,
  κ ≥ 0.40, ≥3 runs, CI excludes 0) — deterministic verifier accuracy excepted.
- No adapter promoted to default without the W2 gate + formal protected-floor
  proof (`tools/promote_adapter.py`).
- Live RLVR and live grounding remain out of scope to *run* in CI; their
  interfaces and offline invariants are the deliverable here.
