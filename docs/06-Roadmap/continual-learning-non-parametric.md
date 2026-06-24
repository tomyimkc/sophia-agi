# Plan: Non-Parametric Continual Learning on the OKF Belief Graph

**Status:** active — branch `claude/continual-learning-catastrophic-forgetting-kyhui2`
**Thesis:** Catastrophic forgetting is a property of *parametric* (weight-stored)
knowledge. Sophia already stores declarative knowledge **non-parametrically** — as
provenance-typed OKF pages in a belief graph (`okf/`). When "learning" is a page
write and "forgetting" is an auditable graph operation (`okf.revise`,
`okf.counterfactual_remove`), catastrophic forgetting of facts is **structurally
impossible**: writing page *N+1* cannot physically alter page *N*. This plan turns
that claim into measured, gate-checked machinery — in Sophia's no-overclaim style.

> **Scope, stated plainly.** This addresses forgetting of *declarative knowledge*
> (facts, attributions, provenance) — most of what Sophia's corpus is about. It does
> **not** by itself cure forgetting of *skills* that must live in weights (language,
> reasoning procedures, perception). For that residue, Experiment 4 consolidates only
> what passes a forgetting-regression gate. No AGI claim; the deliverable is honest,
> measured machinery.

---

## Why this is the right approach (research grounding)

- **Parametric continual learning is partial.** EWC/replay/regularization mitigate but
  never solve forgetting; surgical weight editing (ROME/MEMIT) shows *gradual then
  catastrophic* forgetting and a "disabling edit" can cripple the model, and edits stay
  surface-level (the "ripple effect," ~38–66% on entailed facts).
  - Model Editing → Catastrophic Forgetting — https://arxiv.org/html/2401.07453v3
  - Ripple Effects of Knowledge Editing (TACL 2024) — https://aclanthology.org/2024.tacl-1.16.pdf
- **Non-parametric continual learning is the winning camp for facts.** Keep weights
  frozen; store knowledge externally; retrieve it. Plain vector-RAG is too flat —
  structured (knowledge-graph) external memory is needed.
  - From RAG to Memory: Non-Parametric Continual Learning for LLMs (HippoRAG 2, ICML 2025) — https://arxiv.org/html/2502.14802v1
- **Belief revision / Truth Maintenance Systems** give selective, auditable, reversible
  forgetting — the operation weights cannot do. AGM postulates + hierarchical belief
  importance (axioms > user-stated > system-inferred).
  - Belief Revision & TMS overview — https://cse.buffalo.edu/~shapiro/Papers/br-overview.pdf
  - Practical defeasible reasoning + belief revision — https://link.springer.com/chapter/10.1007/BFb0028183
- **Concept-level neuro-symbolic CL** (COOL) and **brain-inspired complementary learning
  systems** outperform regularization on forgetting by learning stable concepts and
  splitting fast (episodic) from slow (consolidated) memory.
  - Neuro-Symbolic CL / COOL — https://arxiv.org/pdf/2302.01242
  - Hybrid Learners Do Not Forget — https://arxiv.org/pdf/2503.12635

---

## What already exists (we build on, not from scratch)

| Capability | Existing module |
|---|---|
| Obsidian-format pages + wikilinks | `okf/frontmatter.py`, `okf/wikilinks.py`, `okf/page.py` |
| Provenance belief graph + min-over-chain confidence | `okf/graph.py` (`build`, `propagate_confidence`, `belief`) |
| Counterfactual "what if source removed" | `okf/counterfactual.py` (`counterfactual_remove`, `is_grounded`) |
| Belief revision + transitive cascade (TMS) | `okf/revision.py` (`revise`, `claims_to_abstain`) |
| Fail-closed provenance gate | `agent/gate.py`, `agent/grounded_gate.py` |
| Weight-update promotion gate (forgetting-regression test) | `agent/continual_plasticity.py` (`evaluate_update`, protected suites) |
| Memory consolidation scaffold | `agent/memory_consolidation.py`, `agent/layered_memory.py` |
| Live wiki knowledge pages | `wiki/` (concept/event/figure/text/tradition) |

The OKF graph is the substrate; this plan adds the **measurement, conflict-handling,
unlearning, and consolidation** layers on top.

---

## Execution order: smallest → boldest

Each experiment is self-contained, offline, deterministic, dependency-free (matching
`tests/test_okf_*.py` conventions), ships with tests, and emits a machine-readable
JSON report (schema `sophia.<name>.v1`, `level3Evidence: false` until it clears the
no-overclaim gate).

### Experiment 1 — Sequential-retention benchmark (smallest, highest-certainty win)
**Goal:** Prove 0% forgetting of grounded facts under sequential "tasks," and make
forgetting *measurable* (the report's "hard to even measure" point).

- **Module:** `agent/continual_retention.py`
- **Approach:** Treat each "task" as a batch of OKF pages added to the graph in
  sequence. After each batch, snapshot the belief state (grounded claims +
  `effectiveConfidenceRank` per id via `propagate_confidence`). Compute a
  **retention matrix** R[i][j] = fraction of task-i facts still correctly grounded
  after learning task j. Backward-transfer / forgetting = R[i][last] − R[i][i].
- **Headline metric:** `forgottenGroundedClaims` — must be 0 for a pure-additive
  stream (no contradictions).
- **Report:** `agi-proof/continual/retention_report.json`.
- **Tests:** `tests/test_continual_retention.py` — additive stream ⇒ zero forgetting;
  retention matrix is lower-triangular-complete; deterministic.
- **CLI/demo:** `scripts/demo_continual_retention.py` (offline, no keys).
- **Acceptance:** additive 5-task stream ⇒ 0 forgotten grounded claims; snapshot diff
  reproduces exact per-task retention.

### Experiment 2 — Conflict → belief revision (revise-or-abstain, never overwrite)
**Goal:** When new knowledge contradicts old, do AGM-style revision instead of silent
overwrite (which *is* forgetting in weight models).

- **Module:** `agent/belief_revision_policy.py` (orchestrates `okf.revise` +
  `contradiction_ledger` + constitution-ranked importance).
- **Approach:** On ingest, detect contradictions (`contradicts` edges /
  `contradiction_ledger`). Apply a hierarchical importance order
  (constitution/axiom > user-stated > attributed source > system-inferred). Resolve
  by retracting the *weaker* side (with cascade) **or abstaining** if comparable —
  never clobbering. Emit an audit entry per decision.
- **Report:** decision ledger with `kept`, `retracted`, `abstained`, `cascade`.
- **Tests:** `tests/test_belief_revision_policy.py` — weaker claim yields; equal-rank
  conflict ⇒ abstain; constitution always wins; cascade recorded.
- **Acceptance:** baseline "last-write-wins" loses the older fact; policy preserves the
  higher-importance fact and records why.

### Experiment 3 — Forgetting-as-command (reversible, audited unlearning)
**Goal:** Demonstrate selective, reversible forgetting — impossible cleanly in weights
(GDPR deletion, retracting a debunked/poisoned source).

- **Module:** `agent/unlearning.py` (thin policy over `okf.retract` /
  `counterfactual_remove`, with tombstone + restore).
- **Approach:** `forget(source, reason)` → show blast radius via
  `counterfactual_remove`, retract with audit entry, persist a **tombstone** (not a
  delete) so the op is reversible; `restore(source)` re-grounds dependents. Verify the
  runtime gate now refuses the abstain set (`claims_to_abstain`).
- **Report:** before/after grounded-claim counts + reversibility proof.
- **Tests:** `tests/test_unlearning.py` — retract un-grounds the cascade; gate refuses
  abstained ids; restore returns to the exact prior belief state (round-trip).
- **Acceptance:** poisoned-source retraction removes exactly its support cascade and is
  bit-for-bit reversible.

### Experiment 4 — CLS consolidation loop (boldest; touches weights, gated)
**Goal:** Complementary Learning Systems — wiki = fast hippocampus, weights = slow
neocortex. Distill *only* stable, gate-cleared wiki subgraphs into a LoRA candidate,
and promote **only if** it passes the forgetting-regression gate.

- **Module:** `agent/cls_consolidation.py` (selection + candidate assembly) feeding the
  existing `agent/continual_plasticity.py` promotion gate.
- **Approach:** Select wiki subgraphs stable + gate-cleared for ≥ N snapshots → build a
  distillation set → produce an `UpdateCandidate` → `evaluate_update(target_suite=…,
  max_protected_regression=…)`. Protected suites = `source_discipline`,
  `fact_check_false_accept` (the catastrophic-forgetting tripwire). Promote / quarantine
  / reject + ledger (`append_promotion_ledger`).
- **Report:** consolidation candidates + promotion decisions (`agi-proof/continual/`).
- **Tests:** `tests/test_cls_consolidation.py` — only-stable subgraphs selected; an
  adapter that regresses a protected suite is **rejected** (anti-forgetting invariant);
  clean improving adapter promotes. (Distillation itself remains a dry-run candidate —
  `level3Evidence: false` — no real training in CI.)
- **Acceptance:** the protected-suite invariant holds: no adapter that regresses old
  knowledge can be promoted.

---

## Cross-cutting requirements

- **Offline, deterministic, dependency-free** — like the existing `okf` tests; no
  network or API keys in CI.
- **Machine-readable reports** under `agi-proof/continual/`, schema-versioned,
  `level3Evidence: false` until a result clears the full no-overclaim gate (≥2 judge
  families, κ ≥ 0.40, ≥3 runs, CIs) per RESULTS.md.
- **No overclaim** — Experiments 1–3 are exact/structural (provable); only Experiment 4
  could ever make a model-performance claim, and only through the existing gate.
- **Tests + demo per experiment**; update `CHANGELOG.md`. No corpus/weight changes
  without the user's go-ahead.

## Done criteria

1. Four modules + tests, all green, offline.
2. One JSON report per experiment under `agi-proof/continual/`.
3. Demo script for Experiment 1 runnable with `python scripts/demo_continual_retention.py`.
4. This plan kept in sync as scope evolves.
