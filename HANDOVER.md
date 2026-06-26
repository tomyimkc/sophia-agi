# Sophia-AGI — Session Consolidation Handover

**Date:** 2026-06-26
**Branch:** `claude/consolidate-sessions-merge-main-96ld0x` → PR **#115** into `main`
**Repo:** `tomyimkc/sophia-agi`

---

## 1. What this consolidation did

Swept **every live session branch** in the repo, rebuilt a single consolidation branch
on the **current** `main` (after PRs #112/#113/#111/#114/#98 landed), and merged
**35 session branches** into it. Net diff vs `main`: **~497 files, ~100.9k insertions**.

- **~21 branches** (`feat/*`, `legal-*`, `council-*`, `openrouter-judge`, …) had **0
  changed files** vs `main` — already absorbed; nothing to merge.
- **35 branches** merged (clean auto-merge + hand-resolved conflicts).
- **1 branch deferred** — `claude/sophia-agi-architecture-review-ucvzyl` (see §4).

Verification before each push: full Python tree compiles; **169 targeted offline tests
pass**; the four CI artifact-drift gates regenerated (wiki pages, RAG `--local` index,
RESULTS.md, dataset guard) and re-pass locally.

## 2. Capability features added (the "more capable toward AGI" deliverable)

The vision/capability arc — Sophia already owned the *trustworthy* half of an agent
(provenance, verifier-gated, fail-closed); these sessions add the *more-capable* half,
in Sophia's idiom (deterministic, offline-testable, auditable):

- **Agent harness (vision):** KV-cache-aware context manager (`agent/context_manager.py`),
  long-horizon execution engine with durable task tree + recovery memory
  (`agent/long_horizon.py`), subagent delegation (`agent/subagent.py`).
- **Harness↔model co-evolution loop** (`agent-harness-coevolution`).
- **Self-evolving agent loop** — evolve→no-hack→promote→retain→commit, fail-closed
  (`repo-agi-research-alignment`).
- **Lifelong / continual learning** — closed-loop accumulation benchmark
  (`agent/lifelong_accumulation.py`, 561 LOC + tests); catastrophic-forgetting retention
  gate wired into promotion (see §3).
- **AI-search substrate** — query understanding, hybrid dense⨁BM25 RRF recall,
  multi-layer HNSW Rust ANN core + Python↔Rust bridge, pluggable embedder registry, plus a
  deterministic offline lexical-vector retrieval tier (`agent/retrieval.py` combined).
- **Systems/infra** — sharded async Rust KV cache + `sophia-lsm` WAL/io_uring storage,
  NCCL all-reduce bench + cluster scheduling/fault simulator, DeepSeek pretraining-alignment
  + data-engineering (dedup) pipeline, CI path-selection.
- **Evals** — OpenRouter calibration, judged agent-faithfulness, multimodal hallucination
  traps, TruthfulQA lane, free-generation fabrication scorer + domain scorers.
- **Gates/governance (safety-critical):** the adapter-promotion gate now combines the
  **godel-oracle** invariant check with **multi-goal Pareto** promotion AND a
  **catastrophic-forgetting retention** gate (`tools/promote_adapter.py` +
  `agent/continual_plasticity.py`).

## 3. Non-trivial conflict resolutions (review these first)

- **`tools/promote_adapter.py`** — hand-combined HEAD's godel-oracle/content-channel gate
  with v3-gate's multi-goal + retention gate. Both feature sets coexist. Tests adapted to
  the tuple-return `build_candidate` (all call sites unpack `[0]`). 30/30 gate tests green.
- **`agent/retrieval.py`** — hand-combined the registry-aware learned-embedding tier
  (`embed_query_for_index`) with the deterministic lexical-vector tier; `retrieve(mode=…)`
  walks learned → lexical-vector → keyword.
- **Policy:** append-only files (`failure-ledger.md`, `.gitignore`, `CHANGELOG.md`,
  ci path-filters) → union; branches that predate a merged PR → keep `main`'s newer code,
  but their *new* files (benchmarks/scorers) still merge.
- **`RESULTS.md`** regenerated from `published-results.json`; a distributed-storage branch's
  hand-edited kvcache section was dropped (RESULTS.md is generated — do not hand-edit). To
  restore it properly, add a `published-results.json` entry and re-run
  `python tools/build_results_page.py`.

## 4. OUTSTANDING — needs a human decision

**`claude/sophia-agi-architecture-review-ucvzyl` was NOT merged.** It has an irreconcilable
schema collision on `agi-proof/architecture-bets.json`:
- `tests/test_architecture_bets.py` (on `main`) asserts a *module-wiring* registry
  (`claim_router`, `graded_decision`, …) — each bet needs `module`, `live_caller`,
  `status`, `ablation_flag`.
- `tests/test_long_context_runner.py` (on the branch) asserts **exact set-equality** on a
  *different* 7-bet registry (`verifier-gated-long-context`, `council-small-models`, …) with
  `honest_status`/`blocked_on`/`implementation_files`.
- `tools/lint_claims.py` also reads this file.

One file cannot satisfy both test suites without fabricating a hybrid schema that guts
`main`'s governance tracking. **Decision needed:** which schema is canonical going forward?
The branch's *code* (long-context recall eval `tools/run_long_context_*`, intake contract
`sophia_contract/intake.py`, the retrieval lexical-vector tier — already merged separately)
is valuable; only the `architecture-bets.json` schema + its two competing tests conflict.

## 5. How to land / continue

1. **PR #115** → `main` is open. `mergeable_state` was `blocked` only by required CI
   (no merge conflicts — base == current `main`). The post-merge CI failures
   (`build_rag_index --verify`, `wiki_sync check`, `build_results_page --check`) are FIXED
   in commits `db021d6`/`56f3c01`. Once CI is green, merge (squash or merge commit).
2. **Resolve architecture-review (§4)** — pick the canonical `architecture-bets.json`
   schema, update the loser test + `lint_claims.py`, then merge that branch.
3. **Optional:** add the kvcache result to `published-results.json` to restore it in
   RESULTS.md.

## 6. Suggested prompt for the next session

> Continue the Sophia-AGI consolidation. PR #115 (`claude/consolidate-sessions-merge-main-96ld0x`)
> merges 35 session branches into main; confirm CI is green and merge it. Then resolve the
> one deferred branch `claude/sophia-agi-architecture-review-ucvzyl`: it collides on
> `agi-proof/architecture-bets.json` because `tests/test_architecture_bets.py` (main) and
> `tests/test_long_context_runner.py` (branch) demand incompatible registry schemas, and
> `tools/lint_claims.py` reads the file. Decide the canonical schema, reconcile both tests +
> lint_claims, merge the branch, and keep all four CI artifact-drift gates green
> (`wiki_sync.py check`, `build_rag_index.py --verify`, `build_results_page.py --check`,
> `build_local_sophia_dataset.py --check`). Then delete the merged session branches.
