# Sophia AGI-Candidate Proof TODO

This checklist consolidates the proof work into one public tracker. It is
evidence-oriented: checked items mean the repo has an artifact or protocol, not
that Sophia is proven AGI.

## Architecture Claim

- [x] Sophia is documented as an AGI-candidate architecture.
  Evidence: `agi-proof/definition.md`, `agi-proof/README.md`, `docs/09-Agent/Sophia-Agent.md`.
- [x] Repo, docs, MCP tools, benchmark suite, source discipline, council system,
      memory gates, and executor/proof scaffolds exist.
  Evidence: `sophia_mcp/`, `benchmarks/`, `agent/`, `docs/08-Domains/`, `agi-proof/evidence-manifest.json`.
- [x] Publish an architecture diagram tying RAG, council, gate, memory, executor,
      hidden eval, and proof package into one flow.
  Evidence: `docs/09-Agent/Sophia-Architecture.md`.

## General Cognitive Competence

- [x] Visible benchmark suite covers philosophy, psychology, history, and religion.
  Evidence: `tests/benchmark-*.json`, `benchmark/results/leaderboard-*.json`.
- [x] Hidden-test protocol covers philosophy, psychology, history, logic, coding,
      planning, tool-use, and learning.
  Evidence: `agi-proof/hidden-reviewer-packs/schema.json`, `tools/hidden_eval_protocol.py`.
- [x] Full Sophia hidden runner exists with retrieval, gate, repair, tool logs,
      memory diff, web evidence hooks, and rubric review.
  Evidence: `tools/run_hidden_eval_sophia.py`, `agent/web_evidence.py`, `agent/rubric_review.py`.
- [x] Run one fresh self-authored 8-case hidden pack through the full Sophia runner with live DeepSeek and commit artifacts/checksums (execution-health only; not independent/validated evidence).
  Evidence: `agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.public.json`, `agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.checksums.sha256`.
- [ ] Run at least 100 fresh hidden reviewer tasks across four or more domains. (2026-06-26 W1 ran 8 self-authored tasks with live DeepSeek; artifact-backed execution-health evidence, but insufficient as 100-task or independent hidden-reviewer evidence, so this remains open.)
- [x] Complete two-pass manual semantic review for hidden tasks. (2026-06-26 W3: author two-pass on W1 v2 pack; semantic 8/8, strict-pass 3/8 reported distinct from auto; author-only, third-party independence NOT cleared — `agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.W3-review.md`.)
- [x] Publish reviewer-signed aggregate hidden results without exposing private prompts. (2026-06-26: reviewer-signed aggregate in `.W3-review.md` / `.manual-review-completed.json`; author signature, third-party still pending.)

## Baselines And Ablations

- [x] Baseline/ablation protocol is documented.
  Evidence: `agi-proof/baseline-ablation/README.md`.
- [x] Baseline ablation runner exists.
  Evidence: `tools/run_ablation_sophia.py` runs all seven modes over the shared
  `run_case` pipeline; `tests/test_ablation_runner.py`; example
  `agi-proof/baseline-ablation/example-pack.json`.
  Run: `python3.12 tools/run_ablation_sophia.py <pack.json> --backend grok --modes all`.
- [ ] Compare Sophia-full against raw model.
- [ ] Compare Sophia-full against raw model plus tools.
- [ ] Compare Sophia-full against Sophia without knowledge base.
- [ ] Compare Sophia-full against Sophia without council.
- [ ] Compare Sophia-full against Sophia without gate.
- [ ] Compare Sophia-full against Sophia without memory.
- [ ] Compare Sophia-full against Sophia without executor.
- [ ] Publish effect sizes, confidence intervals, and failure examples.

## Learning Under Novelty

- [x] Learning-under-shift protocol is documented.
  Evidence: `agi-proof/learning-under-shift/README.md`.
- [x] Hidden runner records append-only memory diff for learning tasks.
  Evidence: `tools/run_hidden_eval_sophia.py`.
- [x] Protected knowledge hash check exists to prove old records were not changed.
  Evidence: `tools/run_hidden_eval_sophia.py`.
- [x] Standalone learning-under-shift experiment runner exists (pre-test,
      promotion gate, post-test, old-benchmark stability, contamination audit,
      protected-hash proof).
  Evidence: `tools/run_learning_shift.py`, `tests/test_learning_shift.py`,
  `agi-proof/learning-under-shift/example-spec.json`.
- [ ] Run pre-test on an unknown domain.
- [ ] Run append-only learning phase.
- [ ] Run fresh post-test tasks not seen during learning.
- [ ] Publish memory diff and protected-record hash proof.

## Long-Horizon Work

- [x] Long-horizon run protocol is documented.
  Evidence: `agi-proof/long-horizon-runs/README.md`.
- [x] Long-horizon autonomy harness exists (append-only run log, tool/state/
      failure/self-correction events, human-intervention counter, checkpoint/
      resume, tier + autonomy classification).
  Evidence: `tools/run_long_horizon.py`, `tests/test_long_horizon.py`, demo
  `agi-proof/long-horizon-runs/long-horizon-self-test-2026-06-20.public-report.json`.
- [ ] Run a 30-minute task with action/tool/failure/self-correction logs.
- [ ] Run a 2-hour task with action/tool/failure/self-correction logs.
- [ ] Run a 1-day task with action/tool/failure/self-correction logs.
- [ ] Report human interventions and final artifacts for every long-horizon run.

## External Benchmarks

- [x] External benchmark plan is documented.
  Evidence: `agi-proof/external-benchmarks/README.md`.
- [ ] Run ARC-AGI / ARC-AGI-3 for novel reasoning.
- [ ] Run GAIA-style tool-using assistant tasks.
- [ ] Run SWE-bench-style repository tasks.
- [ ] Run expert-blinded philosophy/psychology/history/religion tests.
- [ ] Publish setup, commit hash, model/backend versions, and raw aggregate scores.

## Third-Party Reproduction

- [x] Third-party replication checklist is documented.
  Evidence: `agi-proof/third-party-replication/README.md`.
- [x] Replication harness exists (records commit/env, runs validation + tests,
      emits a reviewer-signature template; cannot self-certify).
  Evidence: `tools/run_replication_check.py`, sample
  `agi-proof/third-party-replication/replication-check-2026-06-20.json`.
- [ ] External reviewer clones the repo from scratch.
- [ ] External reviewer runs validation and proof package build.
- [ ] External reviewer adds hidden questions Sophia has not seen.
- [ ] External reviewer reproduces or falsifies reported results.
- [ ] Publish reviewer identity policy, environment, and signed outcome.

## Data Package

- [x] `agi-proof/definition.md`
- [x] `agi-proof/preregistered-thresholds.md`
- [x] `agi-proof/benchmark-results/`
- [x] `agi-proof/baseline-ablation/`
- [x] `agi-proof/hidden-reviewer-packs/`
- [x] `agi-proof/long-horizon-runs/`
- [x] `agi-proof/learning-under-shift/`
- [x] `agi-proof/failure-ledger.md`
- [x] `agi-proof/third-party-replication/`
- [x] `agi-proof/evidence-manifest.json`
- [x] `agi-proof/TODO.md`
- [x] `agi-proof/mlops/checkpoint-registry.json` (W6, created 2026-06-26: 1 entry referencing the RLVR-math 3-seed N=60 artifact; verdict promote; `canClaimAGI:false`; self-extension rung / not benchmark evidence).

## Current Blocking Gaps

The Level-3 *harnesses* now exist and are unit-tested (ablation, learning-shift,
long-horizon, replication, architecture diagram). The remaining gaps are
evidence runs that need either a live backend or an external human.

- [ ] Fresh independent hidden pack is needed; the current diagnostic packs are spent. W1 live-backend execution-health run 2026-06-26 is artifact-backed, but it used a self-authored pack and therefore does **not** close this independent-pack gap (`agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.public.json`). The earlier v1 artifact-retention failure is also recorded (`agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1.invalid-run-summary.json`).
- [ ] Manual semantic review is still required for strict hidden-test claims.
- [ ] Ablation and learning-shift deltas not yet produced on a live backend
      (grok token expired / `DEEPSEEK_API_KEY` unset); runners are ready.
- [ ] External benchmarks are not yet run.
- [ ] Independent clean-clone replication is not yet run (harness ready; a real
      external reviewer must run and sign it).
- [ ] Long-horizon 30-minute / 2-hour / 1-day runs not yet executed (harness ready;
      only a short self-test demo exists).

## Claim Boundary

Allowed public wording:

> Sophia is an AGI-candidate proof package for provenance-aware reasoning.

Disallowed public wording:

> Sophia is proven AGI.
