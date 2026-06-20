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
- [ ] Publish an architecture diagram tying RAG, council, gate, memory, executor,
      hidden eval, and proof package into one flow.

## General Cognitive Competence

- [x] Visible benchmark suite covers philosophy, psychology, history, and religion.
  Evidence: `tests/benchmark-*.json`, `benchmark/results/leaderboard-*.json`.
- [x] Hidden-test protocol covers philosophy, psychology, history, logic, coding,
      planning, tool-use, and learning.
  Evidence: `agi-proof/hidden-reviewer-packs/schema.json`, `tools/hidden_eval_protocol.py`.
- [x] Full Sophia hidden runner exists with retrieval, gate, repair, tool logs,
      memory diff, web evidence hooks, and rubric review.
  Evidence: `tools/run_hidden_eval_sophia.py`, `agent/web_evidence.py`, `agent/rubric_review.py`.
- [ ] Run at least 100 fresh hidden reviewer tasks across four or more domains.
- [ ] Complete two-pass manual semantic review for hidden tasks.
- [ ] Publish reviewer-signed aggregate hidden results without exposing private prompts.

## Baselines And Ablations

- [x] Baseline/ablation protocol is documented.
  Evidence: `agi-proof/baseline-ablation/README.md`.
- [x] Baseline ablation runner exists.
  Evidence: `tools/sophia_agent.py sophia_baseline_ablation_run`, MCP tool metadata.
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
- [ ] Run pre-test on an unknown domain.
- [ ] Run append-only learning phase.
- [ ] Run fresh post-test tasks not seen during learning.
- [ ] Publish memory diff and protected-record hash proof.

## Long-Horizon Work

- [x] Long-horizon run protocol is documented.
  Evidence: `agi-proof/long-horizon-runs/README.md`.
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

## Current Blocking Gaps

- [ ] Fresh independent hidden pack is needed; the current diagnostic packs are spent.
- [ ] Manual semantic review is still required for strict hidden-test claims.
- [ ] External benchmarks are not yet run.
- [ ] Independent clean-clone replication is not yet run.
- [ ] Long-horizon evidence is protocol-ready but not yet executed.

## Claim Boundary

Allowed public wording:

> Sophia is an AGI-candidate proof package for provenance-aware reasoning.

Disallowed public wording:

> Sophia is proven AGI.
