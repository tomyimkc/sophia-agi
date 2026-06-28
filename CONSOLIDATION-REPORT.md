# Main-branch consolidation report — 2026-06-26

Consolidation pass over `tomyimkc/sophia-agi`, which had drifted to **82 remote
branches** and 6 open PRs from many parallel agent sessions.

## What was done

1. **Merged PR #133 into `main`** (merge commit `67cb249`).
   #133 was an integration branch combining the three open agent-track PRs
   (#129 verifier-as-reward, #131 DGX Spark NVFP4, #132 IP package). It had gone
   stale (`mergeable_state: dirty`) because `main` advanced 3 commits past its base.
   - Re-merged `origin/main` into the branch and resolved the one real conflict in
     `tools/ingest_rlvr_eval.py` (`map_report`): unified #133's task-aware
     metric/protected-axis logic with `main`'s new capability-panel block into a
     single `mapped = {...}` + `return mapped`. Directly-affected tests pass:
     `test_ingest_rlvr_eval.py` (6), `test_rlvr.py` (14), `test_code_rlvr.py` (5).
2. **Closed PR #129** — fully superseded; `git log origin/main..#129` is empty.

## Pre-existing CI hang on `main` — diagnosed and FIXED in this branch

`main`'s required `ci-complete` check had been **failing/cancelled on every recent
push** (head `42a817b` = cancelled; #132 merge = failure; ~10 prior). The `test`
job (`pytest -q`, `timeout-minutes: 20`) **hung**: in CI it reached 6% then made
zero progress for 14 min before being cancelled.

Root cause (reproduced on **clean `origin/main`**): `agent/verifiers.py`
`provenance_faithful()` recompiled ~765 regexes (9 per record×author, 36-record
corpus = 0.31s) on **every call**, and the gate is invoked **once per claim**.
`provenance_bench` scores 319 attribution cases × 2 arms across 5 panel tests
(`test_capability_panel.py`, `test_eval_rlvr_capability_panel.py`) →
≈3,190 rebuilds × 0.31s ≈ **16 min** of pure recompilation. Not an infinite loop —
pathological O(claims) recompilation.

**Fix (this branch):** memoise the compiled specs under a content key
(titles + `doNotAttributeTo`), LRU-bounded to 256 so grounded mode can't grow it
unbounded. 640 build+verify calls dropped from ~200s → **0.18s**; the two panel
test files now pass in **9.5s** (were a 16-min hang); 281 provenance/verifier
tests pass with **zero behavior change** (assertion still fails, carve-out/clean
still pass). PR #131's `fix(ci)` (timeout 20→30 + dropping a redundant
search-quality eval) is complementary headroom, not the root-cause fix.

## Branch cleanup — 36 merged branches (action required: run the script)

36 branches are **fully merged into `main`** (their content is preserved in `main`;
deleting them loses nothing). Deletion could **not** be completed from this
environment: `git push --delete` returns **HTTP 403** (organization egress policy
blocks ref deletion), and there is no GitHub MCP delete-branch tool.

➡️ Run **`scripts/delete-merged-branches.sh`** from a machine with push rights to
delete all 36. They are (also lists the now-merged #133 and #129 branches):

```
agi-validation, chore/math-code-sft-workflow-on-main, ci/parallel-runpod-seeds,
claude/agi-pilots-feasibility-review-nkoqn3, claude/consolidate-sessions-merge-main-96ld0x,
claude/continual-learning-catastrophic-forgetting-kyhui2, claude/error-memory-rag,
claude/fix-promote-adapter-religion-floor-test, claude/github-portfolio-redesign-sk5htt,
claude/gitignore-private-files-q4ljon, claude/hk-bilingual-advisor, claude/hurdle1-truthfulqa-codex,
claude/llm-wiki-agi-expansion-v0eifa, claude/market-driven-roadmap-04qci9,
claude/repo-dev-recommendations-c1zxvy, claude/safe-rsi-brainstorm-v7q5f9, claude/sophia-7b-train-verify,
claude/sophia-agi-architecture-review-ucvzyl, claude/sophia-agi-challenges-j4sj7a,
claude/sophia-agi-feasibility-eval-kzei9d, claude/sophia-agi-handover-merge-kkvjji (#133),
claude/sophia-agi-hurdles-10iqob, claude/sophia-consolidation-inventory-vj5wfm,
claude/sophia-math-code-curriculum, claude/sophia-team-orchestrator, claude/sophia-training-next-steps-ij1kvm,
claude/sophia-v3-gate-analysis-w2lyt4, claude/supercomputing-cluster-job-fdbkks, claude/team-agents-mode,
claude/tool-use-training-impl, claude/tool-use-training-plan, claude/verifier-as-reward-foundation (#129),
feat/activation-steering-pif, feat/capability-retention-mcp, feat/personality-council-pif, feat/roadmap-to-10
```

## Unmerged branches — report only (no deletion, per request)

### Open PRs
| PR | Branch | Note |
|----|--------|------|
| #131 | `claude/agi-repo-optimization-5kpqbc` | **Absorbed into this PR** — its net-new work (hybrid do-no-harm guard, retrieval/curriculum fixes, CI-timeout bump) fixes the 3 tests the hang was masking. Close once this lands. |
| #135 | `feat/two-paths-to-novelty` | DreamerV3 world model + Lean proof search — left open |
| #136 | `claude/repo-ip-protection-qvsxyv` | docs-only: wire Zenodo DOI — left open |
| #137 | `claude/runpod-pod-stall-8rcxvz` | RunPod connection + stop EXITED pod leak — left open |

### Unmerged, NO open PR (~38) — candidates for PR-or-prune
Recent & substantive (review for a PR):
- `feat/governed-sparse-quant` (+9), `claude/dgx-spark-integration` (+7),
  `claude/anthropic-skills-gap-research-ni1zl2` (+6), `claude/ai-search-engine-repo-eup3y9` (+6),
  `claude/spark-moe-workflow` (+4), `claude/agentark-sophia-integration-v9obrf` (+3, SophiaArk Evolve),
  `claude/llm-framework-job-alignment-d0etrm` (+1, governed-scaling), `claude/multimodal-model-roadmap-3fd0yt` (+1).

Likely-stale fixes that `main` may already have addressed (verify, then prune):
- `claude/fix-main-test-failures`, `claude/fix-rag-index-drift`,
  `claude/fix-cluster-package-collision`, `claude/agi-gap-audit-roadmap-12ft30`.

Older long-lived branches (Jun 21–22), large divergence — likely superseded by the
provenance/legal work already in `main`; confirm before pruning:
- `feat/rlvr-and-local-agent-delta` (+129), `claude/reconcile-framing` (+121),
  `claude/council-distillation-spec` (+117), `claude/mac-icloud-git-setup` (+114),
  `claude/council-small-llm` (+112), `claude/legal-faithfulness-hard-bench` (+110),
  `feat/classification-lattice` (+100), `claude/openrouter-judge-support` (+99),
  `claude/sector-gate-and-cantonese` (+96), `claude/legal-faithfulness-gated-bench` (+93),
  `feat/training-safety` (+92), `feat/dataflow-interpreter` (+84),
  `claude/legal-semantic-faithfulness` (+82), `feat/security-firewall` (+67),
  `feat/provenance-delta-benchmark` (+63), `feat/personality-measurement-gate` (+11).

Other roadmap/infra branches (mostly docs/+1–4 commits):
- `claude/repo-agi-research-alignment-onptqa`, `claude/pretraining-data-engineering-8l1z3u`,
  `claude/distributed-storage-repo-dev-w0brhm`, `claude/distributed-storage-repo-dev-c4cvmz`,
  `claude/deepseek-pretraining-alignment-o281ju`, `claude/sophia-wisdom-4b-roadmap-jyesip`,
  `claude/hpc-operator-compiler-roadmap-zumu83`, `claude/ai-cluster-infra-roadmap-mk4yvn`,
  `claude/agent-harness-coevolution`, `claude/agent-harness-vision-6guo28`,
  `claude/follower-repo-brainstorm-5h8ocf`, `claude/sophia-agi-aipp-integration-57x7ip`.

## Suggested next steps
1. Run `scripts/delete-merged-branches.sh` to clear the 36 merged branches
   (blocked from the CI environment by egress policy — run where you have push rights).
2. ~~Fix the `provenance_faithful` regex hang~~ — **done in this branch** (see above).
3. Land or close #131 (it carries real unmerged work, not just a dupe of #133).
4. Decide PR-or-prune on the older long-lived branches above.
