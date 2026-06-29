# Session Handover — 2026-06-29 (sentinel fix → split ADOPTED + fact-check CI + DataAnalyst)

> Continues `SESSION-HANDOVER-2026-06-29-data-analysis-agent.md` (read that first for the
> full Data-Analysis-Agent picture). This session reviewed the staged entity-disjoint
> split, fixed a real defect, then — with human approval — **adopted the split** (sealed +
> CI-gated), added an on-demand Google Fact Check CI workflow, and made the DataAnalyst
> aware of the adoption. `canClaimAGI` stays **false**; nothing here promotes a published
> result (the DHI is operational-only; the split is maintainer-authored, not third-party).

## 0. Git / repo state
- **Branch `claude/data-analysis-agent-strategy-gxoekd-ghg5ad`** (tip `79b80a42`), pushed,
  working tree clean. Carries the 5 prior data-analysis commits + this session's 5 commits.
- **STALE vs origin/main: ~7 ahead / 20+ behind** (older `…-gxoekd` base; `main` moved).
  Rebase onto fresh `origin/main` before any PR/merge — `failure-ledger.md`, `ci.yml`,
  `dataset_guard.py` are contended files and will need conflict reconciliation. Not merged;
  no PR (none requested).
- Commits this session (newest first): `b8a617dd` registry lineage edges + version anchor ·
  `3a771740` handover/changelog · `79b80a42` DataAnalyst adoption-aware ·
  `1ca5f0d5` fact-check CI workflow · `fa4c7449` split ADOPTED (sealed+gated) ·
  `b8b183d2` handover/changelog · `278518bb` sentinel fix (carve 75→73).

## 1. What this session did
1. **Sentinel fix (`278518bb`)** — content-review of the 75-case candidate found 2
   non-attribution finance probes ("burn multiple"); root cause was `attributedAuthor:
   "multiple"` (collective-authorship sentinel) admitted as a named entity. Fixed via a
   `_SENTINELS` stoplist in `build_entity_vocab` (vocab 201→200). Carve → **73 cases**,
   all attribution traps, gold derivable for all, still entity-disjoint.
2. **Split ADOPTED (`fa4c7449`)** — human-approved. Sealed at
   `data/seib_entity_disjoint/heldout_v1.jsonl` + `manifest.json`
   (`sealed:true, candidateOnly:false, humanReviewed:true`); registered in
   `dataset_guard.EVAL_GLOBS`; **`assert_entity_decontam --eval-file
   data/seib_entity_disjoint/heldout_v1.jsonl --fail-covered 0` is now a hard CI gate**
   (`.github/workflows/ci.yml`). Registry 15→16 assets; **DHI 0.6507→0.6518**. Ledger
   `entity-decontam-candidate-staged-not-gated-2026-06-29` → **Closed**.
3. **Google Fact Check CI workflow (`1ca5f0d5`)** — `.github/workflows/google-factcheck-coverage.yml`,
   a `workflow_dispatch` that injects the `GOOGLE_FACTCHECK_API_KEY` secret and runs the live
   coverage probe (uploads artifact, never commits, key never printed). The integration was
   already present + verified live this session (general 6/6, provenance 0/6 — the honest
   domain gap). **ACTION FOR HUMAN:** add the repo secret `GOOGLE_FACTCHECK_API_KEY` (and
   **rotate** the key that was pasted in chat) to use it from CI.
4. **DataAnalyst adoption-aware (`79b80a42`)** — it no longer proposes "carve a split"
   after adoption; it demotes that action and re-points to the residual (a third-party
   pack). Top priority now correctly = `coverage` (0.314).

## 2. Verify (all green this session)
```bash
make claim-check                                   # M3-pilot GO, M3-transfer GO, lint/decontam OK
python tools/build_data_registry.py --check && python tools/data_health_report.py --check && \
python tools/assert_mix_balance.py && python tools/validate_failure_ledger.py --check && \
python tools/assert_decontam.py && \
python tools/assert_entity_decontam.py --eval-file data/seib_entity_disjoint/heldout_v1.jsonl --fail-covered 0
python -m pytest -q tests/test_data_health_report.py tests/test_build_data_registry.py \
  tests/test_assert_entity_decontam.py tests/test_carve_entity_disjoint_split.py \
  tests/test_assert_mix_balance.py tests/test_data_analyst.py tests/test_swarm_router.py   # 54 passed
```

## 3. ▶ Next steps (by leverage; all human-judgment + data + GPU)
1. **Curate toward DHI/coverage targets** — now the DataAnalyst's top priority: records
   210→500, rows 1560→10k, fix the mix (hk_bilingual/moral_gate/tool_mcp <1%→~10%, ZH share)
   via the gated builders + `tools/spark_data_refinery.py`. Track records vs rows; `--update`
   the mix baseline only after a real improvement; watch DHI rise.
2. **Third-party entity-disjoint pack** — the only path that advances an external
   generalization claim (`hidden-review-third-party-not-run`). The adopted split is
   maintainer-authored; `canClaimAGI` stays false until a third-party hidden eval is beaten.
3. **Registry lineage edges** (`data-lineage-graph-partial`) — *first increment landed*
   (`b8a617dd`): the registry now has a `lineage` block (9 declared-only edges +
   `registryVersion` anchor, 0.375 upstream coverage). Remaining: backfill upstream
   declarations in the other 10 manifests (esp. eval surfaces → corpus version + checkpoint),
   and have eval reports stamp the `registryVersion` they ran against.
4. **Rebase onto `origin/main`** before opening any PR (see §0 — contended files).

## 4. Read-first
1. `SESSION-HANDOVER-2026-06-29-data-analysis-agent.md` (base picture)
2. `docs/11-Platform/Entity-Disjoint-Split-Adoption.md` (now marked ✅ ADOPTED)
3. `agi-proof/failure-ledger.md` → entity-decontam item (Closed), data-health/lineage rows

## 5. Don't-break
Gates green this session: `lint_claims`, `assert_decontam`, the new entity `--fail-covered 0`
gate on the sealed split, `build_data_registry --check`, `data_health_report --check`,
`assert_mix_balance`, `validate_failure_ledger --check`, `make claim-check` (both gates GO),
and the data-agent test files (54 passed). No gate relaxation — the adoption STRENGTHENS the
gate. DHI stays operational-only; `canClaimAGI` false.
