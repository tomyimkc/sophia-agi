# Session Handover — 2026-06-29 (entity-vocab sentinel fix + split content-review)

> Continues `SESSION-HANDOVER-2026-06-29-data-analysis-agent.md` (read that first for the
> full Data-Analysis-Agent picture). This session executed the **agent's half of Next-Step
> #1** — the content review of the staged entity-disjoint split — found and fixed a real
> defect, and left adoption (a human curation step) **pending a decision**. `canClaimAGI`
> stays **false**; nothing here promotes a result.

## 0. Git / repo state
- **Branch `claude/data-analysis-agent-strategy-gxoekd-ghg5ad`** (tip `278518bb`), pushed,
  working tree clean. It carries the 5 prior data-analysis commits (from `…-gxoekd`,
  `b85b9f5`) **plus** this session's fix on top.
- Not merged to `main`; **no PR** (none requested).
- Note: the prior branch `…-gxoekd` (`b85b9f5`) is now the *base*; my branch is the live one.

## 1. What this session did (commit `278518bb`)
A `DataAnalyst` content-review pass over the **75-case** staged entity-disjoint candidate
found 2 cases that were **not authorship/attribution probes**: the HK$ SaaS *"burn
multiple"* finance prompts. Root cause: `data/attributions.json` uses
`attributedAuthor:"multiple"` as a **collective-authorship sentinel** (I Ching, Book of
Songs); `build_entity_vocab` admitted `"multiple"` as a named entity, which matched the
literal word "burn **multiple**" and pulled finance prompts into the carve.

**Fix (at source):** a `_SENTINELS` stoplist in `tools/assert_entity_decontam.py` so
authorship-*status* placeholders (multiple, unknown, anonymous, …) are never treated as
entities. This makes the recognizer **more precise** (removes false positives) — it does
**not** weaken any decontam gate (no real entity is named "multiple"). vocab 201→200.
- Re-staged candidate: **73 cases**, all well-formed attribution traps, still
  entity-disjoint (`sharedWithTrain=[]`, `--fail-covered 0` exit 0).
- **Gold verified derivable for all 73** (72 directly in
  `provenance_bench/data/wikidata_snapshot.json` / `misattributions.json`; the
  newton/mahabharata case is a derivable *negative*).
- Regression test added (`tests/test_assert_entity_decontam.py::test_authorship_status_sentinels_are_not_entities`).
- Docs + ledger updated: strategy doc, adoption runbook (agent-review log added),
  `failure-ledger.md` (75→73 + the fix).

## 2. Verify (all green this session)
```bash
python tools/lint_claims.py && python tools/assert_decontam.py && \
python tools/build_data_registry.py --check && python tools/data_health_report.py --check && \
python tools/assert_mix_balance.py && python tools/validate_failure_ledger.py --check && \
make claim-check   # M3-pilot GO, M3-transfer GO
python -m pytest -q tests/test_data_health_report.py tests/test_build_data_registry.py \
  tests/test_assert_entity_decontam.py tests/test_carve_entity_disjoint_split.py \
  tests/test_assert_mix_balance.py tests/test_data_analyst.py tests/test_swarm_router.py
# 53 passed
```

## 3. ▶ Next step — the pending DECISION (human-owned curation)
The split is now clean, agent-reviewed and gold-verified, but **adoption was NOT done** —
sealing a held-out eval surface + flipping a CI gate is curation the contract reserves for
humans, and no affirmative approval was obtained this session (the interactive prompt could
not be delivered). **Default left in place: hold as candidate.**

To adopt when a human approves the 73-case content, run
`docs/11-Platform/Entity-Disjoint-Split-Adoption.md` steps 3–7:
seal under `data/seib_entity_disjoint/` → add glob to `dataset_guard.EVAL_GLOBS` →
`assert_entity_decontam --eval-file data/seib_entity_disjoint/heldout_v1.jsonl --fail-covered 0`
in CI → regenerate registry/DHI → close `entity-decontam-candidate-staged-not-gated`.
Verify-ready now:
```bash
python tools/assert_entity_decontam.py \
  --eval-file agi-proof/data-health/seib_entity_disjoint_candidate/candidate.jsonl \
  --fail-covered 0   # exit 0
```
Alternatives if not adopting: Phase 4 curation toward DHI/mix targets, or registry lineage
edges (`data-lineage-graph-partial`) — both in the prior handover §4.

## 4. Read-first
1. `SESSION-HANDOVER-2026-06-29-data-analysis-agent.md` (the base picture)
2. `docs/11-Platform/Entity-Disjoint-Split-Adoption.md` (adoption runbook + agent-review log §2)
3. `agi-proof/failure-ledger.md` → `entity-decontam-candidate-staged-not-gated-2026-06-29`

## 5. Don't-break
The drift/measurement gates that stayed green: `lint_claims`, `assert_decontam`,
`build_data_registry --check`, `data_health_report --check`, `assert_mix_balance`,
`validate_failure_ledger --check`, the 7 data-agent test files, and `make claim-check`
(both claim gates GO). No gate relaxation; the DHI stays operational-only; `canClaimAGI` false.
