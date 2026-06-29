# Session Handover — 2026-06-29 (Data Analysis Agent + data-management instruments)

> Continuation point for the next session/device. This session turned "I'm missing a
> Data Analysis Agent — research how data is best used and give a feasible plan to
> benchmark + improve data management" into committed, deterministic, CI-gated
> machinery. `canClaimAGI` stays **false**; nothing here promotes a result. The DHI is
> an operational/illustrative metric only (never in `published-results.json`).

## 0. Branch / where things are
- **Feature branch `claude/data-analysis-agent-strategy-gxoekd`** (tip `58dd364`): all
  work below, pushed, all offline gates green, working tree clean.
- **Not merged to `main`** and **no PR opened** (user has not asked). Open one when ready.
- 5 commits: `5950bb5` strategy doc → `68bd67f` Phases 0–3+5 machinery →
  `2e0a469` Phase 4 + staged split → `58dd364` scoped gate + adoption runbook.
  (`ed24b1b` was the pre-session base.)
- Local `main` is the usual stale container lineage; `origin/main` is source of truth.

## 1. What shipped (all committed on the branch)

**Docs**
- `docs/11-Platform/Data-Analysis-Agent-Strategy.md` — thesis (data as governed
  evidence; data quality is the capability lever), ranked ledger-grounded limitations,
  the agent design, the DHI spec, the phased plan, and an implementation-status table.
- `docs/11-Platform/Entity-Disjoint-Split-Adoption.md` — the human-review runbook to
  adopt the staged clean split and flip the gate (see §4 next-steps).

**Instruments (deterministic, offline, fail-closed; each has a test)**
- **DHI scorecard** — `tools/data_health_report.py` → `agi-proof/data-health/report.json`
  (`--check` drift gate). **Baseline DHI = 0.6507.** 7 dims: coverage 0.314,
  mixBalance 0.259, decontamStrength 1.0, dedupHealth 0.85, provenanceCompleteness 0.5,
  lineage 1.0, reproducibility 0.889. Weighted mean, weights declared in the file.
- **Data asset registry** — `tools/build_data_registry.py` →
  `agi-proof/data-health/registry.json` (`--check`). 15 assets, each manifest-sha256-anchored.
- **Entity-level decontam** — `tools/assert_entity_decontam.py`. Surfaces the SEIB leak
  shingles miss: 40 shared entities, **151 eval prompts fully entity-covered by train**.
  Diagnostic by default; `--fail-covered N` / `--fail-shared N` gate; **`--eval-file`**
  scopes the audit to one JSONL (the adoption gate).
- **Entity-disjoint split carver** — `tools/carve_entity_disjoint_split.py`. Dry-run by
  default; staged a **75-case candidate** at
  `agi-proof/data-health/seib_entity_disjoint_candidate/` (proof `sharedWithTrain=[]`,
  freshness-tested).
- **Mix-balance ratchet gate** — `tools/assert_mix_balance.py` +
  `agi-proof/data-health/mix-baseline.json` (in CI). Pins today's skew (**L1=0.7406**,
  worst = settled_fact); fails any PR that worsens overall or per-family
  distance-from-target beyond tolerance. `--update` re-baselines after an improvement.

**The agent**
- `agent/data_analyst.py` — `DataAnalyst`: `assess()` (DHI + entity audit),
  `curation_plan()` (priority-ranked, propose-only), `report()`. Fail-closed (refuses on
  missing tool/manifest). Run it: `python -m agent.data_analyst` (today's top priority =
  `entityDisjointSplit`).
- Registered as the least-privilege **`data`** team in `agent/swarm_router.py` (schema
  enum extended); a data/corpus/decontamination task now fans out to it.

**CI** — new step *"Data health + registry drift gates"* in `.github/workflows/ci.yml`:
runs `build_data_registry --check`, `data_health_report --check`, `assert_mix_balance`,
and the 6 data-agent test files.

**Failure ledger** — 5 honest OPEN/Partial items added (`data-health-index-below-target`,
`data-lineage-graph-partial`, `entity-decontam-candidate-staged-not-gated`,
`mix-balance-gate-ratchet-not-at-target`, `data-analyst-flywheel-not-run`).

## 2. How to verify (all green at handover)
```bash
python tools/lint_claims.py
python tools/build_data_registry.py --check
python tools/data_health_report.py --check
python tools/assert_mix_balance.py
python tools/validate_failure_ledger.py --check
python -m pytest -q tests/test_data_health_report.py tests/test_build_data_registry.py \
  tests/test_assert_entity_decontam.py tests/test_carve_entity_disjoint_split.py \
  tests/test_assert_mix_balance.py tests/test_data_analyst.py tests/test_swarm_router.py
```

## 3. Design decisions / gotchas (don't relearn these)
- **The mlx training files are not committed** (`training/local_sophia_v3/mlx/*` absent);
  `manifest.json` is the source of truth for build stats, and `training/corpus.jsonl`
  (528 rows, messages+metadata) is the committed corpus. The DHI reads those.
- **`assert_decontam.py` runs on absent train globs** (mlx files) → trivially passes
  today. Entity-decontam uses the committed `corpus.jsonl` + `moral_gate_sft.jsonl`, so
  it actually measures something.
- **`agi-proof/data-health/**` is swept by neither eval/train globs nor the registry/DHI
  globs** (`data/*`, `training/*`) — that's why staging the candidate there perturbs
  nothing. Promoting it *into* `data/` (the runbook) WILL change the registry/DHI
  denominators → regenerate them as part of adoption.
- **`build_agi_proof_package.py` runs without `--check` in CI** (regenerate-only) and the
  committed `evidence-manifest.json` is allowed to lag (it was already stale 58 vs 85
  before this session) — so adding ledger rows does NOT break CI. Left it untouched.
- **Mix gate is a ratchet, not an absolute gate** — an absolute gate would red-CI now.
- **DHI determinism**: stdlib only, sorted keys, floats rounded 4dp, no timestamps — the
  `--check` drift gate depends on this. Same for the registry.

## 4. ▶ Next steps (human-judgment + data + GPU — NOT pure code)
Ordered by leverage. Each is in the strategy doc's phase plan + the failure ledger.

1. **Adopt the staged clean split** — follow
   `docs/11-Platform/Entity-Disjoint-Split-Adoption.md`: review all 75 cases → seal under
   `data/seib_entity_disjoint/` → register in `dataset_guard.EVAL_GLOBS` → add
   `assert_entity_decontam --eval-file data/seib_entity_disjoint/heldout_v1.jsonl
   --fail-covered 0` to CI → regenerate registry/DHI → close
   `entity-decontam-candidate-staged-not-gated`, advance `seib-generalization-split-not-validated`.
   *Verify-ready now:* `python tools/assert_entity_decontam.py --eval-file
   agi-proof/data-health/seib_entity_disjoint_candidate/candidate.jsonl --fail-covered 0` → exit 0.
2. **Curate toward targets (Phase 4)** — lift coverage (records 210→500, rows 1560→10k)
   and mix (hk_bilingual/moral_gate/tool_mcp <1%→~10%, ZH ~10% → ~90% EN today) via the
   gated builders + `tools/spark_data_refinery.py`; track records (coverage) separately
   from rows (volume); `--update` the mix baseline only after a real improvement. Watch
   the DHI rise. Holds `abstention-is-reward-positive`.
3. **Lineage edges (Phase 1 extension)** — extend the registry from manifest-level to a
   real source→shard→checkpoint→eval graph; pin eval runs to a corpus/registry version.
4. **Run the continual flywheel (Phase 6)** — schedule the DataAnalyst
   audit→plan→human-approved-gated-build→re-measure loop; re-run decontam + lint + DHI
   `--check` each cycle.

## 5. One rule above all (unchanged)
No gate relaxation. If a target (DHI, mix, volume) is hit, **upgrade the wording /
promote via the existing gate** — never loosen a check. The DHI stays operational-only;
an entity-disjoint split removes one confound, it is **not** a frontier/AGI claim and is
still maintainer-authored (third-party hidden eval still gates `canClaimAGI`).
