# Data Analysis Agent — strategy, limitations, and a benchmark-first improvement plan

> **Status: strategy / proposal (not a measured result).** Every number below is a
> *target* or *proposed threshold*, never a validated claim. Nothing here changes the
> no-overclaim gate. The deliverable of this doc is a *plan and an instrument*, in
> Sophia's idiom: deterministic, offline-testable, fail-closed, auditable.
>
> **Companions:** [`Spark-Data-Refinery.md`](./Spark-Data-Refinery.md) (the gate-filtered
> generation engine), [`Local-Sophia-Training.md`](./Local-Sophia-Training.md),
> [`DGX-Spark-Maximization.md`](./DGX-Spark-Maximization.md) §3.1,
> [`../../agi-proof/measurement-thesis.md`](../../agi-proof/measurement-thesis.md) (the IEC),
> [`../../agi-proof/failure-ledger.md`](../../agi-proof/failure-ledger.md).

---

## 0. TL;DR

The repo is **mature in data *tooling* and data *governance philosophy*** — a
crawl→quality→dedup→shard pipeline, per-row passports, deterministic manifests, a
fail-closed decontamination gate, a gate-filtered synthesis refinery, and an
8-pillar measurement contract. What is missing is a **single agent that closes the
data loop**: one role that continuously *audits corpus health, scores data
management maturity, decides what to curate next, and feeds those decisions back into
the pipeline and the failure ledger* — instead of the current state where these
decisions are scattered across ~30 tools and made ad-hoc by humans.

This is also where the **highest capability ROI** is. Our own
[`Training-Efficiency-Feasibility.md`](./Training-Efficiency-Feasibility.md) concludes
the lever toward stronger models is **data quality, not training speed**. Three facts
make the case sharp:

1. **The corpus is the binding constraint, not compute.** Gate-passed training rows
   are on the order of ~1.5k, built from ~72 structured records via ~7.4×
   templating; the M2 target is ≥10k (failure-ledger
   `sophia-wisdom-4b-m2-volume-below-target-2026-06-25`). More GPUs do not fix this.
2. **The "held-out" split is contaminated at the source.** Contested entities in the
   SEIB-100 eval already appear in the foundational corpus
   (`seib-generalization-split-not-validated-2026-06-23`;
   `third-party-heldout-pack-empty-2026-06-26`). Every "held-out generalization"
   number is therefore suspect until a clean, entity-disjoint split exists.
3. **The mix is severely skewed.** `settled_fact` is over-represented while
   `hk_bilingual`, `moral_gate`, and `tool_mcp` sit <1% against ~10% targets; the
   corpus is ~90% English against a 10% Chinese goal.

A Data Analysis Agent that owns these three numbers — and is graded by the same
measurement contract everything else here is — is the single most leveraged thing the
repo is missing on the data axis.

---

## 1. What already exists (don't rebuild this)

The agent must *orchestrate* the existing surface, not duplicate it.

**Pipeline (`pipeline/`)** — `document.py` (fail-closed schema validation),
`quality_score.py` (provenance-aware scoring: poison-gate pooled confidence +
`authorConfidence` priors + 5 content heuristics), `dedup/minhash.py` (deterministic
blake2b MinHash+LSH) and `dedup/vector.py` (semantic near-dup via the local hash
embedder), `corpus_table.py` (DuckDB-or-stdlib columnar stats with identical output
shape), `quality_regression.py` (fail-closed regression gate), `manifest.py`
(deterministic sha256 shard manifests), `shard_writer.py` + `store/` (object/KV/queue).

**Per-row lineage primitive** — `pretraining/data_passport/passport.py` already
computes `content_hash`, `source`, `license`, `quality_score`, `minhash`,
`dedup_cluster` per row, and `build_passport.py` emits a datasheet summary. This is
the lineage atom; it just isn't wired end-to-end.

**Synthesis + curation tooling (`tools/`)** — `spark_data_refinery.py` (teacher
proposes, intrinsic gate disposes), `build_local_sophia_dataset.py`,
`build_discipline_sft.py`, `build_moral_gate_sft.py`, `build_distill_dpo_pairs.py`
(anti-circular: rejected must trip the gate), `lint_training_rows.py` (habit-not-fact),
`assert_decontam.py` (+ per-virtue variants), `corpus_stats.py` / `pipeline_stats.py`
/ `score_corpus.py`, `mine_hard_negatives.py`, `build_rag_index.py` (sha256-verifiable),
`heldout_seal_guard.py`.

**Governance substrate** — `provenance_bench/dataset_guard.py` (train/eval
disjointness, sealed-benchmark manifests), evidence cards
(`training/context_packing/evidence_card.schema.json`), the IEC
(`agi-proof/measurement-thesis.md`) and its gates (`claim_gate.py`, `eval_stats.py`).

**Existing agent fabric to slot into** — `agent/swarm_router.py` (today routes 6
teams: search, research, math_verify, legal, ontology, redteam — *no data team*),
`agent/subagent.py` (least-privilege, bounded-budget, fail-closed delegation),
`agent/long_horizon.py` (durable task ledger), `agent/corpus_scrub.py`, `agent/dedup.py`.

> **Honest gaps in the as-is** (from the two surface audits): no end-to-end lineage
> (source→shard→checkpoint→eval), no corpus *version* stamps, the row-passport and
> shard-manifest systems are not unified, dedup/quality thresholds are implicit (not
> recorded in manifests), decontam is shingle/exact only (misses entity-level
> contamination), there is no "all corpora" registry, and decontam runs as a CI
> *gate* but not as part of the *build* (fragile when a new benchmark is added).

---

## 2. Thesis: how data should be utilized, in Sophia's idiom

Sophia's contribution is the *trust layer*, not out-training frontier labs
(`VISION.md`). The data thesis follows directly: **data is not fuel to be maximized;
it is evidence to be governed.** Every datum should carry the same discipline the
gate imposes on every claim — provenance, a verifiable quality verdict, and a
fail-closed default. Concretely, data should be utilized along two axes the operator
named.

### Axis A — Data for code development (data as an engineering signal)

The repo *is* a data product (corpora, benchmarks, RAG index, eval splits, the wiki).
Treat data assets with the same rigor as code:

1. **Lineage as a first-class graph.** Unify `data_passport` (row) + `manifest` (shard)
   into one DAG: *source document → shard → training pack → checkpoint → eval result*.
   This makes "which data produced this number" answerable — the same provenance
   promise the gate makes for claims, applied to the repo's own data. It also lets the
   failure ledger point at *specific* contaminated rows, not "the SEIB split".
2. **A data asset registry.** One queryable catalog ("what corpora exist, their size,
   mix, quality, contamination status, version, build commit") so humans and agents
   stop doing git archaeology. This is the missing index over the existing manifests.
3. **Regression gates everywhere data changes.** `quality_regression.py` exists; extend
   the same fail-closed pattern to *mix balance*, *decontam strength*, and *coverage*
   so a PR that skews the corpus or leaks an entity fails CI like a broken test.
4. **Data observability.** Feed the corpus scorecard (§5) into `observability/` so
   corpus health is a dashboard, not a one-off script — and so drift is *noticed*.
5. **Reproducibility anchors.** The RAG index already sha256-verifies; make every data
   artifact `--verify`-able and pin every eval run to a corpus version.
6. **The agent's own dogfooding.** Error/decision memory (`agent/error_rag.py`,
   `agent/memory.py`) is data; the same lineage/quality discipline should govern the
   agent's continual-learning store so self-improvement is auditable, not a black box.

### Axis B — Data for LLM training (corpus as the capability lever)

1. **Break the corpus bottleneck without breaking decontam.** The constraint is ~72
   structured records, not teacher availability. Two safe expansion paths:
   (a) *structured-record growth* — expand philosophy/history/religion/psychology
   ground-truth records (the genuinely manual, high-value work an agent can *triage and
   draft* but a human must verify); (b) *gate-filtered synthesis* via
   `spark_data_refinery.py` — teacher proposes, the **intrinsic** gate (no-question
   path: 0/439 false drops vs 88/564 for the trap grader) disposes. Synthesis raises
   *quality per prompt*; only new records raise *count* — the agent must track both and
   never let synthetic volume masquerade as coverage.
2. **Mix as a controlled variable.** `pretraining/data_mixing/run_mixing.py` already
   demonstrates ratio sweeps; promote mix from an accident of what got built to a
   *target with a tolerance*, gated in CI. Fix the hk_bilingual / moral_gate / tool_mcp
   / Chinese-language shortfalls explicitly.
3. **Decontamination that catches *entities*, not just shingles.** Add an entity/concept
   layer (resolve against the OKF wiki / Wikidata snapshot) so "Socrates in train with
   attribution X, in eval with attribution Y" is caught — the exact hole behind the
   contaminated SEIB split.
4. **A genuinely held-out, entity-disjoint split** + a commissioned third-party pack
   (closes `third-party-heldout-pack-empty`, `hidden-review-third-party-not-run`).
   This is the prerequisite for *any* honest generalization claim.
5. **Curriculum & hard-negative mining as a loop.** `mine_hard_negatives.py` and
   `build_distill_dpo_pairs.py` exist; the agent should run them continuously against
   *current model errors* (RFT/GRPO data), holding the P2→P3 invariant: **abstention is
   reward-positive** — never train out the fail-closed habit.
6. **Every training row keeps teaching a habit, not a fact** (`lint_training_rows.py`),
   and every row carries a passport. Quality of *signal*, not quantity of *tokens*, is
   the metric.

---

## 3. Current limitations on the data aspect (ranked, grounded)

Each item cites the failure-ledger entry or audit finding it comes from, with the
claim it blocks.

| # | Limitation | Evidence | Blocks |
|---|---|---|---|
| 1 | **Eval split contaminated at source** — SEIB-100 contested entities already in the foundational corpus; exact/shingle decontam passes but the split isn't truly held-out. | `seib-generalization-split-not-validated-2026-06-23` | Any "held-out generalization" claim. |
| 2 | **No independent third-party held-out pack** — `caseCount: 0` by design; synthetic packs share authorship with training. | `third-party-heldout-pack-empty-2026-06-26`, `hidden-review-third-party-not-run` | `canClaimAGI`; external validity. |
| 3 | **Corpus size is the binding constraint** — ~1.5k rows from ~72 records via templating; ≥10k target unmet; mix skewed (source_discipline over, hk_bilingual/moral_gate/tool_mcp <1%). | `sophia-wisdom-4b-m2-volume-below-target-2026-06-25` | Sample size / power on most recipes. |
| 4 | **Decontam is lexical only** — exact + word-shingle (Jaccard ≥0.9); misses entity/concept-level contamination and paraphrase below threshold. | `assert_decontam.py` audit | Strength of the disjointness guarantee. |
| 5 | **No end-to-end lineage or corpus versioning** — row passports and shard manifests aren't unified; no source→checkpoint→eval graph; eval runs not pinned to a corpus version. | Pipeline audit | Reproducibility; "which data produced this number". |
| 6 | **No data asset registry / no curator role** — ~30 tools, decisions ad-hoc; swarm_router has no data team; decontam runs as a CI gate but not in the build. | Both audits | Operational data governance at scale. |
| 7 | **Bilingual/domain skew** — ~90% EN vs 10% target; specialist domains starved. | `local_sophia_v3/manifest.json` | Generalization beyond English source-discipline. |
| 8 | **Implicit thresholds** — dedup (Jaccard 0.8 / cosine 0.9) and quality keep-threshold (0.5) not recorded in manifests; re-runs can silently differ. | Pipeline audit | Reproducibility of dedup/quality decisions. |

---

## 4. The Data Analysis Agent — design

A new agent that **fits the existing fabric**, deterministic and offline-testable like
everything else here. It does not invent a parallel stack; it is the *conductor* over
§1's tools and the *author* of §5's scorecard.

**Placement.** Register a 7th team in `agent/swarm_router.py` — `"data"` — backed by a
new `agent/data_analyst.py`, delegated through `agent/subagent.py` (least-privilege:
read corpus/manifests, run analysis tools, **propose** curation; write only via the
existing fail-closed builders + gate). Long-running audits ride
`agent/long_horizon.py`'s durable ledger.

**Fail-closed contract (load-bearing).** The agent **never** silently mutates data.
It (a) *reports* (read-only scorecards/diffs), (b) *proposes* (a curation plan + diff),
(c) *acts only through gated builders* (`spark_data_refinery.py`,
`build_*`, with `assert_decontam.py` + `lint_training_rows.py` re-run as belt-and-braces).
If any analysis tool fails to import or a manifest is missing, it refuses to emit a
verdict — same posture as the refinery and the gate.

**Capabilities (each maps to an existing tool or a thin new one):**

1. **Corpus health audit** → runs `corpus_table.py` / `pipeline_stats.py` /
   `score_corpus.py`, computes the §5 scorecard, diffs against the committed baseline.
2. **Contamination audit** → runs `assert_decontam.py` + the proposed entity-level
   check; produces a *row-level* contamination report (which rows, which entities) for
   the failure ledger — turning item #1/#4 from prose into a list.
3. **Mix & coverage analysis** → measures domain/language mix vs targets; flags the
   shortfalls in items #3/#7; proposes the next batch to build to close the gap.
4. **Curation planning** → given current model errors (from eval + `error_rag`), drafts
   a prioritized curation plan: which records to expand, which hard-negatives to mine,
   which synthesis seeds to feed the refinery. Human approves; agent executes via gated
   builders only.
5. **Lineage maintenance** → keeps the unified passport↔manifest↔checkpoint↔eval graph
   (§Axis A) current; answers "which data produced result R".
6. **Ledger integration** → opens/updates failure-ledger items with measured numbers
   (e.g. attaches the real contamination row-count to item #1) so the ledger reflects
   data reality automatically.

**What it deliberately does NOT do:** generate training signal unfiltered, relax any
gate, author results (Leiden: humans author results), or treat synthetic volume as
coverage.

---

## 5. The instrument: a Data Management Maturity scorecard (benchmark your process)

The operator asked to *benchmark* the data-management process. In this repo, the way
to benchmark anything is to build a deterministic instrument — so define a **Data
Health Index (DHI)**: a single offline-computed scorecard, version-stamped, gated in
CI the way `quality_regression.py` is. This is the agent's headline artifact and the
literal answer to "how do I benchmark my data management process."

Proposed tool: `tools/data_health_report.py` → writes
`agi-proof/data-health/report.json` (+ a `--check` mode for CI drift, mirroring
`build_results_page.py --check`). Seven dimensions, each scored 0–1, deterministic:

| Dim | What it measures (deterministic) | Proposed target |
|---|---|---|
| **Coverage** | unique structured records; rows; rows/record (templating-inflation guard) | records ↑ toward 500; rows/record ≤ 8.0 |
| **Mix balance** | L1 distance of domain+language mix from target vector | ≤ 0.15 |
| **Decontam strength** | exact + shingle + **entity-level** disjointness; row-level leak count | 0 leaks; entity check present |
| **Dedup health** | duplicate-rate from `corpus_table`; thresholds recorded in manifest | dup-rate ≤ baseline; thresholds pinned |
| **Provenance completeness** | % rows with full passport (source+license+quality+hash) | ≥ 0.99 |
| **Lineage** | % artifacts reachable in the source→checkpoint→eval graph | ≥ 0.95 |
| **Reproducibility** | % data artifacts that pass a `--verify` sha256 rebuild | 1.0 |

DHI = a transparent weighted mean (weights declared in the report; no hidden
knobs). **Crucially, the DHI is *illustrative/operational*, not a no-overclaim
result** — it is an internal management metric, labelled as such, and never promoted
to `published-results.json`. It exists to make data-process improvement *measurable
and regression-gated*, which is exactly the IEC thesis applied to the repo's own data.

---

## 6. Feasible, phased plan (each phase = one PR, CI-green, ledger-linked)

Ordered by ROI and dependency. Every phase ships an offline test and updates the
failure ledger; none relaxes a gate. Sizing is rough.

**Phase 0 — Instrument first (the benchmark).** *~1 PR.*
Build `tools/data_health_report.py` + the DHI scorecard (§5) over the *current* corpus,
commit the baseline `report.json`, add a `--check` CI drift gate. *Deliverable:* a
number for today's data health, so every later phase is measured against it. *Risk:*
low (read-only stats over existing tools).

**Phase 1 — Unify lineage.** *~1–2 PRs.*
Wire `data_passport` → `manifest` → a `data_asset_registry.json` (the missing catalog,
item #5/#6). Pin dedup/quality thresholds into manifests (item #8). *Pass bar:* every
shipped corpus resolves to a registry entry; `--verify` round-trips.

**Phase 2 — Strengthen decontamination to entity level.** *~1 PR.*
Add an entity/concept disjointness check (resolve against OKF wiki / `provenance_bench/data/wikidata_snapshot.json`)
to `assert_decontam.py`; emit a row-level leak report. *Pass bar:* re-audit SEIB-100,
attach the measured contaminated-row list to ledger item #1.

**Phase 3 — Clean, entity-disjoint held-out split.** *~1–2 PRs.*
Use the Phase-2 auditor as an oracle to carve a genuinely disjoint split; commission /
draft a third-party pack scaffold. *Pass bar:* `caseCount > 0` with proven entity
disjointness; closes/advances items #1 and #2.

**Phase 4 — Mix & coverage closure loop.** *~ongoing.*
Add a mix-balance regression gate (target vector + tolerance). Run the curation loop:
expand structured records + gate-filtered synthesis to lift hk_bilingual / moral_gate /
tool_mcp / Chinese toward targets and rows toward ≥10k *without* inflating rows/record.
*Pass bar:* DHI mix-balance ≤ 0.15; M2 volume gate progresses (item #3/#7).

**Phase 5 — Stand up the Data Analysis Agent.** *~1–2 PRs.*
`agent/data_analyst.py` + a `"data"` team in `swarm_router.py`, delegated via
`subagent.py`, orchestrating Phases 0–4's tools; long-horizon durable audits; automatic
ledger updates. *Pass bar:* offline test reproduces the DHI and a curation plan from a
fixed fixture; fail-closed when a tool/manifest is absent.

**Phase 6 — Continual data flywheel (governed).** *ongoing.*
Agent runs the audit→plan→gated-build→re-measure loop on a schedule, feeding RFT/GRPO
data from current errors while holding *abstention-is-reward-positive*. Every cycle
re-runs decontam + lint + DHI `--check`; drift fails the build.

---

## 7. Guardrails, risks, non-goals

- **No gate relaxation, ever.** If DHI or volume targets are hit, *upgrade the wording /
  promote via the existing gate*, never loosen a check (repo rule: OPEN→0 means upgrade
  the claim, not relax the gate).
- **Synthetic ≠ coverage.** Refinery rows are `registeredResult:false` iteration fuel;
  the agent must track records (coverage) and rows (volume) separately and label
  synthetic provenance.
- **Human authors results.** The agent proposes and measures; humans approve curation
  and own every published number (Leiden value 2).
- **Fail-closed on missing data.** No manifest / no import → refuse to emit, never
  pass-through.
- **DHI is operational, not a no-overclaim result** — internal management metric,
  labelled illustrative, never in `published-results.json`.
- **Cost discipline** for any GPU synthesis stays under
  [`wisdom-gpu-prebaked`](../../.claude/skills/wisdom-gpu-prebaked/SKILL.md) (cheap
  validation first, watch for restart loops, zero leaked pods).

## 8. Failure-ledger items to open with this work

- `data-health-index-baseline-pending` — DHI defined but no committed baseline yet (Phase 0).
- `data-lineage-graph-not-wired` — passport↔manifest↔checkpoint↔eval graph absent (Phase 1).
- `entity-level-decontam-missing` — decontam is lexical only (Phase 2; subsumes the SEIB row-level audit).
- `mix-balance-gate-absent` — no CI gate on domain/language mix (Phase 4).
- `data-analyst-agent-not-implemented` — the curator role itself (Phase 5).

---

*This document is a plan and an instrument specification. It introduces no validated
claims and changes no gate. Build the benchmark (Phase 0) first; let every subsequent
improvement be measured against it.*
