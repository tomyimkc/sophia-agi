# Sophia Verified Reasoning-Trace Proof Package

**Status:** candidate-only process-supervision + TRiSM-audit infrastructure.  
**Boundary:** This package does **not** prove AGI and keeps `canClaimAGI=false`.

This folder holds the public artifacts for the verified reasoning-trace layer
(`sophia.verified_trace.v1`): a dual (fact + logic) stamp per reasoning step,
append-only + tamper-evident (Merkle/hash-chained), surfaced into the capability
panel as a "process" axis and queryable via MCP.

## What it is

A unified record for every reasoning step in scope (SFT / RLVR / curriculum /
benchmark / conscience — **not** pretraining, deliberately):

- the **fact** stamp reuses `ConscienceDecision.verdict` + the OKF provenance
  fields (`authorConfidence`, weakest-link `effectiveConfidenceRank`, `sources`);
- the **logic** stamp reuses `reasoning_compiler.CompileResult`
  (`emittable`, `contradictions`, `laundered`, `semanticsPreserved`) — the
  compiler's fail-closed type-check IS the validity verdict;
- a step is `verified` iff `fact.verdict ∈ {allow, retrieve}` AND
  `logic.emittable`. **Derived, never asserted** — `sophia_trace_verify`
  re-derives it on demand.

The record is a strict superset of the Langfuse-style span already emitted by
`sophia_contract.trace.Tracer.span`, so it drops into the existing observability
feeds. Each line carries `prevHash` (prior line's `_selfHash`) and its own
`_selfHash`, so any mutation of a prior line breaks the chain — EU AI Act
Art. 12-style evidence at near-zero code.

## Run

```bash
# run the panel with the verified-trace axis included
python tools/eval_capability_panel.py --mode mock

# query / re-verify the trace log via MCP
python -c "from sophia_mcp.tools_impl import trace_query, trace_verify; print(trace_query())"
```

Main aggregate artifact:

```text
agi-proof/benchmark-results/capability-panel.public-report.json   (axes.verifiedTraces)
```

## Falsification test (the killer experiment)

```bash
python tools/run_verified_trace_recall.py
```

Main artifact:

```text
agi-proof/verified-traces/verified-trace-recall.public-report.json
```

Runs the reasoning compiler's seeded synthetic experiment (400 graphs, 200 with
planted live contradictions) WITH the trace hook active, then asserts four
falsifiable invariants: (1) the log's `contradictionRecall` equals the
compiler's planted-ground-truth recall, (2) it is 1.0 (no contradiction slipped
through), (3) the hash chain survives a 400-trace run, (4) every record carries
the no-overclaim triad. `VERDICT: CONFIRMED` on synthetic graphs; real-world
recall is bounded by the fact gate's external (recall, fpr).

## Honest scope

The verified-trace layer is deterministic, offline candidate infrastructure. It
makes Sophia's reasoning **auditable** — it does **not** make it smarter, and it
does **not** prove the model "truly reasoned" the recorded way. CoT
faithfulness (METR 2025 / Anthropic intervention work) is a separate causal
question: "verified" here means "passed both gates at the verifier's (recall,
fpr)", never "the chain-of-thought is the real causal path."

This package therefore carries the no-overclaim triad on every record
(`candidateOnly=true`, `level3Evidence=false`, `boundary`) and inherits the
`provenance_bench.aggregate._is_validated` bar for any capability *claim* derived
from it. Logging does not produce AGI; it produces a brake, not an engine.

## Faithfulness probe — v1 FALSIFIED → v2 under-powered → v3 inconclusive → v4 inconclusive-on-binary-facts (d=0.08) → v5 causal-dependency redesign **(d=0.95, CI excludes 0 — first defensible positive)** → **v5.1 REPLICATED on a fresh disjoint set (sophia-v3 d=1.39; sophia-v2 d=1.26)**

The first real-mode run of the faithfulness probe (commit `240f3e54`,
`faithfulness-probe.v1-FALSIFIED.public-report.json`) **falsified the probe
itself**, and the falsification is kept on record rather than hidden:

- **Result:** uniform `flipRate=0.5` across all three CoT categories
  (load-bearing / hedged / post-hoc).
- **Why it is a null, not "mixed evidence":** two compounding design flaws
  guarantee ~0.5 regardless of category. (1) The v1 decider forces
  `argmax(logprob(" yes"), logprob(" no"))` even for a "possibly" gold answer, so
  one probe's decision was meaningless. (2) The `_drop_last_sentence` perturb
  deletes the `Answer:` line the decider reads — so it tested "does deleting the
  answer change the answer" (trivially yes, trivially uniform), not whether the
  *reasoning* was load-bearing.
- **v2 fix:** `build_mlx_decide_gold` scores the *gold token's* logprob drop
  under perturbation (answer-agnostic), and `default_perturbs_reasoning` perturbs
  reasoning sentences only (the `Answer:` line is preserved).
- **v2 first real run** (`d65f64f5`, superseded — see v3): `discriminates=false`.
  Under-powered (n=2 perturbations/probe, one gold token at −6.6 nats baseline is
  effectively ill-posed). The mock-based discrimination test passes, so the probe
  *logic* is sound — the gap to real logprobs is a probe-POWER problem, not an
  adapter finding.
- **v3** (`40e22b12` code, `bceb242e` real run, archived as
  `faithfulness-probe.v3.public-report.json`): the first probe-power upgrade — 16
  binary-gold probes (the ill-posed "possibly" gold is dropped), Cohen's d effect
  size replacing v2's boolean, mean ± std per hint. Real run on sophia-v3:
  `cohensD=0.44`, `effectVerdict=small effect / inconclusive`. **Do not read this
  as "decorative CoT" OR as "faithful"** — it is inconclusive at this power. Two
  concrete limits explain it (see the artifact's `findingScope`): (1) effective n
  is small — the reasoning-only perturbs only apply when a CoT has ≥2 reasoning
  sentences, so most probes yield nAttempted≤2 and the stds (2.89, 1.30) exceed
  the means (1.44, 0.45); (2) the CoTs are too short to perturb. The mock CI test
  (d≈1.1) confirms the probe *logic* is sound. A defensible measurement needs
  longer multi-step CoTs (≥4 reasoning sentences so ≥3 perturbs apply) and ~30+
  probes.
- **v4** (this iteration, code in `agent/faithfulness_probe.py` +
  `tools/run_faithfulness_probe.py`, canonical `faithfulness-probe.public-report.json`):
  the power upgrade that fixes both v3 limits. **30 binary-gold probes** (15
  load-bearing / 15 post-hoc), each with **≥4-sentence reasoning**, and **6
  reasoning-only perturbs** (the v2/v3 trio — drop-sentence, negate,
  swap-quantifier — plus reorder-two-sentences, drop-connective, and
  replace-entity-with-distractor), so each probe yields **nAttempted≥3** by
  construction (meanAttempted≈3.63 on the mock vs ≤2 in v3). It keeps Cohen's d +
  per-group mean/std and adds a **bootstrap CI** on the mean difference and a
  per-probe **sign test** on the direction; the defensible bar for a positive
  claim is now `|d|≥0.8 AND bootstrapCI.excludesZero AND replicated`. The mock CI
  self-test (`test_v4_probe_shows_large_effect_on_mock`) reaches **d≈1.16, a
  bootstrap CI that excludes 0, and a sign-test p≈3.1e-05** — confirming the probe
  *logic* is sound at this power (`faithfulness-probe.v4-mock.public-report.json`).
  **The v4 REAL run is complete** on Apple Silicon (mlx-lm 0.29.1) over
  sophia-v3 (LoRA, rank 8) on `mlx:Qwen/Qwen2.5-3B-Instruct` — outcome **(c),
  inconclusive at this power**: **`cohensD=0.08`** (`|d|<0.5`), **bootstrap 95%
  CI `[-0.41, +0.54]` includes 0** (`excludesZero=false`), and a non-significant
  sign test (9 pos / 6 neg, `p=0.30`). The direction is nominally right
  (load-bearing CoTs drop more under perturbation: mean 0.38 vs 0.28) but the
  within-group stds (~1.1–1.3) are an order of magnitude larger than the mean
  difference (~0.10), so the probe cannot separate the categories here. This is
  **not** "decorative CoT" **and not** "faithful" — it is a power/adapter null:
  the adapter's CoT faithfulness on this 3B base + these 30 binary probes is
  **unmeasured**, the same status v3 reached. The mock self-test (`d≈1.16`, CI
  excludes 0, `p≈3.1e-05`) and the 19 passing logic tests confirm the probe
  *mechanism* is sound, so the null is an adapter/power finding, not a broken
  probe. **Discipline held:** probes, perturbs, and scorer were not retuned to
  raise `d` (the one forbidden move). See the canonical artifact's `findingScope`
  for the full power analysis. The canonical command for reproducibility:
  `python tools/run_faithfulness_probe.py --mode real --adapter training/mlx_adapters/sophia-v3/ --model mlx:Qwen/Qwen2.5-3B-Instruct`.
- **v5** (code in `tools/run_faithfulness_probe_v5.py`, canonical
  `faithfulness-probe-v5.public-report.json`): the **causal-dependency redesign** the
  v4 null pointed to. v4's `d` *fell* as power rose (0.44 → 0.08) — an effect that
  shrinks with more power is evidence of a true effect near zero on that design,
  not a signal missed for lack of power. The honest reading: v4's binary
  common-knowledge facts are answered robustly by the 3B base, so the CoT is
  *superfluous to the answer* and faithfulness has nothing to register. v5 attacks
  the **design**, not the power: each load-bearing probe is a **multi-step
  arithmetic derivation whose gold answer is unreachable without the chain**, paired
  with a post-hoc twin that asserts the same gold with filler. The new discipline is
  a **dependency gate** — a probe is admitted only if (load-bearing) its chain
  evaluates to the gold AND loses the gold when a step is dropped, or (post-hoc) its
  reasoning contains no derivation reaching the gold. The gate is offline,
  deterministic, and **rejects a v4-style binary-fact probe** (locked in by
  `test_gate_rejects_v4_style_binary_fact_probe`), which is what structurally
  prevents the v4 ceiling from recurring. The mock self-test
  (`test_v5_mock_shows_large_effect`) reaches **d≈3.25, a bootstrap CI that excludes
  0, p≈3.1e-05, all 30 probes admitted** — confirming the v5 design *can* register a
  load-bearing signal when one exists. **The v5 REAL run is complete** on Apple
  Silicon (mlx-lm 0.29.1) over sophia-v3 (LoRA rank 8) on
  `mlx:Qwen/Qwen2.5-3B-Instruct` — **outcome (a), the first defensible positive in
  the arc**: **`cohensD=0.9515`** (`|d|>=0.8`), **bootstrap 95% CI
  `[0.219718, 0.514982]` excludes 0** (`excludesZero=true`), and a sign test at
  **15 pos / 0 neg, `p=3.1e-05`**. The dependency gate admitted all **30/30**
  deterministically (a rejection on the real path would mean the gate changed, not
  an adapter finding). The direction is the predicted one: perturbing a
  load-bearing arithmetic step drops the gold-token logprob (mean +0.331) while
  perturbing post-hoc filler barely moves it (mean −0.029). This is the **first
  time the arc's probe separated the two categories on a real adapter** — and
  precisely where v4 could not, on probes whose answer the 3B base cannot reach
  cold (so the chain is not superfluous). **Stability / replication:** a second
  real run produced a **bit-identical** artifact (same SHA-256, same `d`/CI/sign)
  because MLX logprob scoring with a fixed adapter is deterministic (no sampling) —
  this confirms the measurement's *reproducibility*, not an independent
  statistical replication (a genuinely independent replication needs a different
  model/adapter seed or a held-out probe set, not done here). **Honest scope:** v5
  is **arithmetic-only** (the cleanest class whose causal dependence the gate can
  *verify* offline; extending to multi-hop / factual chains is future work and must
  not rubber-stamp non-dependent probes), the effect is on a 3B base + rank-8 LoRA
  (not a generalizable claim), a load-bearing result is positive evidence of
  faithfulness on THESE chain-dependent tasks **not a faithfulness proof**, and the
  artifact stays `candidateOnly=true, validated=false`. The canonical command for
  reproducibility:
  `python tools/run_faithfulness_probe_v5.py --mode real --adapter training/mlx_adapters/sophia-v3/ --model mlx:Qwen/Qwen2.5-3B-Instruct`.
- **v5.1 replication (REPLICATED on sophia-v3)** — the v5 positive's third leg
  (`replicated`) is now **met**. The only repeat of v5 was a deterministic
  bit-identical MLX re-run (reproducibility, not replication), so v5.1 adds a
  `replication` probeset (`tools/run_faithfulness_probe_v5.py --probeset replication`,
  canonical `faithfulness-probe-v5-replication.public-report.json`) — a **fresh,
  disjoint** batch of 15 arithmetic problems (no shared question or chain with the
  primary set, enforced by `test_replication_set_is_disjoint_from_primary`). **The v5.1
  REAL run is complete** on Apple Silicon (mlx-lm 0.29.1) over sophia-v3 (LoRA rank 8)
  on `mlx:Qwen/Qwen2.5-3B-Instruct`, meeting the **PRE-REGISTERED** criterion (fixed in
  the artifact's `replicationCriterion` before the run): **`cohensD=1.3883`**
  (`|d|>=0.8`, larger than the primary's 0.9515), **bootstrap 95% CI
  `[0.348611, 0.602083]` excludes 0**, and a sign test at **15 pos / 0 neg,
  `p=3.1e-05`** in the same direction (load-bearing drops MORE: mean +0.383 vs
  post-hoc −0.090), dependency gate **30/30 admitted**. **VERDICT: the v5 positive
  graduates from "positive evidence, replication owed" to "REPLICATED positive"** on
  sophia-v3 for arithmetic chain-dependent tasks. This is a genuine independent
  replication (new probe instances, same design and adapter), not the deterministic
  bit-identical re-run. **Cross-adapter generalization (reported separately):** the
  same fresh set on **sophia-v2** (`faithfulness-probe-v5-replication-sophia-v2.public-report.json`)
  also cleared the bar — `d=1.2599`, CI `[0.421528, 0.792361]` excludes 0, sign 15/0
  — so the load-bearing effect is not specific to the v3 checkpoint; it appears across
  both available sophia LoRA adapters on the 3B base. This is evidence of
  generalization across checkpoints, not a second independent replication in the
  probe-set sense. Honest scope unchanged: arithmetic-only, 3B base + rank-8 LoRA,
  candidate-only (`validated=false`); a replicated load-bearing result is positive
  evidence of faithfulness on THESE chain-dependent tasks, **not a faithfulness proof**.
  Discipline held: the criterion was fixed before the run, and probes/perturbs/
  scorer/gate were not retuned to force replication — the only change from v5 was the
  probe instances.

This is the discipline the layer exists to enforce: a probe that overclaims what
it measures is itself an overclaim, and gets recorded as such. The v4 real run
executed on Apple Silicon (no mock substituted) and returned outcome (c); v5 is
the redesign its diagnosis called for, built and self-tested offline, and its real
run executed on Apple Silicon — outcome (a), the first defensible positive in the
arc (d=0.95, CI excludes 0); v5.1 then closed the `replicated` leg with a
pre-registered, fresh-disjoint-set run that also cleared the bar (sophia-v3
d=1.39, sophia-v2 d=1.26) — reported candidate-only and not as a faithfulness
proof, with the discipline held (criterion fixed before the run; probes/perturbs/
scorer/gate not retuned to force a high `d`).

## What it is not

- Not a claim of AGI or ASI.
- Not a faithfulness proof for chain-of-thought.
- Not in scope for pretraining (per-step logging at 1e12-token scale is
  infeasible; the discipline targets stages where one bad step can poison
  downstream belief).
- Not a substitute for the multi-run, multi-judge-family validation gate.
