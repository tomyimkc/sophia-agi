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

## Faithfulness probe — v1 FALSIFIED, v2 in place

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
- **v3** (`40e22b12` code, `bceb242e` real run,
  `faithfulness-probe.public-report.json`): the probe-power upgrade — 16
  binary-gold probes (the ill-posed "possibly" gold is dropped), Cohen's d effect
  size replacing v2's boolean, mean ± std per hint. Real run on sophia-v3:
  `cohensD=0.44`, `effectVerdict=small effect / inconclusive`. **Do not read this
  as "decorative CoT" OR as "faithful"** — it is inconclusive at this power. Two
  concrete limits explain it (see the artifact's `findingScope`): (1) effective n
  is small — the reasoning-only perturbs only apply when a CoT has ≥2 reasoning
  sentences, so most probes yield nAttempted≤2 and the stds (2.89, 1.30) exceed
  the means (1.44, 0.45); (2) the CoTs are too short to perturb. The mock CI test
  (`test_v3_probe_shows_large_effect_on_mock`, d≈1.1) confirms the probe *logic*
  is sound. A defensible measurement needs longer multi-step CoTs (≥4 reasoning
  sentences so ≥3 perturbs apply) and ~30+ probes.

This is the discipline the layer exists to enforce: a probe that overclaims what
it measures is itself an overclaim, and gets recorded as such.

## What it is not

- Not a claim of AGI or ASI.
- Not a faithfulness proof for chain-of-thought.
- Not in scope for pretraining (per-step logging at 1e12-token scale is
  infeasible; the discipline targets stages where one bad step can poison
  downstream belief).
- Not a substitute for the multi-run, multi-judge-family validation gate.
