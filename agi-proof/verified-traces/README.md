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

## What it is not

- Not a claim of AGI or ASI.
- Not a faithfulness proof for chain-of-thought.
- Not in scope for pretraining (per-step logging at 1e12-token scale is
  infeasible; the discipline targets stages where one bad step can poison
  downstream belief).
- Not a substitute for the multi-run, multi-judge-family validation gate.
