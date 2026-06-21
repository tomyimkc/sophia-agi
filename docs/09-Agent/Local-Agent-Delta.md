# Local-agent delta — alone vs +gate vs +MCP-tools

Measures whether a local LLM augmented with Sophia's tools performs better, over
the 87 provenance cases. **Honest finding (2026-06-21): on a strong model
(qwen3:30b-a3b) the gate is neutral and `+mcp-tools` gives a small gain
(gold 90.2%→92.7%) — there is little headroom.** An earlier build *degraded*
accuracy by forcing tools on every case; fixed via selective invocation +
richer tool outputs (see below, and `agi-proof/failure-ledger.md`). Not AGI;
not a general-performance claim.

## Conditions
- **alone** — raw model (reuses `provenance_bench.runner.run_case`).
- **+gate** — provenance verifier → repair/abstain (same).
- **+mcp-tools** — native tool-calling loop: the model is given Sophia's
  read-only MCP tools (`check_claim` / `wiki_search` / `belief`) as OpenAI
  function schemas; it may call them, we dispatch **in-process** via
  `sophia_mcp.tools_impl` (no server), feed results back, take the final answer.

## Result (qwen3:30b-a3b, 87 cases)

| condition | hallucination | gold-affirm (true) | false-positive |
|---|---|---|---|
| alone | 0.0% | 90.2% | 9.8% |
| +gate | 0.0% | 90.2% | 9.8% |
| +mcp-tools | 0.0% | **92.7%** | **7.3%** |

- **+tools now helps** — gold-affirmation 90.2%→92.7%, false-positive 9.8%→7.3%.
  (An earlier build *degraded* gold to 51.2% by forcing tools on every case with
  sparse outputs; fixed — see below.)
- **Small delta, by design** — qwen3:30b-a3b already answers 90%+ correctly, so
  there is little headroom for the gate/tools. Matches RESULTS.md: the delta
  tracks a model's propensity-to-assert, not its size. A visible delta needs a
  weaker/high-assertion model.

## Result (dolphin-llama3:8b, 87 cases)

**Single-judge (lexical, 1 run) — illustrative only:** alone 15.2% → +gate 4.3%
hallucination. *This did NOT survive validation* (below). The `+mcp-tools` 0.0%
was re-generation, NOT tool-use (`toolsUsed: []` — dolphin doesn't emit native
`tool_calls`), so never attribute it to the MCP tools.

**Validated run (3 runs, 2 judge families = `ollama:llama3.2:3b` +
`deepseek:deepseek-chat`):**

| metric | value |
|---|---|
| hallucination alone | 9.4% |
| hallucination gated | 7.2% |
| delta | 2.2% |
| **95% CI on delta** | **[−2.2%, +6.5%] — includes zero** |
| false-positive cost | **0.0%** |
| gate coverage | 46.2% |
| judge agreement | 78% |
| **validated?** | **NO** (`ciExcludesZero` fails) |

- **The honest finding: there is NO quotable validated capability number.** Under
  a proper 2-family consensus judge the delta is positive-leaning but **not
  statistically distinguishable from zero** at N=46 false cases × 3 runs.
- **Why the single-judge 15.2%→4.3% shrank:** judge choice dominates the absolute
  number (see RESULTS.md). The lone lexical judge over-counted "hallucinations";
  stricter consensus judges agree on far fewer, so both the rate and the delta
  fall below significance. *Sophia's own no-overclaim gate caught Sophia's
  optimistic number* — working as designed.
- **What IS quotable today:** 0% false-positive cost across 3 runs / 2 judge
  families (the gate never broke a correct answer) and 46.2% gate coverage.
- **The tension to know:** models that hallucinate (need help) tend to be weak
  tool-callers; models that tool-call well (qwen3) don't hallucinate (no
  headroom). The **gate** is robust to both. A genuine *tool-use* delta needs a
  model both weak and tool-capable (e.g. `qwen2.5:3b-instruct`, `glm-4-9b-chat`).

## Design (the two fixes that turned a degradation into a gain)
1. **Selective invocation** — `+tools` starts from the SAME plain answer as
   `alone` and only runs the tool loop when that answer is low-confidence
   (hallucinated / blank-abstention / true-case miss). This *guarantees* tools
   can never make the answer worse; they only diverge to repair a weak answer.
2. **Richer tool outputs** — `wiki_search` results carry content snippets (not
   just page IDs) and `belief` falls back to a wiki hint on an entity miss, so
   the model gets signal, not bare handles.

## Headline claim (defensible — what survives the gate)
> Across 3 runs and 2 independent judge families, Sophia's provenance gate
> removes ~46% of a small uncensored model's hallucinated attributions at **0%
> false-positive cost** (never breaks a correct answer). The hallucination-rate
> *delta* is positive but not yet statistically significant
> (Δ2.2%, 95% CI [−2.2%, +6.5%]) — a larger N is needed before quoting it.

Do NOT quote the single-judge "15.2%→4.3%" — it did not survive multi-judge
validation. Quote the **0% false-positive cost** and **coverage**, not
the `+mcp-tools` row (which on this model is re-generation, not tool-use). For a
*validated* (not illustrative) headline, run ≥3 runs with ≥2 judge families.

## Reproduce
```bash
python tests/test_local_agent_delta.py                              # offline wiring (CI)
python tools/run_local_agent_delta.py --model mock                  # offline invariants
python tools/run_local_agent_delta.py --model ollama:dolphin-llama3:8b   # the visible delta
python tools/run_local_agent_delta.py --model ollama:qwen3:30b-a3b       # strong model (small delta)
# validated gate delta (>=2 judge families, >=3 runs, kappa + bootstrap CI):
# uses run_provenance_delta.py — it has the full _is_validated machinery.
DEEPSEEK_API_KEY=... python tools/run_provenance_delta.py \
    --models ollama:dolphin-llama3:8b \
    --judges ollama:llama3.2:3b,deepseek:deepseek-chat --runs 3 \
    --out agi-proof/benchmark-results/provenance-delta-dolphin-validated.public-report.json
```
