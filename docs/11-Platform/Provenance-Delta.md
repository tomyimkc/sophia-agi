# The Provenance Delta benchmark

**The one artifact:** a reproducible measurement of how often a model asserts a
**false authorship lineage** when used *alone* versus *behind Sophia's provenance
gate* — on ground truth that is independent of the gate.

> On an independent dataset of attribution claims, a model asserts a false
> lineage **X%** of the time; behind Sophia's gate that drops to **Z%** —
> reproducible in one command.

This closes claim-ladder items 6–7 (external evaluation, independent
replication) for the provenance niche. See the design spec:
[`docs/superpowers/specs/2026-06-21-provenance-delta-design.md`](../superpowers/specs/2026-06-21-provenance-delta-design.md).

## Why it isn't circular

| Concern | Guarantee |
|---|---|
| Labels from the gate's own corpus? | No. Labels come from `provenance_bench/data/*.json` (Wikipedia/Wikidata + **cited** misattributions), physically separate from the gate corpus (`data/*.json`). |
| Gate judging itself? | No. The gate (`agent/verifiers.py`) is the **runtime treatment** only. The **judge** (`provenance_bench/judge.py`) shares no code with it; for headline runs use an independent LLM-judge (a *different* model than the one under test). |
| Coverage assumed? | No. The fraction of false cases the gate fires on is a **measured** metric, not an assumption. |

## The three honest metrics

1. **Hallucinated-attribution rate** — of answers asserting an attribution, the
   fraction contradicting external gold. Reported *alone* vs *behind gate*; the
   difference is the **delta**.
2. **False-positive cost** — of correct answers the model gives *alone*, the
   fraction the gate then breaks. (A gate that abstains on everything would ace
   metric 1; this catches that.)
3. **Coverage / recall** — of false attributions made *alone*, the fraction the
   gate fixes. Names the gate's narrowness instead of hiding it.

## Run it

```bash
# offline smoke run (deterministic mock; no API cost) — plumbing only
python tools/run_provenance_delta.py --models mock

# real headline run, independent LLM-judge (judge ≠ models under test)
python tools/run_provenance_delta.py \
    --models anthropic,openai,grok,ollama:qwen2.5-7b \
    --llm-judge anthropic:claude-opus-4-8

# optionally verify/populate Wikidata QIDs for the true-attribution snapshot
python tools/fetch_wikidata_authors.py            # dry run
python tools/fetch_wikidata_authors.py --write
```

Outputs (git-ignored, regenerable): `agi-proof/benchmark-results/provenance-delta.public-report.json`
and `.md`.

## Components

| File | Role |
|---|---|
| `provenance_bench/data/misattributions.json` | cited FALSE attributions (lineage-merge probes) |
| `provenance_bench/data/wikidata_snapshot.json` | TRUE attributions (gold + false-positive controls) |
| `provenance_bench/dataset.py` | external files → case set |
| `provenance_bench/judge.py` | independent referee (lexical screen; LLM-judge hook) |
| `provenance_bench/runner.py` | per case: model *alone* vs *gated* (`agent/guarded.py`) |
| `provenance_bench/score.py` | the three metrics |
| `provenance_bench/report.py` | JSON + markdown report |
| `tools/run_provenance_delta.py` | CLI |
| `tools/fetch_wikidata_authors.py` | optional QID verification |

## Scope (this phase)

Textual authorship only. Legal/code provenance, a public submission leaderboard,
and gate-corpus expansion are later phases — see the
[checklist](../../agi-proof/external-benchmarks/PROVENANCE-DELTA-CHECKLIST.md).
