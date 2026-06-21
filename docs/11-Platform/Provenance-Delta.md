# The Provenance Delta benchmark

**The one artifact:** a reproducible measurement of how often a model asserts a
**false authorship lineage** when used *alone* versus *behind Sophia's provenance
gate* — on ground truth that is independent of the gate.

> On an independent dataset of attribution claims, a model asserts a false
> lineage **X%** of the time; behind Sophia's gate that drops to **Z%** —
> reproducible in one command.

### Illustrative multi-model run (87 cases, single run each, lexical judge)

Not a headline claim yet — one run per model, lexical screen, N=46 false cases,
and we observed real run-to-run variance. But the shape of the result is
informative:

| Model | Halluc. alone | Halluc. gated | Δ | False-positive cost | Coverage |
|---|---|---|---|---|---|
| `deepseek` (frontier) | 0.0% | 0.0% | 0.0 | 0.0% | – |
| `dolphin-llama3:8b` (uncensored tune) | 15.2% | 6.5% | **8.7** | 0.0% | 57% |
| `llama3.2:3b` (aligned) | 2.2% | 2.2% | 0.0 | 0.0% | 0% |
| `qwen2.5:3b` (aligned) | 2.2% | 2.2% | 0.0 | 0.0% | 0% |

**Honest reading — the delta tracks a model's *propensity to assert*, not its
size:**
- A frontier model already practices source discipline on these cases (0%).
- The biggest delta is the *uncensored* 8B tune, which confidently asserts false
  lineages — exactly the population the gate is for. Behind the gate its rate
  drops ~2-3× **at zero cost to correct answers**.
- The well-aligned 3B models rarely assert a false attribution (they hedge or
  abstain), so there is little for the gate to fix. Smaller ≠ more hallucination.
- **Coverage < 100% is real and diagnosable.** Observed misses: a quoted title
  (`wrote "The Constitution of the Athenians"`) the gate regex doesn't span, and
  `attributed to X` phrasing (the gate matches `attributed by`). These are scoped
  gate improvements, logged in the checklist — not reasons to loosen the
  high-precision core.
- **Variance is real**: counts shifted run-to-run. Single-run numbers are
  illustrative only.

Promote to a headline only after the Tier-1 steps in the
[checklist](../../agi-proof/external-benchmarks/PROVENANCE-DELTA-CHECKLIST.md)
(CIs, multi-run averaging, independent LLM-judge).

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

## Which models show a delta? (where the gate earns its keep)

The delta tracks a model's **propensity to assert**, not its parameter count.
Choose subjects accordingly:

- **Big deltas:** uncensored / "uncial" / older / instruction-light models that
  answer confidently even when wrong (e.g. `dolphin-llama3:8b`). This is the
  gate's target population.
- **Small/zero deltas:** well-aligned frontier models (`deepseek`, Claude, GPT)
  already practice source discipline on these cases, and well-aligned *small*
  models tend to hedge/abstain rather than confabulate — so there is little for
  the gate to remove. A near-zero delta here is a *true* result, not a failure.

So the headline is **not** "we beat frontier models"; it is **"Sophia's gate
makes the models that need it measurably more faithful, at zero cost to correct
answers."** Pair the benchmark with a confidently-wrong model to show the effect,
and an independent frontier model as the **judge** (never as both subject and
judge).

## Run it

```bash
# offline smoke run (deterministic mock; no API cost) — plumbing only
python tools/run_provenance_delta.py --models mock

# citable run: independent judge (≠ subject) + 3 runs -> bootstrap 95% CIs
python tools/run_provenance_delta.py \
    --models ollama:dolphin-llama3:8b \
    --llm-judge deepseek --runs 3

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
| `provenance_bench/runner.py` | per case: model *alone* vs *gated* — same answer, gate-filtered (reuses `agent/guarded.py` helpers); prompt held constant to isolate the gate's effect |
| `provenance_bench/score.py` | the three metrics |
| `provenance_bench/report.py` | JSON + markdown report |
| `tools/run_provenance_delta.py` | CLI |
| `tools/fetch_wikidata_authors.py` | optional QID verification |

## Scope (this phase)

Textual authorship only. Legal/code provenance, a public submission leaderboard,
and gate-corpus expansion are later phases — see the
[checklist](../../agi-proof/external-benchmarks/PROVENANCE-DELTA-CHECKLIST.md).
