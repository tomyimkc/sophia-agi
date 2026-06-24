# CPQA — pre-registered candidate result (NOT yet validated)

Companion to [continual-learning-non-parametric.md](continual-learning-non-parametric.md)
and [continual-learning-limitations.md](continual-learning-limitations.md). This is a
**pre-registration + candidate measurement** for the Continual Provenance QA benchmark,
written in RESULTS.md style. It is **illustrative / candidate only** — it does **not**
clear Sophia's no-overclaim gate yet (see "Gate status" below). RESULTS.md itself is
generated and not hand-edited; this doc is where the candidate lives until it earns a row.

## Pre-registered protocol

- **Benchmark.** `eval/continual_qa/episodes_v2_wiki.jsonl` — 92 queries auto-generated
  from the 67-page `wiki/` corpus (recall / retention / unlearning / control). Frozen and
  committed before the judged runs.
- **Systems under test.** `grounded` (answers only from the retrieved OKF/wiki source) vs
  `raw` (a neutral LLM answering from parametric memory, no source).
- **Answer model.** `gpt-4o-mini` (OpenAI family) — neutral; not on the judge panel.
- **Judge panel.** Two **distinct provider families**: `claude-opus-4-8-low` (Anthropic) +
  `gemini-2.5-flash` (Google), via the LLMHub gateway.
- **Rubric.** Per answer: abstains / answersQuestion / faithful / fabricatesAttribution →
  binary verdict (assert-expected passes only if it answers faithfully without fabrication;
  abstain-expected passes only if it declines). Consensus = both judges pass.
- **Statistics.** 3 runs; pooled bootstrap 95% CI; inter-judge Cohen's κ per run.
- **Pre-registered prediction.** `grounded` consensus pass rate strictly exceeds `raw`.

## Candidate measurement (10-query subset, 3 runs)

| System | mean consensus pass | 95% CI | per-judge (Claude-low / Gemini-flash) | mean κ |
|---|---|---|---|---|
| **grounded** | **0.733** | [0.567, 0.867] | 0.73 / 1.00 | 0.0 (degenerate) |
| **raw** | **0.400** | [0.233, 0.567] | 0.43 / 0.47 | 0.80 |

Prediction confirmed: grounded > raw in every run. Artifact:
`agi-proof/benchmark-results/continual-qa.judged.json`.

**Illustrative second panel (same-provider, strong tier).** Answers `deepseek-chat`,
judges `deepseek-reasoner` + `deepseek-chat` (12 queries): grounded consensus **1.0**,
CI [1.0, 1.0], κ 1.0; raw ~0.33–0.42. Higher absolute numbers, but same provider and
self-grading — strictly weaker evidence than the cross-family panel above.

## Strongest panel — cross-GATEWAY, distinct families (20 queries, 3 runs)

Answers `gpt-4o-mini` (OpenAI, via LLMHub); judges `deepseek/deepseek-chat` (DeepSeek) +
`meta-llama/llama-3.3-70b-instruct` (Meta), **both via OpenRouter — an independent
gateway from the answer model.** This removes the single-gateway and self-grading
caveats and uses the repo's established validated judge pair.

| System | consensus pass | 95% CI | **assert (recall)** | **abstain (traps)** | κ / %-agree |
|---|---|---|---|---|---|
| **grounded** | **0.933** | [0.867, 0.983] | 0.911 (n=45) | **1.000 (n=15)** | 0.33 / 0.93 |
| **raw** | 0.667 | [0.55, 0.783] | 0.889 (n=45) | **0.000 (n=15)** | 0.81 / 0.92 |

**The clean finding (per-expect breakdown):** on *recall* the two are ~equal (0.911 vs
0.889) — a strong raw model already knows well-known wiki facts, so grounding barely
helps. The entire advantage is on the *abstain* traps/retractions: grounded **1.000** vs
raw **0.000** — total, robust separation, with **disjoint overall CIs**. Grounding's
contribution is precisely *fail-closed abstention*, not recall — exactly the thesis, now
measured across distinct provider families on independent infrastructure. Artifact:
`agi-proof/benchmark-results/continual-qa.judged-xgateway.json`.

*κ caveat persists on grounded:* κ=0.33 (< 0.40) because grounded is near-saturated
(both judges pass almost everything → low variance), while raw κ=0.81 is healthy. The
honest agreement number is **percent-agreement 0.93**; report it alongside κ.

## Gate status (no-overclaim) — why this is candidate, not validated

| Criterion | Status |
|---|---|
| ≥2 judge families | ✅ DeepSeek + Meta (cross-gateway panel); also Anthropic + Google |
| Judge ≠ subject | ✅ answers by OpenAI; judges DeepSeek/Meta — no overlap |
| Judge gateway ≠ answer gateway | ✅ judges via OpenRouter, answers via LLMHub (independent infra) |
| ≥3 runs | ✅ 3 runs |
| Confidence intervals | ✅ bootstrap 95% CI (overall CIs disjoint: [0.867,0.983] vs [0.55,0.783]) |
| Inter-judge agreement κ ≥ 0.40 | ⚠️ **raw κ 0.81 ✓**, but **grounded κ 0.33** — depressed by near-saturation (both judges pass almost all grounded answers → low variance). Honest agreement = **percent-agreement 0.93**. |
| Pre-registered sign-off / external replication | ❌ self-authored benchmark, two keys held by one operator, single 20-query subset |

**Verdict: candidate / illustrative only. `validated:false`.** The cross-family
machinery, distinct families, runs, and CIs are in place; what remains is (a) a stronger
judge tier so grounded κ is non-degenerate (or report percent-agreement alongside κ),
(b) the full 92-query set, (c) independent replication off a single gateway, and
(d) formal pre-registration sign-off.

## Honest caveats

- **κ degeneracy.** When a judge passes (or fails) every item, κ → 0 regardless of true
  agreement. Report raw percent-agreement next to κ in future runs.
- **Tier sensitivity.** Absolute grounded pass fell from 1.0 (strong same-family panel) to
  0.73 (fast cross-family panel) — the number depends on answer-model and judge tiers, so
  only the *direction* (grounded > raw) is robust, not the absolute value.
- **Abstain rubric.** Confident refutation of a fictional premise scores as
  non-abstention, understating `raw` on those controls; the unambiguous contrast is on
  retracted *real* facts.
- **One gateway / one key.** Not independent infrastructure.

## Reproduce

```bash
# id-routing control-flow gap (deterministic stand-ins, offline):
python tools/run_continual_qa_validation.py
# live cross-family judged pass (needs LLMHUB_API_KEY):
LLMHUB_API_KEY=... python tools/run_continual_qa_judged.py --limit 10 --runs 3 \
  --answer llmhub:gpt-4o-mini --judge llmhub:claude-opus-4-8-low --judge llmhub:gemini-2.5-flash
```
