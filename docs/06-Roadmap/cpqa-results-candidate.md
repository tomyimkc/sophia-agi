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

## Full 92-query cross-GATEWAY run, 3 runs — the honest result at scale

Answers `gpt-4o-mini` (OpenAI, via LLMHub); judges `deepseek/deepseek-chat` (DeepSeek) +
`meta-llama/llama-3.3-70b-instruct` (Meta), **both via OpenRouter — an independent
gateway from the answer model** (distinct families, no self-grading, the repo's
established judge pair). Artifact: `agi-proof/benchmark-results/continual-qa.judged-xgateway-full92.json`.

| System | overall consensus | 95% CI | **recall (n=261)** | **traps (n=15)** | κ / %-agree |
|---|---|---|---|---|---|
| **grounded** | 0.529 | [0.471, 0.587] | **0.502** | **1.000** | 0.94 / 0.97 |
| **raw** | **0.884** | [0.848, 0.920] | **0.935** | **0.000** | 0.67 / 0.95 |

**The honest headline flips at scale, and that matters more than the flattering subset.**
A 20-query, trap-heavy subset earlier showed grounded 0.93 > raw 0.67. On the full 92
queries the overall winner is **raw (0.88) over grounded (0.53)** — and the per-expect
split shows exactly why:

- **Traps / retractions (n=15): grounded 1.000 vs raw 0.000.** Robust, total separation —
  grounding gives perfect fail-closed abstention; the raw model fabricates/asserts on
  every one. The safety thesis holds.
- **Recall (n=261): grounded 0.502 vs raw 0.935.** Grounding *collapses* on recall because
  the answer is constrained to the retrieved wiki page, and many pages are thin provenance
  stubs that do not contain the answer — so the grounded system abstains/fails where a raw
  model just answers from parametric memory.

This is the **coverage-vs-fabrication tradeoff** named in the limitations doc, now measured:
grounding buys trap-safety **at a recall cost**, not a blanket win, and on a recall-heavy,
thin-source corpus the raw model wins overall. The right reading is *not* "grounding loses"
— it is "grounding is a precision/safety layer whose value depends on (a) how trap-heavy
the workload is and (b) how complete the sources are." Both are improvable (richer wiki
pages; a hybrid that falls back to parametric recall when the gate would abstain).

**κ is now healthy** (grounded 0.94, raw 0.67; percent-agreement 0.97 / 0.95) — the earlier
grounded-κ degeneracy was a small-subset/ saturation artifact and does not appear at scale.

## Recall fix — Steps 1+2+4 hybrid (full 92, 3 runs, same cross-gateway judges)

The recall audit (Step 6) found **58% of recall targets are thin provenance stubs**, so
strict grounding is corpus-capped. The hybrid (Step 2 typed gate + Step 1 graph-neighborhood
retrieval + Step 4 attribution-safe fallback) routes each query by context type:
no grounded source → hard-abstain; answer-bearing → strict; grounded-but-thin → gated
parametric fallback.

| System | overall | 95% CI | recall (n=261) | traps (n=15) | κ |
|---|---|---|---|---|---|
| strict (baseline) | 0.529 | [0.471, 0.587] | 0.502 | 1.00 | 0.94 |
| **hybrid + neighborhood** | **0.685** | [0.630, 0.739] | **0.678** | 0.80* | 0.84 |
| raw | 0.906 | [0.870, 0.938] | 0.958 | 0.00 | 0.80 |

**Recall recovered 0.50 → 0.68 (+35% relative); overall 0.53 → 0.69.** The raw model still
wins overall (0.91), but the gap narrowed from 0.38 to 0.22 — and grounding keeps the
property raw never has: it abstains on traps instead of fabricating.

**Trap-safety held at the behavior level (the guardrail that matters).** Policy telemetry:
`{abstain_no_source: 15, grounded_strict: 120, grounded_fallback: 141}` — **all 15 trap
evaluations took the hard-abstain path; 0 traps ever reached the parametric fallback** (it
fired only on the 141 grounded-but-thin facts). *The trap *score* of 0.80 (vs strict 1.0) is
judge measurement noise on the fixed "I don't know" abstention string — not gate leakage; the
behavior is identical to the strict run. Re-running would re-roll that noise; the policy-count
proof is the durable guarantee.

**Honest read:** the hybrid is a real, measured recall recovery with the fabrication guardrail
intact by construction. Remaining gap to raw is the thin corpus itself — Step 5 (enrich the
wiki page bodies from the `data/*.json` records the frontmatter already cites) is the lever
that would close it without any parametric fallback at all. Artifact:
`agi-proof/benchmark-results/continual-qa.judged-hybrid-full92.json`.

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
