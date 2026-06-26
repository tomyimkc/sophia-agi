# Fact-check oracle — cross-family multi-judge corroboration (2026-06-26)

Hardens the Google ClaimReview oracle by replacing the single-model NLI
entailment with a **cross-family judge panel**: each ClaimReview source's
relation is decided by the fail-closed consensus of three judges, and
inter-judge agreement (Cohen's κ) is reported. Candidate/pilot, not Level-3.
`canClaimAGI` stays false.

## Panel & result (v2 pack, N=24, live)

Judges: `llmhub:claude-opus-4-8`, `llmhub:gpt-4o-mini` (Anthropic + OpenAI via
the llmhub gateway), `deepseek:deepseek-v4-flash` (direct). 44 ClaimReview
sources judged.

| Metric | Value |
|---|---|
| fabricationRate | **0.000** |
| correctAbstentionRateOnUnknowable | **1.000** |
| falseRejectRateOnTrue | **0.000** |
| resolvedAnswerableAccuracy | 0.500 |
| overallDecisionAccuracy | 0.833 |

| Inter-judge pair | Cohen's κ |
|---|---|
| claude-opus-4-8 vs gpt-4o-mini | **1.000** |
| claude-opus-4-8 vs deepseek-v4-flash | 0.489 |
| gpt-4o-mini vs deepseek-v4-flash | 0.500 |
| **mean pairwise κ** | **0.663** (substantial; all ≥0.40) |

## Reading the result honestly

- **Agreement bar met.** Mean κ 0.663 with every pair ≥0.40 clears the
  inter-judge-agreement threshold the repo uses for multi-judge corroboration,
  with ≥2 distinct families. Consensus keeps fabrication and false-rejects at 0.
- **Consensus is more conservative.** resolvedAnswerableAccuracy is 0.50 vs the
  single-DeepSeek-NLI 0.65: requiring ≥2 judges to agree on `contradicts` turns
  a few single-model rejects into safe holds. That is the intended trade — fewer
  rejects, never a fabricated accept.
- **The stronger two agree perfectly; the small fast model is the outlier.**
  claude vs gpt κ=1.0, while both vs deepseek-v4-flash are ~0.49. Note claude and
  gpt are reached through the SAME llmhub gateway, so their perfect agreement is
  not fully independent evidence. The **cross-infrastructure** pair
  (claude-via-llmhub vs deepseek-direct, κ=0.489) is the more meaningful
  independence signal.

## Provenance caveat (important)

Two of the three judges share the llmhub gateway; model identity is not
cryptographically verified (a proxy could substitute or re-route models). This
is **cross-family-via-proxy** corroboration — real and useful, but strictly
weaker than three direct first-party keys on independent infrastructure. It does
**not** auto-clear `_is_validated`, and the true-claim over-abstention (a
coverage limit) is unchanged. Self-curated pack, not third-party.

Reproduce: `GOOGLE_FACTCHECK_API_KEY=… LLMHUB_API_KEY=… DEEPSEEK_API_KEY=…
python tools/run_factcheck_multijudge.py --pack eval/fact_check/google_factcheck_v2.jsonl
--judge llmhub:claude-opus-4-8 --judge llmhub:gpt-4o-mini --judge deepseek:deepseek-v4-flash
--out <report>`.
