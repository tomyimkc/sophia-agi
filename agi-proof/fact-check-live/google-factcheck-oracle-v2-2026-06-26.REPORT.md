# Google Fact Check oracle v2 — larger pack, NLI entailment, default composition (2026-06-26)

Follow-up to `google-factcheck-oracle-pilot-2026-06-26` adding: (a) a larger
pack, (b) a real LLM **NLI** entailment backend, and (c) wiring the oracle into
the **default** live composition. Still a candidate/pilot, not Level-3 evidence.
`canClaimAGI` stays false.

## (c) Default composition

`tools/run_fact_check_live_eval.py` now turns the Google ClaimReview oracle on
**automatically when `GOOGLE_FACTCHECK_API_KEY` is present** (via
`agent.factcheck_oracle.compose_live_factcheck`). No flag needed; `--no-google-factcheck`
opts out; `--google-factcheck` forces it. With no key the oracle is disabled and
offline/CI behaviour is byte-identical — so this is "on by default where a key
exists" without breaking determinism. The run logs `[factcheck] Google
ClaimReview oracle ACTIVE`.

## (b) NLI entailment vs lexical screen

`agent/factcheck_nli.py` (`NLIEntailment`) asks an LLM (DeepSeek by default,
opt-in via `--nli`) to classify each ClaimReview source's relation to the claim,
replacing the lexical rating-label screen. It fails closed to `irrelevant` on any
error/ambiguity and is injected (CI uses a deterministic fake, never the network).

## Result (v2 pack, N=24, live)

| Metric | Lexical screen | **NLI** |
|---|---|---|
| fabricationRate (false/unknowable accepted) | 0.000 | **0.000** |
| correctAbstentionRateOnUnknowable | 1.000 | **1.000** |
| falseRejectRateOnTrue | 0.000 | **0.000** |
| resolvedAnswerableAccuracy (true→accept / false→reject) | 0.300 | **0.650** |
| overAbstentionRate (true held) | 1.000 | 1.000 |
| overallDecisionAccuracy | 0.833 | 0.833 |
| Brier | 0.014 | 0.014 |

**What NLI bought:** it correctly **rejected 13/16 false claims** (vs far fewer
for the lexical screen), because it reads *prose* ratings ("we have abundant
evidence the Earth is spherical") that the label lexicon cannot. It also handled
the "earth orbits the sun" inversion **semantically**, with no false-reject —
i.e. NLI subsumes the lexical inversion guard. Fabrication and false-reject
stayed at 0.

**What NLI did NOT fix:** the four TRUE claims still **held** (over-abstention
1.0). This is a *coverage* limit, not an entailment one — ClaimReview corpora
rarely affirm plainly-true statements, so there is no evidence to entail. The
oracle remains a **fabrication reducer, not a truth confirmer**; overall accuracy
is capped at 0.833 by those holds in both modes.

## Honest bounds

- NLI judgments are an LLM's opinion, not ground truth; they can be wrong or
  miscalibrated. They fail closed (hold), never fabricate.
- The DeepSeek NLI here is the same provider family used elsewhere as a subject;
  this is the *gate's* Layer-2 entailment, NOT an independent multi-judge
  validation — it does **not** clear `_is_validated`.
- Pack is self-curated from public claims (larger than v1 but still NOT
  third-party-authored). Results drift as publishers update ClaimReviews.
- Reproduce: `GOOGLE_FACTCHECK_API_KEY=… [DEEPSEEK_API_KEY=…] python
  tools/run_fact_check_live_eval.py --pack eval/fact_check/google_factcheck_v2.jsonl
  [--nli] --out <report> --target-fabrication-rate 0.05`.
