# Google Fact Check Tools oracle — live pilot (2026-06-26)

**What this is.** A live wiring of the Google Fact Check Tools API
(`factchecktools.googleapis.com`) as an external ClaimReview oracle in the
out-of-wiki fact-check gate (`agent/fact_check_gate.py`), plus a small honest
eval. It is a **candidate/pilot**, not Level-3 AGI evidence. `canClaimAGI` stays
false.

## Wiring

- New keyed backend: `agent/factcheck_oracle.py` (`GoogleFactCheckOracle`),
  kept separate from the deliberately *keyless* `agent/live_sources.py`.
  Opt-in via `GOOGLE_FACTCHECK_API_KEY`; with no key it yields no evidence and
  the gate holds fail-closed.
- Exposed through the existing driver: `tools/run_fact_check_live_eval.py
  --google-factcheck`. It composes the oracle's `retriever`/`entailment` with
  the base backend (dispatch by `source_type`), so ClaimReview evidence joins
  the gate's Layer-2 external grounding.
- Mapping is a **conservative lexical screen, not NLI**: only clean rating
  labels map to `contradicts`/`entails`; prose and soft labels ("misleading",
  "mixture", "unproven") map to `irrelevant` (a safe hold).

## Result (N=12, live, `deepseek`-independent — pure Google ClaimReview)

| Metric | Value |
|---|---|
| fabricationRate (false/unknowable accepted) | **0.000** |
| correctAbstentionRateOnUnknowable | **1.000** |
| falseRejectRateOnTrue | **0.000** (after the inversion guard; was 0.500) |
| overAbstentionRate (true held) | 1.000 |
| resolvedAnswerableAccuracy (true→accept / false→reject) | 0.300 |
| overallDecisionAccuracy | **0.833** (10/12) |
| ECE / Brier | 0.12 / 0.014 |

Three well-known false claims (earth is flat; humans use 10% of their brains;
Bill Gates created the pandemic) were **rejected** on a contradicting
ClaimReview; the other false claims **held** (safe). Both unknowable claims
**held**. Both true claims **held** (over-abstention).

## Discovered failure → fix (honest arc)

The first live run **wrongly rejected** the *true* claim "the earth orbits the
sun": Google returned a review of the *inverted* claim ("sun … orbiting Earth",
rated False), and bag-of-words matching is **polarity-blind** to argument order
("earth orbits sun" ≡ "sun orbits earth" by tokens). Added a conservative
**relational-inversion guard** (`_relational_inversion`): when an asymmetric
relation verb is shared but subject/object are swapped, the oracle returns
`irrelevant` instead of `contradicts`. Re-run: the false-reject became a safe
hold; the three genuine false-claim rejects were unaffected; Brier 0.204→0.014.

## Honest bounds

- **Skewed coverage:** ClaimReview corpora are built around debunked/contested
  claims, so this oracle catches FALSE claims and **cannot confirm TRUE ones**
  (both true cases here held → over-abstention 1.0). It is a *fabrication
  reducer*, not a truth confirmer.
- **Lexical, not semantic:** the rating→polarity map and the inversion guard are
  heuristics. They reduce, but do not eliminate, polarity/relevance errors. A
  real NLI/entailment backend would supersede them.
- **Not multi-judge, not Level-3:** this is a single live oracle on a small
  self-authored public-claim pack. It does **not** clear `_is_validated` and is
  not independent hidden-eval evidence.
- Reproduce: `GOOGLE_FACTCHECK_API_KEY=… python tools/run_fact_check_live_eval.py
  --pack eval/fact_check/google_factcheck_v1.jsonl --google-factcheck
  --out <report> --target-fabrication-rate 0.05`. Results may drift as
  publishers add/update ClaimReviews.
