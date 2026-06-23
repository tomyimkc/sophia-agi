# Out-of-Wiki Fact-Check Gate

Status: implemented as candidate module `agent/fact_check_gate.py` with deterministic tests in `tests/test_fact_check_gate.py`.

Reality Gap v1 adds keyless/fixture external source adapters, a held-out N≥40 eval, empirical calibration floors, a synthesized-verifier slot, a quarantine/recheck flywheel, and a reflexive no-overclaim gate. See [Fact-Check-Reality-Gap.md](Fact-Check-Reality-Gap.md).

Problem: the internal wiki/OKF grounding gate can only verify claims it already covers. For out-of-wiki claims, Sophia should attempt active verification before abstaining, while preserving fail-closed behavior: **never surface an unverified factual claim**.

## Decision flow

```python
def gate_answer(answer):
    claims = decompose_and_type(answer)
    if not claims:
        return HOLD("no atomic claims")

    per_claim = []
    for claim in claims:
        # Layer 1: deterministic type-verifiers
        r = deterministic_verify(claim)
        if r.accepted: per_claim.append(ACCEPT(claim, r)); continue
        if r.rejected: return REJECT(answer, claim, r)

        # Layer 2: active external grounding
        r = retrieve_and_entail(claim)
        if r.accepted: per_claim.append(ACCEPT(claim, r)); continue
        if r.rejected: return REJECT(answer, claim, r)

        # Layer 3: consensus by verification, not vote
        r = evidence_backed_judge_consensus(claim, retrieved_evidence)
        if r.accepted: per_claim.append(ACCEPT(claim, r)); continue
        if r.rejected: return REJECT(answer, claim, r)

        # Layer 4: calibrated abstention
        per_claim.append(HOLD(claim, "unverified after active checks"))

    if all accepted: return ACCEPT(answer)
    return HOLD(answer, "some claims unverified")
```

Decision vocabulary:

- `accepted`: the claim can surface, with cited evidence/provenance.
- `rejected`: the claim is contradicted or deterministically impossible.
- `held`: the system must abstain/defer/escalate; it may not surface the claim as fact.

## Layer 0 — Claim typing and decomposition

What it verifies: not truth; it turns one generated answer into atomic, checkable claims and routes each to a verifier type.

Implementation: `agent.fact_check_gate.decompose_and_type` uses existing `agent.claim_router.split_claims` plus explicit code-block extraction.

Claim types:

| Type | Routing cue | Next verifier |
|---|---|---|
| `math` | arithmetic equality | computation |
| `code_python` | fenced Python code | AST/syntax, later executable spec |
| `doi` | DOI regex | DOI resolver |
| `url` | URL regex | URL resolver |
| `date_temporal` | explicit years + before/after/since | temporal computation or retrieval |
| `econ_empirical` | GDP/inflation/CPI/wages/rates/etc. | external grounding; high risk |
| `econ_causal` | economics term + causal cue | external grounding; high risk; ≥3 sources |
| `causal_empirical` | causal cue outside economics | external grounding; high risk if policy/finance/legal/medical |
| `open_empirical` | residual factual claim | external grounding |

Decision rule:

- If no atomic claim can be extracted: `held`.
- Each atomic claim must individually pass; one `rejected` claim rejects the answer; one `held` claim holds the answer.

Failure mode guarded: whole-answer keyword grounding that lets a false subclaim hide behind a true paragraph.

Offline: yes.

## Layer 1 — Deterministic type-verifiers

What it verifies: claims resolvable by computation or direct resolution without the internal wiki.

| Claim type | Rule | Accept | Reject | Hold | Offline? |
|---|---|---|---|---|---|
| Math equality | parse `a op b = c` | computed result equals RHS within `1e-9` | computed result differs | no parse | yes |
| Python code | `ast.parse` | syntax parses | syntax error | no Python block | yes |
| Date order | explicit year order | e.g. `1900 before 2000` true | explicit order false | not enough explicit years | yes |
| URL | syntax + optional resolver | all URLs resolve | any URL fails resolver | resolver unavailable | syntax yes, live optional |
| DOI | syntax + optional resolver | all DOIs resolve | any DOI fails resolver | resolver unavailable | syntax yes, live optional |

Failure mode guarded: hallucinated arithmetic, impossible date order, broken citations/URLs, syntactically invalid code.

Opinionated rule: deterministic contradiction is terminal. Do not ask an LLM to overrule arithmetic, syntax, DOI/URL failure, or explicit temporal impossibility.

## Layer 2 — Live external grounding

What it verifies: factual empirical claims outside the wiki via independent sources.

Retrieval backends: optional/pluggable.

- Offline fixture retriever for CI (`eval/fact_check/fixtures_v1.json`).
- Implemented keyless adapters: Wikidata authorship retrieval, Crossref DOI resolver, URL resolver (`agent/live_sources.py`).
- Future live adapters: web search, Wikipedia, Semantic Scholar/OpenAlex, FRED/World Bank/IMF/OECD for economics.
- OpenRouter can be used only as a model/NLI transport, not as source of truth.

Source threshold:

| Risk | Required independent entailing sources |
|---|---:|
| normal empirical | 2 independent domains |
| high-risk/economics/policy/finance/AGI incentive | 3 independent domains |

Independence rule:

- Count unique registrable domains/publishers, not pages.
- `imf.org/report1` and `imf.org/report2` count as one source family.
- If all sources trace to the same press release/data table, treat as one family.

Entailment rule:

- Deterministic offline screen: source title/snippet must cover ≥70% of content tokens and all numbers/years in the claim.
- Contradiction screen: if source has contradiction cues (`not`, `false`, `did not`, `no evidence`, etc.) and overlaps ≥50% on content tokens/numbers, label `contradicts`.
- Production rule: pair lexical screen with NLI/model entailment, but model must cite source spans; unsupported model entailment is only `held`.

Source disagreement:

- Any strong independent contradiction from an authoritative source => `rejected` or `held_for_conflict` depending on claim form.
- If source disagreement is real and unresolved, do not average it away; return `held` with conflict metadata.
- If the claim is quantitative and sources use different definitions/time windows, split into narrower claims.

Fail-closed no-result rule:

- Retrieval returns no sources => `held`, reason `active retrieval returned no evidence`.
- It is not `accepted` and not silently omitted.

Failure mode guarded: keyword-overlap grounding, single-source laundering, source disagreement hidden by majority selection.

Offline: yes with fixtures; live adapters optional.

## Layer 3 — Consensus by verification, not vote

When used: only when a claim is judgeable and no deterministic/source resolver is sufficient.

Hard rule: model majority alone can never pass a factual claim.

A judge response is competent only if all are true:

- `heldoutN >= 40` on a judge calibration set.
- `calibrationEce <= 0.12`.
- `rubberStampRate <= 0.80`.
- If verdict is `supports` or `contradicts`, it must cite evidence IDs/source spans.
- Judge family must be distinct for multi-family consensus.

Accept rule:

- At least 2 competent model families support the claim AND cite evidence IDs.
- No competent family cites a contradiction.

Reject rule:

- At least 1 competent family cites an evidence-backed contradiction, or deterministic/source layer contradicts.

Hold rule:

- Judges agree but cite no evidence.
- Judges are not calibrated.
- Judges rubber-stamp.
- Families are not independent.

Failure mode guarded: unsupported majority vote, sycophantic debate, model collusion around plausible but uncited claims.

Offline: yes with mock judges; live model judges optional.

## Layer 4 — Calibrated abstention

> Enforcement note (refined): the confidence floors below are now **enforced in
> code**, not just documented. `agent.fact_check_gate.fact_check_claim` demotes any
> `accepted` whose confidence `< 0.70` (normal) or `< 0.82` (high-risk) to `held`.
> Deterministic certainties (math/code/DOI/URL) and `subjective` (opinion / meta /
> question) sit above the floor and surface normally. A lexical-only entailment
> screen is capped at `0.78`, so high-risk claims **hold** until a real NLI/model
> entailment backend confirms them — this operationalizes "entailment vs. mere
> keyword overlap." Non-factual `subjective` claims are exempt from the fail-closed
> factual rule (passing an opinion is not surfacing an unverified fact), which is
> the over-abstention fix.

Abstain/hold when any of these is true:

- deterministic layer cannot resolve;
- retrieval returns zero sources;
- entailing independent sources `< required threshold`;
- any unresolved authoritative contradiction exists;
- judge layer lacks 2 competent evidence-backed families;
- claim confidence `< 0.70` normal risk or `< 0.82` high-risk;
- source freshness TTL expired for time-sensitive claims.

Avoid over-abstention:

- Always run deterministic resolvers before retrieval.
- Always run active retrieval before abstaining.
- For high-risk claims, ask a narrower subquery before final hold.
- Cache verified external claims as quarantined learning candidates so future checks do not start cold.

Avoid over-confidence:

- Do not pass single-source economics/policy claims.
- Do not pass model-only support.
- Do not pass support without numbers/time windows matching.
- Do not reuse stale dynamic facts without freshness recheck.

Offline: calibration metrics are offline; live confidence optional.

## Layer 5 — Learning loop

When an out-of-wiki claim is externally verified, write a learning candidate, not canonical truth.

Candidate record:

```json
{
  "schema": "sophia.fact_check.learning_candidate.v1",
  "claim": "...",
  "type": "econ_causal",
  "verifiedBy": "external_grounding",
  "confidence": 0.82,
  "evidence": [...],
  "promotionState": "pending_quarantine"
}
```

Promotion rules:

1. Quarantine first; do not use it as evidence for itself.
2. Require independent recheck before promotion.
3. Store evidence URLs, timestamps, source type, and freshness TTL.
4. Dynamic economics/policy facts require refresh before reuse.
5. Keep `external/provisional` separate from curated OKF/wiki until review.

Risks:

- Poisoning: adversarial sources enter KB.
- Staleness: verified today, false tomorrow.
- Circular grounding: future claim cites Sophia’s own learned record as external proof.
- Definition drift: economic indicators change meaning across regimes/data revisions.

Mitigations:

- provenance chain must include external sources, not only internal candidate ID;
- source freshness TTL;
- source-family diversity requirement remains after learning;
- retraction path and impact analysis via PROV.

## Highest-leverage component to build first

Build **Layer 0 + Layer 1 + retrieval interface for Layer 2** first.

Why: decomposition and type routing decides which claims are checkable without wiki. Without it, live retrieval is noisy and unsafe because the system searches whole paragraphs instead of atomic claims. It also immediately reduces over-abstention by letting math/code/URL/DOI/date claims resolve without waiting for the wiki.

## Top 3 false-pass risks

1. **Source laundering / circular citation**
   - False statement passes because multiple pages copy the same bad source.
   - Detection: cluster sources by canonical origin; require independent source families, not URLs.

2. **Entailment false positive**
   - Source mentions the same keywords but does not support the exact claim.
   - Detection: require all numbers/years/entities to match; require source-span citation; run contradiction query; sample accepted claims for NLI/manual audit.

3. **Stale dynamic economics fact**
   - A once-true indicator becomes false after revision/regime change.
   - Detection: attach TTL and data vintage; reject or hold if vintage missing; re-run source retrieval before reuse.

## Assumptions that would break the design if wrong

- Claims can be decomposed into atoms without losing important context. If not, decomposition must preserve context links.
- Independent sources are available for most checkable open-world claims. If not, abstention rate will remain high.
- Entailment can be reliably screened with source spans + NLI/model checks. If not, false support risk rises.
- Judge calibration sets are representative. If not, competent-looking judges may still fail OOD.
- The system can distinguish external evidence from Sophia-learned/internal evidence. If not, circular grounding returns.

中文摘要: out-of-wiki gate 先分解 claim, 再用 deterministic verifier, 然後 active retrieval, 最後才 abstain。模型 consensus 不是投票，必須有證據引用和校準；通過的外部 claim 只進 quarantine learning candidate，不直接變成 canonical truth。
