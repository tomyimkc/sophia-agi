# Trustworthiness Capstone — what the verification toolkit does, and does not, guarantee

This session built a layered independent-verification toolkit and ran every piece **live**.
The honest conclusion up front, because it is the whole point:

> **Trustworthiness is not "catch every fabrication." No single verifier does that — every one
> is coverage-bounded.** Trustworthiness is a *composition*: layered independent verifiers, each
> labelled with its independence and coverage; **fail-closed abstention** when no high-independence
> verifier can confirm a claim ("never vouch for the unverifiable"); and **honest reporting** of
> exactly what each layer catches and misses. That posture — not a magic catcher — is what makes
> the repo trustworthy. `canClaimAGI` stays false.

## The verification toolkit (all built + live-tested 2026-06-28)

| Layer | Module | Covers | Independence | Live result |
|---|---|---|---|---|
| Professional fact-check | `GoogleFactCheckBackend` | viral/general myths | **high** | 4/21 debunk-pack coverage (Great Wall, Edison…) |
| Provenance/authorship | `LiveFactBackend` (Wikidata/Crossref) | "X wrote Y" attributions | **high** | 4/4 authorship misattributions (Dickens/Shakespeare…) |
| **Citation existence** | `citation_existence_verifier` | fabricated *studies/citations* | **high** | 0% over-block; catches specific fake studies; 9.3% overall (model debunks; vague cites missed) |
| Layered core-claim | `layered_verifier` | routes to the best of the above | high→low | 19/21 verified-debunk (4 high-indep + 15 flagged) |
| Core-claim contamination | `core_claim_source_verifier` | verbose grounded answers | depends on refs | over-block 0%; catch coverage-bounded |
| Hybrid | `hybrid_source_verifier` | core-claim + authoritative oracles | high where covered | over-block 0%; auth-only catch 0/43 on this pack |
| Model-knowledge tail | (flagged) LLM judge | the long tail | **low** (flagged) | extends catch, can err (Mozart myth) |

## What the live runs actually proved

1. **Detection is solvable; verification is the hard part.** Swapping the keyword debunk-detector
   for an LLM/NLI one took debunk *detection* 0→100%. But *verifying* a verbose answer is the
   binding constraint (atomic all-claims is too strict; over-blocks/under-confirms).
2. **No free lunch in verification.** Catching contamination needs one of: an oracle that covers
   the claim (sparse), fail-closed strictness (over-blocks clean answers), or model knowledge
   (low independence). The Cluster C matrix and the hybrid's 0/43 authoritative coverage are this
   law in data.
3. **Strong models mostly self-correct.** Across the debunk and contamination packs, the strong
   answer model usually *debunks* the injected falsehood rather than repeating it — so there is
   often no contamination in the output to catch. High "catch" rates from fail-closed verifiers
   are partly over-abstention, not detection. The honest gate abstains only on genuinely
   unverifiable output (citation verifier: 9.3% catch, **0% over-block**).
4. **Independence must be reported, not assumed.** Self-judging flattered an earlier number
   (a withdrawn 70.6% → really 5.9%); a model-knowledge verdict (Mozart) can be wrong. Every
   verdict now carries an `independence` tier.

## What "totally trustworthy" actually requires (honest checklist)

- [x] **Layered independent verifiers** with per-verdict independence labels — built.
- [x] **Fail-closed abstention** — never vouch for an unverifiable citation/attribution — built.
- [x] **Honest coverage reporting** — each layer's catch/miss documented, negatives in the ledger.
- [x] **A correction discipline** — a wrong number (70.6%) was found and withdrawn this session.
- [ ] **Multi-run CIs + answer≠judge as the default protocol** on every headline (partially done;
      single-run live numbers remain candidates, not validated).
- [ ] **An attribution-swap verifier** — the citation verifier confirms a study *exists* but not
      that it says what is claimed; swaps of real works still need an entailment check.
- [ ] **Third-party replication** of the whole gate — the standing top-level gap (`canClaimAGI:false`).

## Bottom line
The repo is *more* trustworthy than at the start of this session: it now has a labelled, layered,
fail-closed verification toolkit and an honest, falsifiable account of each layer's limits — and
it caught and corrected one of its own over-claims. It is **not** "totally trustworthy" in the
sense of catching every fabrication; the honest framing — the one that *is* trustworthy — is that
the system abstains rather than vouch for what it cannot independently verify, and says so.
