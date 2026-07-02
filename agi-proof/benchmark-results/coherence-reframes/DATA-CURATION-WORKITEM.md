# Work item — curated sophia-domain grounding pack (prerequisite to closing the qualifier)

Status: **SPEC only** (not built). This is the single prerequisite the coherence→entailment→retrieval
arc identified: to close the sophia-domain qualifier ("retrieval-bound GO holds on FEVER only"), the
verifier line needs a pack the current material cannot supply. Building it is a **data-curation
decision** (effort + scope), not another verifier variant — hence a spec for maintainer sign-off.

## Why it's needed (what the arc pinned down)
Three constraints, each demonstrated with a gated result — see `ARC-SUMMARY.md` + the D/D2/D3 reports:
1. **Data scarcity.** Non-forge evidence-groundable sophia claims cap at n=106 (30 supported / 76
   refuted). MDE ~0.21 ≫ the +0.05 floor → underpowered by ~4×.
2. **Evidence too terse.** sophia's own wiki prose is ~3 sentences and often doesn't state the
   attribution clearly enough for the NLI cross-encoder to entail → NLI answerable-coverage only 0.43
   even with hedge-matched claims (D3). FEVER's full sentences → GO; terse prose → NO-GO.
3. **Circularity.** The wiki prose is derived from the same attribution records as the claims, so the
   supported class is near-tautological; the load-bearing signal is refuted-rejection only.

## The pack to build
- **Size / power:** ≥150 cases, roughly balanced supported/refuted, so the paired ΔF1 @ matched
  coverage gate can power the pre-registered +0.05 floor. **Pre-compute the MDE from the assembled pack
  and publish it before running** (target MDE ≤ 0.05; if not met, say so before running, per the standing rule).
- **Domain:** philosophy / religion / history attribution & provenance (sophia's core), where the
  no-overclaim + `doNotAttributeTo` discipline already defines gold.
- **Claims:** hedge-matched to the evidence epistemics ("X is the traditionally attributed author of Y"),
  both supported (canonical attribution) and refuted (`doNotAttributeTo` / sibling-author negatives).
  The binding scarcity is the **supported class** — curation must add real, clearly-attributable works
  beyond the current 30 records.
- **Evidence — the critical upgrade:** **INDEPENDENT, multi-sentence, fact-bearing** passages, NOT
  sophia's own wiki (breaks circularity + fixes terseness). Sources: external encyclopedic passages
  (e.g. Wikipedia lead paragraphs), scholarly abstracts, or reference-work entries — sealed +
  sha256-hashed at retrieval so both arms score identical pairs.
- **Provenance discipline:** treat `authorConfidence: legendary|compiled|none_extant` as its own tier;
  keep traditions separate (儒家 vs 道家); no attribution to a figure in `doNotAttributeTo`.
- **Forge:** a paraphrase-expanded arm may ride along **explicitly labeled internal-validity-only**
  (cases correlated → CI optimistic); it never substitutes for the non-forge primary.

## The gate it feeds (already frozen + implemented — reuse, don't re-spec)
`tools/nli_grounding_gate.py` / `nli_healthy_incumbent_D2.py` / `nli_hedge_matched_D3.py`:
frozen NLI verifier (`agent.nli_grounding.build_nli_entailment`) vs the **healthy semantic incumbent**
(`build_semantic_entailment`), through the real `fact_check_gate.external_ground`. Guards, unchanged:
- primary paired ΔF1 (NLI − semantic) @ matched coverage ≥ +0.05, 95% CI excluding 0, ≥3 seeds;
- incumbent-health (semantic coverage ≥ 0.10);
- **answerable-coverage** drop ≤ 0.01 (the guard D3's terse evidence failed — the load-bearing test);
- fail-closed on no-evidence; two-family κ ≥ 0.40; latency/cost per claim (escalation-tier economics).

## Pre-declared outcomes
- **GO** (ΔF1 ≥ +0.05 CI∌0, answerable-coverage guard passes, incumbent healthy, manipulation ≥50%):
  the sophia-domain qualifier is **closed** — entailment beats coherence on sophia's own domain with
  adequate evidence → propose the escalation-tier default via `evaluate_update` (protected suites +
  answerable-coverage guarded). First default-on candidate.
- **NO-GO** (healthy incumbent, manipulation passed, adequate power): the mechanism genuinely does not
  transfer to sophia's domain even with rich independent evidence → bank + close the line for good.
- **Under-powered / manipulation-failed:** say so before running; the constraint is still data.

## Cost / decision
This is human-or-curated data effort (assembling ≥150 clearly-attributable works with independent
sourced evidence), not compute. Fund only if closing the sophia-domain qualifier is worth that curation
cost; otherwise the arc's banked result stands (retrieval-bound GO on FEVER; sophia-domain characterized,
not closed). `candidateOnly` / `canClaimAGI=false` until the gate itself speaks.
