# NLI production acceptance gate — pre-registered 2026-07-02 (before any run)

> Registers the acceptance test for promoting the NLI entailment verifier
> (`agent/nli_grounding.py`, on `feat/oscillatory-crosspollination`) from
> "validated on clean gold evidence" (FEVER AUROC 0.962 vs 0.650 coherence)
> to a default-on candidate inside `fact_check_gate` / `realtime_grounding`.
> candidateOnly:true / canClaimAGI:false regardless of outcome. Threshold
> changes after the first run constitute a NEW pre-registration.
> Unblocked by the py3.10 regex fix in agent/fact_check_gate.py (this branch).

## Setup
- C1 fact pack through the REAL retrieval pipeline. Retrieved evidence is
  snapshotted and sha256-sealed so both arms score identical claim/evidence pairs.
- Arms: incumbent lexical screen vs NLI-injected gate (EntailmentFn). Nothing
  else varies. Promotion branch must be a THIN cherry-pick off main
  (nli_grounding.py + regex fix + this harness), not the omnibus feature branch.

## Decision rule (all required)
1. PRIMARY: paired ΔF1 at MATCHED COVERAGE ≥ +0.05, 95% bootstrap CI over
   cases excluding 0; ≥3 seeds on any stochastic component. (The FEVER +0.31
   was specialist-on-gold-evidence; the floor is deliberately modest.)
2. ABSTENTION GUARD: answerable-coverage drop vs the lexical arm ≤ 0.01 —
   a "win" bought by sliding toward abstain-everything is a FAIL. Report
   selective risk / AURC alongside F1, never instead of it.
3. PROTECTED: religion/history suite regression ≤ 0.01 — hard reject.
4. FAIL-CLOSED PROPERTY: zero retrieved evidence ⇒ the gate abstains, never
   entails against nothing. Asserted in the harness, not assumed.
5. TWO FAMILIES: a second independent NLI family (different cross-encoder or
   LLM-NLI) scores the same pairs; verdict agreement kappa ≥ 0.4. Disagreements
   are dumped to a failure-taxonomy file (the seed corpus for boundary tests).
6. ECONOMICS: latency + cost per NLI call logged (sizes the escalation tier).

## Pre-declared outcomes
- GO: propose default-on through agent/continual_plasticity.evaluate_update,
  then the conformal/abstention integration (NLI as the ESCALATION tier for
  claims where the cheap screen is uncertain).
- NO-GO: ledger row with the numbers, then boundary stress tests (noisy /
  adversarial retrieval, multi-hop claims) to locate where entailment breaks.
Either way the coherence-arc negatives (~15 instruments; "coherence measures
confidence, not truth") remain first-class evidence in the failure ledger.
