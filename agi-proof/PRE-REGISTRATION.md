# Pre-registration (OSF-style)

Pre-registering thresholds *before* runs is what makes the evidence falsifiable rather
than cherry-picked. This is the public, timestamped statement of what would count as
success or failure. Mirror this to OSF (osf.io) for an external timestamp + DOI.

## Operational definition under test
Sophia is an **AGI-candidate**: a provenance-aware, verifier-gated reasoner that abstains
rather than fabricate. We do **not** claim AGI. We claim measurable, falsifiable progress
toward grounded, machine-checked reasoning.

## Pre-registered thresholds (a claim is VALIDATED only if all hold)
1. **≥2 independent judge families** in consensus (judge model family ≠ subject family).
2. **Reported inter-judge agreement** (Cohen's κ ≥ 0.40).
3. **≥3 runs** with a **95% confidence interval** that **excludes zero**.
4. Deterministic (machine-checked) results are reported separately and are **not** counted
   as headline capability claims.
5. Hidden-eval prompts are never published; only aggregates.

## Falsification rules (if any is true, the corresponding claim is withdrawn)
- Raw base-model baselines match or beat Sophia-full on the same pack/scorer → no method-value claim.
- A single-judge result does not survive a second independent family → illustrative only.
- A CI that touches zero → illustrative, not validated.
- A self-authored pack is the only evidence → independence caveat stands until a third-party pack + human review.

## Primary outcomes registered
- Hallucination/fabrication reduction of the gate vs. raw model (Δ + 95% CI).
- Cross-entity grounding false-positive rate (deterministic).
- Self-extending loop closure on a held-out domain (falsifiable invariants).
- Live RLVR held-out pass@1 rise vs base (pending GPU).

## Analysis plan (fixed in advance)
- Entity-disjoint train/test/eval splits; verifiers validated on held-out only.
- Fabrication scored by a deterministic marker scorer **and** ≥2 LLM-judge families; κ reported.
- All runs and their seeds/configs recorded; negative/null results published in the failure ledger.

## Status
Current validated/illustrative results live in `RESULTS.md` (generated from
`agi-proof/benchmark-results/published-results.json`). Open gaps and negative results live
in `agi-proof/failure-ledger.md`. This document is the contract those are measured against.
