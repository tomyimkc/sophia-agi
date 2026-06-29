# Thesis: Can an AI-driven score system auto-approve everything that clears the safety gates?

**Status:** Research note / brainstorm. Not a capability claim. Nothing here is validated by
the measurement contract; treat every proposal as a design hypothesis to be gated, not a result.

**Scope question.** This repo already runs a 15-gate, fail-closed verification pipeline
(`lint_claims` → `assert_decontam` → `eval_stats`/`claim_gate` GO/NO-GO receipt →
`promote_adapter` invariant suite → published-results label). The question this note explores:
*can we put an AI-driven scoring layer on top that auto-approves any artifact which passes all
gates, with no human in the loop — and is that wise on a path toward very capable systems?*

The short answer is **yes within a bounded envelope, no in general, and the boundary itself is the
whole design problem.** The rest of this note argues why, and then brainstorms concrete mechanisms.

---

## 1. Reframing the question precisely

"Auto-approve everything that passes the gates" hides three separate claims. Pull them apart:

1. **Decision automation** — *given* a verdict, let a machine apply the promotion instead of a
   human clicking approve. This is cheap, already mostly true here (CI is the approver), and not
   where the risk lives.
2. **Verdict automation** — let an AI *produce* the verdict (the score) rather than a deterministic
   tool. This is the hard part: an AI judging AI.
3. **Gate sufficiency** — the belief that "passed all current gates" ⟹ "actually safe/correct."
   This is the dangerous part: the gates are a *filter with residual*, never a guarantee.

The repo's existing doctrine already encodes the right instinct for (3): `lint_claims` forbids the
literal phrase that would assert a guarantee, precisely because the gate is "a filter (23.6%
residual)." An auto-approval score system must be built *around that residual*, not in denial of it.

---

## 2. What theory says is possible

### 2.1 The verification–generation asymmetry (the reason this can work at all)

For many tasks, *checking* a candidate is dramatically cheaper and more reliable than *producing*
one. Checking a satisfying assignment, a Lean proof, a unit-test pass, a citation against a source
snapshot — all are near-deterministic and far cheaper than the search that found them. Wherever this
asymmetry holds, an automated approver is sound *because it is not doing the hard part*; it is only
re-running a cheap, trusted check on the artifact the generator already produced. The repo's
`godel_oracle` (Z3-checked decidable invariants) and `grounded_gate` (offline Wikidata resolution)
are exactly this regime: the verdict is a re-derivation, not an opinion.

**Design rule:** auto-approval is legitimate precisely to the degree the check is *independent of and
cheaper than* the generation. Catalogue every claim type by its verify/generate cost ratio and only
automate the favourable ones.

### 2.2 Scalable oversight (the reason it can extend beyond cheap checks)

For claims too big to check directly, the literature offers decomposition: break a hard claim into
sub-claims each individually checkable (recursive reward modeling / iterated amplification), or pit
two systems against each other so a weaker judge only has to evaluate the *disagreement* (debate),
or train a strong verifier from weak-but-trusted supervision (weak-to-strong generalization). The
repo's `claim_router` already does the first move — it decomposes an answer into atomic claims and
routes each to a specialised verifier rather than scoring the whole blob. That is the seed of
scalable oversight; the open research is *verifying the decomposition itself*.

### 2.3 Proof-carrying artifacts

Borrow proof-carrying-code: require every submission to ship a machine-checkable **certificate**, so
the approver runs a tiny, audited checker rather than reasoning about the artifact. Verification cost
collapses and the trusted base shrinks to the checker. The repo's pre-registration-via-git-history
(`claim_gate --assert-prereg`) and measurement receipts are embryonic certificates. Pushing more
claim types into certificate form is the single highest-leverage move for safe automation.

---

## 3. What theory says is impossible (the hard limits — read these before scaling)

These are not engineering gaps to be closed; they are walls.

1. **Rice's theorem.** Every non-trivial *semantic* property of arbitrary programs is undecidable.
   So a fully general, sound, complete, automatic verifier for "is this artifact safe/correct" cannot
   exist. Auto-approval can only be sound on *restricted, decidable, or empirically-bounded* domains.
   Outside them, abstain — do not approve.
2. **Soundness vs completeness.** For any non-trivial gate you choose at most one of: never approves a
   bad thing (sound) / never rejects a good thing (complete). For a safety gate, **choose soundness
   and eat the friction** of false rejections. A complete-but-unsound auto-approver is a liability.
3. **Self-reference (Gödelian).** A system cannot fully certify its own consistency from inside. An
   auto-approval pipeline cannot be its own ultimate ground of trust; it needs an *external anchor* —
   formal proof checkers small enough to audit, fresh human-generated tests, or physical reality.
4. **Goodhart's law / specification gaming.** Once a score becomes the *target* a capable optimizer is
   trained or selected against, it stops measuring what it measured. The more capable the generator,
   the harder it games a fixed scorer. This is the load-bearing danger on a path to very capable
   systems: **the auto-approval score must never become the objective of the system being judged**,
   or the gate becomes theatre. Keep generation and verification incentive-separated and keep the
   trust root out of the optimization loop.
5. **Verifier collapse / autophagy.** AI-judging-AI in a closed loop with no external ground truth
   drifts and degenerates (the model-collapse failure mode). A tether to fresh, non-AI signal is
   mandatory, not optional.

The honest synthesis: **automate the decidable and the cheaply-checkable; abstain-and-escalate on
everything else; and never let the artifact's author optimize against the very score that approves
it.**

---

## 4. Creative idea menu (mapped to what already exists here)

Grouped by function. Each idea names the closest existing hook so it is incremental, not greenfield.

### A. Make the score honest about itself

1. **Multi-dimensional score, never a scalar.** A single number is the most Goodhart-vulnerable
   object you can build. Emit a vector — correctness, calibration, novelty, decontam-distance,
   protected-floor margin, judge-agreement — and require *all* coordinates to clear their own
   threshold (AND, not weighted-sum). Weighted sums let a high score on one axis buy a failure on
   another. *Hook: the `claim_gate` already returns per-pillar `ok` booleans — keep them disjoint.*
2. **Calibrated, abstaining score with explicit uncertainty.** Auto-approve only when score is high
   **and** uncertainty is low **and** the item is in-distribution. Add **conformal prediction** to get
   a distribution-free guarantee on the false-approval *rate* (you choose ε; the method bounds the
   error at the cost of an abstention region). The abstention region is routed to humans, by design.
   *Hook: `eval_stats` already refuses a verdict when the probe is underpowered — generalise that
   "refuse-when-can't-resolve" stance to the auto-approver itself.*
3. **Capability-conditioned strictness.** Scale gate strength with generator strength. A stronger
   generator is a stronger adversary against a fixed gate, so the more capable the producer, the more
   adversarial verifiers, higher consensus thresholds, and larger held-out probes you require.

### B. Decorrelate the verifiers (the single biggest reliability lever)

4. **Diverse-failure ensembles.** Approve on consensus of verifiers chosen to fail *differently*:
   different model families, a deterministic/symbolic checker, a retrieval-grounded checker, and an
   LLM judge. Correlated agreement is near-worthless; **measure inter-verifier correlation and discount
   agreement between verifiers that share a failure mode.** *Hook: the contract already demands ≥2
   distinct-family constructs (Pillar 5) — extend "distinct family" to the approval ensemble.*
5. **Adversarial dual-net (prover/skeptic / debate).** One agent argues *for* approval; an independent
   one is rewarded only for finding a defect; auto-approve only when the skeptic concedes (or its best
   objection is itself refuted by a check). Prompt verifiers to *refute by default* and treat "couldn't
   refute" as the signal, not "looks good."
6. **Perspective-diverse verification.** When a claim can fail in several ways (wrong, unsafe, leaked
   from training, doesn't reproduce), assign each verifier a distinct lens rather than N identical
   reviewers. Diversity catches failure modes redundancy cannot.

### C. Make gaming structurally hard (anti-Goodhart)

7. **Secret / rotating / portfolio rubrics.** Do not expose the exact scoring function to the
   generator; rotate metrics and hold a portfolio so optimizing any one does not pass. *Hook: the
   private eval split + content-shingle decontam (`assert_decontam`) is the start of "the test is not
   visible to the trainee."*
8. **Freshness / provenance gate as a first-class score input.** A high correctness score is *void* if
   the evidence is not independent of training. Make decontam-distance a coordinate of the score, not a
   pre-filter, so a contaminated win simply cannot reach the approval threshold.
9. **Tripwires / canaries.** Seed the stream with known-good and known-bad planted items. If the
   auto-approver ever approves a planted bad item or rejects a planted good one, **trip a circuit
   breaker** that halts autonomy and reverts to human approval. This continuously tests the verifier,
   not just the artifacts.
10. **Mutation/differential testing of the gate itself.** Inject mutations into real artifacts and
    confirm the gate catches them; the gate's *sensitivity* becomes a measured, regression-tracked
    quantity. A gate whose catch-rate drifts down is itself a NO-GO.

### D. Bound autonomy and keep it revocable

11. **Trust envelope / graduated autonomy.** Auto-approve only inside a region of demonstrated
    competence (claim types × difficulty × stakes where the track record is audited). Expand the
    envelope only as evidence accrues; high-stakes/novel/out-of-envelope always escalates.
12. **Asymmetric thresholds.** Price a false-approval as far costlier than a false-rejection and tune
    thresholds to err toward escalation. "Unsafe until proven safe," not the reverse.
13. **Human-on-the-loop importance sampling.** Humans audit a *statistically chosen* sample (oversample
    high-stakes/novel) to get an unbiased, CI-bounded estimate of the auto-approver's real error rate.
    The approver's own accuracy is then itself a gated, published metric — measured with the same
    contract it enforces on everything else.
14. **Revocable approvals + shadow regression.** Approvals are not permanent. Re-run past approvals
    against new tests; if a previously-approved claim now fails, **auto-demote it and quarantine
    everything downstream that depended on it.** *Hook: the failure ledger is already a revocation
    record — wire demotion to it.*

### E. Shrink and externalise the trust root

15. **A tiny, immutable, human-legible trust kernel (a TCB for claims).** Everything above it can be
    AI, but every approval must ultimately reduce to checks this small kernel can run, and the kernel
    is **not** AI-optimized and **not** writable by the generator. The moment a capable system can edit
    its own gates, the gates stop meaning anything — so make them external and tamper-evident.
16. **Proof-carrying promotions.** Push every claim type you can into certificate form (Lean/SMT proof,
    reproducible script + fixed seed, citation against a frozen snapshot) so the approver verifies a
    certificate instead of forming a judgment. *Hook: `godel_oracle` + `--assert-prereg` are the first
    certificates; expand the set.*
17. **Tamper-evident, append-only approval ledger.** Every auto-approval logged with full evidence,
    reproducible end-to-end, replayable. Post-hoc auditability means a slipped-through bad approval is
    *traceable and fixable*, and the gate that missed it gets a new mutation test.

### F. Incentive and economic designs (more speculative)

18. **Skin-in-the-game staking.** Each approval stakes reputation/credit that is *slashed* on later
    disconfirmation. A verifier that approves a thing that later fails loses standing; this aligns the
    scorer with long-run truth rather than short-run pass-rate.
19. **Internal prediction market on claims.** Aggregate many independent estimators' bets on
    "will this claim survive re-test in N weeks?" as the score; markets resist single-point gaming and
    surface confidence directly. (Speculative; needs a real disconfirmation feed to settle bets.)
20. **Completeness critic as a standing role.** A final agent whose only job is "what did we *not*
    check — which modality, which claim, which source went unread?" Its findings become the next round
    of gates. Silent coverage gaps are how unsafe things pass; make "what's missing" an explicit output.

---

## 5. A concrete proposed architecture (incremental on this repo)

A staged "Auto-Approval Trust Score" (AATS) that bolts onto the existing pipeline:

```
artifact + self-carried certificate
  └─ existing lint/decontam/claim_gate  →  GO/NO-GO receipt        (unchanged, deterministic)
        └─ AATS score vector  (only computed on GO artifacts):
             correctness | calibration | decontam-distance | protected-margin |
             verifier-consensus(diverse families) | judge-agreement(κ) | uncertainty
        └─ DECISION:
             all-coordinates-clear AND in-envelope AND uncertainty-low
                 → AUTO-APPROVE (logged to append-only ledger, sampled for human audit)
             else → ESCALATE to human (the abstention region)
        └─ CONTINUOUS: canary tripwires, mutation tests of the gate,
                       shadow re-test of past approvals, importance-sampled human audit
                       → any anomaly trips the circuit breaker back to human-only
```

The key properties: the AI score *never replaces* the deterministic GO/NO-GO receipt — it only
decides, among already-GO artifacts, which are safe to promote *without* a human. The human audit
sample continuously measures the auto-approver's true error rate under the same contract. The trust
root (lint + decontam + receipt + canaries) stays deterministic and outside the optimization loop.

### Suggested first experiments (each gated, none claiming success in advance)

1. **Verify/generate cost census.** Classify existing claim types by cost ratio; the favourable ones
   are the initial auto-approval envelope. Pure measurement, no model needed.
2. **Diverse-ensemble κ study.** Add a second distinct-family verifier to one claim type; measure
   inter-verifier agreement and, crucially, *correlation of their errors* on planted defects.
3. **Conformal abstention calibration.** On a labelled set, fit a conformal threshold for a chosen
   false-approval ε and report the resulting abstention (escalation) rate — that rate is the price of
   the guarantee, and it should be published, not hidden.
4. **Canary circuit-breaker harness.** Seed known-bad items; verify the breaker trips and reverts to
   human-only. This is a safety mechanism, so it ships *before* any autonomy is enabled.

### Status: implemented offline (candidate-only)

All four are built and run deterministically with no model (`make aats-experiments`). They are
candidate-only machinery — the model-gated arms (two-model-family judges, real labeled outcome rows)
are *not* run, so nothing here is a validated capability. The harnesses:

| # | Experiment | Tool / module | Offline result on the shipped demo |
|---|---|---|---|
| 1 | Verify/generate cost census | `tools/verify_generate_census.py` | Envelope = arithmetic, authorship.temporal, authorship.provenance, legal, code (each MEASURED deterministic + offline + independent); `other` escalates |
| 2 | Diverse-ensemble agreement | `tools/ensemble_agreement_study.py` (core `evaluate_ensemble`) | Two real distinct-family authorship verifiers: error-correlation −0.25, bothWrong 0, so AND-consensus false-approval 0.0 vs 0.33 single |
| 3 | Conformal abstention calibration | `tools/aats_conformal_calibration.py` (drives `agent/conformal_gate.py`) | Price-of-guarantee curve; on the synthetic set no operating point meets ε=0.05 → escalate-everything (refuses rather than over-approves) |
| 4 | Canary circuit-breaker | `agent/auto_approval_breaker.py` + `tools/run_canary_breaker.py` | Latching breaker: sound approver stays armed; a leaky (rubber-stamp) approver trips it to human-only; fail-closed on corrupt state |

Tests: `tests/test_aats_experiments.py` (deterministic, dependency-free). These exercise the *machinery*;
promotion past candidate requires the model-gated arms under the standard measurement contract
(≥2 judge families, κ≥0.40, ≥3 seeds, CIs, decontam).

---

## 6. Bottom line

A verifying score that auto-approves is **achievable and worthwhile for the decidable and
cheaply-checkable fraction of work**, and this repo already has the right bones for it: fail-closed
gates, distinct-family triangulation, decontamination, pre-registration, a failure ledger, and a
refuse-when-underpowered stance. The way to extend it safely is not to make the scorer *smarter* but
to make it *narrower, more decorrelated, more abstaining, and more externally anchored* — and to
design from the start for the fact that a more capable generator is a more capable adversary against
any fixed gate. Full, general, unconditional auto-approval is provably out of reach (Rice, Goodhart,
self-reference); a bounded, revocable, audited trust envelope is not. Build the envelope, measure its
edges honestly, and let humans own everything outside it.

---

*Open questions to push next: (a) how to verify a claim *decomposition* itself, not just the leaves;
(b) whether staking/market mechanisms settle fast enough to be useful given slow disconfirmation;
(c) how large the human audit sample must be to keep the false-approval CI tight as throughput grows.*
