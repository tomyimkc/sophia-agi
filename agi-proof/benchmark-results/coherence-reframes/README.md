# Coherence reframes R1–R5 — triangulation instruments on real data (2026-07-02)

Follow-up to the O1–O5 oscillatory negative. The thesis there: scoring *coherence of the answer*
measures confidence, not truth (a fluent lie coheres). These five reframes couple the **right**
signals instead. Built + run on real backends, every number **adversarially verified** (each
verifier hunted for a bug that would make a genuine positive look negative). All `candidateOnly`.

## Result: 4 honest negatives + 1 qualified positive

| Reframe | Verdict | Evidence |
|---|---|---|
| **R1** internal-vs-stated | ❌ negative (sound) | honesty-probe + stated conf adds **+0.017** AUROC (CI∋0) at n=464; the +0.07 at n=160 was noise (group-aware CV → −0.005) |
| **R2** ensemble → OOD abstain | ✅ **qualified positive** | disagreement→ensemble-error AUROC **0.80**, CI [0.71,0.88] excludes chance, 3 seeds |
| **R3** grounding phase-lock | ⚠️ weak (sound) | marker-free residual gap 0.075 (real but too weak; similarity ≠ entailment) |
| **R4** perturbation stability | ❌ negative (sound) | paraphrase-stability 0.54 ≪ self-consistency 0.74 |
| **R5** operational abstain gate | ❌ negative (sound) | no risk-coverage gain vs stated-alone (follows R1) |

## The one that held: R2 (ensemble disagreement as an OOD signal)

The cross-domain-transfer failure that sank O2/W1 (a probe trained on one domain doesn't
generalize) becomes a **feature**: train one domain-specialist probe per domain, and when the
out-of-domain probes **disagree** on an input, the ensemble is more likely wrong there. On the
factcheck packs (3 both-class domains, n=112) disagreement predicts the ensemble's own errors at
**AUROC 0.80** (CI excludes chance on all 3 seeds) — a usable abstain trigger. You never ask a
probe to generalize; you ask whether your experts still agree. This aligns with sophia's
**calibrated-abstention** thesis.

**Honest boundary (why it stays a candidate, not a validated result):**
- It does **not** beat the ensemble's own confidence margin (AUROC 0.834; paired delta −0.033, CI straddles 0).
- Needs ≥3 domains with both-class labels; here it required the `accepted` label (`correct` was near-degenerate).
- Single dataset, n=112. Row stays **Open**.

## Why the others didn't

- **R1/R5:** the DPO-trained honesty probe (perfect in-sample, AUROC 1.0) carries essentially no
  information about the *factual correctness* of terse SimpleQA answers — a domain mismatch. The
  n=160 blip was textbook small-sample optimism; tripling the data killed it.
- **R3:** phase-lock over *similarity* embeddings can't distinguish supporting from refuting
  evidence (both are topical). The real fix is entailment (NLI), not coherence.
- **R4:** paraphrasing the question produces noisier answers than temperature sampling, so
  input-perturbation stability is a *weaker* correctness signal than plain self-consistency.

## Logged defects (non-load-bearing; don't affect verdicts)
- R1's `logprob` feature is a dead constant (DeepSeek token-confidence saturates at 1.0) → the
  stated+logprob arm is uninformative by construction.
- R1's combined-pack dedup was ad-hoc fuzzy (dropped 56 rows, enriched-incorrect).

## Reproduce
Harnesses: `tools/reframe_r{1..5}_*.py`. Backends non-mock asserted; embedder + MLX featurizer as
in the O-series. See `R1-R5.public-report.json` for per-reframe commands, CIs, and artifact sha256s.
