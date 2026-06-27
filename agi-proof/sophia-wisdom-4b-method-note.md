# Sophia-Wisdom-4B: a method note on training — and *honestly measuring* — a source-discipline habit

**Status:** narrow, corpus-bound, `candidate_only`. **Not** market-beating, **not** a general LLM,
**not** validated by a third party, **not** AGI (`canClaimAGI: false`). This note leads with the part
that generalizes — the **measurement discipline** — and reports the model result inside its honest bounds.

## TL;DR

The headline contribution is **not the adapter** — it is a **fail-closed train-and-measure contract**
that refuses any claim an instrument cannot resolve. Applied to a 4B LoRA, it shows you can teach a
*transferable* source-discipline habit (qualify contested claims, refuse false attributions, keep
traditions distinct) — and, just as importantly, it **caught us almost shipping a false "12-point
forgetting" verdict** from an underpowered 34-item probe. The durable lesson: *the dominant risk in
LLM work is mismeasuring the model, not building it.*

## The architecture: truth outside the weights

```
   ground truth (corpus + tools)            weights (LoRA)
        │                                        │
   external GATE ── enforces truth ──►  practices HABIT (route-first, qualify, refuse, separate)
        │                                        │
   admission: teacher → gate → {accept | correct-abstain} → SFT   ·   {fabricate} → preference "rejected"
```

Facts are enforced *outside* the weights (the gate/tools); the weights learn only *judgment and
discipline*. That is how a small model becomes reliable on contested questions without "knowing"
everything.

## The measurement contract (the reusable part)

A claim is admissible only with a receipt proving all of:

1. **Pre-registration before training.** Criteria + a GO/NO-GO gate committed *before* the data
   (enforced by a git-ancestry check, not the honor system).
2. **Power to the threshold.** A probe whose Minimum Detectable Effect exceeds your decision
   threshold *cannot test it*. We refuse a directional verdict when `MDE(N) > effect`.
3. **Uncertainty, always — and anytime-valid when you peek.** Every number carries a CI; any metric
   iterated across runs uses a time-uniform confidence sequence (fixed-n CIs lie under optional stopping).
4. **Triangulate ≥2 *independent* constructs.** Deterministic markers + an LLM-judge panel +
   behavioral transfer. Two scorers of the *same* family are not corroboration.
5. **External validity on novel entities.** Test the habit on works/traditions never seen in
   training to separate a *habit* from a *memorized format*.
6. **Decontaminate automatically.** Exact + content-shingle disjointness between train and every eval
   surface, re-checked independently of the build.
7. **Volume is corpus-bound.** Headline the count of *ground-truth records*, not templated rows; flag
   row inflation.
8. **Simplest recipe first, then measure.** Rank recipes only on a *powered* axis with the simple
   baseline in the table.

Every pillar is a deterministic check that runs in CI (`tools/claim_gate.py`, `tools/eval_stats.py`,
`tools/assert_decontam.py`, `tools/lint_training_rows.py`, `tools/lint_claims.py`). See
[`measurement-thesis.md`](measurement-thesis.md).

## What the contract certified for Sophia-Wisdom-4B (with bounds)

| Finding | Evidence | Honest bound |
|---|---|---|
| Source-discipline is **learnable** by a 4B model | qualification **+0.475** (3-seed +0.475/+0.371/+0.383), tradition-merge +0.143, false-attribution +0.014 — all CI-clean | deterministic-marker primary; ~3.3k rows |
| Gains are **semantic**, not just format | 3-judge-family blind A/B unanimous **169–5**; Gwet AC1 0.68–0.79 | Cohen κ < 0.40 (prevalence paradox) — **not** formally validated |
| It is a **transferable habit** | transfer probe on **160 novel entities**: qualification **+0.432**, `claim_gate` GO | single seed; corpus-bound |
| **No catastrophic forgetting** | **powered N=970** probe: Δ **−0.001**, fixed-n CI [−0.020,+0.018]; **anytime-valid** CS [−0.030,+0.028] | single seed |
| **ORPO does not beat SFT** | from-base ≈ coin-flip; on-SFT loses ~19% of the primary | small preference corpus |

**The instructive failure:** the forgetting verdict matured **N=34 → 70 → 970**. At N=34 the probe's
MDE was ~0.34 — it could not resolve a 5-point criterion — yet it produced a scary **−0.118**. Powering
the probe (not changing the model) showed Δ−0.001. *The fix was a better instrument.*

## What this is **not** (the ceiling)

- **Not market-beating.** The adapter is already *saturated* on its trained axis (qualification 0.978,
  false-attribution 0.000), so there is no headroom to "beat" frontier models that do source-discipline
  natively — the realistic head-to-head is **parity**, and that comparison is still pending
  (`tools/reality_check.py`).
- **Not a general LLM.** Narrow capability, single base, single retention seed, corpus-bound.
- **Not third-party validated, not a hallucination guarantee, not AGI.**

## How to reuse it

The model is a *feasibility proof of the method*, not a product. For a larger base or corpus, carry the
**measurement contract unchanged** (it is model-agnostic), scale the **corpus** (the real bottleneck)
with the fail-closed accuracy gate, and re-run the **same gates** before any claim. The honest output of
all this is not a leaderboard headline — it is a result you can hand to a hostile reviewer and keep.
