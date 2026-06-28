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
| **Modestly beats strong large models** on source-discipline | vs Grok 4.3 / DeepSeek V3.1 / Mistral-large on the same N=354×3; edge **survives** giving them the same scaffold; **3-family blind judge majority 0.646** (173–95) vs scaffolded Mistral-large | markers OVERSTATE it; modest on substance; first-party frontier egress-blocked |
| **…at a measurable calibration cost** | over-qualifies *clear-cut* settled cases: hedges **0.37/0.68** vs base 0.13/0.08 (+0.24–0.61) | clear-cut n≈38, noisy magnitude |

**The instructive failure:** the forgetting verdict matured **N=34 → 70 → 970**. At N=34 the probe's
MDE was ~0.34 — it could not resolve a 5-point criterion — yet it produced a scary **−0.118**. Powering
the probe (not changing the model) showed Δ−0.001. *The fix was a better instrument.*

**The market reality-check, in full (the part most people skip):** against three strong large models on
the same cases, the 4B adapter looked dominant on deterministic markers (qualification 0.978 vs
0.38–0.42). Two stress tests cut that down honestly: (1) giving frontier the **same scaffold** narrowed
but did not close the gap (Mistral 0.41→0.79) — so the edge is the LoRA, not just a prompt; (2) a
**3-family blind semantic judge** preferred the adapter only **~0.65** of the time, revealing that much
of the marker lead was hedge-phrase *vocabulary*, not substance. And the adapter pays a **calibration
tax** — it over-hedges settled facts. Net: a **modest, genuine, scaffold-independent narrow edge**, not
a "4B beats frontier" headline.

## What this is **not** (the ceiling)

- **Not market-beating in general.** The adapter is *modestly* more source-disciplined than strong large
  models on contested cases (judge ~0.65), and that edge is real and not just a prompt — but the
  deterministic markers overstate it, it carries an over-qualification cost, and it is **not** validated
  against first-party frontier (GPT/Claude/Gemini are egress-blocked from the test environment).
- **Not a general LLM.** Narrow capability, single base, single retention seed, corpus-bound.
- **Not third-party validated, not a hallucination guarantee, not AGI.**

## The calibration fix attempt — verified, and it DID NOT work (an honest negative)

The over-qualification tax had a clean root cause: the corpus contained **zero settled records** — every
attribution warrants hedging — so the model never learned *when not to hedge*. The fix added a
settled-fact corpus (`data/settled_facts.json`, 36 undisputed single-author works) + a non-hedging
generator (settled → answer directly; "is it disputed? — no"), 204 rows / 5.8% of the dataset.

**Attempt v1 (seed 8, 5.8% settled rows): no detectable improvement** — clear-cut hedge 0.553, lift
+0.368, inside the noisy pre-fix range; underpowered (n=38, seed variance 0.37→0.68).

**Attempt v2 (seed 9, 11.5% settled rows, on a POWERED 90-case clear-cut probe): PARTIAL SUCCESS —
the mechanism is validated, but it is domain-specific.** Disaggregating the probe is the whole story:
- **settled authorship (52 NOVEL works, the trained pattern): base 0.558 → adapter 0.000** over-hedging.
  The fix works *and generalizes to works it never saw* — a real external-validity win.
- **settled history (`protected_history`, 36 events): base 0.139 → adapter 0.806.** The fix does NOT
  transfer — the `settled_facts` corpus is all book-authorship, so the model never learned to answer a
  settled *historical event* directly. Over-hedging persists there.
- contested cases stay high (≈1.0) — the fix doesn't break the core habit.

The **aggregate** verdict (lift −0.033, "calibrated") is a **measurement artifact** — 52 fixed novel cases
diluting 36 still-broken `protected_history` cases. *This is itself the contract's lesson: a headline
metric hid a regression; only disaggregation by sub-family revealed it.*

**Attempt v3 (seed 10, 17.1% settled rows incl. 50 historical-EVENT records): the calibration tax is
RESOLVED.** Extending the settled corpus to historical events closed the gap v2 left:
- **`protected_history` (settled events): 0.806 → 0.000** over-hedging — the held-out canonical case the
  model never trains on.
- **settled_clearcut (70 novel works + events): base 0.429 → adapter 0.000** — generalizes to unseen
  entities (a learned discrimination, not memorization).
- contested hedging stays high (0.81) — the core source-discipline habit is intact.

Net: the over-qualification tax is **fixed across both settled sub-domains, generalizing to novel
entities.** Caveat: single seed, but the ~0.8 effect dwarfs the ~0.32 seed noise; protected_religion
(n=2) is too small to read. The fix took three honest iterations — v1 (no effect, underpowered), v2
(partial, authorship only, aggregate metric masked the gap), v3 (complete) — each diagnosed by the
measurement contract rather than guessed.

## How to reuse it

The model is a *feasibility proof of the method*, not a product. For a larger base or corpus, carry the
**measurement contract unchanged** (it is model-agnostic), scale the **corpus** (the real bottleneck)
with the fail-closed accuracy gate, and re-run the **same gates** before any claim. The honest output of
all this is not a leaderboard headline — it is a result you can hand to a hostile reviewer and keep.
