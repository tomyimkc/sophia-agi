# Teaching — and Honestly Measuring — a Transferable Source-Discipline Habit in a 4B Model

**Author:** tomyimkc · **Repository:** https://github.com/tomyimkc/sophia-agi · **License:** Apache-2.0
(code) / CC-BY-4.0 (this note). **Status:** narrow, corpus-bound, `candidate_only`. **Not**
market-beating, **not** a general LLM, **not** third-party validated, **not** AGI. `canClaimAGI: false`.

> This is a defensive public disclosure of the method and findings below, with a dated commit history
> as the record of authorship and priority. Cite as: tomyimkc, "Teaching and Honestly Measuring a
> Transferable Source-Discipline Habit in a 4B Model," sophia-agi, 2026.

## Abstract

We show that a small (4B) open model can be taught a genuine, *transferable* **source-discipline
habit** — qualify contested claims, refuse false attributions, keep distinct traditions distinct — by
supervised fine-tuning on **gate-passed** data, with factual truth enforced *externally* (by tools and
a gate) rather than memorised in weights. The contribution that generalises beyond this one model is a
**fail-closed train-and-measure contract** that refuses any claim an instrument cannot resolve. Three
findings: (1) the habit is real and *transfers to entities the model never trained on*; (2) against
three strong large models it is *modestly* better at source-discipline, but deterministic markers
**overstate** the gap and a semantic-judge panel cuts it to ~0.65; (3) the same training induced an
**over-qualification tax** (reflexive hedging on settled facts) which we then *measured and fixed* over
three honest iterations. The durable lesson: **the dominant risk in LLM work is mis-measuring the
model, not building it** — we nearly shipped a false "12-point forgetting" verdict from an
underpowered 34-item probe.

## 1. The architecture: truth outside the weights

```
   ground truth (corpus + tools)            weights (LoRA)
        │                                        │
   external GATE ── enforces truth ──►  practices HABIT (route-first, qualify, refuse, separate)
        │                                        │
   admission: teacher → gate → {accept | correct-abstain} → SFT   ·   {fabricate} → preference "rejected"
```

Only gate-passed rows become training targets; the weights learn *judgment/discipline*, never bare
facts. This is how a small model becomes reliable on contested questions without "knowing" everything.

## 2. The measurement contract (the reusable contribution)

A claim is admissible only with a machine-checked receipt proving all of:

1. **Pre-registration before training** + a GO/NO-GO gate (git-ancestry proves the spec predates the data).
2. **Power to the threshold** — a probe whose Minimum Detectable Effect exceeds the decision threshold
   *cannot test it*; we refuse a directional verdict when `MDE(N) > effect`.
3. **Uncertainty always; anytime-valid when you peek** — every number has a CI; metrics looked at
   repeatedly use a time-uniform confidence sequence (fixed-n CIs lie under optional stopping).
4. **Triangulate ≥2 *independent* constructs** — deterministic markers, an LLM-judge panel, behavioural
   transfer, and (here) third-party fact-checkers. Two scorers of the same family are not corroboration.
5. **External validity on novel entities** — to separate a *habit* from a *memorised format*.
6. **Decontaminate automatically** — exact + content-shingle disjointness, re-checked independently.
7. **Volume is corpus-bound** — headline *records*, not templated rows.
8. **Simplest recipe first, then measure.**

Each pillar is a deterministic CI gate (`tools/claim_gate.py`, `eval_stats.py`, `assert_decontam.py`,
`lint_training_rows.py`, `lint_claims.py`). See [`measurement-thesis.md`](measurement-thesis.md).

## 3. Results (with honest bounds)

| Finding | Evidence | Bound |
|---|---|---|
| Source-discipline is **learnable** by 4B | qualification **+0.475** (3-seed), tradition-merge +0.143, false-attr +0.014 — CI-clean | marker primary |
| It **transfers** to novel entities | 160 held-out novel-entity cases: qualification **+0.432**, `claim_gate` GO | single seed |
| **No catastrophic forgetting** | powered **N=970** probe: Δ **−0.001**, anytime-valid CS [−0.030,+0.028] | single seed |
| **Modestly** better than strong large models | vs Grok 4.3 / DeepSeek V3.1 / Mistral-large; edge **survives** the same scaffold; 3-family judge majority **0.646** | markers overstate; first-party frontier egress-blocked |
| It carried an **over-qualification tax**, since fixed | over-hedged settled cases (protected_history 0.81 vs base 0.14); fixed in 3 iterations → **0.81→0.00**, generalising to novel entities | single seed |
| Third-party **fact-check corroboration** (one slice) | Google Fact Check Tools API: **6/15** pop-myth claims have a published fact-check, *all* rating them false; **0/3** on the source-discipline core | fact-checkers don't cover authorship — the gap is the point |

**The instructive failures.** (a) A "12-point forgetting" verdict came from an N=34 probe whose
MDE (~0.34) could not resolve the 5-point criterion; powering it (N=970) showed Δ−0.001 — *the fix was
a better instrument, not a model change*. (b) The over-qualification fix's v2 "calibrated" verdict was
a **measurement artifact** — a saturated-cases average masked a regression visible only on
disaggregation. (c) Independent fact-checkers corroborate the corpus's *myths* but provide **zero**
coverage of authorship/provenance — quantifying exactly why a source-discipline layer is needed.

## 4. What this is not (the ceiling)

Narrow, single-base, single-seed on the headline axes, corpus-bound, `candidate_only`. The deterministic
markers reward the trained format (the judge is the semantic cross-check). The frontier comparison omits
GPT/Claude/Gemini (egress-blocked from the test environment). No market, product, or AGI claim is made.

## 5. Reproducibility

All code, gate-passed data, eval artifacts, and receipts are in the repository; the model is reproducible
from the committed data + code + seed (the adapter weights are deliberately not shipped — the work measured
*behaviour*, not a product). Run `make claim-check` to re-run the full contract.

## 6. Related work (selected)

Adding error bars to evals (Miller, arXiv:2411.00640); anytime-valid inference / confidence sequences
(Howard et al. 2021; Ramdas et al., arXiv:2210.01948); construct validity for ML benchmarks
(arXiv:2510.23191); benchmark contamination (arXiv:2406.04244); LLM-as-judge bias (arXiv:2406.07791);
Gwet's AC1 vs the κ prevalence paradox.

## 7. How to cite / priority

This note + its dated commit history establish public disclosure and authorship priority. For an
academic DOI, archive a tagged release via Zenodo (GitHub→Zenodo integration mints a citable DOI) or
submit to arXiv. **Note:** public disclosure is a defensive publication — it creates prior art (others
cannot patent these findings) but may also forfeit your own patent rights outside any grace period;
consult a patent attorney *before* publishing if patenting matters to you.
