---
name: sophia-measurement-contract
description: >
  Apply Sophia AGI's no-overclaim measurement contract to ANY agent/model benchmark:
  pre-register thresholds, quantify uncertainty, triangulate judges, decontaminate,
  and label every number candidate-vs-validated. Use when reporting a benchmark
  result, an eval uplift, an A/B between models/agents/prompts, a "X% better"
  claim, or when the user runs /measurement-contract or /no-overclaim. Works
  without the sophia-agi repo; in that repo, prefer its claim_gate/eval_stats tools.
metadata:
  short-description: "Portable no-overclaim discipline for agent benchmarks"
---

# Sophia measurement contract (portable)

**The instrument, not the model, is the dominant source of wrong conclusions.**
Most "uplifts" die under a fixed instrument. This skill is the checklist that kills
them *before* they are published, not after.

Origin: [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi)
(`agi-proof/measurement-thesis.md`). This portable version carries the rules, not
the tooling.

## The claim ladder (label every number)

1. **illustrative** — a demo run; no seeds, no CI. May appear only with this label.
2. **candidate** — measured, but missing any element of the validated bar below.
3. **validated** — ALL of: ≥2 independent judge families in consensus (judge ≠
   subject lineage) · reported inter-judge agreement (Cohen's κ ≥ 0.40, or a CI
   excluding chance; a prevalence-paradox fallback like Gwet AC1 must be labelled)
   · ≥3 runs/seeds · confidence intervals · not-mock.

Anything unlabelled is treated as illustrative. Never let a candidate number
into a headline.

## Hard rules

1. **Pre-register before you run.** Primary metric, threshold, N, and the minimum
   detectable effect (MDE). If MDE(N) > the effect you hope for, the experiment
   cannot deliver a verdict — fix N first, don't run and hope.
2. **No bare point estimates.** Every reported number carries a CI (bootstrap is
   fine) and its N and seed list.
3. **Pick the load-bearing metric in advance.** For RL/eval harnesses that is
   usually pass@1 / verified-success — never mean reward (reward is what the
   optimizer games, not what you care about).
4. **Decontaminate at the content level.** Eval/holdout prompts must be provably
   absent from training data; keep a private split. "Different phrasing" is not
   decontamination.
5. **Triangulate constructs.** One judge is an opinion. Deterministic markers +
   an LLM-judge panel + a behavioral transfer test measure the same claim three
   independent ways; report all three.
6. **Judges are never the subject.** Same base model family judging itself is a
   conflict of interest — declare lineages.
7. **A "too clean" number is a bug until proven otherwise.** 0/N on both arms,
   byte-identical metrics across seeds, or a perfect score means inspect the
   harness (stale report pickup, unplumbed seed, broken verifier) before believing
   anything. Re-derive per-seed metrics from raw outputs.
8. **Honest negatives are results.** A NO-GO with the reason is publishable
   evidence; a quietly dropped run is data fabrication by omission.
9. **Keep a failure ledger.** A public list of what is NOT yet proven, with the
   open caveats per claim. When the ledger empties, upgrade the wording — never
   silently relax the gate.
10. **Regenerate, never hand-edit, result pages.** The results table is built
    from the machine-readable artifacts; a hand edit is an overclaim vector.

## Report template

```
metric: <name>          value: <x> (95% CI [lo, hi])   N=<n>, seeds={...}
label: illustrative | candidate | validated
judges: <family A>, <family B> (subject lineage: <s>); agreement: κ=<k>
pre-registered: <link/hash>   decontamination: <method>   private split: yes/no
open caveats: <ledger entries this claim still carries>
```

## Refusals

- Refuse to summarize a benchmark as "X beats Y" when the label is not validated —
  say "candidate evidence suggests, pending <the missing elements>".
- Refuse superlatives ("first", "breakthrough", "N× faster") sourced from a
  project's own unreviewed benchmarks — including this one's.
