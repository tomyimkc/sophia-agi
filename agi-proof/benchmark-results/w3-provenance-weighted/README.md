# W3 provenance-weighted training — local take-live (2026-07-01 → strengthened 2026-07-02)

> **UPDATE 2026-07-02 — strengthened with TRUE per-example loss weighting; row STAYS OPEN.**
> A custom MLX loop with genuine per-example weighted next-token CE (loss = Σ wᵢ·CEᵢ / Σ wᵢ,
> NOT replication) gives **weighted 1.0 vs uniform 0.45–0.55 correct across 2 seeds** (beats the
> replication proxy 0.95/0.50). Still Open: the "held-out suite" is the SAME 20 invented facts
> used in training (in-sample memorization of which source won, not held-out generalization),
> N=20, and influence is a model-free proxy (validated via LOO, not a real TracIn backend).
> Artifact: `w3-weighted-loss-2026-07-02.candidate.json`.

> **candidateOnly:true · gateMet:false** (original replication run, 2026-07-01). Result + sha256:
> `w3-provenance-weighted-2026-07-01.candidate.json`.

## Design (conflicting-provenance, verifier-checkable, decontaminated)

20 **invented** facts ("the capital of Zorbia is Xantel"). For each, a HIGH-trust `okf://`
source (real `rank_source` = 0.95) teaches the **correct** answer and a LOW-trust `model:`
source (0.20) teaches a **wrong** one. Loss weight = the real `agent.source_ranking.rank_source`,
realized as **replication** (mlx_lm has no per-example loss weight): weighted = 5:1
correct:wrong, uniform = 1:1. Two LoRA adapters trained; a third drops the poison for an
8-fact slice (leave-one-out).

## Results

| | correct (high-prov) | wrong (low-prov) |
|---|---|---|
| base (untrained) | 0.15 | 0.00 |
| **uniform** (1:1) | 0.50 | 0.50 |
| **provenance-weighted** (5:1) | **0.95** | 0.05 |

- **(a) weighted beats uniform:** 0.95 vs 0.50 correct (**+0.45**). Provenance-weighting steers
  learning to the trusted source.
- **(b) influence proxy agrees with LOO:** dropping the influence-implicated low-trust
  (`model:`) rows for an 8-fact slice flips that slice **0.25 → 0.75** correct, while the 12
  untouched (still-poisoned) facts stay ~stable (0.67 → 0.50, noise). The proxy fingers exactly
  the rows whose removal fixes the answer.
- **(c) no register collapse:** weighted diversity ratio **1.0**, 20/20 distinct answers.

## Verdict — gate NOT cleared (row stays Open)

All three mechanisms work end-to-end, but this is a **controlled synthetic conflict**: the eval
facts are the *trained* facts (in-sample "which source wins", not held-out generalization to
unseen facts); **N=20, single seed**; weighting is **replication** (a proxy for true per-example
loss weighting); and the influence proxy is simple (it fingers the `model:` sources by
construction). A strong demonstration of the mechanism, not a cleared general gate.

## To close it

A real held-out generalization suite (train the weighting *pattern*, test unseen conflicting
facts), ≥2 seeds, true per-example loss weighting (custom loop, not replication), and a real
TracIn/influence-function backend validated against full leave-one-out across the set.

## v3 (2026-07-02) — HELD-OUT GENERALIZATION on a rule task (addresses the core gap)

The earlier fact-conflict version was in-sample. This version uses a learnable **rule** (a number
is "zibbo" iff N>50) with clean (okf) vs **flipped** (model) labels, and tests on **unseen N**
(`w3-rulelearning-heldout-2026-07-02.candidate.json`). Provenance-weighted training **generalizes**:
weighted **0.834** vs uniform **0.500** (chance) mean over 2 seeds. Uniform is confused by the 50%
flipped labels; provenance-weighting (okf 0.95 ≫ model 0.20) recovers the true rule and it transfers
to held-out inputs — **genuine held-out generalization, not fact-memorization**. Remaining for
promotion: ≥3 seeds + more rule tasks + a real TracIn/influence backend, then sign-off. Row **Open**.
