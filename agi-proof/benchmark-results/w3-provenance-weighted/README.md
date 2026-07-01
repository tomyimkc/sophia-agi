# W3 provenance-weighted training — local take-live (2026-07-01)

> **candidateOnly:true · level3Evidence:false · canClaimAGI:false · gateMet:false**
> The `w3-provenance-weighting-not-validated-vs-loo` ledger row **stays Open.** All three gate
> mechanisms fire strongly on a controlled surface — a strong *candidate*, not a cleared gate.
> Result + sha256: `w3-provenance-weighted-2026-07-01.candidate.json`.

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
