# MLOps Checkpoint Registry

`checkpoint-registry.json` records gated training candidates with a config hash,
eval-artifact references (with SHA-256), seeds, explicit promotion verdict, and
`canClaimAGI: false`. It was CREATED for W6 of the AGI gap-closure roadmap; the
directory did not previously exist.

## Boundary (THE ONE RULE)

Entries may reference **training-oracle / self-extension rung** passes (sympy/exec
verifiers, promotion gates). Those are curriculum gates only and are **never**
benchmark (MATH/GSM8K/HumanEval/Vectara/hidden-pack) evidence. Each entry carries
`evidenceOracleClaim: false` to make that explicit. `canClaimAGI` stays false.

## First entry

`math-rlvr-glm4-9b-3seed-n60-2026-06-25` — references the already-completed RLVR-math
3-seed N=60 artifact (`agi-proof/self-extension/math-rlvr-3seed-n60/`). No new GPU run
was performed for registration. Verdict: `promote` (self-extension rung gate cleared;
modest/narrow ~10% absolute gain on a judge-free held-out synthetic math set).
