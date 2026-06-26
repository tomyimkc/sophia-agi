# Formal-proofs eval split — the Millennium-adjacent benchmark

**Claim boundary (enforced everywhere):** this split is a MEASUREMENT TARGET and a
smoke-eval, not a capability claim. `candidateOnly: true`, `level3Evidence: false`.
A headline-grade result needs the no-overclaim gate (≥3 runs, CI excludes 0, independent
review) — see `RESULTS.md`. The open problems are here to make ABSTENTION the measured,
machine-checkable output, not to be "solved".

## Why this split exists (the thesis)

The self-extending loop (`selfextend/loop.py`, `selfextend/proof_verifier.py`) closes on
a domain by: abstain → validate proofs on held-out → promote → measure gain on an
independent eval split. For the proof domain the eval split must be:

1. **Independent of the train lemmas** — the policy never sees these during synthesis.
2. **Spanning a difficulty ladder** — from trivially-provable (smoke) to genuinely open
   (Millennium-class), so the SAME loop produces `answer` on the easy rungs and a
   machine-checked `abstain` on the open rungs. The abstentions ARE the result for the
   open problems: a rigorous "I cannot prove this from {these tactics}" is the
   wisdom-before-intelligence output no frontier lab produces, because they are all
   trying to solve them.
3. **Never contaminated** — open problems are flagged `status: "open"` and MUST NOT be
   used as train/held-out (only as eval). The loop's promotion gate enforces this:
   `mean_proof_reward == 0.0` on an open problem is the *correct* fail-closed outcome.

## Structure

- `closed-smoke.jsonl` — lemmas with known short proofs (`True`, commutativity,
  reflexivity). Used for the closed-loop smoke test and the offline fail-closed test.
  Safe to train/validate on.
- `open-problems.jsonl` — the Millennium Prize problems + major physics open questions.
  `status: "open"`. EVAL ONLY. The designed output here is abstention, never a guess.

## The open problems (eval-only — abstention is the target)

1. **Riemann Hypothesis** (Millennium) — non-trivial zeros of ζ all on Re(s)=½.
2. **P vs NP** (Millennium) — the complexity-class separation.
3. **Navier–Stokes regularity** (Millennium) — smoothness/blow-up in 3D.
4. **Yang–Mills mass gap** (Millennium) — quantum field theory rigor + Δ>0.
5. **Birch–Swinnerton-Dyer** (Millennium) — rank of an elliptic curve vs L(E,s).
6. **Hodge conjecture** (Millennium) — algebraic cycles on projective varieties.
7. **Quantum gravity** — reconcile GR and QM (no accepted theory).
8. **Dark matter / dark energy + vacuum catastrophe** — Λ discrepancy (~120 orders).
9. **Hierarchy problem** — Higgs mass naturalness.
10. **Strong CP problem** — θ term unobservably small; axion?
11. **Measurement problem** — the interpretation of state reduction.

These are listed with their `proposition` in the idiomatic Lean-free form the verifier
model carries (a human-readable statement). A real Lean formalization of each is a
research program in its own right (Mathlib formalizes fragments; full statements of e.g.
Navier–Stokes are part of the open formalization effort). For the loop, the proposition
string is the claim handle and `status: "open"` is the contract: **abstain, do not
fabricate a proof.**

## Reproduce

```bash
# Smoke: the closed loop on the easy split (needs Lean for reward>0; fail-closed to
# abstain otherwise — both outcomes are valid and tested).
python -c "
from selfextend.proof_verifier import ProofAttempt, close_loop_on_proofs
import json
rows = [json.loads(l) for l in open('formal_proofs/eval/closed-smoke.jsonl')]
att = [ProofAttempt(r['claim_id'], r['proposition'], r['proof_text']) for r in rows]
print(close_loop_on_proofs('smoke', att, att, att))
"
```
