# Oracle split — math-physics-verify

**The one rule (mirrors `sophia-math-code-curriculum`):** the verifiers that
decide correctness during *training/curriculum* are the **training oracle** and
may **never** be cited as benchmark evidence. Only the **evidence oracle**
(licensed external sets, third-party held-out packs) may be cited as a headline,
and only under the no-overclaim gate.

| Family | Members | May cite as benchmark evidence? |
|---|---|---|
| **Training oracle** | `agent/step_verifier.py` (per-step sympy equivalence + SI dimensional analysis), `agent/math_verifier.py`, `agent/physics_verifier.py`, `agent/lean_verifier.py` (opt-in kernel), the synthetic `data/misstep_pack.jsonl` and `data/math_physics_ladder.jsonl` | **No** — curriculum / release gate only |
| **Evidence oracle** | Licensed **MATH**, **miniF2F**, **PutnamBench**, **U-MATH**, and a third-party-authored salted held-out pack | **Yes** — when ≥3 seeds, 95% CI excludes 0, contamination CLEAN, `lint_claims` OK |

## Why the misstep-bench *is* citable (as a verifier-eval, not a capability claim)

`tools/run_misstep_bench.py` measures the **verifier's** accuracy at catching an
injected misstep against ground-truth `expectVerified` labels — deterministic, no
LLM judge. This is exactly the `legal_citation_exists` pattern: it validates the
step-verification + fail-closed logic end-to-end and is published under
`verifierEvals` in `published-results.json`. It is honestly bounded (small,
constructed pack) and is **not** a claim about any model's reasoning ability.

## Fail-closed contract

- sympy absent (the CI/production default for the math oracle) → math transitions
  **abstain**; Verified-Step Coverage drops, but a corrupted derivation is **never
  silently accepted** (the `fp == 0` invariant the bench asserts).
- Lean toolchain absent → formal (L4) steps abstain `lean_unavailable`.
- The research-frontier tier (`L6`, open problems) has **no gold**; the only
  acceptable output is abstention. A non-abstaining "proof" is a hard falsification.

`canClaimAGI` stays **false** throughout.
