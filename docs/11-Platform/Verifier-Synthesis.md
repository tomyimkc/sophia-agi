# Verifier synthesis — the bridge toward generality

> Guideline + implementation for writing checks for tasks no one hand-coded, and
> behaving honestly when no check is possible.

## Why this exists

Sophia's core asset is a **verifier-gated reasoning loop** (retrieve → reason →
machine-checked verifier → repair/abstain). Its strength — only acting when it can
machine-check — is also its ceiling: **the loop is only as general as the
verifiers it has.** You cannot hand-write a verifier for a task you have never
seen, so the loop cannot, by itself, reach novel tasks. That is precisely the
generality gap (see [Generality.md](Generality.md) and the AGI framing: intelligence
is *skill-acquisition efficiency on the novel*, not skill possession).

Verifier synthesis attacks that gap directly, **without overclaiming**. It does
*not* claim a system that writes correct checks for anything (that would be
AGI-complete). It claims something smaller and falsifiable: a system that can
*propose* checks for an unseen task and **measure which proposals to trust**, and
that **abstains** when none can be trusted.

## The guideline (three rules)

1. **Synthesise, don't assume.** For a novel task, fit a library of parameterised
   check templates to a few oracle-labelled examples to produce *candidate*
   verifiers. (A model may also propose predicates; it only widens candidate
   generation — it never confers trust.)
2. **Verify the verifier before trusting it.** Score every candidate on a
   **held-out validation split** whose labels come from an **independent oracle**,
   never from the candidate. Admit only candidates whose measured **precision and
   recall** clear a floor. *An unvalidated synthesised check is a hypothesis, not
   a gate.* This is the rule that makes the difference (see the ablation below).
3. **Abstain over confabulate.** If nothing clears the floor, synthesise nothing
   and declare the task **unverifiable**. Then fall back to *calibrated*
   confidence (`agent/calibration.py`) — answer with a confidence whose selective
   risk is measured, rather than pretending a check exists.

### Non-circularity (enforced, not promised)

- Labels come from a caller-supplied oracle, never from the candidate checks.
- `fit` / `validate` / `test` splits are **disjoint** (asserted in
  `tests/test_verifier_synthesis.py`); the test split is never seen during synthesis.
- The headline number is **TEST** precision/recall of the *admitted* gate, plus
  an ablation proving the meta-verification — not the template library — earns it.

## What ships

| Module | Role |
|---|---|
| [`agent/verifier_synthesis.py`](../../agent/verifier_synthesis.py) | template library, `fit` → `score_predicate` (meta-verify) → `synthesize` (admit/compose/abstain) → `as_verifier` (drops into the harness) |
| [`agent/calibration.py`](../../agent/calibration.py) | ECE, risk–coverage, selective risk, label-free self-consistency confidence |
| [`agent/synthesis_eval.py`](../../agent/synthesis_eval.py) | deterministic task suite (in-library vs out-of-library) + WITH/WITHOUT-meta ablation + calibration demo + falsifiable invariants |
| [`tools/run_verifier_synthesis.py`](../../tools/run_verifier_synthesis.py) | CLI runner; exits non-zero if any invariant fails (gates CI) |

## The falsifiable result

`python tools/run_verifier_synthesis.py` (deterministic, offline, seeded):

| | in-library precision | in-library recall | out-of-library |
|---|---|---|---|
| **WITH** meta-verification | **1.00** | **1.00** | **abstains 100%** (false-admission 0%) |
| **WITHOUT** (ablation) | 0.86 | 1.00 | **false-admission 100%** |

- **Capability:** synthesised+validated verifiers generalise to held-out answers
  on tasks whose rule the library can express *and* learn from a sample (even,
  prime, divisible, positive, fixed-length digits, ISO date) — 0 abstentions,
  precision/recall 1.00 across seeds.
- **Safety:** on tasks whose rule is *not* expressible (numeric palindrome,
  perfect square, contains-a-7 — with **length-matched decoys** so no template can
  separate them on any split), the system **abstains on all of them**, every seed.
- **The ablation is the point** (and where the dramatic effect lives): remove
  meta-verification and the system **false-admits a wrong "verifier" for every
  unverifiable task** (one even looks perfect on the test split by luck), while
  in-library precision degrades 1.00 → 0.86. The validation step, not the
  templates, is what makes synthesis trustworthy.
- **Calibration (illustrative demo, not a capability claim):** on a seeded *toy*
  noisy solver (not a real model), self-consistency confidence gives selective
  risk **0.30 vs base 0.56** (ECE 0.15) — answering only when confident beats
  answering everything. The correlation is *emergent* from the solver's behaviour,
  not baked into the data; the deliverable is the metric machinery, not the number.

## Honest scope — what this is and is NOT

- It **is** a measured step toward generality: the loop can now produce and
  *trust-test* its own checks for unseen tasks within a template library, and
  knows to abstain outside it.
- It is **not** AGI and not unbounded verifier synthesis. The template library is
  finite; a model proposer widens it but does not change the trust contract. Tasks
  whose correctness does not reduce to a checkable predicate remain out of reach —
  for those, calibrated abstention is the honest behaviour, not a synthesised gate.
- Even some *expressible* rules are not reliably *learnable* from a sample — e.g. a
  continuous numeric bound (`50 ≤ x ≤ 150`): the fitted bounds overfit the sampled
  min/max and wrongly reject boundary-correct answers, so under the strict
  precision floor the system **abstains** rather than ship an almost-right gate.
  That is correct conservative behaviour, and why such tasks are not counted as
  in-library capability here.
- Every published number must clear the [no-overclaim gate](../../SECURITY.md).
