# Informative-abstention reports — design (REVIEW ONLY, gated behind Formal-Proofs-Eval-Design)

**Status:** DESIGN — pre-implementation. No code, no eval run, no committed result.
`canClaimAGI = false`.

**Hard dependency:** this design is **gated behind**
`docs/06-Roadmap/Formal-Proofs-Eval-Design.md`. Nothing here may be implemented until
the held-out split (miniF2F-v2) eval described there exists AND has cleared the
no-overclaim gate. Abstention reports are only meaningful on top of a real held-out
eval — abstention on hand-picked smoke lemmas proves nothing and will not be published.

---

## 0. What "informative abstention" means (concretely, no hype)

Today, when the formal-proof verifier cannot prove a proposition, it returns
`status: unprovable_here`, `verdict: held` — a *binary* "I have no proof here". That is
correct fail-closed behavior, but it is **uninformative**: it does not say *why* the
proof failed.

"Informative abstention" = annotating that `held` verdict with a **diagnostic**: which
specific tactic-library gap or proof-state obstruction blocked the proof. Concretely,
from a kernel error trace, classify the failure into a small, auditable taxonomy:

- **missing lemma** — the search needed a Mathlib lemma not in the tactic library
  (e.g. `error: unknown identifier ...` / `unknown theorem`).
- **type mismatch** — the goal's expected type doesn't unify with what a tactic
  produced (a unification failure, often the most common).
- **stuck rewrite / simp** — a normalization tactic (`rw`/`simp`) left residual goals.
- **timeout / search-budget** — no error; the proposer ran out of budget before closing.
- **genuine hard obstruction** — the goal is (locally) not provable from the available
  axioms; a model-of-the-obstruction would be required to claim this, and we likely
  cannot assert it — default to "search-budget / unknown".

This is a **diagnostic label**, not a claim about the *truth* of the proposition. It
does not assert "this problem is unprovable"; it asserts "the proposer failed here, and
the most-informative reason the kernel gave was X".

---

## 1. Why the gating matters (the honest constraint)

Abstention on **smoke lemmas** is meaningless: those lemmas were *selected to be
provable*, so abstaining on them would be a bug, not a signal. Abstention is only
meaningful on a **held-out set where some problems are genuinely hard** — i.e. miniF2F
items the proposer *should* sometimes fail. Therefore:

- The informative-abstention harness **requires** the held-out split from
  `Formal-Proofs-Eval-Design.md` to exist and be sealed.
- The taxonomy labels are computed **only over the held-out eval's abstention cases**,
  never over smoke or training data.
- No abstention report is published until the parent eval's pass-rate / abstention-rate
  numbers have cleared the no-overclaim gate. An abstention breakdown on top of
  un-validated numbers would inherit their invalidity.

---

## 2. What the diagnostic produces (artifact shape)

For each held-out problem where the verifier returned `held`, the report emits:

```json
{
  "claimId": "minif2f-aime-2003-...",
  "verdict": "held",
  "status": "unprovable_here",
  "abstentionDiagnosis": {
    "category": "type_mismatch",
    "kernelErrorTail": "type mismatch ... (truncated, full trace in <hash>.txt)",
    "lastSuccessfulTactic": "rw [add_comm]",
    "residualGoalCount": 2
  }
}
```

Plus an **aggregate** over the held-out abstention set: a histogram of categories.
That aggregate is the candidate "result" — and it is a *characterization of the
proposer's failure modes on this split*, nothing more.

---

## 3. What this is NOT (the framing guardrails)

These statements must **not** appear in any committed artifact, report, README, or PR
description produced by this work, regardless of how the abstention numbers look:

- "Wisdom-defining" / "the system's wisdom" / "wise".
- "No frontier lab produces this" / "uniquely" / "the only system that...".
- Framing the abstention breakdown as a proof of the *proposition's* unprovability.
- Framing it as a Millennium-problem result. The open-problems split is explicitly
  **out of scope** for the published artifact until a held-out eval exists *and* a
  reviewer asks for it.
- Any claim that moves the Level ladder. `canClaimAGI = false`, permanently for this
  work, regardless of the abstention-quality numbers.

The artifact reports: "on held-out split X, with proposer Y at budget Z, the verifier
abstained on N problems; of those, the diagnosed failure categories were {…}". That is
the entire claim. `lint_claims.py` will be extended to reject the phrases above in any
report file, the same way it already enforces no-overclaim copy.

---

## 4. The honest skeptic's questions (pre-answered)

- **"Is the diagnosis reliable?"** — Partially. The kernel error trace is ground truth
  for *what the kernel said*; mapping it to a category is heuristic and will have edge
  cases. We report the raw trace alongside the label, so a reviewer can audit. We do
  **not** claim the category labels are perfect.
- **"Couldn't you relabel failures to look good?"** — The taxonomy is fixed before the
  run (pre-registered), the raw traces are committed, and the aggregate is a simple
  count. A relabeling would be visible in the diff. This is the same auditability the
  provenance gate enforces elsewhere.
- **"What's the falsifier?"** — If a held-out problem is later proved (by any system),
  its prior "abstained" label is not invalidated (we never claimed unprovability), but
  the *category* assigned (e.g. "missing lemma") becomes checkable against the actual
  proof — a form of delayed validation.

---

## 5. Open questions for review — ANSWERED (design-refinement pass)

### Q1. Taxonomy granularity? → **5 categories is right; bias-control by construction**

The 5-category set (`missing_lemma` / `type_mismatch` / `stuck_rewrite` /
`search_budget` / `genuine_obstruction`) is the right resolution for a *first* report.
Two controls keep it honest:

- **A `unknown` / `unclassified` bucket is mandatory.** Anything the classifier can't
  map confidently lands there, not force-fit into a category. The aggregate reports the
  `unknown` fraction explicitly — a high `unknown` fraction is itself a finding (the
  taxonomy is too coarse) and is published, not hidden.
- **The classifier is rule-based (regex over the kernel error string), deterministic,
  unit-tested against fixed error-trace fixtures** — no model judges the category, so
  there's no "the model relabeled its own failures to look good" attack surface. A
  human can re-bucket any case from the committed raw trace; the rules + fixtures make
  re-bucketing auditable.
- **Bias risk is real but bounded:** the rules will reflect *this proposer's* common
  errors initially. We acknowledge that in the report ("categories reflect failure
  modes of proposer B at this budget; not a universal taxonomy of proof difficulty").

### Q2. LeanDojo for richer state? → **Dependency already declared; richer state is a Phase-2 upgrade**

Revised: `lean-dojo>=4.0` is **already** in `requirements-theorem.txt`, and
`agent/proof_search.LeanProofSession` already threads LeanDojo `proof_state` objects
(see parent design §6 correction). So:

- **Phase 1 diagnosis uses the kernel *error string*** (cheap, always available, from
  `lean_backend.verify_proof`). This gives the 5-category taxonomy above.
- **Phase 2 upgrade uses the *proof state at failure*** (goal + context), which
  `LeanProofSession` already captures per-node. This enables a richer diagnosis
  ("the search reached *this* subgoal and proposed no tactic that closed it") — but it
  is an *upgrade*, gated behind the held-out eval existing, not a prerequisite.

The richer-state path is therefore not "needs a new LeanDojo integration" — the
integration exists. It's "needs the eval to exist so the per-node states have somewhere
to be recorded."

### Q3. Should the aggregate ever be a headline? → **No. Confirmed.**

Stays a *diagnostic section* inside the held-out eval report. Headline = pass@1 (A and
B columns); abstention breakdown = supporting characterization with its `unknown`
fraction named. No abstention number is quoted standalone in any README, RESULTS.md,
or PR title — `lint_claims` enforces this.

---

## 6. What gets built (only after both this doc AND the parent eval are approved)

- `selfextend/abstention_diagnosis.py` — the kernel-error-trace → category classifier
  (deterministic, pure-Python, unit-tested against fixed error-trace fixtures).
- A diagnostic section in the held-out eval report (`tools/run_formal_proofs_eval.py`
  extension) emitting the per-problem labels + aggregate histogram.
- A `lint_claims` rule rejecting the banned phrases in §3 from any formal-proofs report.

Nothing here is implemented until §5 is answered, the parent design is approved, **and**
the held-out eval has a cleared result.
