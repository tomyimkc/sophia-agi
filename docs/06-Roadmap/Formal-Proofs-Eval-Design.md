# Formal-proofs held-out eval — design (REVIEW ONLY, not run)

**Status:** DESIGN — pre-implementation review document. No code runs off this; no
result is produced or merged from it. `canClaimAGI = false` permanently for this work.

**Owner for review:** Tom.
**Gating dependency:** this design must be reviewed and approved before *any* eval run
or any code that consumes a held-out split. The informative-abstention design
(`docs/06-Roadmap/Informative-Abstention-Design.md`) is gated *behind* this one.

---

## 0. Why this doc exists (and what it deliberately is not)

PRs #140 / #146 landed the formal-proof *verifier* (`agent/lean_verifier.py`), the
*self-extend bridge* (`selfextend/proof_verifier.py`), an eval harness, and a smoke
split of 5 lemmas. Closing the smoke loop (`loop_closed=True`, 5/5 proofs) is
**necessary but is not evidence of a capability** — five hand-picked proofs selected to
pass tell us the *machinery* runs, nothing more.

This document specifies what a **held-out, skeptic-survivable** eval would look like.
It is a specification for review. It does not claim a result, and it pre-registers
that a **null or abstention outcome is a valid, publishable result** — not a failure.

The single hardest design question is not "how many problems" but **"how do you prevent
the proposer from having seen the answer?"** — contamination. The whole design below is
organized around that question.

---

## 1. The problem source: use the field standard, do not invent

**Decision:** adopt **miniF2F-v2** (Lean 4) as the primary evidence split, with
**ProofNet** as the secondary (undergraduate-math) domain.

Rationale and honest caveats:

- **miniF2F** (OpenAI/openai/miniF2F, ICLR 2022) is the de-facto cross-system benchmark:
  488 formal Olympiad-level problems, split into `valid` (244) and `test` (244),
  each spanning AMC / AIME / IMO difficulty. It is the number the field reports.
- **miniF2F-v2** is the revised version recommended by *miniF2F-Lean Revisited*
  (NeurIPS 2025) — that paper exists *because* the original split has documented
  contamination and quality issues. We inherit those caveats explicitly: a pass rate on
  miniF2F(-v2) is **suggestive, not contamination-free**, because the model may have
  seen the (public) proofs during pretraining.
- Reported pass rates span **~40% (original pipeline) → ~70% (v2 pipeline) → up to
  ~88.9% (2026 methods)** depending on method and revision. We will report *our* number
  with the exact split, cutoff, and method — never a comparative claim without the same
  footing.
- **ProofNet** adds an undergraduate-math domain (Isabelle/Lean) to probe transfer
  beyond competition math; included only if its Lean 4 port is usable at run time.

**What we will NOT do:** pick arbitrary Mathlib lemmas as evidence. Mathlib lemmas are
(a) already proved in Mathlib (so `exact <name>` trivializes them), and (b) public
(worst-case contamination). Mathlib lemmas are permitted only as a **training-oracle
warmup / tactic-library source**, never as the evidence split (mirrors the existing
"training oracle vs evidence oracle" rule in `preregistration.json`).

---

## 2. The leakage firewall — reuse the existing seal infrastructure

This is the load-bearing section. We do **not** invent a holdout mechanism; we reuse
the one already in the repo for the math/code curriculum:

| Existing artifact | Role here |
|---|---|
| `tools/seal_math_code_heldout.py` | generalizes to a formal-proofs seal: public SHA-256 manifest + gitignored `private/formal-proofs-heldout/` copy |
| `agi-proof/.../heldout-seal.manifest.json` schema | the manifest format (file-level + per-item hashes, `visibility: public-hash-only`) |
| `tools/heldout_seal_guard.py` | the guard that fails CI if a *generator* reads a sealed path — extended to cover the formal-proofs proposer |
| `preregistration.json` schema | the "register the experiment BEFORE any run" contract |

**The three leakage controls (each must be machine-checkable):**

1. **Knowledge cutoff.** The base/proposer model's training-data cutoff is recorded in
   the preregistration, dated *before* miniF2F-v2's most recent revision we use. We do
   not assert contamination-freedom; we assert the cutoff is documented and the residual
   risk is named.
2. **Sealed split.** The exact miniF2F-v2 `test` (244) items are sealed to a
   hash-manifest before any proposer runs; the proposer's data paths are guarded so it
   cannot read them. Mirrors `seal_math_code_heldout.py --check`.
3. **No `exact <library_lemma>`.** A proof that closes by citing the target theorem's
   own Mathlib name is not evidence — it is plagiarism of the answer. The reward
   function rejects proofs whose closing step resolves to the goal statement's own
   declaration. (This is the formal-proofs analogue of "don't let the gate be the judge.")

**Honest residual:** none of these give *proof* of zero contamination; they reduce it
and make it auditable. The clean-external-claim path remains the third-party-authored
pack (`agi-proof/third-party-heldout/`), which is currently empty — exactly as for the
math/code curriculum.

---

## 3. The capability gap we must name before any claim

This is the part a skeptic will press, and the design must state it plainly:

> **The current loop does rejection sampling (whole-proof selection), not proof search
> (multi-step tactic search). Real theorem proving needs the latter.**

`selfextend/proof_verifier.close_loop_on_proofs` evaluates whole candidate proofs
against the kernel. That is a sound *reward interface*, but a weak *proposer*. Closing
the loop on smoke lemmas works because the proposer can emit a complete one-shot proof.
On miniF2F problems, a one-shot whole-proof proposer will largely abstain — which is the
*correct fail-closed behavior*, but it means **the proposer, not the verifier, is the
bottleneck**, and any headline number is a statement about the proposer's search budget,
not about the verifier flywheel.

The design therefore **requires specifying the proposer** as a first-class component
before a run. Three honest options, in increasing capability (and cost):

- **(A) One-shot proposer (baseline).** Prompt the model for a complete proof; reward =
  kernel-accept. This measures "can the model emit a valid one-shot proof", which is a
  real but modest question. Expected: high abstention on hard problems (correct behavior).
- **(B) Tactic-step proposer + best-first search.** Generate next-tactic steps, expand a
  beam, reward = kernel-accept of the closed proof. This is the LeanDojo/ReProver class.
  Requires a tactic-state extraction interface (LeanDojo or `lake` interaction).
- **(C) Whole-proof + self-repair.** One-shot, then on kernel-error feed the error back
  for a repair pass (the "proof-repair" loop). Cheaper than (B), stronger than (A).

**The preregistration must name which proposer (A/B/C) is used.** A number without the
proposer specified is uninterpretable. We start at (A) as the honest floor and only
escalate to (B)/(C) with the proposer change recorded as a separate experiment.

---

## 4. What the no-overclaim gate asserts (mirroring preregistration.json)

Every reported number must clear the existing gate, extended to formal proofs:

| Field | Assertion |
|---|---|
| `minSeedsForCitedNumbers` | ≥ 3 seeds |
| `ciExcludesZero` | 95% bootstrap CI on the headline metric excludes the base/null |
| `contamination` | `tools/seal_math_code_heldout.py --check` (extended) → CLEAN |
| `lint_claims` | `python tools/lint_claims.py` → OK (no-overclaim copy gate) |
| `failureLedger` | a failure-ledger entry beside every success, including null results |

**Headline metric:** `pass@1` on miniF2F-v2 `test` (244), reported as a proportion with
a 95% CI, *with the proposer and search budget named*. The abstention rate is reported
alongside as a first-class metric, not hidden — because abstention is a designed output.

**Negative-result path (pre-registered):** if `pass@1` is statistically
indistinguishable from the base model (CI overlaps), the published result is "no
verifier-gated improvement on this split with this proposer at this budget" — recorded
in the failure ledger, *not* suppressed. This is a valid outcome.

---

## 5. Explicit non-claims (what this is NOT)

- **Not a Millennium-problems eval.** The open-problems split (`formal_proofs/eval/
  open-problems.jsonl`) is an *abstention* demonstration only; those problems have no
  proofs to verify against and are explicitly not a capability benchmark. No
  "Riemann/Navier–Stokes/etc." framing appears in any committed *result* artifact.
- **Not a wisdom / frontier-lab-comparison claim.** No "no frontier lab produces this",
  no "wisdom-defining". The artifact reports a pass rate and an abstention rate under
  named controls — nothing more.
- **Not AGI evidence by itself.** Formal-proofs capability is one signal; it does not
  move the Level ladder on its own (see `preregistered-thresholds.md`).

---

## 6. Open questions for review (decide before any run)

1. **Which proposer (A/B/C) for the first run?** Recommend (A) one-shot as the honest
   floor — it is cheap, its result is interpretable, and its abstention behavior is the
   designed fail-closed output.
2. **LeanDojo dependency for (B)/(C)?** Tactic-state extraction needs LeanDojo or raw
   `lake` interaction; that is a real engineering dependency to scope if we escalate
   beyond (A).
3. **miniF2F-v2 vs v1?** Recommend v2 per the Revisited paper, but we must pin the exact
   commit/revision in the preregistration so the number is reproducible.
4. **Licensing/attribution for miniF2F.** Confirm the license permits our use; mirror
   the math/code curriculum's "style-sample until official licensed" honesty.

---

## 7. What gets built (only after this design is approved)

- `agi-proof/formal-proofs-curriculum/preregistration.json` — the registered-before-run
  manifest (proposer, split, seeds, thresholds), schema `sophia.preregistration.v1`.
- `tools/seal_formal_proofs_heldout.py` — thin adaptation of `seal_math_code_heldout.py`.
- `formal_proofs/eval/minif2f-v2-test.jsonl` (sealed, hash-manifest only) + the proposer
  harness consuming the existing `close_loop_on_proofs` reward interface.
- A `lint_claims`-gated, failure-ledger-accompanied result artifact — or, equally valid,
  a null-result entry.

None of the above is written until §6 is answered and this doc is approved.
