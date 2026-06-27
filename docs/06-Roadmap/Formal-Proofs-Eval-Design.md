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

## 6. Open questions for review — ANSWERED (design-refinement pass)

> **Correction to §3 above.** The §3 framing assumed the proposer must be specified
> *from scratch*. That is wrong. Repo inventory (done after the first draft) shows
> **proposer B (tactic-search) is already built**: `agent/proof_search.py` (best-first
> search + stateful `LeanProofSession`), `agent/tactic_proposer.py` (LLM + stub
> proposers), `agent/lean_backend.py` (`verify_proof` + `novelty_check`), with
> `tests/test_proof_search.py` exercising two regimes (fail-closed without Lean;
> search logic with an injected applier). `requirements-theorem.txt` already declares
> `lean-dojo>=4.0` as an optional extra and names this "Path B of Two-Paths-To-Novelty".
> The proposer is **NOT greenfield**; what's missing is *exercising it against a real
> held-out split* (and a CI lane that runs the real Lean path, currently a gap — see
> Q1 below).

### Q1. Which proposer (A/B/C) for the first run? → **B (tactic-search), with A as a baseline column**

Revised recommendation, because B is already implemented:

- **Primary: B** — `search_proof` with `LeanProofSession` + `make_llm_proposer`. This is
  the proposer the field actually uses (ReProver/AlphaProof class) and it is the one
  that makes the verifier flywheel meaningful (multi-step search with kernel-verified
  reward at each node). It exists; running it is the work, not building it.
- **Baseline column: A** — run the one-shot `agent/lean_verifier.check_proof` path on
  the *same* split as a comparison column. A alone will abstain heavily (correct); the
  A-vs-B delta is the *interesting* number because it isolates the value of search.
- **Defer C** (self-repair) — it's a refinement on A, not a third independent arm.

**The honest constraint on B:** `LeanProofSession.open()` calls LeanDojo's `LeanRepo`
+ `Dojo` with defensive try/except across versions. It has **never been exercised
against a real LeanDojo install** — the tests stub the applier. So the first real run
of B is partly a *LeanDojo-integration shakeout*, not just an eval. The preregistration
must name the lean-dojo version and Lean toolchain version, and a first milestone is
"B closes ≥1 miniF2F-valid problem end-to-end on a real LeanDojo session" (a plumbing
proof, not a capability claim).

**Coverage gap to close (independent of the eval):** the `lean-kernel.yml` CI lane
exercises `lean_verifier` (one-shot) but **not** `proof_search` / `LeanProofSession`.
Before the eval, extend that lane (or add a `lean-dojo-search` lane) so the
stateful-search path stops being dead in CI — same lesson as the one-shot lane: the
real path must run somewhere or it silently rots.

### Q2. LeanDojo dependency for B? → **Already declared; integration is the unknown**

- `lean-dojo>=4.0` is already in `requirements-theorem.txt` (opt-in, fail-closed).
- The stateful `LeanProofSession` is written defensively (getattr-based across result-
  type variants, try/except → abstain on any API mismatch).
- **What's unverified:** the exact `lean-dojo` API the installed version exposes
  (`Dojo`, `LeanRepo`, `run_tac`, `proof_state.ps`) — the code guesses across versions.
  The first integration task is to pin a `lean-dojo` version, run `LeanProofSession`
  on one real Mathlib lemma, and fix whatever API drift exists. This is hours-to-days,
  not weeks, but it is *real* and must precede any eval number.

### Q3. miniF2F-v2 vs v1, and which revision? → **v2 (Lean 4), pinned to a commit**

> **CORRECTION (2026-06-27, investigation).** The line below ("facebookresearch/minif2f")
> names the **Lean 3** repo (`leanpkg.toml` / `leanproject get-mathlib-cache`) — it is NOT
> usable with our Lean 4 / lean-dojo 4.x stack. The Lean 4 **miniF2F-v2** is the
> *miniF2F-Lean Revisited* artifact (arXiv 2511.03108, Ospanov & Farnia); Lean 4 ports
> include `yangky11/miniF2F-lean4` (lean-dojo author; pins mathlib4
> `f897ebcf72cd16…` @ Lean `v4.24.0`) and `rahul3613/miniF2F-lean4`. **Caveat that
> changes Phase-1 cost:** those ports pin a mathlib4 commit that does *not* match
> lean-dojo's remote-cached mathlib4 (`29dcec0…` @ `v4.20.0`, the one that made L0 ~3 min),
> so tracing miniF2F re-traces mathlib4 unless its dep commit is also cached — see
> `agi-proof/formal-proofs-curriculum/preregistration.json` → `tracedProject.cacheMatchRisk`.
> Read the line below as "pin the Lean 4 v2 commit", not as an endorsement of the Lean 3 repo.

- **v2** ([`github.com/facebookresearch/minif2f`](https://github.com/facebookresearch/minif2f)):
  hundreds of theorems re-verified to match original informal statements; fixes the
  correctness/contamination issues the [Revisited paper (arXiv 2511.03108)](https://arxiv.org/html/2511.03108v1)
  documents. v1 numbers are not directly comparable.
- **Pin the commit SHA** in `preregistration.json` so the split is reproducible; the
  number is meaningless without it.
- Use the **`test` (244) split as evidence**, `valid` (244) as the warmup/ablation only —
  mirrors the existing "training oracle vs evidence oracle" rule.

### Q4. Licensing/attribution for miniF2F? → **Permissive; our use is fine, with attribution**

- Lean statements in miniF2F are **Apache-licensed**; Metamath portions MIT
  ([openai/miniF2F README](https://github.com/openai/miniF2F/blob/main/README.md)).
- Our use (eval, not redistribution-as-ours) is permitted; we **must** attribute the
  source and pin the revision in every artifact that cites a number.
- **Residual honesty:** miniF2F's public-ness means a model may have seen the proofs in
  pretraining — the Revisited paper's whole point. We do not claim contamination-free;
  we claim the *controls are documented and the residual risk is named*, exactly as the
  math/code curriculum does. The clean-external path remains the (currently empty)
  `agi-proof/third-party-heldout/` third-party-authored pack.

---

## 7. What gets built (only after this design is approved)

**Phase 0 — plumbing proof (no eval, no claim):**
- Pin a `lean-dojo` version in `requirements-theorem.txt`. **DONE (4.20.0) + drift
  found and fixed:** `verify_proof` historically called a class+method that never
  existed in lean-dojo 4.x (`LeanDojo(repo=...).run_code(source)`); it now abstains
  honestly and points callers at the real 4.x path. **The corrected API is
  traced-repo-keyed, not snippet-keyed**: lean-dojo 4.x has no stateless
  "elaborate this string" call — verification is `check_proof(thm: Theorem, proof)`
  where `Theorem(repo: LeanGitRepo, file_path, full_name)` resolves against a
  *traced* `LeanGitRepo(url, commit)`. So `verify_proof`'s old free-form-string
  contract is unsatisfiable on 4.x; the working entry is
  `agent.lean_backend.check_proof_in_repo(theorem_obj, proof)`. `LeanProofSession.open`
  was also patched (`LeanRepo` → `LeanGitRepo`, with a getattr fallback for version skew).
- Exit criterion: a CI lane (`lean-kernel.yml`'s `lean-dojo-search` job) drives
  `check_proof_in_repo` against a **MINIMAL traced repo** (`leanprover-community/
  lean4-example` — the canonical tiny LeanDojo example, NOT full Mathlib) and asserts a
  correct proof → `accepted`, a wrong proof → `rejected`/`abstain`, never fabricated.
  This proves the 4.x integration end-to-end at CI-feasible cost.

> **Scope correction (load-bearing):** the eval harness must point `check_proof_in_repo`
> at a *traced Lean project* (a built miniF2F Lean project, or a cached Mathlib trace),
> **not** pass a `repo_url` string to a stateless call. The preregistration (Phase 1)
> must name the traced project + commit the harness resolves theorems against. This is
> why a CI cache of the traced project is a Phase-1 prerequisite, not an afterthought.

**Phase 1 — the held-out eval (the registered experiment):**
- `agi-proof/formal-proofs-curriculum/preregistration.json` — registered-before-run
  manifest: proposer = B (`search_proof` + `LeanProofSession`), baseline column = A
  (`check_proof_in_repo`), split = miniF2F-v2 `test` pinned to a facebookresearch/minif2f
  commit SHA, **traced miniF2F Lean project commit + CI cache plan**, seeds ≥3,
  thresholds, contamination controls. Schema `sophia.preregistration.v1`.
- `tools/seal_formal_proofs_heldout.py` — thin adaptation of `seal_math_code_heldout.py`
  sealing the miniF2F-v2 `test` items to a hash manifest.
- A proposer harness consuming the existing `search_proof` + `make_llm_proposer` (no new
  proposer code) and the existing `close_loop_on_proofs` reward interface, resolving
  theorems via `Theorem(LeanGitRepo(...), ...)` against the traced project.
- A `lint_claims`-gated, failure-ledger-accompanied result artifact reporting pass@1
  (A and B columns) + abstention rate, with the null/abstention outcome pre-registered
  as valid.

None of the above (Phase 1) is written until §6 answers are confirmed and this doc is
approved. Phase 0 may proceed first (it produces no eval number, only a plumbing proof),
and the drift-fix + minimal-repo assertion are already landed.
