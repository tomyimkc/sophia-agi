# HANDOVER — formal-proofs arc (for the next AI session)

> **Purpose:** hand this entire document to the next AI session (Claude / Grok / ZCode /
> other) as the first thing it reads. It is self-contained: it states the project
> context, the strict style rules the user (Tom) requires, the exact current state of
> the formal-proofs workstream, the one open blocker, and the prioritized action plan.
> The next AI should not re-derive any of this — read, ground against the repo, then act.
>
> **Author of this handover:** GLM-5.2 (ZCode), 2026-06-27.
> **User:** Tom (tomyimkc). Address him as Tom. Sole author of the repo.
> **Repo:** https://github.com/tomyimkc/sophia-agi · **Branch:** work on a fresh branch
> off `origin/main` (see "workflow rules" below).

---

## 0. Non-negotiable style rules (follow in EVERY response)

- **Address the user as Tom.** Be critical and honest about AGI claims — no hype, no
  sugar-coating gaps. The thesis is *wisdom-before-intelligence*: provenance discipline,
  0% fabrication on traps, fail-closed abstention, epistemic integrity benchmarks.
- **Base every recommendation on the latest deployed repo state.** Verify against
  `origin/main` before asserting anything — multiple concurrent AI sessions edit this
  repo, so working-tree state is unreliable; always `git fetch origin main` and diff
  against `origin/main`, never the local working tree.
- **No-overclaim discipline is the core constraint.** `canClaimAGI` stays `false` for
  all formal-proofs work. Every result artifact carries `candidateOnly: true` and
  `level3Evidence: false` until a held-out eval clears the no-overclaim gate (≥3 seeds,
  95% CI excludes 0, contamination CLEAN, `lint_claims` OK, failure-ledger entry).
- **Forbidden framing** in any committed artifact / report / PR / README: "wisdom-
  defining", "no frontier lab produces this", "the only system", "world-first",
  Millennium-problem result framing, or any Level-ladder claim — until a held-out result
  exists AND the gate is green. `lint_claims.py` will be extended to enforce this; do
  not write these phrases even as aspiration.
- **Workflow rules** (a concurrent-session hazard bit Tom hard; see
  `docs/12-Setup/Concurrent-Sessions-Worktrees.md`): do ALL work in a dedicated
  `git worktree` off `origin/main`, keep the main checkout clean, never `--force` push
  (use `--force-with-lease`), and rebase onto current `origin/main` before merging.

---

## 1. What the formal-proofs arc is (the thesis-level goal)

Sophia's roofline result bounds every output to (train ∪ retrieved). Formal proof is
the ONE domain where novelty is reachable *under* that ceiling: a Lean-verified proof is
self-certifying. The arc builds the open AlphaProof-class stack (LeanDojo + ReProver-
style search + Lean verification) into Sophia's existing verifier-gated, fail-closed
architecture, measured against a held-out benchmark (miniF2F-v2).

**The heterodox framing Tom has endorsed (but which must NOT appear in committed
artifacts until gated):** the Millennium/open problems are an *abstention* demonstration,
not a solve-target. A rigorous, machine-checked "I cannot prove this from {these
tactics}" is the designed output on open problems. That reframing is the long-term
direction; it only becomes publishable once it sits on a real held-out eval, not the
smoke set.

---

## 2. Exact current state on `origin/main` (verified 2026-06-27)

### Infrastructure — LANDED and CI-exercised

| Component | File | Status |
|---|---|---|
| One-shot Lean verifier (fail-closed, optional-dep) | `agent/lean_verifier.py` | ✅ merged (#140) |
| Self-extend flywheel bridge (kernel = held-out oracle) | `selfextend/proof_verifier.py` | ✅ merged (#140) |
| Smoke eval split + harness (contamination guard) | `formal_proofs/eval/*.jsonl`, `tools/run_formal_proofs_eval.py` | ✅ merged (#140) |
| **lean-dojo 4.x API fix + `check_proof_in_repo`** | `agent/lean_backend.py`, `agent/proof_search.py` | ✅ merged (#175) |
| Tactic-search proposer (ReProver/AlphaProof class) | `agent/proof_search.py`, `agent/tactic_proposer.py` | ✅ pre-existing, CI-exercised |
| Real-kernel CI lane (one-shot) | `.github/workflows/lean-kernel.yml` `lean-kernel` job | ✅ merged (#146) |
| lean-dojo-search CI lane (stateful search) | `.github/workflows/lean-kernel.yml` `lean-dojo-search` job | ✅ merged (#166, #175) |
| Spark lane → manual dispatch (clean green signal) | `.github/workflows/spark-smoke.yml` | ✅ merged (#151) |

### Design docs — LANDED, approved

- `docs/06-Roadmap/Formal-Proofs-Eval-Design.md` — the held-out eval spec. miniF2F-v2,
  traced-project-keyed, proposer B (tactic-search) primary + A (one-shot) baseline.
  Read §6 (all 4 questions answered) and §7 (Phase-0 done, Phase-1 plan) first.
- `docs/06-Roadmap/Informative-Abstention-Design.md` — gated BEHIND the eval. Diagnostic
  labels on `held` verdicts; never a headline.
- `docs/06-Roadmap/Lean-L0-Trace-Deadlock.md` — **the open blocker** (see §3 below).

### The one open blocker — `trace()` deadlock (READ THIS FIRST)

`docs/06-Roadmap/Lean-L0-Trace-Deadlock.md` documents a **reproduced, deterministic**
blocker: `lean_dojo.trace()` on a minimal local lake fixture **deadlocks** at 1515/1518
of the Lean prelude extraction on Tom's Mac (arm64). 0% CPU, no exception, no trace dir.
Reproduces on two independent fixtures (a `trivial_true` theorem AND a def-only fixture),
so it is NOT proof-extraction-specific — it is a systematic stall in the tracer's final
stage. Because `check_proof_in_repo` (#175) resolves a `Theorem` against a *traced*
repo, this deadlock blocks the entire L0 step ("one real Lean green check") locally.

**The deadlock is NOT a code bug in Sophia.** `lean_available()` stays `True` (the
import probe is honest), and `verify_proof`/`check_proof_in_repo` honor the fail-closed
contract. It is documented, not papered over.

**The cheapest next signal (named in the doc, NOT yet run):** reproduce `trace()` in CI
on Linux (`ubuntu-latest`). If mathlib4 or lean4-example traces there, L0 is a
CI-only demonstration and the Mac deadlock is a platform limitation. **This is the
single highest-leverage unblocked action.**

---

## 3. The viability ladder (where L0 fits)

Per the critique doc and `Lean-L0-Trace-Deadlock.md` §0:

- **L0** — `lean-dojo` traces a bundled trivial proof and `check_proof_in_repo` returns
  `accepted`. First non-`candidateOnly` Lean artifact. **BLOCKED by the trace() deadlock
  locally; CI reproduction is the unblock.**
- **L1** — reproduce a known Mathlib proof (not novel, but verifies the search loop).
- **L2** — a novel lemma (not in the training corpus; passes the strict novelty probe).
- **L3** — namespace-disjoint novelty. Speculative until L0–L2.

Everything above L0 is speculative until L0 is demonstrated. Do not work on L1+ before
L0 unblocks.

---

## 4. Phase-1 action plan (the held-out miniF2F eval) — DO NOT START until L0 is green

This is the action plan the user asked to be handed off. It is **gated behind L0** and
behind Tom's preregistration sign-off. Sequence:

### Step 1 — Unblock L0 via CI trace reproduction (HIGHEST PRIORITY, unblocked now)
- In a fresh worktree off `origin/main`, extend the `lean-dojo-search` CI job (or add a
  new one) to call `lean_dojo.trace()` on a MINIMAL local fixture (the `trivial_true`
  theorem in a git-init'd lake project, lean-toolchain pinned to v4.20.0), then assert
  `check_proof_in_repo` returns `accepted` on a correct proof and `rejected` on a wrong
  one. The `tests/test_lean_dojo_check_proof.py` real-lean test already attempts this
  against `lean4-example` — it's the right harness; it just needs the trace to complete.
- **Success criterion:** the `lean-dojo-search` lane goes GREEN on main. That single
  green is the L0 plumbing proof. If it deadlocks in CI too (Linux), the blocker is
  deeper (lean-dojo version issue) and needs a lean-dojo issue + version pin
  investigation — surface to Tom, do not paper over.
- This produces **no eval number, no capability claim**. It's pure plumbing.

### Step 2 — Write the Phase-1 preregistration (BEFORE any eval run)
- File: `agi-proof/formal-proofs-curriculum/preregistration.json`, schema
  `sophia.preregistration.v1` (mirror the existing
  `agi-proof/sophia-math-code-curriculum/preregistration.json`).
- Must name: proposer = B (`search_proof` + `LeanProofSession`), baseline column = A
  (`check_proof_in_repo`), split = miniF2F-v2 `test` (244) pinned to a
  **facebookresearch/minif2f commit SHA**, the **traced miniF2F Lean project commit +
  CI cache plan** (load-bearing — verification is traced-repo-keyed), seeds ≥3,
  thresholds, contamination controls. Read `Formal-Proofs-Eval-Design.md` §4 + §7.
- **Tom must approve this file before any eval run.** Do not run anything until he signs off.

### Step 3 — Seal the held-out split (leakage firewall)
- `tools/seal_formal_proofs_heldout.py` — thin adaptation of the existing
  `tools/seal_math_code_heldout.py`. Seal the miniF2F-v2 `test` items to a public
  SHA-256 hash manifest + gitignored `private/` copy. Reuse the existing
  `tools/heldout_seal_guard.py` so no proposer reads sealed paths. Three controls
  (per the design doc §2): knowledge cutoff, sealed split, no `exact <library_lemma>`.

### Step 4 — Run the eval (only after Steps 1–3)
- Proposer harness consuming the EXISTING `search_proof` + `make_llm_proposer` (no new
  proposer code) + the EXISTING `close_loop_on_proofs` reward interface, resolving
  theorems via `Theorem(LeanGitRepo(...), ...)` against the traced project.
- ≥3 seeds. Report pass@1 (A and B columns) + abstention rate.
- **Null/abstention is a pre-registered VALID outcome** — record in the failure ledger,
  do not suppress. The first eval will very likely be a low pass rate + high abstention
  (the verifier doing its job), which is a GOOD result, not disappointing.

### Step 5 — Gate before publishing anything
- `lint_claims` OK, failure-ledger entry beside any success, 95% CI excludes 0, no
  banned framing (§0). `canClaimAGI=false`. Only THEN does any number appear in
  RESULTS.md or a PR title.

---

## 5. Open PRs / branches the next AI should be aware of (verify with `gh pr list`)

Multiple concurrent sessions work this repo. At handover time, relevant open items
included a Path-B lean-proof-setup branch (#174) and a faithfulness-probe-v3 branch
(#177). **Always `gh pr list --state open` and `git fetch origin` before starting** —
state moves fast and another session may have already done part of what you plan.

---

## 6. What NOT to do

- Do NOT merge or publish any result framing until the held-out eval (Step 4) clears the
  gate. Smoke-loop closure is machinery validation, not evidence.
- Do NOT rewrite `verify_proof` to "verify standalone snippets" — lean-dojo 4.x has no
  such API; the traced-repo-keyed `check_proof_in_repo` is the correct path (#175 settled
  this; do not regress it).
- Do NOT attempt the Mathlib trace on Tom's Mac — it deadlocks (§3). The path is CI.
- Do NOT use `--force` push, do NOT work in the main checkout, do NOT leave uncommitted
  work for another session to inherit (`docs/12-Setup/Concurrent-Sessions-Worktrees.md`).
- Do NOT ship a fragile green. If a lane is red, diagnose; don't relax the assertion.

---

## 7. First message to send the next AI (copy-paste)

> You are taking over the Sophia-AGI formal-proofs arc. Read
> `docs/06-Roadmap/HANDOVER-Formal-Proofs-Arc.md` first — it is the authoritative
> handover (context, style rules, exact state, the one blocker, the action plan). Then
> verify state against `origin/main` (the working tree is unreliable; concurrent
> sessions edit this repo). Your first action is Step 1 of §4: unblock L0 by
> reproducing the `lean_dojo.trace()` call in the `lean-dojo-search` CI lane on Linux —
> if it traces there, L0 is unblocked and the Mac deadlock is a platform limitation.
> Do not start Phase 1 (the miniF2F eval) until L0 is green AND Tom approves the
> preregistration. Address the user as Tom; be critical; `canClaimAGI=false` throughout.
