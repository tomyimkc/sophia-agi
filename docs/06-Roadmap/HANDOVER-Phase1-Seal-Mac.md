# HANDOVER — Phase-1 seal + traced-project (for an AI running on Tom's Mac)

**You are taking over a scoped, network-dependent slice of the formal-proofs arc.**
Read this whole file first, then ground against `origin/main` before doing anything.
Author of this handover: Claude (Opus 4.8) session, 2026-06-27. User: **Tom** — address
him as Tom.

Why you (the Mac) and not the cloud session that wrote this: the cloud environment's
egress proxy **denies external git** (403, org policy, not retryable), so it could not
clone `facebookresearch/minif2f` or any miniF2F Lean 4 port. Your Mac has external git,
so you can do the parts that need it. **One hard Mac constraint — see §"Mac trap".**

---

## 0. Non-negotiable style + discipline (every response)

- Address the user as **Tom**. Be critical and honest; no hype.
- **No overclaim. `canClaimAGI = false` throughout.** Everything stays `candidateOnly`.
  Forbidden framing in any committed artifact/PR/report: "wisdom-defining", "no frontier
  lab produces this", "world-first", Millennium-problem-solve framing, any Level-ladder
  claim. `python tools/lint_claims.py` must stay OK.
- **Do NOT run any eval and do NOT flip the preregistration `status` from `DRAFT` to
  `OPEN`.** Tom approves that. Your job is pre-run prep only.
- Ground against `origin/main` (concurrent sessions edit this repo): `git fetch origin
  main` and diff against it; the working tree is unreliable. Work on a branch; never
  `--force` (use `--force-with-lease`); rebase onto current origin before pushing.

## What is already DONE (do not redo)

- **L0 achieved** (real `check_proof` through the Lean kernel) — merged in #187. See
  `docs/06-Roadmap/Lean-L0-Trace-Deadlock.md` §1c. It used lean-dojo's *remote-cached*
  mathlib4 (`29dcec0…` @ Lean `v4.20.0`), check_proof in ~3 min.
- **Phase-1 preregistration DRAFT**: `agi-proof/formal-proofs-curriculum/preregistration.json`
  (schema `sophia.preregistration.v1`, `status: DRAFT`). Proposer B (`agent.proof_search`
  + `LeanProofSession`) primary, baseline A (`agent.lean_backend.check_proof_in_repo`).
- **Seal tooling built + tested**: `tools/seal_formal_proofs_heldout.py` +
  `tools/heldout_seal_guard.py` (extended) + `tests/test_formal_proofs_seal.py` (passes).
- **Source correction**: `facebookresearch/minif2f` is **Lean 3** — NOT the Lean 4 source.
  Corrected in the prereg + design doc Q3.

## Read these, in order

1. `docs/06-Roadmap/HANDOVER-Formal-Proofs-Arc.md` — the arc-level handover + style rules.
2. `docs/06-Roadmap/Formal-Proofs-Eval-Design.md` — the approved eval design (§2 firewall,
   §4 gate, §6 Q3 with the 2026-06-27 correction, §7 phases).
3. `agi-proof/formal-proofs-curriculum/preregistration.json` — esp. `split`,
   `tracedProject.cacheMatchRisk`, and `openChecklist`. This is your task list.
4. `docs/06-Roadmap/Lean-L0-Trace-Deadlock.md` §1 (the Mac deadlock) and §1c (L0 win).

---

## ⚠️ Mac trap — local `trace()` DEADLOCKS on this machine

`lean_dojo.trace()` deadlocks deterministically on Tom's Mac (arm64) — stalls at
~1515/1518 of prelude extraction, 0% CPU (Lean-L0-Trace-Deadlock.md §1). So:

- **DO NOT attempt local cold tracing of mathlib4 or any Lean project on the Mac.** It
  will hang. The remote-cache *download* path (what L0 used) is fine; local *extraction*
  is not.
- **The seal task (§1 below) needs NO tracing** — it is git clone + text extraction +
  SHA-256 hashing. Fully safe on the Mac.
- Any step that requires actually tracing a new mathlib4 commit must be delegated to
  Linux CI or a 32 GB Linux box — NOT done on the Mac.

---

## Your tasks (all pre-run; none flips status to OPEN)

### TASK 1 — Stage + seal the real miniF2F-v2 (Lean 4) test split  [needs external git; Mac-safe]

This fills `split.source` + `split.commitSha` and is the leakage firewall.

1. **Confirm the Lean 4 v2 source.** miniF2F-v2 is from *miniF2F-Lean Revisited*
   (arXiv 2511.03108, Ospanov & Farnia). Candidate Lean 4 repos:
   `yangky11/miniF2F-lean4` (lean-dojo author) and `rahul3613/miniF2F-lean4`. Determine
   which carries the **v2-corrected** statements (the Revisited authors' own release is
   authoritative; a generic port may be v1-era). Record the exact repo + commit you pick.
2. **Clone it** (shallow is fine) and locate the **test (244)** theorem statements.
   Older miniF2F aggregates them in `test.lean`/`valid.lean`; Lean 4 ports keep them under
   a `MiniF2F/` dir. You want the **statements only — NO proofs** (the proof is what the
   eval must find; sealing the proof would leak the answer).
3. **Write the sealed inputs** (gitignored — `private/` is in `.gitignore`):
   ```
   private/formal-proofs-heldout/source.json
       {"repo": "<url>", "commit": "<sha>", "split": "test",
        "paper": "miniF2F-Lean Revisited (arXiv 2511.03108)",
        "license": "Lean statements Apache-2.0; Metamath MIT"}
   private/formal-proofs-heldout/minif2f-v2-test.jsonl
       one JSON object per line, e.g.
       {"claim_id": "<id>", "proposition": "<informal or name>",
        "lean_statement": "theorem <id> ... : ... := by"}
   ```
4. **Seal**: `python tools/seal_formal_proofs_heldout.py`  → writes the committed
   hashes-only manifest `agi-proof/formal-proofs-curriculum/heldout-seal.manifest.json`
   and refreshes the gitignored private copy. Then `--check` must print OK.
5. **Commit ONLY the hash manifest** (the `private/` payload is gitignored and must
   stay out of git — verify with `git status` that no statements are staged). Update
   `preregistration.json` `split.source` + `split.commitSha` with the real values and
   tick that `openChecklist` item.
6. Sanity: `python -m pytest tests/test_formal_proofs_seal.py -q` and
   `python tools/lint_claims.py` both green.

### TASK 2 — Resolve `tracedProject.cacheMatchRisk`  [investigation; Mac-safe if no local trace]

L0 was cheap because it hit lean-dojo's *cached* mathlib4 (`29dcec0…` @ v4.20.0).
`yangky11/miniF2F-lean4` pins mathlib4 **`f897ebc…` @ v4.24.0** — a different commit, so
tracing miniF2F re-traces mathlib4 (~1 hr / 32 GB) unless that dep is also cached.
Pick ONE and record it in `tracedProject`:

- **(a)** Find/produce a miniF2F-v2 Lean 4 build whose mathlib4 dep matches a
  **lean-dojo-cached** commit (ideal: `29dcec0`/v4.20.0) → cheap CI trace like L0.
- **(b)** Trace the port's mathlib4 **once on a Linux 32 GB box** (NOT the Mac), persist
  `~/.cache/lean_dojo` via `actions/cache`, then per-run tracing only touches miniF2F.
  **READY-MADE:** dispatch `.github/workflows/formal-proofs-trace-cache.yml` (calls
  `tools/trace_minif2f_project.py`) with `runner_label` set to a ≥32 GB runner and
  `minif2f_commit` set — it traces + saves the cache (restore/save split so a partial
  trace can't poison it). Optional `verify_file`/`verify_theorem` run a no-fabrication
  sanity check (a deliberately wrong proof must NOT be accepted).
- **(c)** Confirm lean-dojo already publishes a remote cache for the port's mathlib4
  commit (download-only path, like L0). Check `https://dl.fbaipublicfiles.com/lean-dojo`.

Whichever you pick, fill `tracedProject.{leanProjectUrl, leanProjectCommit, leanToolchain}`.

### TASK 3 — Fill remaining `PIN-BEFORE-OPEN` anchors  [Tom decides the model]

- `proposer.primary.searchBudget.*` (beam width, max depth, tactic-call cap, per-problem
  timeout) — propose concrete values; a pass@1 is uninterpretable without them.
- `baseModel`, `recipe.modelSpec`, `recipe.knowledgeCutoff`, `recipe.leanToolchain` —
  **the model is Tom's call** (cost/contamination); ask him, don't pick a frontier API
  model unilaterally.

### TASK 4 — (optional, recommended) LeanDojo-integration shakeout  [NOT on the Mac]

`LeanProofSession.open()` has never run against a real lean-dojo install. The first
milestone is "B opens one real `LeanProofSession` and closes ≥1 miniF2F-v2 problem
end-to-end" — a *plumbing* proof, not a capability claim. Because it needs tracing, run
it on Linux CI / a 32 GB box, not the Mac. Do this only after Tasks 1–2.

---

## Definition of done (for this handover)

- Real miniF2F-v2 test split sealed; `split.{source,commitSha}` filled; `--check` OK;
  only the hash manifest committed (no statements in git).
- `tracedProject.cacheMatchRisk` resolved with a chosen option + filled
  `leanProjectUrl/commit/toolchain`.
- Search-budget anchors proposed; model anchors raised with Tom.
- `status` STILL `DRAFT`. Report to Tom what's left before he can flip it to `OPEN`.
- `lint_claims` OK; `canClaimAGI=false`; nothing run as an eval.

## Workflow / git

- Develop on a dedicated branch off current `origin/main` (the cloud session used
  `claude/formal-proofs-handover-f9zrh6`; rebase it onto `origin/main` first, or branch
  fresh). Commit with clear messages. Never `--force` (use `--force-with-lease`).
  Do NOT open a PR unless Tom asks.
- Keep the `private/` payload OUT of git (it is `.gitignore`d — confirm before every commit).
