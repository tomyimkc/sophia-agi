# Lean L0 — the `trace()` deadlock blocker (documented gap)

**Status:** evaluation, not a claim. Records a concrete, reproduced blocker to
the L0 viability-ladder step (critique §5.1: "get to L0 this month — one real
Lean green check"). This is **not** a capability statement; it is an honest
record of where the formal-proof path is *blocked*, so no advisor ships a
fragile green on top of it.
**Author:** GLM-5.2 (ZCode), 2026-06-27. Read alongside
`docs/06-Roadmap/Two-Paths-To-Novelty.md` and PR #175.

---

## 0. What L0 is, and why it matters

The viability ladder (critique §4) keys each level to *what has actually run*:

- **L0** — `lean-dojo` traces a bundled trivial proof in the local env and
  `check_proof_in_repo` returns `accepted`. The first non-`candidateOnly` Lean
  artifact. Everything above (L1 reproduce, L2 novel lemma, L3 namespace-disjoint
  novelty) is speculative until L0 is demonstrated.

PR #175 landed the *correct* 4.x API path: `verify_proof` now fail-closed
abstains on standalone snippets (4.x removed the old `run_code` API), and the
real verification path is `check_proof_in_repo(theorem_obj, proof)` over a
**traced** `LeanGitRepo`. That design is sound and disciplined. L0 then reduces
to one thing: **does `trace()` complete on a minimal local repo?**

## 1. The blocker — reproduced, deterministic

On this machine (Tom's Mac, arm64), `lean_dojo.trace()` on a minimal local lake
fixture **deadlocks** deterministically. It does not complete; it stalls at the
final proof-state extraction step with the tracer process at 0% CPU.

Reproduction (Python 3.12.13, Lean v4.20.0, lean-dojo 4.20.0 — the versions
lean-dojo itself pins in its own `lean-toolchain`):

```python
# minimal fixture: MinLean.lean = `theorem trivial_true : True := by trivial`,
# lakefile.toml = one `[[lean_lib]]` target, lean-toolchain = v4.20.0, git-init'd.
from lean_dojo import LeanGitRepo, trace
repo = LeanGitRepo.from_path("/path/to/fixture")   # local, git, one commit
traced = trace(repo, dst_dir="/path/to/trace")      # <- DEADLOCKS HERE
```

Observed behavior (cold trace, run under a tracked background process so the
shell timeout could not kill it):

- The tracer compiles its bundled `ExtractData.lean` against Lean v4.20.0 and
  begins extracting proof states. Progress bar iterates **1518 items** (this is
  Lean's prelude, not the fixture — same count regardless of fixture content).
- At **1515–1516/1518** the process **stalls**: `ps` shows `%CPU 0.0`,
  `STAT Ss` (sleeping), no further progress for 10+ minutes, no trace dir
  produced. No exception, no error message — a silent deadlock.
- **Reproduces on two independent fixtures**: (a) the `trivial_true` theorem
  fixture and (b) a *def-only* fixture (`def answer : Nat := 42`, no tactic
  proof at all). Both stall at the identical 1515/1518 boundary. So the hang is
  **not** proof-state-extraction-specific to my theorem — it is a systematic
  stall in the tracer's final extraction stage.

Because `check_proof_in_repo` (PR #175) resolves a `Theorem` against a traced
repo, and tracing does not complete, **the real 4.x verification path cannot
run on this machine today**. L0 is blocked at the trace step.

## 1a. Now reproduced on Linux / ubuntu-latest CI (2026-06-26) — NOT macOS-specific

The deadlock is **not** macOS-arm64-specific, and **not** a `from_path`-local-repo
artifact. Both of §4's leading unblock hypotheses (options 1–2: "use a real GitHub
repo instead of `from_path`" and "run on Linux") were exercised *together* by the
existing CI and **both are falsified as fixes**: the trace still hangs.

Evidence — `lean-kernel.yml` `lean-dojo-search` job on `main`@`fd2f59a` (the #175
merge), run `28270535053`, job `83766792273`, ubuntu-latest, Python 3.12,
Lean v4.20.0, lean-dojo 4.20.0:

- The job installs the Lean toolchain + lean-dojo, then runs
  `tests/test_lean_dojo_check_proof.py`, whose real-lean case
  `test_check_proof_in_repo_accepts_correct_rejects_wrong` traces
  **`leanprover-community/lean4-example`** — a *real GitHub* minimal repo fetched
  by `LeanGitRepo`, **not** a local `from_path` fixture.
- pytest reached **95%** (the two fail-closed cases PASSED) at `23:16:39`, then the
  real-trace case produced **no further output for 29m33s** until
  `##[error]The operation was canceled` at `23:46:10` — killed by the job's
  `timeout-minutes: 30`, *not* by concurrency-cancel (no superseding `lean-kernel`
  run exists on `main` after this one; the ~30-min duration is the timeout).
- At cleanup the runner had to `Terminate orphan process` a tree of hung **`lake`**
  and **`lean`** subprocesses (pids 2613/2616, 5630/5634, 5988/5992, 6596/6600)
  plus two `python` workers — i.e. lean-dojo's tracer had spawned its build/extract
  subprocess fan-out and **stalled with them alive at 0 progress**. Same fingerprint
  as the macOS stall in §1.

**Honesty bound:** this is **one** Linux CI run (§1's macOS result was reproduced on
two fixtures; the Linux side has one so far). It is nonetheless strong: the hang is
deterministic in the sense of "the trace step never returns within 30 min," and the
orphaned-`lake`/`lean` evidence shows the tracer — not the test harness — is where it
stalls. Combined with §1, the blocker is now **platform-independent (macOS-arm64 AND
Linux-x64) and repo-source-independent (`from_path` AND real GitHub repo)**, which
points at the lean-dojo 4.20.0 tracer itself rather than any local environment.

## 2. What is NOT the blocker (ruled out empirically)

- **Python version.** lean-dojo's metadata declares `Requires-Python: <3.13,>=3.9`;
  Python 3.12 is officially supported (the docs' older `<3.12` note is stale).
  Import and `trace()` invocation both succeed on 3.12. No 3.11 pin is needed.
- **Lean version.** The machine had Lean 4.31.0 (too new — the tracer's
  `ExtractData.lean` fails to compile against it). lean-dojo 4.20.0 pins
  `leanprover/lean4:v4.20.0` in its own repo; installing that via elan lets the
  tracer compile. The deadlock is *not* a Lean-version problem — it occurs with
  the exact pinned versions.
- **macOS multiprocessing spawn.** An initial `RuntimeError` ("start a new
  process before bootstrapping") from lean-dojo's progress-bar subprocess was
  fixed by the standard `if __name__ == "__main__":` guard. That crash is gone;
  the deadlock remains and is upstream of it.
- **Fixture validity.** `lake build` succeeds on both fixtures (Lean compiles
  `trivial_true` cleanly). The fixture is not the problem.
- **Remote cache.** lean-dojo has `REMOTE_CACHE_URL`, but it only serves repos
  lean-dojo has *pre-traced and published* (e.g. mathlib4). A custom minimal
  fixture is not there, so the cache cannot bypass the local tracer for L0.

## 3. Honest assessment & what this is NOT claiming

- This is **not** "lean-dojo v4 is broken in general" — it is "lean-dojo v4's
  `trace()` deadlocks on this machine on a minimal local repo, reproduced
  deterministically." It may be macOS-arm64-specific, lean-dojo-version-specific,
  or a known upstream issue; it has not been root-caused here.
- It is **not** a regression from #147/#157 (the tactic-DAG / G1-G2 work) — those
  are on `main` and untouched by this. The deadlock is in upstream lean-dojo,
  exercised by #175's (correct) trace-dependent path.
- It does **not** affect the fail-closed contract: with the tracer deadlocked,
  `check_proof_in_repo` cannot be reached, so verification abstains. No fragile
  green is possible — which is exactly why this is documented rather than
  papered over.

## 4. Options to unblock L0 (status updated 2026-06-26 after the CI repro in §1a)

1. ~~**Trace a real GitHub repo instead of a `from_path` local repo.**~~
   **FALSIFIED (§1a):** CI traces `leanprover-community/lean4-example` (a real
   GitHub repo via `LeanGitRepo`) and still hangs. Repo-source is not the cause.
2. ~~**Run the same reproduction on Linux** (CI is ubuntu-latest).~~
   **FALSIFIED (§1a):** ubuntu-latest hangs identically (orphaned `lake`/`lean` at
   the 30-min timeout). The deadlock is not macOS-arm64-specific, so there is no
   "CI-only L0" shortcut.
3. **Bypass `trace()` entirely with a lighter verification path.** Since L0 only
   needs "one real green check + one real reject," a `lake env lean` /
   direct-tactic-exec path (no lean-dojo tracer) could satisfy L0 without the
   deadlocking trace step. Most code, but it removes the dependency on the broken
   tracer and keeps the fail-closed contract intact. **Highest expected value** —
   the only option that doesn't route through the hung tracer.
4. **Root-cause the tracer stall** in `ExtractData.lean`'s extraction
   (`ps`/`py-spy` on the hung `lean` orphan to capture where it blocks), then file
   a precise upstream bug. Smallest scope, payoff is an upstream fix on someone
   else's timeline.
5. ~~**Bump lean-dojo to a newer version.**~~ **DEAD END:** verified 2026-06-26 that
   **4.20.0 is the latest lean-dojo on PyPI** (next-newest is the 2.x line, which
   predates the 4.x `check_proof`/`LeanGitRepo` API this codebase requires). There
   is nothing to bump *to*, and pinning *older* loses the 4.x API. Pursue an
   upstream fix (option 4) rather than a version bump.
6. **Trace mathlib4 instead of a minimal repo.** Still untried, lowest priority:
   §1a shows the stall is in the tracer's subprocess fan-out on a *minimal* repo, so
   a heavier repo is not obviously a fix, and it costs a multi-GB clone + trace.

Recommended order: **(3) → (4) → (6)**. Do **not** claim L0 until a green-on-valid
AND reject-on-invalid run both pass on a clean cache-miss.

**CI guard (2026-06-26):** because the real-trace case in
`tests/test_lean_dojo_check_proof.py` deadlocks the `lean-dojo-search` lane to its
30-min timeout on *every* PR touching the Lean paths, that one assertion is now
**skipped by default** and runs only when `SOPHIA_LEAN_TRACE_DEADLOCK_PROBE=1` is
set (opt-in, for someone actively working options 3/4). The skip is honest — L0
remains blocked, not green — it just stops the lane burning 30 CI-minutes per PR.
This is a test-hygiene change, not a masking of the deadlock (§5): the verification
contract is untouched; `check_proof_in_repo` still abstains when the trace hangs.

## 5. Non-action taken (discipline)

Per the project guardrail ("if the v4 trace-after-write flow forces anything
unclean behind the `lean_available()` gate, stop and document the gap rather
than shipping a fragile green"), **no code was changed to mask the deadlock**.
`lean_available()` still returns `True` on this machine (the import probe is
honest — lean-dojo *is* importable), and `verify_proof` / `check_proof_in_repo`
honor the fail-closed contract (they abstain when the trace is unreachable).
The gap is recorded here so that L0 is not silently claimed on a hung trace.
