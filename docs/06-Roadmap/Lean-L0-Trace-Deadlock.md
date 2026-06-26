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

## 4. Options to unblock L0 (none attempted yet — for the next session)

1. **Trace mathlib4 instead of a minimal repo** and target a real mathlib
   theorem for the L0 smoke check. mathlib4 is the path lean-dojo is *built and
   tested* against; its trace is also remote-cacheable. Heaviest, but most
   likely to dodge the minimal-repo deadlock. (Cost: multi-GB clone + trace.)
2. **Reproduce against a lean-dojo version newer than 4.20.0** (if released) —
   the deadlock may be fixed upstream. Pin that version in
   `requirements-theorem.txt`.
3. **Run the same reproduction on Linux** (CI is ubuntu-latest). If the deadlock
   is macOS-arm64-specific, CI may trace fine — in which case L0 is a CI-only
   demonstration and the local-Mac path is documented as unsupported.
4. **Root-cause the 1515/1518 stall** in `ExtractData.lean`'s final extraction
   (the proof-state serialization for the last few items). Smallest scope,
   most uncertain payoff.

The cheapest signal is **(3)**: if CI traces mathlib4 cleanly, L0 is achievable
in CI and the local deadlock becomes a known platform limitation, not a project
blocker. Recommend that as the next attempt.

## 5. Non-action taken (discipline)

Per the project guardrail ("if the v4 trace-after-write flow forces anything
unclean behind the `lean_available()` gate, stop and document the gap rather
than shipping a fragile green"), **no code was changed to mask the deadlock**.
`lean_available()` still returns `True` on this machine (the import probe is
honest — lean-dojo *is* importable), and `verify_proof` / `check_proof_in_repo`
honor the fail-closed contract (they abstain when the trace is unreachable).
The gap is recorded here so that L0 is not silently claimed on a hung trace.
