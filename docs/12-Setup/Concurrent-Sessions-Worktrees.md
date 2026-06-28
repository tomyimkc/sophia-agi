# Concurrent sessions, worktrees, and checkout hygiene

**Status:** operational note · **Audience:** anyone running multiple AI coding
sessions (Claude / Grok / ZCode / Cursor) on this repo in parallel.

## The setup that already works

This repo uses `git worktree` to isolate parallel feature work — each session
gets its own directory + branch, so two agents never edit the same files:

```
git worktree list
/Users/tom/Documents/GitHub/sophia-agi            main / fix/...
/private/tmp/sophia-agi-pr100                      claude/error-memory-rag
/private/tmp/sophia-spark-moe                      claude/spark-moe-workflow
/private/tmp/sophia-tactic-dag                     feat/tactic-dag-novelty-hash
... (one per active feature)
```

This is the right pattern and it is already in use. New feature work should go
in a **fresh worktree**, not in the main checkout.

## The hazard this note exists to prevent

The main checkout (`/Users/tom/Documents/GitHub/sophia-agi`) can become a
**collision point** when multiple sessions stack *uncommitted* changes in its
working tree. Symptoms observed in practice:

- Uncommitted files from one session's in-flight feature appear in another
  session's `git status`, blurring "what is mine" vs "what is concurrent".
- `git checkout` / `git switch` is refused because of another session's dirty
  tree, forcing awkward `git stash` or `reset --soft` workarounds that risk
  losing the other session's work.
- A commit can land on the wrong branch when the working branch is switched
  underneath a session mid-edit.
- Published-result artifacts (`agi-proof/benchmark-results/*.json`) can appear
  to "drift" when one session regenerates them while another is mid-analysis.

Every one of these is recoverable, but each costs time and risks lost work.

## The rule

**Keep the main checkout clean.** It should hold only committed work on its
checked-out branch. All in-flight / speculative / uncommitted feature work
belongs in a dedicated worktree.

### Starting a new feature (do this every time)

```bash
# 1. From anywhere, create an isolated worktree on a fresh branch off main:
git fetch origin main
git worktree add -b feat/<my-feature> /private/tmp/sophia-<my-feature> origin/main

# 2. Do all your work there:
cd /private/tmp/sophia-<my-feature>
# ... edit, test, commit ...

# 3. Push + open a PR from that worktree:
git push -u origin feat/<my-feature>
gh pr create --base main --head feat/<my-feature>

# 4. After the PR merges, clean up:
cd /Users/tom/Documents/GitHub/sophia-agi
git worktree remove /private/tmp/sophia-<my-feature>
git branch -d feat/<my-feature>
```

### If you must use the main checkout

If a session has to operate in the main checkout (e.g. to triage state), it
must **commit or stash before switching branches**, and must never leave
uncommitted files that another session will inherit. The safe check before any
branch operation:

```bash
git status --porcelain   # must be empty before `git checkout`/`git switch`
```

### Re-basing a PR onto a moved main

`main` moves fast (often dozens of commits while a feature is in flight).
Before merging, rebase the feature branch onto current main **inside its own
worktree** so the main checkout is never disturbed:

```bash
cd /private/tmp/sophia-<my-feature>
git fetch origin main
git rebase origin/main
# resolve conflicts, test, then:
git push --force-with-lease origin feat/<my-feature>
```

`--force-with-lease` (not `--force`) aborts if the remote moved, so a
concurrent push can't be silently overwritten.

## Lean 4 toolchain on this machine (optional, fail-closed)

The formal-proof verifier (`agent/lean_verifier.py`, `selfextend/proof_verifier.py`)
is **fail-closed without Lean** — every check returns `verdict: held` /
`lean_unavailable` and never `accepted`, and CI is green without a toolchain (the
fail-closed path IS the tested contract). A Lean install is OPTIONAL and changes
no gate's logic; it only lets the two real-kernel test cases run (otherwise they
skip) and lets `tools/run_formal_proofs_eval.py` actually type-check proofs.

Lean 4.31 was installed on this Mac (Tom's machine) via `scripts/install_lean.sh`
while debugging the lean-kernel CI lane. It lives at:

```
/Users/tom/.elan/bin/{lean,lake,elan}
```

The installer deliberately does **not** modify shell rc files. To make
`lean`/`lake` available in a shell, add to `~/.zshrc` (or the relevant rc file):

```bash
export PATH="/Users/tom/.elan/bin:$PATH"
```

Verify with `lean --version`. A future session that needs the kernel should check
this path first (`command -v lean`) before re-running `scripts/install_lean.sh`
(which is idempotent — it no-ops if Lean is already present).

Note: this path is **machine-local to Tom's Mac**, not portable — it is recorded
here only so concurrent sessions on this checkout know the kernel is already
available and don't re-download ~100 MB unnecessarily. On any other machine (CI
included), run `scripts/install_lean.sh` fresh.

## Why it matters for this repo specifically

Sophia's discipline is provenance and auditability — every claim traceable,
every result reproducible. Concurrent-session collisions are the
*negative-space* version of that: work whose provenance becomes ambiguous
("did I write this or did another session?") is exactly the kind of
untraceable state the repo's own gates exist to prevent in *outputs*. Keeping
the main checkout clean extends that discipline to the *process*.
