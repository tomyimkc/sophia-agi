---
name: pr-merge-verification
description: Verification discipline for reviewing, triaging, or merging GitHub PRs in this high-churn, multi-agent repo. Use whenever you are asked to review a PR, check if a branch/PR is "superseded" or "divergent", merge anything, diagnose a failing or red CI check, or diagnose a "bug" in a PR's code or CI assertions — even if the user does not say "verify first". This repo has 80+ branches and parallel agents merging constantly; the three false-call patterns below (divergence misread, cancel-artifact false-fails, missing-precondition bugs) have all happened here. Always trigger before any irreversible action (merge, close, push a fix).
---

# PR merge verification (this repo)

This is a high-churn, multi-agent repo: 80+ branches, parallel agents and the human merging through the UI constantly, and CI jobs that get cancelled mid-run by the churn. Three specific false-positive diagnoses have caused wrong actions here. Before you merge, close, or push a "fix," run the checks below. They are cheap and they prevent the three errors.

**The governing rule:** *verify before you act.* A merge, a branch close, and a pushed "fix" are all hard to reverse and outward-facing. Read the actual state — don't infer it from a status label, a line count, or a single failing assertion.

## Trap 1 — "divergent / competing branches" (the line-count fallacy)

**The wrong reasoning:** two PRs touch the same file, the file has very different line counts between them (e.g. 166 vs 530 lines), so you conclude they are "divergent evolutions" or "competing approaches" and treat one as superseding the other.

**Why it's wrong:** in a stacked-PR workflow, the newer branch is built *on top of* the older one's merge. The line-count difference is **additive stacking**, not divergence. Treating a clean stack as a conflict causes you to close or force-rebase work that actually merges cleanly.

**The right check — ancestry, not line count:**

```bash
# Is the older PR's merge commit an ANCESTOR of the newer branch's base?
# (If yes → the newer branch is stacked ON the older one, not divergent from it.)
git merge-base --is-ancestor <older-pr-merge-sha> <newer-branch-base-sha> && echo "STACKED (clean)" || echo "divergent"

# Then confirm: does the merge-base already contain the older branch's content?
# (0-line diff between merge-base and main for that file = older branch already landed as the base)
git diff $(git merge-base origin/main <newer-branch>) origin/main -- <file> | wc -l

# And the decisive merge-cleanliness test — actually try the merge:
git checkout -b trial origin/main
git merge --no-ff --no-commit <newer-branch>
git diff --name-only --diff-filter=U   # empty = no conflicts, regardless of line counts
git merge --abort && git branch -D trial
```

Line count is never evidence of divergence. Only ancestry + a trial merge are.

## Trap 2 — "test: fail" that is actually a cancelled run (cancel-artifacts)

**The wrong reasoning:** `gh pr checks` shows `test fail` (or `ci-complete fail`), so the PR has a real regression and must not merge (or must be "fixed").

**Why it's wrong:** in this repo, when main moves during a long test run, GitHub *cancels* the in-flight run. A cancelled job reports as `fail` in the rollup, but its actual **job conclusion is `cancelled`, not `failure`** — there was no test failure at all. Merging on this is fine; "fixing" it is chasing a ghost.

**The right check — read the job conclusion, not the status rollup:**

```bash
# 1. Is the failing job actually FAILED, or CANCELLED?
JOB_URL=$(gh pr checks <PR> | grep '^test' | grep -oE 'runs/[0-9]+/job/[0-9]+' | head -1)
gh run view <run-id> --json jobs --jq '.jobs[] | "\(.name): \(.status)/\(.conclusion)"'
#   conclusion: "cancelled" → NOT a real failure. conclusion: "failure" → real.

# 2. Is the run stale (computed against an older main / since-superseded)?
gh run view <run-id> --json created_at,head_sha --jq '"run for \(.head_sha[0:8]) at \(.created_at)"'
#   compare head_sha to the PR's current head; compare time to now.

# 3. If cancelled or stale, RE-TRIGGER and wait for a fresh run before deciding:
git checkout -b trigger origin/<pr-branch>
git commit --allow-empty -m "chore: re-trigger CI (prior run was cancelled/stale)"
git push origin trigger:<pr-branch>
# then poll: for i in $(seq 1 40); do ... gh run view <new-run> --json status,conclusion; sleep 30; done
```

Never treat a red check as a failure until you have read `conclusion: "failure"` on a *current* run. `cancelled` ≠ `failure`.

## Trap 3 — diagnosing a "bug" without checking the PR's own preconditions

**The wrong reasoning:** a PR's CI assertion fails (or an asserted condition doesn't hold in an artifact), so the assertion is "over-claiming" or "wrong" and should be fixed/weakened.

**Why it's wrong:** the assertion may be correct *given a precondition the PR itself is responsible for providing* — a toolchain install, a runner, an environment variable, a data file. If you inspect the failure in an environment that *lacks* the precondition, you'll wrongly conclude the assertion is broken. Real example: a "loop must close" assertion looked broken (held-out reward 0.6 < 1.0) only because the **Lean kernel wasn't installed**; the PR's own purpose was the `install_lean.sh` bootstrap that makes the kernel present, after which the smoke lemmas genuinely prove (1.0) and the assertion passes.

**The right check — find the precondition before you call it a bug:**

```bash
# 1. What does the failing assertion ACTUALLY require? Read it + the surrounding setup steps.
gh run view <run-id> --log-failed | grep -B3 -A3 "assert\|Assert"

# 2. Does the PR ADD the thing the assertion depends on? (a setup step, an install script,
#    a workflow that provides it). Look in the PR's own diff for it:
gh pr diff <PR> | grep -iE "install|setup|bootstrap|runs-on|requirements|pip install|elan|toolchain"

# 3. Was the failing run MISSING that precondition? (e.g. different workflow, no kernel,
#    wrong runner). Compare the failing run's environment to what the PR sets up.
gh run view <run-id> --json jobs --jq '.jobs[].name'   # did the providing job even run?

# 4. Only if the assertion fails IN THE PR'S OWN INTENDED ENVIRONMENT is it a real bug.
#    Otherwise the fix is "run it where the precondition holds," not "weaken the assertion."
```

An assertion that fails without its precondition is not necessarily wrong. The precondition may be the entire point of the PR.

## Operating rules for this repo specifically

- **Never admin-merge past a check you haven't diagnosed.** If a gate is pending/failing, run Trap 2's conclusion-check first. Admin-merge is justified only when the sole block is a *non-required* check (empty `required_status_checks`) that is itself stalled/cancelled — not to skip a real failure.
- **Stale branches leave "BLOCKED" merge-state.** A perpetually-`pending` job that's not a required check usually means no matching runner is registered (`gh api repos/<owner>/<repo>/actions/runners` → `total_count`). Diagnose, don't override blindly.
- **`canClaimAGI` must stay `false`** after every merge. Verify it: `grep canClaimAGI agi-proof/architecture-bets.json`. If a merge flipped it true, something overclaimed — stop and report.
- **Work from an isolated worktree** (`git worktree add --detach /private/tmp/<name> origin/main`), never from a shared checkout that another agent may be mid-edit in. Other agents' uncommitted changes are not yours to touch.
- **Every merge, close, and pushed fix is reported honestly** — including when your own diagnosis turns out wrong. State corrections plainly ("I was wrong; verified X instead"). Do not quietly reverse course.

## When you are unsure — stop and ask

These three traps exist because acting on a misread is expensive. If ancestry is ambiguous, if a run's conclusion can't be confirmed, or if you can't tell whether a precondition is the PR's job or the environment's — surface the ambiguity to the user with the specific evidence and let them decide. Asking is cheaper than a wrong merge or a weakened assertion.
