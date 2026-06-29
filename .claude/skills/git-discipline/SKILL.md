---
name: git-discipline
description: >
  Run BEFORE any git write or merge in this multi-agent repo: commit, push, branch, switch,
  rebase, merge, open/close/merge a PR, re-trigger CI, "unblock #NNN", or diagnose a failing
  CI check or a "superseded/divergent" branch. This repo (tomyimkc/sophia-agi) has many
  concurrent advisors (Claude/Copilot/GLM/human) on many worktrees pushing to main around the
  clock, so a local checkout is a STALE snapshot the moment it is taken. Use even if the
  conversation already contains a diagnosis or a "you are on branch X" line — that may be stale.
metadata:
  short-description: "Stale-snapshot / merge-blocker / verify-before-act discipline for this high-churn repo"
---

# Git & merge discipline (this repo)

This repo has **many concurrent advisors** pushing and merging to `main` constantly. Every
expensive mistake here came from acting on an **unverified local picture**: a fix PR built off
`main` that was 28 commits behind; a duplicate PR for a problem another advisor already fixed;
a "merge" blocked by unresolved review threads (not CI); a cancelled CI run misread as a real
failure. This skill is the cheap pre-flight that prevents that whole class of waste.

> The canonical, fuller playbooks + ready-to-run scripts live in `.agents/skills/` (shared with
> the other agent tools). This skill is the Claude-Code entry point — it routes you to them and
> states the rules. Do not re-implement the scripts; call them.

## 1. Before ANY git write — situational awareness (1 s, read-only)

```bash
python .agents/skills/git-operations/scripts/git_situational_awareness.py        # human
python .agents/skills/git-operations/scripts/git_situational_awareness.py --json # machine
```
Exit code is the verdict: **0** safe · **1** warnings (review, then proceed) · **2** blockers (stop).
It answers the six questions: where am I, is a sibling worktree on this branch, what's uncommitted,
am I stale vs `origin/main`, is there a competing PR on this branch, is `main` directly pushable.
It does **not** fetch — run `git fetch origin --prune` first when you need fresh remote truth.

## 2. Before any merge / unblock / CI-fix — merge pre-flight

```bash
git fetch origin --prune
python .agents/skills/multi-agent-merge-preflight/scripts/merge_blockers.py NNN
```
Re-verify the thing is **still** broken on `origin/main` (another advisor may have fixed it),
check for a **competing PR** before opening your own, and enumerate **all** merge blockers — not
just CI. The silent killer here is `required_review_thread_resolution: true`: green checks but
unresolved review threads still block the merge.

## 3. Verify before you act (the three false calls that happened here)

- **"Divergent/superseded" by line count** → wrong. Test ancestry + a trial merge, never line counts.
- **"test: fail"** → may be a **cancelled** run (job conclusion `cancelled`, not `failure`) caused by
  main moving mid-run. Read the job conclusion and re-trigger before "fixing" a ghost.
- **A "bug" in a PR** → reproduce it on the PR's *current* head against fresh `origin/main` first.

Full reasoning + the exact `gh api graphql` thread-resolution recipe:
`.agents/skills/pr-merge-verification/SKILL.md` and `.agents/skills/multi-agent-merge-preflight/SKILL.md`.

## 4. When you commit, isolate your changes

The working tree is frequently contaminated by other advisors' uncommitted work.
- `git add <explicit file list>` — never `git add -A` / `git add .`.
- `git diff --cached --stat` to confirm only your files are staged.
- Confirm `git branch --show-current` and `git log --oneline -1` before pushing.
- Develop on the branch the user named; direct pushes to `main` are blocked by protection.

## Branch protection (quick reference)

One ruleset (`main-protection`) on `main`: required checks **`fast`** + **`ci-complete`**;
**`required_review_thread_resolution: true`** (the silent blocker); `required_approving_review_count: 0`;
`required_linear_history: true`; direct pushes to `main` blocked — go through a PR.

## Stop and ask the human

This skill governs *mechanics* (stale? blocked? duplicated?). If an action would change what is
**claimed** (a number's status, "validated", a `RESULTS.md` entry), that is the no-overclaim
contract — stop and ask before acting (see the `ci-artifact-drift` and `sophia-agi` skills).
