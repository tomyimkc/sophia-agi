---
name: multi-agent-merge-preflight
description: Mandatory pre-flight gate before ANY merge / unblock / CI-fix / branch-protect work in this repo. Re-queries the LIVE remote and checks for competing advisors before acting — this repo has many concurrent advisors (Claude/Copilot/GLM/human) pushing and merging constantly, and acting on a stale local snapshot wastes enormous effort. Use whenever the task touches a PR, a merge, a CI failure, branch protection, a re-trigger, or "unblock X" — even if the conversation already contains a diagnosis from earlier.
---

# Multi-agent merge pre-flight

This repo (`tomyimkc/sophia-agi`) has **many concurrent advisors** — multiple Claude/Copilot/GLM agents plus the human, all pushing and merging to `main` around the clock. A local checkout is a **stale snapshot** the moment it's taken. Diagnoses, file contents, PR states, and CI results you captured even 20 minutes ago are very likely **already obsolete**.

Three real, expensive failures happened because an advisor acted on a stale picture:

1. **Built an entire fix PR off local `main` that was 28 commits behind `origin/main`** — the failures were already fixed by another advisor's merge; the whole PR was obsolete.
2. **Opened a competing fix PR without checking that another advisor had already opened one** doing the identical diagnosis (`#138`). Both timed out / duplicated work.
3. **Diagnosed CI as failing, planned a merge — but never discovered the branch protection also requires `required_review_thread_resolution`** (8 unresolved Copilot threads), which was the *actual* blocker after checks went green.

This skill exists so that doesn't happen again.

## When to run this skill

Run it **first**, before doing any of:
- Merging a PR, re-triggering CI, "unblocking #NNN"
- Fixing a CI/test failure on `main`
- Opening a fix/feature PR
- Bisecting or diagnosing a "blocking" issue

It is cheap (~15 s of API calls) and catches the entire class of stale-snapshot waste. Do it even if a prior advisor's prompt already contains a diagnosis — that diagnosis may be minutes or hours stale.

## The pre-flight (do all of this before acting)

### 1. Refresh the ground truth, then trust only the live remote

```bash
git fetch origin --prune          # ALWAYS fetch first; local refs are stale
```

After fetching, compare local vs remote. **Never** reason off `git log` of a local branch you haven't fetched. Specifically:
- `git rev-list --left-right --count <branch>...origin/<branch>` — if right side > 0, your local branch is behind and anything you "know" about main is suspect.
- Treat `origin/main` as the source of truth, not `main`.

### 2. Re-verify the thing you're about to act on is STILL broken/blocked

Before building a fix, confirm the failure still reproduces **on `origin/main`**, not on your stale local checkout:
- For a CI failure: is the latest run on `origin/main` still red, or did another advisor already fix it? `gh run list --branch main --workflow CI --limit 3`
- For "blocked PR #NNN": re-fetch its state — `gh pr view NNN --json mergeable,mergeStateStatus,headRefOid`. If `mergeStateStatus` is `CLEAN`, it's unblocked; if `BLOCKED`, find out *why* (see step 4) before assuming it's a CI failure.

**If another advisor already fixed it on main: stop.** Don't open a duplicate PR. Report the finding instead.

### 3. Check for competing work before opening any PR

Before opening a fix/feature PR, list **all** open PRs and skim titles + branches for overlap:
```bash
gh pr list --state open --json number,title,headRefName,mergeable,mergeStateStatus
```
If a PR already targets the same problem (same failing test, same module, same diagnosis), **do not duplicate**. Either review/help that PR (comment, push to its branch if the human allows), or defer. Duplicating wastes CI minutes and the human's review bandwidth.

### 4. Discover ALL merge blockers, not just CI

`mergeStateStatus: BLOCKED` with green checks is common and means something *else* is blocking. The blockers here are not always obvious. Enumerate them:

```bash
# Run the bundled script — it prints a structured blocker report:
python .agents/skills/multi-agent-merge-preflight/scripts/merge_blockers.py NNN
```

It checks, for a given PR number:
- required status checks vs. the PR's current check states (mismatch = a check is missing/failed/pending that protection requires)
- **`required_review_thread_resolution`** — any *unresolved* review threads block the merge even with 0 required approvals and all checks green. This is the easy-to-miss one. Use `gh api graphql` to count `reviewThreads { isResolved }`.
- `required_approving_review_count` and `reviewDecision`
- merge conflicts (`mergeable: CONFLICTING`)

**Do not assume "checks green → mergeable."** In this repo, unresolved Copilot review threads are the most common silent blocker. Read every unresolved thread — Copilot's comments are frequently real correctness bugs (this session: space-joined Lean tactics, an unwired real-run path, a fail-closed break). Fix the valid ones in code, reply, and resolve the threads.

### 5. When you DO act, isolate your changes

The working tree in this repo is frequently **contaminated** with uncommitted work from other advisors (121 dirty files were once seen). When you commit:
- `git add <explicit file list>` — never `git add -A` / `git add .`
- Verify with `git diff --cached --stat` that only your files are staged before committing.
- Another advisor's local branch can get checked out under you mid-task; verify `git branch --show-current` and that your commit is on the right branch (`git log --oneline -1`) before pushing.

## How to resolve review threads (the API is fiddly)

`gh pr merge` will refuse with "base branch policy prohibits the merge" when `required_review_thread_resolution` is set and threads are open. Resolve each thread with a reply explaining the fix, then resolve:

```python
import json, subprocess
# ID! type requires JSON-input, NOT gh -F (which forces String and 422s)
reply = {"query": "mutation($t:ID!,$b:String!){ addPullRequestReviewThreadReply(input:{pullRequestReviewThreadId:$t,body:$b}){ comment{ id } } }",
         "variables": {"t": THREAD_ID, "b": "Fixed in <sha>: <one-line>"}}
subprocess.run(["gh","api","graphql","--input","-"], input=json.dumps(reply), text=True)
resolve = {"query": "mutation($t:ID!){ resolveReviewThread(input:{threadId:$t}){ thread{ isResolved } } }",
           "variables": {"t": THREAD_ID}}
subprocess.run(["gh","api","graphql","--input","-"], input=json.dumps(resolve), text=True)
```

Get thread IDs with: `gh api graphql -f query='{ repository(owner:"tomyimkc",name:"sophia-agi"){ pullRequest(number:NNN){ reviewThreads(first:50){ nodes{ id isResolved path } } } } }'`

## Quick reference: this repo's protection

- One active branch ruleset (`main-protection`) on `main`. Required checks: **`fast`** + **`ci-complete`**.
- **`required_review_thread_resolution: true`** — the silent killer. All review threads must be resolved.
- `required_approving_review_count: 0` — no approvals needed, which makes people *think* it should merge on green checks. It won't, if threads are open.
- `required_linear_history: true`; `allowed_merge_methods: merge, squash, rebase`.
- Direct pushes to `main` are blocked (the `pull_request` rule) — you must go through a PR.

## When to stop and ask the human

Per the repo's no-overclaim discipline: if a decision changes what gets *claimed* (a number's status, whether something is "validated", a RESULTS.md entry), **stop and ask** before acting. This pre-flight is about *mechanics* (is it stale? is it blocked? is it duplicated?), not about relaxing any research-integrity rule.
