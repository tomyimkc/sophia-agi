---
name: git-operations
description: Read-only situational-awareness snapshot to run BEFORE any git write (commit, push, branch, rebase, merge) in this multi-agent repo. Answers the six questions an agent must resolve to avoid wasted work — where am I, is anyone else on this branch, what's uncommitted, am I stale vs origin/main, is there competing PR work, is main directly pushable. Use whenever the task involves committing, pushing, creating/switching a branch, rebasing, opening a PR, or "before I touch git" — even if the conversation already seems current. This repo has many concurrent advisors (Claude/Copilot/GLM/human) on many worktrees; a checkout is a stale snapshot the moment it is taken.
---

# Git operations (situational awareness first)

This repo (`tomyimkc/sophia-agi`) has **many concurrent advisors** — multiple Claude/Copilot/GLM agents plus the human — frequently working from **separate worktrees** of the same clone, all pushing and merging to `main` around the clock. A local checkout is a **stale snapshot** the moment it is taken, and several adjacent worktrees can be checked out to the same or related branches without you knowing.

The expensive git mistakes here all come from acting on an unverified local picture:

1. **Committed straight to `main`** with uncommitted experimental work, then discovered direct pushes are blocked and the working tree is now contaminated and hard to unwind.
2. **Built a feature branch that was 20+ commits behind `origin/main`**, so the "current" file contents being reasoned about were already obsolete and the resulting PR was dead on arrival.
3. **Pushed to a branch that already had an open PR owned by another advisor**, silently rewriting that PR's head and confusing the review.
4. **Rebased/committed in a worktree while a sibling worktree was on the same branch**, colliding on checkout.

This skill prevents all four by answering six questions **before** the first write.

## When to run this skill

Run it **first**, before any of:
- Committing, pushing, stashing, or creating/switching a branch
- Rebasing, merging, or opening a PR
- Touching git in a worktree you did not create yourself this session

It is cheap (~1 s locally; a few seconds if the PR-existence check hits `gh`), dependency-free, and never writes anything. Run it even if the conversation already contains a "you are on branch X" line from earlier — that line may be stale.

## The snapshot — run the script

```bash
python .agents/skills/git-operations/scripts/git_situational_awareness.py
```

For machine consumption (compose into a larger flow):

```bash
python .agents/skills/git-operations/scripts/git_situational_awareness.py --json
```

**Exit code is the verdict — branch on it:**

| exit | meaning       | action                                             |
|------|---------------|----------------------------------------------------|
| 0    | clean         | safe to proceed with the git write                 |
| 1    | warnings      | review the printed warnings, then proceed if understood |
| 2    | blockers      | do NOT push/merge/commit until resolved            |

The script is dependency-free (Python stdlib + `git` + optional `gh`). It performs **only reads** — `git fetch` is NOT done for you (fetch is a network call you should opt into; see below).

### The six questions it answers

1. **WHERE am I?** — branch, HEAD, worktree path, whether this is a side worktree, and how many worktrees exist on this machine. Tells you whether you're in the main checkout (where another agent may be mid-edit) or an isolated worktree.
2. **Is anyone ELSE on this branch?** — scans all local worktrees for a sibling checked out to the same branch. Commits are still safe, but a rebase/checkout there will collide.
3. **What's UNCOMMITTED?** — porcelain status. Uncommitted changes **on `main`** are a hard blocker (never commit experimental work on main here).
4. **Am I STALE vs `origin/main`?** — left-right count (`git rev-list --left-right --count`). `behind > 0` means your `git log` and file contents are suspect — fetch before reasoning about main. `behind > 20` is a hard blocker: do not open a PR off a base that stale.
5. **COMPETING work?** — `gh pr list --head <branch>`. If an open PR already targets this branch, pushing updates *that* PR; confirm it's yours or that you intend to.
6. **Is `main` directly pushable?** — being ON `main` warns: branch protection blocks direct pushes, so create a feature branch before committing work you intend to land.

## How to use the snapshot

### After the snapshot, fetch if it says you're stale

The script deliberately does not run `git fetch` (a network write to your refs). If the snapshot reports `behind > 0`, fetch first, then reason off `origin/main` as the source of truth — never off a local branch you haven't fetched:

```bash
git fetch origin --prune
git rev-list --left-right --count <branch>...origin/main   # re-confirm after fetch
```

### Always isolate your writes from other advisors' work

The working tree in this repo is frequently **contaminated** with uncommitted work from other advisors. When you commit:
- `git add <explicit file list>` — never `git add -A` / `git add .`
- Verify with `git diff --cached --stat` that only your files are staged.
- Before pushing, confirm `git branch --show-current` and `git log --oneline -1` show your commit on the intended branch (another advisor's checkout can move under you).

### Prefer a side worktree for any non-trivial work

```bash
git worktree add --detach /private/tmp/<name> origin/main
cd /private/tmp/<name>
```

The snapshot's "side wt?" line confirms you landed in the isolated worktree, not the shared main checkout.

## How this skill relates to the other two git skills

Three layered skills; pick by what you're about to do:

- **`git-operations`** (this one) — the **first** read, before *any* git write. Cheap local snapshot; answers "am I safe to touch git at all?".
- **`multi-agent-merge-preflight`** — the gate for **merges / CI-fixes / "unblock #NNN"**. Re-queries the LIVE remote, enumerates *all* merge blockers (including the silent `required_review_thread_resolution`), checks for competing fix PRs. Run this *after* the snapshot, when the action is specifically a merge or CI fix.
- **`pr-merge-verification`** — the discipline for **reviewing/triaging a PR**: the three misread traps (line-count divergence fallacy, cancelled-run false-fails, missing-precondition "bugs"). Run this when judging whether a PR is sound.

Typical flow for landing work: `git-operations` snapshot → write on an isolated worktree → (if merging) `multi-agent-merge-preflight` → (if reviewing) `pr-merge-verification`.

## When to stop and ask the human

The snapshot is about *mechanics* (am I stale? blocked? duplicated? on the wrong branch?), not about any research-integrity claim. If acting on the snapshot would change what gets *claimed* (a status, a "validated" verdict, a RESULTS.md entry), stop and ask before acting. If the blockers are unclear or you cannot tell which worktree/PR is "yours," surface the specific evidence from the snapshot and let the human decide.
