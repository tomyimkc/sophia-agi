#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""
Situational-awareness snapshot for safe git operations in this multi-agent repo.

One command answers the six questions an agent MUST resolve before ANY git write
(commit, push, branch, rebase, merge). Run it first; act on what it says.

    python .agents/skills/git-operations/scripts/git_situational_awareness.py
    python .agents/skills/git-operations/scripts/git_situational_awareness.py --json   # machine-readable

It is dependency-free (stdlib + git + gh). It never writes anything. Exit code:
  0 = clean / safe to proceed
  1 = WARNINGS printed (review before acting)
  2 = BLOCKERS printed (do NOT push/merge until resolved)

Designed for `tomyimkc/sophia-agi` but works on any GitHub repo with `gh` auth.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

# Avoid the zsh readonly-variable trap: never name a var 'status', 'path', etc.


# --------------------------------------------------------------------------- #
# tiny helpers
# --------------------------------------------------------------------------- #
def run(cmd: list[str], *, cwd: str | None = None, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 127, "", str(e)


def git(*args: str, cwd: str | None = None) -> tuple[int, str, str]:
    return run(["git", *args], cwd=cwd)


def gh_json(args: list[str]) -> object | None:
    """Run `gh ... --json ...` and parse. Returns None on failure (never raises)."""
    rc, out, _ = run(args)
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def is_truthy(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# the snapshot
# --------------------------------------------------------------------------- #
def snapshot(repo_root: str) -> dict:
    """Collect the full situational snapshot. Pure reads, no writes."""
    snap: dict = {"warnings": [], "blockers": [], "repo_root": repo_root}
    inside, root, _ = git("rev-parse", "--show-toplevel", cwd=repo_root)
    snap["inside_worktree"] = inside == 0
    if inside != 0:
        snap["blockers"].append("not inside a git worktree (no .git found)")
        return snap
    snap["root"] = root

    # --- 1. WHERE am I? (branch, worktree, HEAD) ---
    _, branch, _ = git("branch", "--show-current", cwd=repo_root)
    _, head, _ = git("rev-parse", "--short", "HEAD", cwd=repo_root)
    snap["branch"] = branch or "(detached HEAD)"
    snap["head"] = head
    snap["detached"] = branch == ""

    # worktree path (distinguishes main checkout from a side worktree)
    _, wt_path, _ = git("rev-parse", "--show-toplevel", cwd=repo_root)
    snap["worktree_path"] = wt_path

    # all worktrees on this machine
    _, wt_list_raw, _ = git("worktree", "list", "--porcelain", cwd=repo_root)
    worktrees = []
    cur: dict = {}
    for line in wt_list_raw.splitlines():
        if line.startswith("worktree "):
            if cur:
                worktrees.append(cur)
            cur = {"path": line.split(" ", 1)[1]}
        elif line.startswith("HEAD "):
            cur["head"] = line.split(" ", 1)[1][:10]
        elif line.startswith("branch "):
            cur["branch"] = line.split(" ", 1)[1]
        elif line.startswith("detached"):
            cur["branch"] = "(detached)"
    if cur:
        worktrees.append(cur)
    snap["worktrees"] = worktrees
    snap["is_side_worktree"] = wt_path not in ("", root) and len(worktrees) > 1

    # --- 2. Is ANYONE ELSE on this branch? (worktree-vs-worktree) ---
    if branch and branch != "(detached)":
        sharing = [w for w in worktrees if w.get("branch") == f"refs/heads/{branch}" and w.get("path") != wt_path]
        # worktree list reports full refpaths for linked worktrees, bare name varies; double-check:
        if not sharing:
            sharing = [w for w in worktrees if w.get("branch", "").endswith(f"/{branch}") and w.get("path") != wt_path]
        snap["branch_shared_locally"] = bool(sharing)
        if sharing:
            snap["warnings"].append(
                f"another local worktree is ALSO on branch '{branch}': {sharing[0]['path']} — "
                "commits here are safe, but a rebase/checkout there will collide"
            )

    # --- 3. What's UNCOMMITTED? (the safety check before any write) ---
    _, dirty, _ = git("status", "--porcelain", cwd=repo_root)
    snap["uncommitted"] = [d[3:] for d in dirty.splitlines()] if dirty else []
    snap["dirty_count"] = len(snap["uncommitted"])
    if snap["uncommitted"] and branch == "main":
        snap["blockers"].append(
            f"{len(snap['uncommitted'])} uncommitted change(s) on 'main' — commit/stash to a feature "
            "branch FIRST; never commit experimental work directly on main (it blocks fast-forward)"
        )

    # --- 4. Am I STALE vs origin? (the #1 cause of wasted work in this repo) ---
    snap["has_remote"], _, _ = git("rev-parse", "--verify", "origin/main", cwd=repo_root)
    if snap["has_remote"] == 0:
        # left-right count: left=local-only commits, right=remote-only commits
        _, lr, _ = git("rev-list", "--left-right", "--count", f"{branch or 'HEAD'}...origin/main", cwd=repo_root)
        try:
            left_s, right_s = lr.split()
            ahead, behind = int(left_s), int(right_s)
        except (ValueError, AttributeError):
            ahead, behind = -1, -1
        snap["ahead_of_origin_main"] = ahead
        snap["behind_origin_main"] = behind
        if behind > 0:
            snap["warnings"].append(
                f"local branch is {behind} commit(s) BEHIND origin/main — your 'git log' and "
                "file contents are stale; FETCH before reasoning about main (fetch is a read)"
            )
        if behind > 20:
            snap["blockers"].append(
                f"{behind} commits behind origin/main — do NOT open a PR off this branch; "
                "fetch + rebase onto origin/main first (a PR built on a 20+ stale base is likely obsolete)"
            )
    else:
        snap["ahead_of_origin_main"] = None
        snap["behind_origin_main"] = None

    # --- 5. COMPETING work? (open PRs on the same branch / same area) ---
    owner_repo = _owner_repo(repo_root)
    snap["owner_repo"] = owner_repo
    if owner_repo and branch and branch != "main" and branch != "(detached)":
        prs = gh_json([
            "gh", "pr", "list", "--state", "open",
            "--head", branch, "--json", "number,title,mergeable,mergeStateStatus",
        ])
        snap["open_pr_for_this_branch"] = prs
        if prs:
            snap["warnings"].append(
                f"an open PR already exists for branch '{branch}' (#{prs[0]['number']}) — "
                "pushing here updates THAT PR; confirm it's yours or that you intend to"
            )

    # --- 6. Is main DIRECTLY PUSHABLE? (it isn't, in this repo) ---
    if branch == "main":
        snap["warnings"].append(
            "you are ON 'main' — direct pushes to main are blocked by branch protection here; "
            "create a feature branch before committing work you intend to land"
        )

    return snap


def _owner_repo(repo_root: str) -> str | None:
    rc, url, _ = git("config", "--get", "remote.origin.url", cwd=repo_root)
    if rc != 0 or not url:
        return None
    # normalize: git@github.com:owner/repo(.git)  or  https://github.com/owner/repo(.git)
    url = url.replace("git@github.com:", "https://github.com/").replace(".git", "")
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return None


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render(snap: dict) -> str:
    """Human-readable report."""
    W = snap.get("warnings", [])
    B = snap.get("blockers", [])
    out: list[str] = []

    out.append("═" * 64)
    out.append("  GIT SITUATIONAL AWARENESS")
    out.append("═" * 64)

    # 1. where
    out.append("")
    out.append("▸ WHERE am I?")
    out.append(f"    branch      : {snap['branch']}")
    out.append(f"    HEAD        : {snap['head']}")
    out.append(f"    worktree    : {snap['worktree_path']}")
    side = "yes (isolated)" if snap.get("is_side_worktree") else "no (main checkout)"
    out.append(f"    side wt?    : {side}")
    n_wt = len(snap.get("worktrees", []))
    out.append(f"    total wt    : {n_wt} on this machine")

    # 2. shared?
    out.append("")
    out.append("▸ Is anyone ELSE on this branch?")
    shared = snap.get("branch_shared_locally", False)
    if shared:
        out.append("    ⚠️  YES — another local worktree shares this branch (see warnings)")
    else:
        out.append("    no — this branch is exclusive to this worktree locally")

    # 3. uncommitted
    out.append("")
    out.append("▸ What's UNCOMMITTED?")
    n = snap.get("dirty_count", 0)
    if n == 0:
        out.append("    clean working tree")
    else:
        out.append(f"    {n} change(s):")
        for f in snap.get("uncommitted", [])[:8]:
            out.append(f"      • {f}")
        if n > 8:
            out.append(f"      …and {n - 8} more")

    # 4. stale?
    out.append("")
    out.append("▸ Am I STALE vs origin/main?")
    ahead = snap.get("ahead_of_origin_main")
    behind = snap.get("behind_origin_main")
    if ahead is None:
        out.append("    (no origin/main reference — first push pending, or no remote)")
    else:
        flag = " ⚠️ STALE" if behind and behind > 0 else ""
        out.append(f"    ahead: {ahead}   behind: {behind}{flag}")

    # 5. competing PR?
    out.append("")
    out.append("▸ COMPETING work?")
    prs = snap.get("open_pr_for_this_branch")
    if prs:
        out.append(f"    ⚠️  open PR #{prs[0]['number']} on this branch: {prs[0].get('title','')[:50]}")
    elif snap.get("owner_repo"):
        out.append("    no open PR targets this branch")
    else:
        out.append("    (couldn't determine repo for PR check)")

    # verdict
    out.append("")
    out.append("─" * 64)
    if B:
        out.append(f"  ✗ BLOCKERS ({len(B)}) — resolve before any push/merge:")
        for b in B:
            out.append(f"      • {b}")
    elif W:
        out.append(f"  ⚠  WARNINGS ({len(W)}) — review before acting:")
        for w in W:
            out.append(f"      • {w}")
    else:
        out.append("  ✓ clean — safe to proceed")
    out.append("─" * 64)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Git situational-awareness snapshot (read-only).")
    ap.add_argument("--repo", default=os.getcwd(), help="repo path (default: cwd)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    a = ap.parse_args()

    snap = snapshot(a.repo)
    if a.json:
        print(json.dumps(snap, indent=2, default=str))
    else:
        print(render(snap))

    if snap["blockers"]:
        return 2
    if snap["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
