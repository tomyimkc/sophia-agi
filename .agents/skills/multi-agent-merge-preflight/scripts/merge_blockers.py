#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a structured report of what blocks PR <N> from merging in tomyimkc/sophia-agi.

Catches the easy-to-miss blockers that green checks hide:
  - required status checks vs the PR's current check states (protection requires X; is X green?)
  - required_review_thread_resolution (unresolved threads block even at 0 approvals)
  - required approvals / reviewDecision
  - merge conflicts

This is a read-only diagnostic. It does not attempt merges or thread resolution.

Usage:
    python .agents/skills/multi-agent-merge-preflight/scripts/merge_blockers.py <PR_NUMBER>
    python .agents/skills/multi-agent-merge-preflight/scripts/merge_blockers.py 149

Exits 0 if no blockers found, 1 if any blocker is present (so it can gate CI/scripts).
"""
from __future__ import annotations

import json
import subprocess
import sys

OWNER = "tomyimkc"
NAME = "sophia-agi"


def gh_graphql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"gh graphql failed: {r.stderr.strip()[:300]}")
    data = json.loads(r.stdout)
    if "errors" in data:
        raise RuntimeError(f"graphql errors: {json.dumps(data['errors'])[:400]}")
    return data["data"]


def gh_rest(path: str) -> dict:
    r = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True
    )
    if r.returncode != 0:
        # 404 etc. -> return the message envelope so callers can degrade gracefully
        try:
            return json.loads(r.stdout)
        except Exception:
            return {"_error": r.stderr.strip()[:200], "_rc": r.returncode}
    return json.loads(r.stdout)


def fetch_ruleset_required_checks() -> set[str]:
    """Return the set of required status-check contexts across all active branch rulesets."""
    contexts: set[str] = set()
    rulesets = gh_rest(f"repos/{OWNER}/{NAME}/rulesets")
    if not isinstance(rulesets, list):
        return contexts
    for rs in rulesets:
        if rs.get("target") != "branch" or rs.get("enforcement") != "active":
            continue
        detail = gh_rest(f"repos/{OWNER}/{NAME}/rulesets/{rs['id']}")
        for rule in detail.get("rules", []):
            if rule.get("type") == "required_status_checks":
                for c in rule.get("parameters", {}).get("required_status_checks", []):
                    contexts.add(c.get("context"))
            if rule.get("type") == "pull_request":
                # stash the PR-rule params on the set as a side channel via an attr-ish dict
                pull_req_params.update(rule.get("parameters", {}))
    return contexts


# module-level side channel for the pull_request rule params (thread resolution, approvals)
pull_req_params: dict = {}


def main(pr_number: int) -> int:
    blockers: list[str] = []

    # 1. PR basic state
    pr = gh_rest(f"repos/{OWNER}/{NAME}/pulls/{pr_number}")
    if "_error" in pr or "message" in pr and not pr.get("number"):
        # GitHub returns {"message": "...", "status": "404"} for missing/forbidden PRs.
        msg = pr.get("_error") or pr.get("message")
        print(f"ERROR: could not fetch PR #{pr_number}: {msg}")
        return 2
    state = pr.get("state")
    mergeable = pr.get("mergeable")        # true / false / null(unknown)
    mstate = pr.get("mergeable_state")     # clean / blocked / dirty / unknown
    head = (pr.get("head") or {}).get("sha", "")[:10]
    # REST returns state lowercase ("open"/"closed"); normalize for comparison.
    print(f"PR #{pr_number}: state={state} mergeable={mergeable} mergeable_state={mstate} head={head}")
    if str(state).lower() != "open":
        blockers.append(f"PR is not OPEN (state={state}) — nothing to unblock.")
    if mergeable is False or mstate == "dirty":
        blockers.append("Merge CONFLICT — rebase/merge main into the branch first.")

    # 2. Required checks vs current check states
    required = fetch_ruleset_required_checks()
    if required:
        print(f"\nRequired checks ({len(required)}): {sorted(required)}")
        checks = gh_graphql(
            """query($o:String!,$n:String!,$p:Int!){
              repository(owner:$o,name:$n){
                pullRequest(number:$p){
                  commits(last:1){ nodes{ commit{
                    statusCheckRollup{
                      state
                      contexts(first:100){ nodes{
                        ... on StatusContext{ context state }
                        ... on CheckRun{ name conclusion status }
                      } }
                    } } } }
                  } } }""",
            {"o": OWNER, "n": NAME, "p": pr_number},
        )
        ctx_states: dict[str, str] = {}
        roll = (
            checks["repository"]["pullRequest"]["commits"]["nodes"][0]["commit"][
                "statusCheckRollup"
            ]
        )
        for c in roll.get("contexts", {}).get("nodes", []):
            name = c.get("context") or c.get("name")
            st = c.get("state") or c.get("conclusion") or c.get("status") or "UNKNOWN"
            ctx_states[name] = str(st).upper()
        for req in sorted(required):
            st = ctx_states.get(req)
            if st in (None, "", "PENDING", "QUEUED", "IN_PROGRESS", "NEUTRAL"):
                blockers.append(f"Required check '{req}' is missing/pending (state={st}).")
            elif st not in ("SUCCESS", "PASS", "PASSED"):
                blockers.append(f"Required check '{req}' is {st} (not success).")
    else:
        print("\nNo required status checks found on any active branch ruleset.")

    # 3. Review threads (the silent blocker)
    if pull_req_params.get("required_review_thread_resolution"):
        threads = gh_graphql(
            """query($o:String!,$n:String!,$p:Int!){
              repository(owner:$o,name:$n){
                pullRequest(number:$p){
                  reviewThreads(first:100){ nodes{ isResolved path } } } } }""",
            {"o": OWNER, "n": NAME, "p": pr_number},
        )
        nodes = threads["repository"]["pullRequest"]["reviewThreads"]["nodes"]
        unresolved = [t for t in nodes if not t["isResolved"]]
        print(f"\nReview threads: {len(nodes)} total, {len(unresolved)} UNRESOLVED "
              f"(required_review_thread_resolution=true)")
        if unresolved:
            blockers.append(
                f"{len(unresolved)} unresolved review thread(s) block the merge "
                f"(required_review_thread_resolution=true). Paths: "
                + ", ".join(sorted({t["path"] for t in unresolved}))
            )
    else:
        print("\nrequired_review_thread_resolution NOT set.")

    # 4. Approvals
    need = pull_req_params.get("required_approving_review_count", 0)
    decision = gh_rest(f"repos/{OWNER}/{NAME}/pulls/{pr_number}/reviews")
    approved = sum(1 for r in decision if r.get("state") == "APPROVED") if isinstance(decision, list) else 0
    if need and approved < need:
        blockers.append(f"Needs {need} approving review(s), has {approved}.")
    print(f"\nApprovals: {approved}/{need} required.")

    # 5. Verdict
    print("\n" + "=" * 60)
    if blockers:
        print(f"BLOCKED — {len(blockers)} blocker(s):")
        for b in blockers:
            print(f"  - {b}")
        return 1
    print("NO BLOCKERS DETECTED — the PR should merge if mergeable_state is CLEAN.")
    print("(If merge still refuses, re-run after GitHub recomputes mergeable_state.)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print(__doc__)
        sys.exit(2)
    sys.exit(main(int(sys.argv[1])))
