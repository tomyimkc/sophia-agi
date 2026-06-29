#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cloud-side client for the Spark git bridge (the operator half of bridge/PROTOCOL.md).

The Spark runs ``tools/github_bridge_poll.py`` and talks ONLY to GitHub (the cloud session cannot
reach the Spark's Tailscale Funnel — egress returns ``connect_rejected 502``). This is the cloud
half: it **validates + composes** a command for ``bridge/commands/<id>.json`` and **reads** status +
results from the ``spark-bridge`` branch, so a cloud agent can drive the Spark safely and the same
discipline is reusable across sessions.

Safety, enforced here AND by the poller:
  * ``args`` tokens must ALL be in the allowlist; anything else is rejected before submit.
  * ``--execute`` / ``--run-train`` are GATED: a command carrying them MUST have a non-empty
    ``approvedBy`` (a human handle) or this client refuses to build it — mirroring the poller, which
    runs gated flags only with human approval. An AI does not self-approve a GPU job.
  * ``gpu_is_free(status)`` lets the operator hold the one-GPU-job invariant before an execute.

Reads use ``git show origin/spark-bridge:<path>`` (read-only); the actual commit of a command file
is done by the caller via the GitHub API. Pure validation/compose logic is deterministic + offline
(tested); the git reads are exercised live, not in CI. canClaimAGI stays false.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BRANCH = "spark-bridge"
ALLOWLIST = {"--dry-run", "--bench-a", "--bench-b", "--all", "--execute", "--run-train"}
GATED = {"--execute", "--run-train"}  # require a human approvedBy


def validate_args(args: str) -> "tuple[bool, list[str]]":
    """Every token must be allowlisted. Returns (ok, problems)."""
    problems: list[str] = []
    tokens = [t for t in (args or "").split() if t]
    if not tokens:
        problems.append("empty args")
    for t in tokens:
        if t not in ALLOWLIST:
            problems.append(f"token not allowlisted: {t!r}")
    return (not problems), problems


def is_gated(args: str) -> bool:
    return any(t in GATED for t in (args or "").split())


def build_command(command_id: str, args: str, *, created_by: str, approved_by: str = "",
                  note: str = "", created_at: str = "") -> dict:
    """Validate + compose a bridge command dict. Raises ValueError on an allowlist violation or on a
    GATED command with no human ``approved_by`` (the no-self-approval rule)."""
    ok, problems = validate_args(args)
    if not ok:
        raise ValueError(f"args rejected: {problems}")
    if is_gated(args) and not approved_by.strip():
        raise ValueError("gated flag (--execute/--run-train) requires a non-empty human approvedBy "
                         "— an AI does not self-approve a GPU job")
    if not command_id or any(c in command_id for c in "/ \t"):
        raise ValueError("id must be filesystem-safe and non-empty")
    return {
        "id": command_id, "args": args, "createdBy": created_by,
        "createdAt": created_at, "approvedBy": approved_by, "note": note,
    }


def gpu_is_free(status: dict) -> bool:
    """The one-GPU-job invariant guard: free iff nothing running and nothing pending."""
    return (status or {}).get("running") in (None, "", "null") and not (status or {}).get("pendingCommands")


# --- live reads (git, read-only) --------------------------------------------------------------
def _git_show(path: str) -> "str | None":
    try:
        subprocess.run(["git", "fetch", "origin", BRANCH, "--quiet"], cwd=ROOT,
                       check=False, capture_output=True, timeout=60)
        out = subprocess.run(["git", "show", f"origin/{BRANCH}:{path}"], cwd=ROOT,
                             check=True, capture_output=True, text=True, timeout=30)
        return out.stdout
    except Exception:
        return None


def read_status() -> "dict | None":
    raw = _git_show("bridge/STATUS.json")
    return json.loads(raw) if raw else None


def read_result(command_id: str) -> "dict | None":
    raw = _git_show(f"bridge/results/{command_id}.json")
    return json.loads(raw) if raw else None


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    checks["allow_dryrun"] = validate_args("--dry-run --all")[0]
    checks["reject_unknown"] = not validate_args("--rm -rf /")[0]
    checks["reject_empty"] = not validate_args("")[0]
    checks["gated_detected"] = is_gated("--bench-a --execute") and not is_gated("--dry-run --all")

    # build refuses a gated command with no approval, allows it with one
    refused = False
    try:
        build_command("x", "--bench-a --execute", created_by="claude")
    except ValueError:
        refused = True
    checks["gated_without_approval_refused"] = refused
    ok_cmd = build_command("x", "--bench-a --execute", created_by="claude",
                           approved_by="user: 'run it' (2026)")
    checks["gated_with_approval_built"] = ok_cmd["approvedBy"] != ""
    checks["dryrun_no_approval_ok"] = build_command("y", "--dry-run --all", created_by="claude")["args"] == "--dry-run --all"

    checks["gpu_free_logic"] = gpu_is_free({"running": None, "pendingCommands": []}) and \
        not gpu_is_free({"running": "olmoe", "pendingCommands": []}) and \
        not gpu_is_free({"running": None, "pendingCommands": ["z"]})

    # id safety
    bad_id = False
    try:
        build_command("a/b c", "--dry-run", created_by="claude")
    except ValueError:
        bad_id = True
    checks["unsafe_id_refused"] = bad_id

    return all(checks.values()), {"checks": checks}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--self-test", action="store_true")

    pc = sub.add_parser("compose", help="validate + print a command JSON (does NOT submit)")
    pc.add_argument("--id", required=True)
    pc.add_argument("--args", required=True)
    pc.add_argument("--created-by", default="claude-web")
    pc.add_argument("--approved-by", default="")
    pc.add_argument("--note", default="")
    pc.add_argument("--created-at", default="")

    sub.add_parser("status", help="print the live Spark STATUS.json")
    pr = sub.add_parser("result", help="print a result by id")
    pr.add_argument("--id", required=True)

    args = ap.parse_args(argv)
    if args.self_test or args.cmd is None and not any(vars(args).get(k) for k in ("cmd",)):
        if args.self_test:
            ok, detail = offline_invariants()
            print("spark_bridge invariants:", "PASS" if ok else "FAIL")
            for k, v in detail["checks"].items():
                print(f"  [{'ok' if v else 'XX'}] {k}")
            return 0 if ok else 1

    if args.cmd == "compose":
        try:
            cmd = build_command(args.id, args.args, created_by=args.created_by,
                                approved_by=args.approved_by, note=args.note, created_at=args.created_at)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(cmd, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "status":
        print(json.dumps(read_status(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "result":
        print(json.dumps(read_result(args.id), indent=2, ensure_ascii=False))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
