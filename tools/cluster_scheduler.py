#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multi-node cluster scheduler for the DGX-Spark git bridge (PROTOCOL-v2 plumbing).

The single-node bridge (`tools/spark_bridge.py` + `bridge/PROTOCOL.md`) drives ONE Spark over the
`spark-bridge` branch. This module is the foundational, **offline + deterministic** plumbing that
lets 2/4/8/16 Sparks safely share ONE job queue over GitHub — **no peer-to-peer network needed**.
Every node talks only to GitHub; coordination is git-mediated.

The hard parts and how they stay safe:

  * **Per-node status** — each node publishes `bridge/nodes/<node_id>/status.json`. `node_is_free`
    mirrors `spark_bridge.gpu_is_free`: the one-GPU-job invariant now holds PER NODE.
  * **Decentralized assignment** — `assigned_node(cmd_id, node_ids)` is a deterministic stable hash
    (sha1 of the cmd id) modulo the SORTED node list, so each node independently knows which commands
    are "its" to claim. No central dispatcher, no `random`, no clock.
  * **Claim / lease as git-push compare-and-swap** — a node claims a command by creating its
    `bridge/claims/<cmd_id>.json` and PUSHING. The first push fast-forwards (wins); losers get a
    non-fast-forward rejection, re-fetch, and see the winner's claim. The git ref is the mutex. A
    lease has a TTL so a crashed node's command can be re-queued (`lease_is_expired`).
  * **Fan-out** — `split_command` shards a parent command (e.g. over SEEDS) into sub-commands, each
    carrying its shard + `parentId` and INHERITING the parent's human `approvedBy`. The gated rule is
    inherited: a gated parent with no `approvedBy` cannot fan out (`ValueError`). The scheduler never
    self-approves GPU work — it reuses `spark_bridge`'s allowlist + gating untouched.
  * **Aggregation** — `collect_shard_results` tells a collector when every shard has landed.

All core logic is pure (timestamps/`now` are passed in as args, like the rest of the repo). The only
git touch is a thin read wrapper (`_git_show_claim`) isolated from the logic, mirroring
`spark_bridge._git_show`. This is design/infra; no capability claim; canClaimAGI stays false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the single-node safety rules verbatim — do NOT widen them here.
from tools.spark_bridge import (  # noqa: E402
    ALLOWLIST,
    BRANCH,
    GATED,
    is_gated,
    validate_args,
)

NODES_DIR = "bridge/nodes"
CLAIMS_DIR = "bridge/claims"


# --- node id safety (SAME unsafe-id rule spark_bridge uses for command ids) --------------------
def is_safe_id(node_id: str) -> bool:
    """Filesystem-safe and non-empty — the exact rule `spark_bridge.build_command` applies to ids."""
    return bool(node_id) and not any(c in node_id for c in "/ \t")


def _require_safe_id(node_id: str, *, what: str = "node id") -> None:
    if not is_safe_id(node_id):
        raise ValueError(f"{what} must be filesystem-safe and non-empty")


# --- per-node status -----------------------------------------------------------------------------
def node_status_path(node_id: str) -> str:
    """`bridge/nodes/<node_id>/status.json` — where node <node_id> publishes its heartbeat."""
    _require_safe_id(node_id)
    return f"{NODES_DIR}/{node_id}/status.json"


def build_node_status(node_id: str, *, updated_at: str, running="", gpu_free_bytes: int = 0,
                      heartbeat_at: str = "") -> dict:
    """Compose a per-node status dict. `running` is a command id or null when idle."""
    _require_safe_id(node_id)
    return {
        "nodeId": node_id,
        "updatedAt": updated_at,
        "running": running if running else None,
        "gpuFreeBytes": int(gpu_free_bytes),
        "heartbeatAt": heartbeat_at or updated_at,
    }


def node_is_free(node_status: dict) -> bool:
    """The one-GPU-job invariant, PER NODE — mirrors `spark_bridge.gpu_is_free`: free iff nothing
    running and nothing pending on this node."""
    s = node_status or {}
    return s.get("running") in (None, "", "null") and not s.get("pendingCommands")


# --- decentralized, deterministic assignment ----------------------------------------------------
def assigned_node(cmd_id: str, node_ids: "list[str]") -> str:
    """Which node owns `cmd_id`. Deterministic + pure: sha1(cmd_id) mod the SORTED node list, so
    every node computes the SAME owner with no dispatcher, no `random`, no clock. Duplicate ids in
    `node_ids` collapse (the list is de-duplicated before sorting) so the owner is stable."""
    nodes = sorted({n for n in (node_ids or []) if n})
    if not nodes:
        raise ValueError("node_ids must be a non-empty list of node ids")
    for n in nodes:
        _require_safe_id(n)
    digest = hashlib.sha1(cmd_id.encode("utf-8")).hexdigest()
    return nodes[int(digest, 16) % len(nodes)]


# --- claim / lease (git-push compare-and-swap) --------------------------------------------------
def claim_path(cmd_id: str) -> str:
    """`bridge/claims/<cmd_id>.json` — the file whose creation+push is the cluster mutex for a cmd."""
    if not is_safe_id(cmd_id):
        raise ValueError("cmd id must be filesystem-safe and non-empty")
    return f"{CLAIMS_DIR}/{cmd_id}.json"


def build_claim(cmd_id: str, node_id: str, leased_at: str, ttl_seconds: int) -> dict:
    """A claim: `{cmdId,nodeId,leasedAt,ttlSeconds}`. A node writes this file then PUSHES; the first
    push fast-forwards and wins (the git ref is the compare-and-swap). Losers get a non-ff rejection,
    re-fetch, and defer to the winning claim. `node_id` is validated with the spark_bridge id rule."""
    _require_safe_id(node_id)
    if not is_safe_id(cmd_id):
        raise ValueError("cmd id must be filesystem-safe and non-empty")
    if int(ttl_seconds) <= 0:
        raise ValueError("ttlSeconds must be positive")
    return {
        "cmdId": cmd_id,
        "nodeId": node_id,
        "leasedAt": leased_at,
        "ttlSeconds": int(ttl_seconds),
    }


def lease_is_expired(claim: dict, now: float) -> bool:
    """True once `leasedAt + ttlSeconds < now` — then the cmd may be re-queued (e.g. a node crashed
    mid-lease). `leasedAt` is epoch seconds and `now` is passed in (no clock in core logic)."""
    if not claim:
        return True
    try:
        leased_at = float(claim.get("leasedAt"))
        ttl = float(claim.get("ttlSeconds"))
    except (TypeError, ValueError):
        return True
    return (leased_at + ttl) < float(now)


# --- fan-out (split a parent command into shards) -----------------------------------------------
def split_command(parent_cmd: dict, shards: "list") -> "list[dict]":
    """Split a parent bridge command into one sub-command per shard (e.g. a SEED list).

    Each sub-command inherits the parent's `args`, `createdBy`, and human `approvedBy`, and carries
    `parentId` + its `shard`. The GATED rule is inherited verbatim from spark_bridge: if the parent's
    args are gated (`--execute`/`--run-train`) and `approvedBy` is empty, fan-out is REFUSED
    (`ValueError`) — the scheduler never self-approves a GPU job. Sub-command ids are
    `<parentId>--shard-<shard>` and validated with the spark_bridge id rule."""
    parent_id = (parent_cmd or {}).get("id", "")
    args = (parent_cmd or {}).get("args", "")
    if not is_safe_id(parent_id):
        raise ValueError("parent id must be filesystem-safe and non-empty")
    ok, problems = validate_args(args)
    if not ok:
        raise ValueError(f"parent args rejected: {problems}")
    approved_by = (parent_cmd or {}).get("approvedBy", "") or ""
    if is_gated(args) and not approved_by.strip():
        raise ValueError("gated parent (--execute/--run-train) with no human approvedBy cannot fan "
                         "out — an AI does not self-approve a GPU job")
    if not shards:
        raise ValueError("shards must be a non-empty list")
    created_by = (parent_cmd or {}).get("createdBy", "")
    created_at = (parent_cmd or {}).get("createdAt", "")
    subs: "list[dict]" = []
    for shard in shards:
        sub_id = f"{parent_id}--shard-{shard}"
        if not is_safe_id(sub_id):
            raise ValueError(f"shard {shard!r} yields an unsafe sub-command id {sub_id!r}")
        subs.append({
            "id": sub_id,
            "args": args,
            "createdBy": created_by,
            "createdAt": created_at,
            "approvedBy": approved_by,
            "parentId": parent_id,
            "shard": shard,
        })
    return subs


# --- aggregation ---------------------------------------------------------------------------------
def collect_shard_results(parent_id: str, results: "list[dict]") -> dict:
    """Given the sub-commands' results, report which shards landed. A collector polls this until
    `complete` is true. A result belongs to `parent_id` iff its `parentId` matches; `missing` lists
    the shard ids of sub-commands with no result yet, derived from the `expectedShards` field that
    the fan-out emitter records (or, absent that, from results present)."""
    mine = [r for r in (results or []) if (r or {}).get("parentId") == parent_id]
    landed = {r.get("shard") for r in mine if "shard" in r}
    expected = set()
    for r in mine:
        for s in (r.get("expectedShards") or []):
            expected.add(s)
    expected |= landed
    missing = sorted([s for s in expected if s not in landed], key=lambda x: str(x))
    return {
        "parentId": parent_id,
        "shards": sorted(landed, key=lambda x: str(x)),
        "complete": bool(expected) and not missing,
        "missing": missing,
    }


# --- live read (git, read-only) — isolated thin wrapper, like spark_bridge._git_show ------------
def _git_show_claim(cmd_id: str) -> "dict | None":
    """Read a claim file off the bridge branch (read-only). Exercised live, not in CI."""
    path = claim_path(cmd_id)
    try:
        subprocess.run(["git", "fetch", "origin", BRANCH, "--quiet"], cwd=ROOT,
                       check=False, capture_output=True, timeout=60)
        out = subprocess.run(["git", "show", f"origin/{BRANCH}:{path}"], cwd=ROOT,
                             check=True, capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout) if out.stdout else None
    except Exception:
        return None


# --- offline invariants (same shape as spark_bridge.offline_invariants) -------------------------
def offline_invariants() -> "tuple[bool, dict]":
    checks: "dict[str, bool]" = {}

    # 1. allowlist/gating reused verbatim from spark_bridge (not widened)
    checks["allowlist_reused"] = ALLOWLIST == {
        "--dry-run", "--bench-a", "--bench-b", "--all", "--execute", "--run-train"} and \
        GATED == {"--execute", "--run-train"}

    # 2. no-double-claim: assigned_node is a SINGLE owner per cmd, and deterministic across calls
    nodes = ["spark-d", "spark-a", "spark-c", "spark-b"]
    a1 = assigned_node("cmd-42", nodes)
    a2 = assigned_node("cmd-42", list(reversed(nodes)))  # order must not matter (sorted internally)
    checks["assignment_single_owner"] = a1 in nodes
    checks["assignment_stable"] = (a1 == a2)

    # 3. lease expiry works
    claim = build_claim("cmd-1", "spark-a", leased_at="1000", ttl_seconds=60)
    checks["lease_live"] = not lease_is_expired(claim, now=1030)
    checks["lease_expired"] = lease_is_expired(claim, now=1100)

    # 4. gated fan-out without approval is refused; with a human handle it inherits to every shard
    refused = False
    try:
        split_command({"id": "p1", "args": "--bench-a --execute", "createdBy": "claude"}, [1, 2, 3])
    except ValueError:
        refused = True
    checks["gated_fanout_without_approval_refused"] = refused
    subs = split_command(
        {"id": "p2", "args": "--bench-a --execute", "createdBy": "claude",
         "approvedBy": "user: 'go' (2026)"}, [1, 2, 10])
    checks["gated_fanout_inherits_approval"] = (len(subs) == 3 and
                                                all(s["approvedBy"] == "user: 'go' (2026)" for s in subs) and
                                                all(s["parentId"] == "p2" for s in subs))

    # dry-run fan-out needs no approval
    dry = split_command({"id": "p3", "args": "--dry-run --all", "createdBy": "claude"}, [1, 2])
    checks["dryrun_fanout_ok"] = (len(dry) == 2)

    # 5. unsafe node id refused (same rule as command ids)
    bad = False
    try:
        assigned_node("cmd-x", ["a/b", "spark-a"])
    except ValueError:
        bad = True
    checks["unsafe_node_id_refused"] = bad

    bad2 = False
    try:
        build_node_status("a b", updated_at="t")
    except ValueError:
        bad2 = True
    checks["unsafe_node_status_id_refused"] = bad2

    # 6. node_is_free mirrors gpu_is_free
    checks["node_free_logic"] = node_is_free({"running": None}) and \
        not node_is_free({"running": "cmd-9"}) and \
        not node_is_free({"running": None, "pendingCommands": ["q"]})

    # 7. aggregation completeness
    agg = collect_shard_results("p2", [
        {"parentId": "p2", "shard": 1, "expectedShards": [1, 2, 10]},
        {"parentId": "p2", "shard": 2},
    ])
    checks["aggregation_missing"] = (agg["missing"] == [10] and agg["complete"] is False)
    agg2 = collect_shard_results("p2", [
        {"parentId": "p2", "shard": 1, "expectedShards": [1, 2]},
        {"parentId": "p2", "shard": 2},
    ])
    checks["aggregation_complete"] = (agg2["complete"] is True and agg2["missing"] == [])

    return all(checks.values()), {"checks": checks}


# --- balance check used by tests + CLI ----------------------------------------------------------
def assignment_distribution(cmd_ids: "list[str]", node_ids: "list[str]") -> "dict[str, int]":
    """How many of `cmd_ids` each node owns — for inspecting load balance. Pure + deterministic."""
    nodes = sorted({n for n in node_ids if n})
    counts = {n: 0 for n in nodes}
    for c in cmd_ids:
        counts[assigned_node(c, nodes)] += 1
    return counts


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    # NOTE: dest must NOT be "cmd" — the assign/claim subparsers define a `--cmd` option, which would
    # otherwise clobber the chosen subcommand name on the same namespace attribute.
    sub = ap.add_subparsers(dest="subcmd")

    pa = sub.add_parser("assign", help="print which node owns a command (deterministic)")
    pa.add_argument("--cmd", required=True)
    pa.add_argument("--nodes", required=True, help="comma-separated node ids")

    pc = sub.add_parser("claim", help="compose a claim JSON for <cmd> by <node> (does NOT push)")
    pc.add_argument("--cmd", required=True)
    pc.add_argument("--node", required=True)
    pc.add_argument("--leased-at", default="0")
    pc.add_argument("--ttl-seconds", type=int, default=3600)

    pf = sub.add_parser("plan-fanout", help="split a parent command into shard sub-commands")
    pf.add_argument("--parent", required=True, help="parent command id")
    pf.add_argument("--args", required=True, help="parent args (allowlisted)")
    pf.add_argument("--shards", required=True, help="comma-separated shard values")
    pf.add_argument("--created-by", default="claude-web")
    pf.add_argument("--approved-by", default="")

    args = ap.parse_args(argv)

    if args.self_test:
        ok, detail = offline_invariants()
        print("cluster_scheduler invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1

    if args.subcmd == "assign":
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
        try:
            owner = assigned_node(args.cmd, nodes)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(owner)
        return 0

    if args.subcmd == "claim":
        try:
            claim = build_claim(args.cmd, args.node, leased_at=args.leased_at,
                                ttl_seconds=args.ttl_seconds)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps({"path": claim_path(args.cmd), "claim": claim}, indent=2,
                         ensure_ascii=False))
        return 0

    if args.subcmd == "plan-fanout":
        shards: "list" = [s.strip() for s in args.shards.split(",") if s.strip()]
        parent = {"id": args.parent, "args": args.args, "createdBy": args.created_by,
                  "approvedBy": args.approved_by}
        try:
            subs = split_command(parent, shards)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(subs, indent=2, ensure_ascii=False))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
