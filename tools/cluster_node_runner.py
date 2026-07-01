#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Per-node multi-node runner for the DGX-Spark git bridge (PROTOCOL-v2 node daemon).

This is the daemon that runs ON each Spark so the cluster self-coordinates over GitHub with NO
Claude session attached. It generalizes the single-node poller (`tools/github_bridge_poll.py`,
branch `spark-bridge`) to many nodes sharing ONE job queue, using the already-landed
`tools/cluster_scheduler.py` plumbing — every node talks only to GitHub; coordination is
git-mediated (the claim file's push is the cluster mutex). There is no central dispatcher, no
peer-to-peer network, no `random`, and no clock in the decision logic.

How the parts split (the same discipline as `spark_bridge._git_show`):

  * **PURE core (`decide`)** — given `(node_id, all_node_ids, pending_command_ids, existing_claims,
    my_node_status, now, commands_by_id)` it returns the next ACTION as a plain dict. No git, no
    exec, no GPU, no clock — `now`/timestamps are passed in. Deterministic and unit-tested in CI.
    The decision:
      - find pending commands `assigned_node(cmd_id, sorted node ids) == node_id` (decentralized,
        deterministic ownership; each node independently knows what is "its");
      - among those, drop any with a LIVE (unexpired) claim — `lease_is_expired` requeues a crashed
        node's command;
      - a command carrying `--execute`/`--run-train` whose dict lacks a human `approvedBy` is
        REFUSED (`{"kind": "refuse-gated"}`) — reusing `spark_bridge` gating; an AI never
        self-approves a GPU job;
      - if I already hold a live claim on an assigned, free command -> `{"kind": "run"}`;
      - else if the node is free -> claim the highest-priority eligible command -> `{"kind":
        "claim"}`;
      - else (busy, or nothing eligible) -> `{"kind": "idle"}`.

  * **THIN impure wrappers** (clearly separated, exercised live, NOT unit-tested in CI) — modelled
    on `github_bridge_poll` but per-node: `_sync` (ff-only fetch+merge of the branch),
    `_read_pending`/`_read_claims`/`_read_command`, `_write_claim` (+commit/push as the
    compare-and-swap), `_publish_node_status` (writes `bridge/nodes/<id>/status.json`), and `_run`
    (Popen `scripts/run_local_benchmarks.sh` with allowlisted toks, cwd=root). `tick()` composes
    them.

Safety is INHERITED verbatim from `spark_bridge` + `cluster_scheduler` — the allowlist is NOT
widened, gated flags require a human `approvedBy`, the one-GPU-job invariant holds PER NODE
(`node_is_free`), and only the assigned node acts on a given command. This is design/infra; no
capability claim; canClaimAGI stays false.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the single-node + multi-node safety rules verbatim — do NOT widen them here.
from tools.cluster_scheduler import (  # noqa: E402
    assigned_node,
    build_claim,
    build_node_status,
    claim_path,
    lease_is_expired,
    node_is_free,
    node_status_path,
)
from tools.spark_bridge import (  # noqa: E402
    BRANCH,
    is_gated,
    validate_args,
)

RUNNER = "scripts/run_local_benchmarks.sh"
DEFAULT_TTL_SECONDS = 3600
MAX_TAIL = 64 * 1024


# =================================================================================================
# PURE CORE — no git, no exec, no GPU, no clock. `now`/timestamps passed in. Unit-tested in CI.
# =================================================================================================
def _command_is_gated_without_approval(cmd: "dict | None") -> bool:
    """True iff this command carries a gated flag (`--execute`/`--run-train`) but has no human
    `approvedBy`. Reuses `spark_bridge.is_gated`; an AI does not self-approve a GPU job."""
    cmd = cmd or {}
    args = str(cmd.get("args", ""))
    approved_by = str(cmd.get("approvedBy", "") or "")
    return is_gated(args) and not approved_by.strip()


def _live_claim_owner(cmd_id: str, existing_claims: "dict", now: float) -> "str | None":
    """The node holding a LIVE (unexpired) claim on `cmd_id`, or None. Pure: `now` passed in."""
    claim = (existing_claims or {}).get(cmd_id)
    if not claim:
        return None
    if lease_is_expired(claim, now):
        return None
    return claim.get("nodeId")


def my_assigned_pending(node_id: str, all_node_ids: "list[str]",
                        pending_command_ids: "list[str]") -> "list[str]":
    """The subset of pending command ids that THIS node owns, in priority order (the order they
    appear in `pending_command_ids`, which the caller sorts — typically lexicographic by id).

    Pure + deterministic: `assigned_node` is a stable sha1-mod over the SORTED node list, so every
    node computes the same owner with no dispatcher and no clock."""
    return [c for c in (pending_command_ids or [])
            if assigned_node(c, all_node_ids) == node_id]


def decide(node_id: str, all_node_ids: "list[str]", pending_command_ids: "list[str]",
           existing_claims: "dict", my_node_status: "dict", now: float,
           commands_by_id: "dict | None" = None) -> dict:
    """PURE decision: what should THIS node do next? Returns an action dict.

    Args (all passed in — no git, no clock):
      node_id              -- this node's id
      all_node_ids         -- the known node-id set (sorted internally by assigned_node)
      pending_command_ids  -- command ids with no result yet, in priority order
      existing_claims      -- {cmd_id: claim_dict} of currently published claims
      my_node_status       -- this node's status dict (running / pendingCommands)
      now                  -- epoch seconds, for lease expiry (NO clock in core)
      commands_by_id       -- {cmd_id: command_dict} for gating checks (args/approvedBy)

    Action kinds:
      {"kind": "run",          "cmdId": id} -- I hold a live claim on an assigned free cmd; run it.
      {"kind": "claim",        "cmdId": id, "claimNodeId": node_id} -- claim the top eligible cmd.
      {"kind": "refuse-gated", "cmdId": id} -- assigned cmd is gated with no human approvedBy.
      {"kind": "idle"}                       -- busy, or nothing eligible for me.
    """
    commands_by_id = commands_by_id or {}
    mine = my_assigned_pending(node_id, all_node_ids, pending_command_ids)
    free = node_is_free(my_node_status or {})

    # 1. If I already hold a LIVE claim on one of my assigned, still-pending commands, run it
    #    (only when free — the one-GPU-job invariant holds PER NODE). A gated-without-approval
    #    command is refused even if somehow claimed.
    for cmd_id in mine:
        if _live_claim_owner(cmd_id, existing_claims, now) == node_id:
            if _command_is_gated_without_approval(commands_by_id.get(cmd_id)):
                return {"kind": "refuse-gated", "cmdId": cmd_id}
            if free:
                return {"kind": "run", "cmdId": cmd_id}
            # I hold the claim but I'm busy with another job -> idle until it frees.
            return {"kind": "idle"}

    # 2. Otherwise look for the top eligible command to CLAIM: assigned to me, not gated-without-
    #    approval, and with no LIVE claim held by anyone.
    for cmd_id in mine:
        owner = _live_claim_owner(cmd_id, existing_claims, now)
        if owner is not None:
            # someone (me or a peer) holds a live claim; if it's a peer, defer; if me, handled above.
            continue
        if _command_is_gated_without_approval(commands_by_id.get(cmd_id)):
            return {"kind": "refuse-gated", "cmdId": cmd_id}
        if not free:
            # eligible work exists but I'm busy -> idle (don't break the per-node GPU invariant).
            return {"kind": "idle"}
        return {"kind": "claim", "cmdId": cmd_id, "claimNodeId": node_id}

    return {"kind": "idle"}


# =================================================================================================
# THIN IMPURE WRAPPERS — git / exec / GPU. Exercised LIVE on the Spark, NOT unit-tested in CI.
# Modelled on tools/github_bridge_poll.py but per-node.
# =================================================================================================
def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_epoch() -> float:
    return _dt.datetime.now(_dt.timezone.utc).timestamp()


def _git(*args, cwd, check=False):
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _sync(root: Path, branch: str) -> None:
    """ff-only sync (avoids re-smudging git-crypt files); rebase then reset as fallbacks.
    Same as github_bridge_poll._sync."""
    _git("fetch", "origin", branch, cwd=root)
    if _git("merge", "--ff-only", f"origin/{branch}", cwd=root).returncode == 0:
        return
    if _git("pull", "--rebase", "origin", branch, cwd=root).returncode == 0:
        return
    sys.stderr.write("[node-runner] ff + rebase failed; last-resort reset --hard\n")
    _git("reset", "--hard", f"origin/{branch}", cwd=root)


def _push_with_retry(root: Path, branch: str) -> bool:
    delay = 2
    for attempt in range(4):
        if _git("push", "origin", branch, cwd=root).returncode == 0:
            return True
        sys.stderr.write(f"[node-runner] push failed (try {attempt+1}); rebasing\n")
        time.sleep(delay)
        delay *= 2
        _git("pull", "--rebase", "origin", branch, cwd=root)
    sys.stderr.write("[node-runner] push giving up this tick; retry next tick\n")
    return False


def _read_command(root: Path, cmd_id: str) -> "dict | None":
    p = root / "bridge" / "commands" / f"{cmd_id}.json"
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _read_pending(root: Path) -> "list[str]":
    """Pending command ids (no result yet), sorted lexicographically for a stable priority order."""
    cmds = sorted((root / "bridge" / "commands").glob("*.json"))
    done = {p.stem for p in (root / "bridge" / "results").glob("*.json")}
    return [p.stem for p in cmds if p.stem not in done]


def _read_claims(root: Path) -> "dict":
    """All published claims as {cmd_id: claim_dict}, read off the working tree (synced ff-only)."""
    claims: "dict" = {}
    cdir = root / "bridge" / "claims"
    if not cdir.exists():
        return claims
    for p in sorted(cdir.glob("*.json")):
        try:
            claims[p.stem] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
    return claims


def _read_node_status(root: Path, node_id: str) -> "dict":
    p = root / node_status_path(node_id)
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _discover_nodes(root: Path) -> "list[str]":
    """Read the known node-id set from bridge/nodes/<id>/ directories."""
    ndir = root / "bridge" / "nodes"
    if not ndir.exists():
        return []
    return sorted(p.name for p in ndir.iterdir() if p.is_dir())


def _write_claim(root: Path, branch: str, cmd_id: str, node_id: str, *,
                 ttl_seconds: int = DEFAULT_TTL_SECONDS) -> bool:
    """Create bridge/claims/<cmd_id>.json and PUSH — the git ref is the compare-and-swap. The first
    push fast-forwards (wins); a loser gets a non-ff rejection on push, re-syncs next tick, and sees
    the winner's claim via decide(). Returns True iff the push landed (i.e. we won the claim)."""
    claim = build_claim(cmd_id, node_id, leased_at=str(int(_now_epoch())), ttl_seconds=ttl_seconds)
    p = root / claim_path(cmd_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(claim, indent=2) + "\n")
    _git("add", str(p), cwd=root)
    if _git("commit", "-m", f"bridge: {node_id} claims {cmd_id} @ {_now_iso()}",
            cwd=root).returncode != 0:
        return False
    return _push_with_retry(root, branch)


def _publish_node_status(root: Path, branch: str, node_id: str, *, running: str = "",
                         pending_ids: "list[str] | None" = None) -> None:
    """Write bridge/nodes/<id>/status.json and commit/push (per-node heartbeat)."""
    status = build_node_status(node_id, updated_at=_now_iso(), running=running)
    status["pendingCommands"] = list(pending_ids or [])
    p = root / node_status_path(node_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, indent=2) + "\n")
    _git("add", str(p), cwd=root)
    if _git("commit", "-m", f"bridge: {node_id} status @ {_now_iso()} (running {running or '—'})",
            cwd=root).returncode == 0:
        _push_with_retry(root, branch)


def _run(root: Path, cmd: dict):
    """Launch scripts/run_local_benchmarks.sh for `cmd` with ALLOWLISTED toks, cwd=root. Returns a
    Popen (non-blocking, like github_bridge_poll._start). Raises ValueError on a disallowed token or
    a gated flag without a human approvedBy (the no-self-approval rule, enforced again at exec)."""
    args = str(cmd.get("args", ""))
    ok, problems = validate_args(args)
    if not ok:
        raise ValueError(f"args rejected: {problems}")
    if _command_is_gated_without_approval(cmd):
        raise ValueError("gated flag (--execute/--run-train) requires a non-empty human approvedBy "
                         "— an AI does not self-approve a GPU job")
    toks = args.split()
    fd, outpath = tempfile.mkstemp(prefix=f"node-{cmd.get('id', 'job')}-", suffix=".out")
    fh = open(fd, "w")
    proc = subprocess.Popen(["bash", RUNNER, *toks], cwd=str(root), stdout=fh,
                            stderr=subprocess.STDOUT, text=True)
    return proc, outpath, fh


# =================================================================================================
# tick() — composes the wrappers around the pure decide(). The one impure orchestration point.
# =================================================================================================
def tick(root: Path, branch: str, node_id: str, all_node_ids: "list[str]", *,
         dry: bool = False) -> dict:
    """One coordination tick. Syncs, reads the shared state, runs the PURE decide(), and (unless
    `dry`) acts on the decision. Returns the decided action dict."""
    if not dry:
        _sync(root, branch)
    (root / "bridge" / "commands").mkdir(parents=True, exist_ok=True)
    (root / "bridge" / "results").mkdir(parents=True, exist_ok=True)

    nodes = all_node_ids or _discover_nodes(root) or [node_id]
    pending = _read_pending(root)
    claims = _read_claims(root)
    my_status = _read_node_status(root, node_id)
    commands_by_id = {cid: (_read_command(root, cid) or {}) for cid in pending}

    action = decide(node_id, nodes, pending, claims, my_status, _now_epoch(), commands_by_id)

    if dry:
        return action

    if action["kind"] == "claim":
        won = _write_claim(root, branch, action["cmdId"], node_id)
        sys.stderr.write(f"[node-runner] {node_id} claim {action['cmdId']} -> "
                         f"{'won' if won else 'lost (peer claimed first)'}\n")
        # status published next tick; the claim commit already pushed.
        return action

    if action["kind"] == "run":
        cmd = _read_command(root, action["cmdId"]) or {"id": action["cmdId"]}
        try:
            proc, outpath, fh = _run(root, cmd)
        except ValueError as exc:
            sys.stderr.write(f"[node-runner] {node_id} refuses to run {action['cmdId']}: {exc}\n")
            return {"kind": "refuse-gated", "cmdId": action["cmdId"]}
        _publish_node_status(root, branch, node_id, running=action["cmdId"], pending_ids=pending)
        sys.stderr.write(f"[node-runner] {node_id} started {action['cmdId']} pid={proc.pid}\n")
        # Non-blocking: this tick returns; the long job runs detached. A result-collector (the
        # single-node poller pattern) reaps it. We keep one GPU job per node by only ever running
        # when node_is_free.
        return action

    if action["kind"] == "refuse-gated":
        sys.stderr.write(f"[node-runner] {node_id} REFUSES gated cmd {action['cmdId']} "
                         f"(no human approvedBy)\n")
        return action

    # idle: still publish a heartbeat so peers see this node alive.
    _publish_node_status(root, branch, node_id, running="", pending_ids=pending)
    return action


# =================================================================================================
# offline invariants over the PURE core (same shape as spark_bridge.offline_invariants).
# =================================================================================================
def offline_invariants() -> "tuple[bool, dict]":
    checks: "dict[str, bool]" = {}
    nodes2 = ["spark-a", "spark-b"]
    nodes4 = ["spark-a", "spark-b", "spark-c", "spark-d"]
    free = {"running": None}
    busy = {"running": "other-cmd"}

    # 1. Assignment ownership is EXCLUSIVE: across a set of cmds, every cmd has exactly one owner,
    #    and only that owner gets a non-idle action for it.
    pending = [f"cmd-{i}" for i in range(20)]
    owners = {c: assigned_node(c, nodes4) for c in pending}
    exclusive = True
    for c in pending:
        actors = []
        for n in nodes4:
            act = decide(n, nodes4, [c], {}, free, now=1000.0,
                         commands_by_id={c: {"id": c, "args": "--dry-run --all"}})
            if act["kind"] != "idle":
                actors.append(n)
        # exactly one node acts, and it is the assigned owner
        if actors != [owners[c]]:
            exclusive = False
            break
    checks["assignment_exclusive_owner"] = exclusive

    # 2. A LIVE claim by one node blocks a SECOND node (contention): the non-owner always idles, and
    #    even the owner idles while a live claim is held by a peer.
    c = "cmd-7"
    owner = assigned_node(c, nodes2)
    other = [n for n in nodes2 if n != owner][0]
    live_claim = {c: build_claim(c, other, leased_at="1000", ttl_seconds=600)}
    cmds = {c: {"id": c, "args": "--dry-run --all"}}
    # owner sees a peer's live claim -> idle (defers)
    a_owner = decide(owner, nodes2, [c], live_claim, free, now=1100.0, commands_by_id=cmds)
    # the claim-holder (other) does NOT own c by assignment -> idle (not its cmd)
    a_other = decide(other, nodes2, [c], live_claim, free, now=1100.0, commands_by_id=cmds)
    checks["live_claim_blocks_second_node"] = (a_owner["kind"] == "idle" and
                                               a_other["kind"] == "idle")

    # 3. Owner with NO live claim claims it; with its OWN live claim runs it.
    a_claim = decide(owner, nodes2, [c], {}, free, now=1100.0, commands_by_id=cmds)
    own_claim = {c: build_claim(c, owner, leased_at="1000", ttl_seconds=600)}
    a_run = decide(owner, nodes2, [c], own_claim, free, now=1100.0, commands_by_id=cmds)
    checks["owner_claims_then_runs"] = (a_claim["kind"] == "claim" and
                                        a_claim["cmdId"] == c and
                                        a_run["kind"] == "run" and a_run["cmdId"] == c)

    # 4. EXPIRED lease frees the cmd: the owner re-claims it.
    expired = {c: build_claim(c, other, leased_at="1000", ttl_seconds=60)}
    a_requeue = decide(owner, nodes2, [c], expired, free, now=5000.0, commands_by_id=cmds)
    checks["expired_lease_requeues"] = (a_requeue["kind"] == "claim" and a_requeue["cmdId"] == c)

    # 5. GATED command with no human approvedBy is REFUSED (never claimed, never run).
    cg = "cmd-gate"
    gowner = assigned_node(cg, nodes2)
    gcmds = {cg: {"id": cg, "args": "--bench-a --execute", "createdBy": "claude"}}
    a_gate = decide(gowner, nodes2, [cg], {}, free, now=1000.0, commands_by_id=gcmds)
    checks["gated_without_approval_refused"] = (a_gate["kind"] == "refuse-gated")
    # with a human approvedBy, the SAME gated command is claimable
    gcmds_ok = {cg: {"id": cg, "args": "--bench-a --execute", "createdBy": "claude",
                     "approvedBy": "user: 'go' (2026)"}}
    a_gate_ok = decide(gowner, nodes2, [cg], {}, free, now=1000.0, commands_by_id=gcmds_ok)
    checks["gated_with_approval_claimable"] = (a_gate_ok["kind"] == "claim")

    # 6. A BUSY node idles even on its own assigned cmd (one GPU job per node).
    cb = "cmd-busy"
    bowner = assigned_node(cb, nodes2)
    a_busy = decide(bowner, nodes2, [cb], {}, busy, now=1000.0,
                    commands_by_id={cb: {"id": cb, "args": "--dry-run --all"}})
    checks["busy_node_idles"] = (a_busy["kind"] == "idle")

    # 7. Only the assigned node acts on a given cmd (non-owner always idles, no claim).
    non_owners_idle = all(
        decide(n, nodes4, [c], {}, free, now=1000.0,
               commands_by_id={c: {"id": c, "args": "--dry-run --all"}})["kind"] == "idle"
        for c in ["cmd-1", "cmd-2", "cmd-3"]
        for n in nodes4 if n != assigned_node(c, nodes4)
    )
    checks["only_assigned_node_acts"] = non_owners_idle

    return all(checks.values()), {"checks": checks}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline invariants over the PURE core and exit")
    ap.add_argument("--node-id", help="this node's id (required to run/tick)")
    ap.add_argument("--branch", default=BRANCH, help="bridge branch (default spark-bridge)")
    ap.add_argument("--interval", type=int, default=30, help="seconds between ticks")
    ap.add_argument("--nodes", default="",
                    help="comma-separated known node-id set; if empty, read from bridge/nodes/")
    ap.add_argument("--trainwatch", default="http://127.0.0.1:8420/api/runs",
                    help="local TrainWatch URL (for status mirroring by the collector)")
    ap.add_argument("--repo-dir", default=str(ROOT), type=Path)
    ap.add_argument("--once", action="store_true",
                    help="one DRY tick: print the decided action without syncing/claiming/running")

    args = ap.parse_args(argv)

    if args.self_test:
        ok, detail = offline_invariants()
        print("cluster_node_runner invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1

    if not args.node_id:
        ap.error("--node-id is required to run (or use --self-test)")

    root = args.repo_dir.resolve()
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    if args.once:
        # DRY: decide using the on-disk state with NO sync/claim/run side effects, print the action.
        node_set = nodes or _discover_nodes(root) or [args.node_id]
        pending = _read_pending(root)
        claims = _read_claims(root)
        my_status = _read_node_status(root, args.node_id)
        commands_by_id = {cid: (_read_command(root, cid) or {}) for cid in pending}
        action = decide(args.node_id, node_set, pending, claims, my_status, _now_epoch(),
                        commands_by_id)
        print(json.dumps({
            "nodeId": args.node_id,
            "knownNodes": sorted(set(node_set)),
            "pending": pending,
            "action": action,
        }, indent=2, ensure_ascii=False))
        return 0

    sys.stderr.write(f"[node-runner] node={args.node_id} branch={args.branch} "
                     f"interval={args.interval}s nodes={nodes or 'auto'} root={root}\n")
    while True:
        try:
            action = tick(root, args.branch, args.node_id, nodes)
            sys.stderr.write(f"[node-runner] tick @ {_now_iso()} -> {action}\n")
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[node-runner] tick error: {exc}\n")
        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
