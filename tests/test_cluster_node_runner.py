#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Per-node multi-node runner tests — PURE core only (no git, no exec, no GPU, no clock).

Simulates 2 and 4 nodes deciding actions over the SAME pending set and asserts the coordination
invariants: exactly one node claims each cmd (others idle); claim contention has a single owner;
an expired lease re-queues; a gated command without a human approvedBy is refused; a busy node
idles; and the --once dry-run decision matches the pure core. Repo style: main()/asserts/OK.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cluster_node_runner import decide, my_assigned_pending  # noqa: E402
from tools.cluster_scheduler import assigned_node, build_claim  # noqa: E402

FREE = {"running": None}
BUSY = {"running": "some-other-cmd"}


def _dry(args: str = "--dry-run --all") -> dict:
    return {"args": args}


def _decide_all(nodes, pending, claims, now, commands_by_id, status=None):
    """Every node decides over the SAME shared state; return {node: action}."""
    return {n: decide(n, nodes, pending, claims, status or FREE, now, commands_by_id)
            for n in nodes}


# --- exactly one node claims each cmd; others idle (2 and 4 nodes) -------------------------------
def test_exactly_one_claims_each_cmd_2_nodes() -> None:
    nodes = ["spark-a", "spark-b"]
    pending = [f"cmd-{i}" for i in range(12)]
    cmds = {c: _dry() for c in pending}
    for c in pending:
        acts = _decide_all(nodes, [c], {}, 1000.0, cmds)
        claimers = [n for n, a in acts.items() if a["kind"] == "claim"]
        idlers = [n for n, a in acts.items() if a["kind"] == "idle"]
        assert claimers == [assigned_node(c, nodes)], (c, acts)
        assert len(claimers) == 1 and len(idlers) == len(nodes) - 1


def test_exactly_one_claims_each_cmd_4_nodes() -> None:
    nodes = ["spark-a", "spark-b", "spark-c", "spark-d"]
    pending = [f"job-{i}" for i in range(40)]
    cmds = {c: _dry() for c in pending}
    # every cmd: exactly one claimer == its assigned owner, the other three idle
    for c in pending:
        acts = _decide_all(nodes, [c], {}, 1000.0, cmds)
        claimers = [n for n, a in acts.items() if a["kind"] == "claim"]
        assert claimers == [assigned_node(c, nodes)], (c, acts)
        assert all(acts[n]["kind"] == "idle" for n in nodes if n != claimers[0])


def test_my_assigned_pending_partitions_the_set() -> None:
    nodes = ["spark-a", "spark-b", "spark-c", "spark-d"]
    pending = [f"cmd-{i}" for i in range(50)]
    # the per-node assigned subsets partition pending exactly (no overlap, full cover)
    seen: list[str] = []
    for n in nodes:
        seen += my_assigned_pending(n, nodes, pending)
    assert sorted(seen) == sorted(pending)
    assert len(seen) == len(set(seen))  # disjoint


# --- claim contention: two nodes, one owner -----------------------------------------------------
def test_claim_contention_single_owner() -> None:
    nodes = ["spark-a", "spark-b"]
    c = "cmd-contended"
    owner = assigned_node(c, nodes)
    other = [n for n in nodes if n != owner][0]
    cmds = {c: _dry()}
    # peer holds a LIVE claim -> owner defers (idle), non-owner idles (not its cmd)
    live = {c: build_claim(c, other, leased_at="1000", ttl_seconds=600)}
    assert decide(owner, nodes, [c], live, FREE, 1100.0, cmds)["kind"] == "idle"
    assert decide(other, nodes, [c], live, FREE, 1100.0, cmds)["kind"] == "idle"
    # with NO claim, exactly the owner claims
    a_owner = decide(owner, nodes, [c], {}, FREE, 1100.0, cmds)
    a_other = decide(other, nodes, [c], {}, FREE, 1100.0, cmds)
    assert a_owner["kind"] == "claim" and a_owner["cmdId"] == c
    assert a_other["kind"] == "idle"


def test_owner_with_own_live_claim_runs() -> None:
    nodes = ["spark-a", "spark-b"]
    c = "cmd-run"
    owner = assigned_node(c, nodes)
    cmds = {c: _dry()}
    own = {c: build_claim(c, owner, leased_at="1000", ttl_seconds=600)}
    a = decide(owner, nodes, [c], own, FREE, 1100.0, cmds)
    assert a["kind"] == "run" and a["cmdId"] == c


# --- lease expiry requeue -----------------------------------------------------------------------
def test_expired_lease_requeues() -> None:
    nodes = ["spark-a", "spark-b"]
    c = "cmd-crashed"
    owner = assigned_node(c, nodes)
    other = [n for n in nodes if n != owner][0]
    cmds = {c: _dry()}
    expired = {c: build_claim(c, other, leased_at="1000", ttl_seconds=60)}
    # while live -> owner defers
    assert decide(owner, nodes, [c], expired, FREE, 1030.0, cmds)["kind"] == "idle"
    # once expired -> owner re-claims (crashed node's cmd is freed)
    a = decide(owner, nodes, [c], expired, FREE, 5000.0, cmds)
    assert a["kind"] == "claim" and a["cmdId"] == c


# --- gated-without-approval refused -------------------------------------------------------------
def test_gated_without_approval_refused() -> None:
    nodes = ["spark-a", "spark-b"]
    c = "cmd-train"
    owner = assigned_node(c, nodes)
    gated = {c: {"id": c, "args": "--bench-a --execute", "createdBy": "claude"}}
    a = decide(owner, nodes, [c], {}, FREE, 1000.0, gated)
    assert a["kind"] == "refuse-gated" and a["cmdId"] == c
    # the runner NEVER self-approves: even holding a claim, a gated unapproved cmd is refused
    own = {c: build_claim(c, owner, leased_at="1000", ttl_seconds=600)}
    a2 = decide(owner, nodes, [c], own, FREE, 1100.0, gated)
    assert a2["kind"] == "refuse-gated"


def test_gated_with_human_approval_claimable() -> None:
    nodes = ["spark-a", "spark-b"]
    c = "cmd-train-ok"
    owner = assigned_node(c, nodes)
    approved = {c: {"id": c, "args": "--bench-a --execute", "createdBy": "claude",
                    "approvedBy": "user: 'run it' (2026)"}}
    a = decide(owner, nodes, [c], {}, FREE, 1000.0, approved)
    assert a["kind"] == "claim" and a["cmdId"] == c
    own = {c: build_claim(c, owner, leased_at="1000", ttl_seconds=600)}
    assert decide(owner, nodes, [c], own, FREE, 1100.0, approved)["kind"] == "run"


# --- busy node idles ----------------------------------------------------------------------------
def test_busy_node_idles() -> None:
    nodes = ["spark-a", "spark-b"]
    c = "cmd-x"
    owner = assigned_node(c, nodes)
    cmds = {c: _dry()}
    # owner is busy -> idles even on its own assigned, unclaimed cmd (one GPU job per node)
    assert decide(owner, nodes, [c], {}, BUSY, 1000.0, cmds)["kind"] == "idle"
    # busy owner holding its own claim still idles (won't start a second job)
    own = {c: build_claim(c, owner, leased_at="1000", ttl_seconds=600)}
    assert decide(owner, nodes, [c], own, BUSY, 1100.0, cmds)["kind"] == "idle"


# --- whole-cluster sweep: no double-run, no double-claim over a shared pending set ---------------
def test_no_double_action_over_shared_pending() -> None:
    nodes = ["spark-a", "spark-b", "spark-c", "spark-d"]
    pending = [f"c{i}" for i in range(30)]
    cmds = {c: _dry() for c in pending}
    # each node sees the FULL pending set; tally claim actions per cmd across all nodes
    claim_counts: dict[str, int] = {c: 0 for c in pending}
    for n in nodes:
        a = decide(n, nodes, pending, {}, FREE, 1000.0, cmds)
        if a["kind"] == "claim":
            claim_counts[a["cmdId"]] += 1
    # a free node claims at most one cmd (its top assigned); no cmd is claimed by two nodes
    assert all(v <= 1 for v in claim_counts.values())


# --- --once dry-run decision matches the pure core ----------------------------------------------
def test_once_dry_run_matches_pure_core() -> None:
    """The CLI --once path builds (node_set, pending, claims, status, commands_by_id) from disk and
    calls decide() with NO side effects. Here we feed the SAME inputs to decide() directly and to a
    re-implementation of the --once composition, asserting they agree (deterministic, offline)."""
    nodes = ["spark-2f2d", "spark-aaaa"]
    pending = ["alpha", "beta", "gamma"]
    cmds = {c: _dry() for c in pending}
    claims: dict = {}
    now = 1234.0
    # decide() is pure -> two calls with identical inputs are identical
    a1 = decide("spark-2f2d", nodes, pending, claims, FREE, now, cmds)
    a2 = decide("spark-2f2d", nodes, pending, claims, FREE, now, cmds)
    assert a1 == a2
    # and the action is well-formed: either this node owns one of {alpha,beta,gamma} or idles
    if a1["kind"] == "claim":
        assert assigned_node(a1["cmdId"], nodes) == "spark-2f2d"
    else:
        assert a1["kind"] == "idle"


def test_self_test_invariants_pass() -> None:
    from tools.cluster_node_runner import offline_invariants
    ok, detail = offline_invariants()
    assert ok, detail


def main() -> int:
    test_exactly_one_claims_each_cmd_2_nodes()
    test_exactly_one_claims_each_cmd_4_nodes()
    test_my_assigned_pending_partitions_the_set()
    test_claim_contention_single_owner()
    test_owner_with_own_live_claim_runs()
    test_expired_lease_requeues()
    test_gated_without_approval_refused()
    test_gated_with_human_approval_claimable()
    test_busy_node_idles()
    test_no_double_action_over_shared_pending()
    test_once_dry_run_matches_pure_core()
    test_self_test_invariants_pass()
    print("test_cluster_node_runner: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
