#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multi-node git-bridge cluster scheduler tests: deterministic assignment, git-CAS claim race,
lease expiry, gated fan-out inheritance, unsafe id refusal.

Deterministic, offline — pure logic, no git, no network, no clock. (Named ``*_bridge`` to avoid the
unrelated, pre-existing ``tests/test_cluster_scheduler.py``, which tests the clustersim simulator.)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cluster_scheduler import (  # noqa: E402
    assigned_node,
    assignment_distribution,
    build_claim,
    build_node_status,
    claim_path,
    collect_shard_results,
    is_safe_id,
    lease_is_expired,
    node_is_free,
    node_status_path,
    offline_invariants,
    split_command,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_assignment_single_and_stable() -> None:
    nodes = ["spark-b", "spark-a", "spark-c"]
    owner = assigned_node("cmd-7", nodes)
    assert owner in nodes
    # stable across calls and independent of input order (sorted internally)
    assert assigned_node("cmd-7", nodes) == owner
    assert assigned_node("cmd-7", list(reversed(nodes))) == owner
    assert assigned_node("cmd-7", ["spark-a", "spark-b", "spark-c"]) == owner


def test_assignment_balanced_2_4_8() -> None:
    cmds = [f"cmd-{i}" for i in range(800)]
    for k in (2, 4, 8):
        nodes = [f"spark-{i}" for i in range(k)]
        dist = assignment_distribution(cmds, nodes)
        assert set(dist) == set(nodes)
        assert sum(dist.values()) == len(cmds)
        # every node gets a nontrivial share — within +/-40% of an even split (sha1 is well spread)
        even = len(cmds) / k
        for n, c in dist.items():
            assert 0.6 * even <= c <= 1.4 * even, (k, n, c, even)


def test_cas_race_only_assigned_node_should_claim() -> None:
    # SIMULATE the compare-and-swap race: two nodes independently compute claims for the SAME cmd.
    # Decentralized assignment says exactly ONE of them owns it; the other must defer (not push).
    nodes = ["spark-a", "spark-b"]
    cmd = "cmd-train-42"
    owner = assigned_node(cmd, nodes)
    loser = [n for n in nodes if n != owner][0]
    # both could *build* a claim object, but only the owner *should* push it
    claim_owner = build_claim(cmd, owner, leased_at="1000", ttl_seconds=3600)
    claim_loser = build_claim(cmd, loser, leased_at="1000", ttl_seconds=3600)
    assert claim_owner["nodeId"] == owner
    assert claim_loser["nodeId"] == loser
    # the git ref is the mutex: both target the SAME path, so the first push wins, the second is
    # non-ff rejected. The deterministic owner is the one that should attempt it.
    assert claim_path(cmd) == f"bridge/claims/{cmd}.json"
    # both nodes agree on the owner with no coordination -> no double-claim
    assert assigned_node(cmd, nodes) == owner
    assert owner != loser


def test_lease_expiry_requeue() -> None:
    claim = build_claim("cmd-9", "spark-a", leased_at="1000", ttl_seconds=120)
    assert not lease_is_expired(claim, now=1100)   # still leased -> do NOT requeue
    assert not lease_is_expired(claim, now=1120)   # exactly at boundary, still held
    assert lease_is_expired(claim, now=1121)       # expired -> requeue allowed
    # a malformed/empty claim is treated as expired (safe to requeue)
    assert lease_is_expired({}, now=0)
    assert lease_is_expired({"leasedAt": "x", "ttlSeconds": "y"}, now=10)


def test_gated_fanout_without_approval_raises() -> None:
    parent = {"id": "p1", "args": "--bench-a --execute", "createdBy": "claude"}
    try:
        split_command(parent, [1, 2, 10])
        raised = False
    except ValueError:
        raised = True
    assert raised, "a gated parent with no human approvedBy must refuse to fan out"


def test_gated_fanout_with_approval_inherits_to_every_shard() -> None:
    parent = {"id": "bench-a-exec", "args": "--bench-a --execute", "createdBy": "claude",
              "approvedBy": "user: 'run bench-a over seeds' (2026-06-30)"}
    subs = split_command(parent, [1, 2, 10])
    assert len(subs) == 3
    for s, seed in zip(subs, [1, 2, 10]):
        assert s["approvedBy"] == parent["approvedBy"]   # human approval inherited verbatim
        assert s["parentId"] == "bench-a-exec"
        assert s["shard"] == seed
        assert s["args"] == "--bench-a --execute"
        assert s["id"] == f"bench-a-exec--shard-{seed}"


def test_dry_run_fanout_needs_no_approval() -> None:
    subs = split_command({"id": "p", "args": "--dry-run --all", "createdBy": "claude"}, [1, 2])
    assert len(subs) == 2 and all(s["approvedBy"] == "" for s in subs)


def test_fanout_rejects_non_allowlisted_args() -> None:
    try:
        split_command({"id": "p", "args": "--rm -rf /", "createdBy": "claude"}, [1])
        raised = False
    except ValueError:
        raised = True
    assert raised, "fan-out must reject non-allowlisted parent args"


def test_unsafe_node_ids_refused() -> None:
    for bad in ("a/b", "a b", "a\tb", ""):
        assert not is_safe_id(bad)
    # a structurally-unsafe id in the node set is refused (empty ids are dropped, not validated,
    # so they cannot inject a claimer — exercised separately below)
    for bad in ("a/b", "a b", "a\tb"):
        try:
            assigned_node("cmd-x", [bad, "spark-a"])
            raised = False
        except ValueError:
            raised = True
        assert raised, f"unsafe node id {bad!r} must be refused in assignment"
    # an empty id is silently dropped from the membership set (never becomes an owner)
    assert assigned_node("cmd-x", ["", "spark-a"]) == "spark-a"
    # node status path + builder enforce the same rule
    try:
        node_status_path("a/b")
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert node_status_path("spark-a") == "bridge/nodes/spark-a/status.json"


def test_node_is_free_mirrors_gpu_is_free() -> None:
    st = build_node_status("spark-a", updated_at="t", running="", gpu_free_bytes=120_000_000_000)
    assert st["nodeId"] == "spark-a" and st["running"] is None
    assert node_is_free(st)
    busy = build_node_status("spark-a", updated_at="t", running="cmd-1")
    assert busy["running"] == "cmd-1" and not node_is_free(busy)
    assert not node_is_free({"running": None, "pendingCommands": ["q"]})


def test_aggregation_complete_and_missing() -> None:
    results = [
        {"parentId": "p2", "shard": 1, "expectedShards": [1, 2, 10]},
        {"parentId": "p2", "shard": 2},
        {"parentId": "other", "shard": 99},  # different parent, ignored
    ]
    agg = collect_shard_results("p2", results)
    assert agg["parentId"] == "p2"
    assert agg["shards"] == [1, 2]
    assert agg["missing"] == [10]
    assert agg["complete"] is False
    # once shard 10 lands, complete
    results.append({"parentId": "p2", "shard": 10})
    agg2 = collect_shard_results("p2", results)
    assert agg2["complete"] is True and agg2["missing"] == []


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} cluster_scheduler_bridge tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
