#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the communication-efficient belief all-reduce thesis (feature #5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.belief_allreduce import (  # noqa: E402
    _base_council,
    _key,
    all_to_all,
    main,
    majority_vote_consensus,
    recursive_doubling,
    reduce_beliefs,
    ring_allreduce,
    run_experiment,
)


def test_reduce_is_commutative_and_unions_provenance():
    a = {"x": {"conf": 1, "holders": {0}, "label": "public"}}
    b = {"x": {"conf": 3, "holders": {1}, "label": "public"}}
    ab = reduce_beliefs(a, b)
    ba = reduce_beliefs(b, a)
    assert _key(ab) == _key(ba)
    assert ab["x"]["conf"] == 3                     # max confidence
    assert ab["x"]["holders"] == {0, 1}             # union of provenance


def test_ring_and_tree_match_all_to_all_consensus():
    for n in (4, 8, 16):
        agents = _base_council(n, seed=1)
        a2a, m_a2a = all_to_all(agents)
        ring, m_ring = ring_allreduce(agents)
        tree, m_tree = recursive_doubling(agents)
        target = _key(a2a[0])
        assert all(_key(x) == target for x in a2a)
        assert all(_key(x) == target for x in ring)
        assert all(_key(x) == target for x in tree)
        assert m_ring < m_a2a and m_tree <= m_a2a   # fewer messages, same result


def test_minority_belief_survives_reduce_but_not_vote():
    agents = _base_council(8, seed=1)
    agents[0]["rare_truth"] = {"conf": 3, "holders": {0}, "label": "public"}
    ring, _ = ring_allreduce(agents)
    assert all("rare_truth" in f for f in ring)             # provenance-preserving keeps it
    assert "rare_truth" not in majority_vote_consensus(agents)  # vote drops the minority


def test_firewall_blocks_secret_to_uncleared():
    r = run_experiment(n=8, seed=4)
    fw = r["firewall"]
    assert fw["forbidden_transmissions"] == 0
    assert not fw["secret_leaked"]
    assert fw["cleared_have_secret"]
    assert fw["public_consensus_everywhere"]


def test_message_counts_are_subquadratic():
    r = run_experiment(n=16, seed=4)
    m = r["messages"]
    assert m["ring"] == 2 * (16 - 1)
    assert m["recursive_doubling"] == 16 * 4
    assert m["all_to_all"] == 16 * 15


def test_cli():
    assert main(["--self-test"]) == 0
    assert main(["--run", "--n", "8"]) == 0
    assert main(["--json", "--n", "4"]) == 0
