#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the G9C cryptographic provenance chain (Merkle lineage) gate.

Offline, pure stdlib, no torch. Covers: a promote path via demo_bundle(); each reject
reason (chain tampering, ungated ancestor); each quarantine/fail-closed reason
(missing chain, non-list chain, missing leafId, leaf absent, broken parent link,
bundle None); and the standing invariants (canClaimAGI False, candidateOnly True,
level3Evidence False, verdict in the allowed set, honest non-empty boundary).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_provenance_chain import (  # noqa: E402
    GATE_ID,
    SCHEMA,
    append_entry,
    build_chain,
    demo_bundle,
    evaluate,
    lineage,
    verify_chain,
)

_ALLOWED = {"promote", "quarantine", "reject"}


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert d["gate"] == GATE_ID
    assert d["schema"] == SCHEMA
    assert d["metrics"]["chainVerified"] is True
    assert d["metrics"]["ungatedAncestors"] == []
    assert d["metrics"]["lineageDepth"] == 3


def test_append_entry_and_verify_roundtrip() -> None:
    """append_entry links to the tip; a built chain verifies and is hmac-stable."""
    content = [
        {"id": "g", "spec": {"a": 1}, "metric": 0.0, "gateVerdict": "promote"},
        {"id": "c1", "spec": {"a": 2}, "metric": 0.5, "gateVerdict": "promote"},
    ]
    chain = build_chain(content)
    assert chain[0]["parentHash"] == "0" * 64
    assert chain[1]["parentHash"] == chain[0]["entryHash"]
    ok, idx = verify_chain(chain)
    assert ok is True and idx == -1
    # HMAC signing changes the digests but still verifies under the key.
    signed = build_chain(content, hmac_key="secret")
    assert signed[1]["entryHash"] != chain[1]["entryHash"]
    assert verify_chain(signed, hmac_key="secret") == (True, -1)
    # ...and fails to verify without the key (forgery resistance).
    assert verify_chain(signed)[0] is False


def test_lineage_walks_to_root() -> None:
    b = demo_bundle()
    path = lineage(b["chain"], "round-2")
    ids = [e["id"] for e in path]
    assert ids == ["round-2", "round-1", "genesis"]


def test_promote_with_hmac_signing() -> None:
    content = [
        {"id": "genesis", "spec": {"k": "base"}, "metric": 0.0, "gateVerdict": "promote"},
        {"id": "leaf", "spec": {"k": "v1"}, "metric": 0.7, "gateVerdict": "promote"},
    ]
    chain = build_chain(content, hmac_key="k3y")
    d = evaluate({"chain": chain, "leafId": "leaf", "hmacKey": "k3y"})
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["signed"] is True


def test_reject_tampering() -> None:
    """Editing an entry's content after the fact invalidates its stored hash -> reject."""
    b = copy.deepcopy(demo_bundle())
    b["chain"][1]["metric"] = 0.999999  # mutate content without rehashing
    d = evaluate(b)
    assert d["verdict"] == "reject", d["reasons"]
    assert any("failed verification" in r for r in d["reasons"])
    assert d["metrics"]["firstBadIndex"] == 1


def test_reject_reorder_breaks_parent_link() -> None:
    """Reordering entries breaks the parentHash linkage -> reject at the swap point."""
    b = copy.deepcopy(demo_bundle())
    b["chain"][1], b["chain"][2] = b["chain"][2], b["chain"][1]
    d = evaluate(b)
    assert d["verdict"] == "reject", d["reasons"]
    assert any("failed verification" in r for r in d["reasons"])


def test_reject_ungated_ancestor() -> None:
    """An ancestor with a non-promote gateVerdict -> reject (not gated lineage)."""
    content = [
        {"id": "genesis", "spec": {"k": "base"}, "metric": 0.0, "gateVerdict": "quarantine"},
        {"id": "round-1", "spec": {"k": "v1"}, "metric": 0.7, "gateVerdict": "promote"},
    ]
    chain = build_chain(content)
    d = evaluate({"chain": chain, "leafId": "round-1"})
    assert d["verdict"] == "reject", d["reasons"]
    assert any("ungated ancestor" in r for r in d["reasons"])
    assert "genesis" in d["metrics"]["ungatedAncestors"]


def test_reject_ungated_leaf_itself() -> None:
    """The leaf's own verdict must be promote too."""
    content = [
        {"id": "genesis", "spec": {"k": "base"}, "metric": 0.0, "gateVerdict": "promote"},
        {"id": "leaf", "spec": {"k": "v1"}, "metric": 0.7, "gateVerdict": "reject"},
    ]
    chain = build_chain(content)
    d = evaluate({"chain": chain, "leafId": "leaf"})
    assert d["verdict"] == "reject", d["reasons"]
    assert "leaf" in d["metrics"]["ungatedAncestors"]


def test_fail_closed_missing_chain() -> None:
    b = copy.deepcopy(demo_bundle())
    b.pop("chain")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("'chain'" in r for r in d["reasons"])


def test_fail_closed_chain_not_list() -> None:
    d = evaluate({"chain": {"not": "a list"}, "leafId": "x"})
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("not a list" in r for r in d["reasons"])


def test_fail_closed_missing_leaf_id() -> None:
    b = copy.deepcopy(demo_bundle())
    b.pop("leafId")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("'leafId'" in r for r in d["reasons"])


def test_fail_closed_leaf_absent() -> None:
    b = copy.deepcopy(demo_bundle())
    b["leafId"] = "does-not-exist"
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("not present in chain" in r for r in d["reasons"])


def test_lineage_broken_parent_link_raises() -> None:
    """A leaf whose parentHash points at no present entry -> ValueError from lineage().

    This is the helper-level guarantee behind the gate's broken-link quarantine branch.
    A fully verifying chain can never trigger it (every non-genesis parentHash equals a
    present prior hash), so we exercise the detached case directly on the helper.
    """
    detached = [{
        "id": "orphan", "parentHash": "a" * 64, "entryHash": "b" * 64,
        "spec": {"k": "v"}, "metric": 0.5, "gateVerdict": "promote",
    }]
    try:
        lineage(detached, "orphan")
        raise AssertionError("expected broken-link ValueError")
    except ValueError as exc:
        assert "broken parent link" in str(exc)


def test_quarantine_empty_chain_leaf_absent() -> None:
    """A present-but-empty chain with a requested leaf -> quarantine (leaf absent)."""
    d = evaluate({"chain": [], "leafId": "solo"})
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("not present in chain" in r for r in d["reasons"])


def test_solo_genesis_leaf_promotes() -> None:
    """A single promote-gated genesis entry is itself a valid (depth-1) lineage."""
    chain = build_chain([{"id": "solo", "spec": {"k": "v"}, "metric": 0.1, "gateVerdict": "promote"}])
    d = evaluate({"chain": chain, "leafId": "solo"})
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["lineageDepth"] == 1


def test_fail_closed_bundle_none() -> None:
    d = evaluate(None)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("bundle is None" in r for r in d["reasons"])


def test_standing_invariants() -> None:
    """Every decision: canClaimAGI False, candidateOnly True, level3Evidence False,
    verdict in the allowed set, honest non-empty boundary, and candidateId echoed."""
    tampered = copy.deepcopy(demo_bundle())
    tampered["chain"][0]["metric"] = 9.9
    bundles = [
        demo_bundle(),
        None,
        {"chain": None, "leafId": "x"},
        tampered,
    ]
    for b in bundles:
        d = evaluate(b, candidate_id="g9c-test")
        assert d["canClaimAGI"] is False
        assert d["candidateOnly"] is True
        assert d["level3Evidence"] is False
        assert d["verdict"] in _ALLOWED
        assert isinstance(d["boundary"], str) and d["boundary"]
        assert d["candidateId"] == "g9c-test"


def main() -> int:
    test_demo_bundle_promotes()
    test_append_entry_and_verify_roundtrip()
    test_lineage_walks_to_root()
    test_promote_with_hmac_signing()
    test_reject_tampering()
    test_reject_reorder_breaks_parent_link()
    test_reject_ungated_ancestor()
    test_reject_ungated_leaf_itself()
    test_fail_closed_missing_chain()
    test_fail_closed_chain_not_list()
    test_fail_closed_missing_leaf_id()
    test_fail_closed_leaf_absent()
    test_lineage_broken_parent_link_raises()
    test_quarantine_empty_chain_leaf_absent()
    test_solo_genesis_leaf_promotes()
    test_fail_closed_bundle_none()
    test_standing_invariants()
    print("test_ssil_provenance_chain: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
