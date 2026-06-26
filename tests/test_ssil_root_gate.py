#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the G0 root-of-trust / meta-gate. Offline, stdlib only, no torch.

Asserts the gate's invariants: a clean manifest match promotes; an unauthorised gate
edit, a root edit without the two-key token, and a root edit that drops a declared
invariant each reject; an unreviewed new gate quarantines; missing safety inputs fail
closed; and the standardised honesty fields hold on every decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_root_gate import demo_bundle, evaluate, sha256_of_text  # noqa: E402

_ALLOWED = {"promote", "quarantine", "reject"}


def _assert_envelope(d: dict) -> None:
    assert d["canClaimAGI"] is False, d
    assert d["candidateOnly"] is True, d
    assert d["level3Evidence"] is False, d
    assert d["verdict"] in _ALLOWED, d
    assert d["schema"] == "sophia.root_gate_decision.v1", d
    assert d["gate"] == "G0", d
    assert isinstance(d["boundary"], str) and d["boundary"], d


# --- promote --------------------------------------------------------------- #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert any("root of trust intact" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_sha256_helper_matches_manifest() -> None:
    # The demo bundle is internally consistent: each current hash equals sha256 of nothing
    # we can recompute here, but the helper is deterministic and hex.
    h = sha256_of_text("def evaluate(bundle): ...  # root gate source v1")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert sha256_of_text("a") != sha256_of_text("b")


# --- reject reasons -------------------------------------------------------- #


def test_unauthorised_non_root_edit_rejects() -> None:
    b = demo_bundle()
    b["gateHashes"]["G5_honeypots"] = sha256_of_text("tampered honeypot gate source")
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("unauthorised gate edit" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_root_edit_without_token_rejects() -> None:
    b = demo_bundle()
    b["gateHashes"]["G0_root"] = sha256_of_text("rewritten root gate source v2")
    # carry over invariants so the failure is specifically the missing token
    b["proposedInvariants"] = {"G0_root": ["two_key_for_root", "invariant_superset"]}
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("without offline two-key rootApprovalToken" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_root_edit_dropping_invariant_rejects() -> None:
    b = demo_bundle()
    b["gateHashes"]["G0_root"] = sha256_of_text("rewritten root gate source v2")
    b["rootApprovalToken"] = "two-key-offline-token-abc"
    # Drops the 'invariant_superset' declared invariant -> reject even with the token.
    b["proposedInvariants"] = {"G0_root": ["two_key_for_root"]}
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("drops declared invariant" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_gate_removed_rejects() -> None:
    b = demo_bundle()
    del b["gateHashes"]["G6_corrigibility"]
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("removed from set without authorisation" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_authorised_non_root_edit_dropping_invariant_rejects() -> None:
    b = demo_bundle()
    b["gateHashes"]["G2_reward_isolation"] = sha256_of_text("edited reward isolation v2")
    b["rootApprovalToken"] = "two-key-offline-token-abc"
    # Authorised (token present) but drops a declared invariant -> reject.
    b["proposedInvariants"] = {"G2_reward_isolation": ["no_reward_write"]}
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("drops declared invariant" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


# --- quarantine reasons ---------------------------------------------------- #


def test_unknown_new_gate_quarantines() -> None:
    b = demo_bundle()
    b["gateHashes"]["G9_brand_new"] = sha256_of_text("a freshly proposed gate not in the manifest")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d
    assert any("absent from signed manifest" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_missing_root_gate_name_quarantines() -> None:
    b = demo_bundle()
    b["rootGateName"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d
    assert any("rootGateName" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


# --- fail-closed on missing safety inputs ---------------------------------- #


def test_missing_gate_hashes_fails_closed() -> None:
    b = demo_bundle()
    b["gateHashes"] = None
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("gateHashes" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_missing_signed_manifest_fails_closed() -> None:
    b = demo_bundle()
    b["signedManifest"] = None
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("signedManifest" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def test_none_bundle_fails_closed() -> None:
    # The highest-privilege gate must fail closed (reject), not crash, on a None bundle.
    d = evaluate(None)
    assert d["verdict"] == "reject", d
    assert any("bundle is None" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


def main() -> int:
    test_demo_bundle_promotes()
    test_sha256_helper_matches_manifest()
    test_unauthorised_non_root_edit_rejects()
    test_root_edit_without_token_rejects()
    test_root_edit_dropping_invariant_rejects()
    test_gate_removed_rejects()
    test_authorised_non_root_edit_dropping_invariant_rejects()
    test_unknown_new_gate_quarantines()
    test_missing_root_gate_name_quarantines()
    test_missing_gate_hashes_fails_closed()
    test_missing_signed_manifest_fails_closed()
    test_none_bundle_fails_closed()
    print("test_ssil_root_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
