#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the out-of-prompt data-flow firewall (M2).

The security property is deterministic and code-enforced: tainted (untrusted) data
cannot reach a write/egress sink without human approval, regardless of what the
model "decides"; the airgap profile blocks all egress; unknown tools default-deny.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dataflow import (  # noqa: E402
    Effect,
    FirewallBlocked,
    ToolCap,
    cap_for,
    combine,
    decide,
    firewalled,
    guard_call,
    taint_of,
    trusted,
    untrusted,
)


def test_taint_propagation() -> None:
    assert taint_of("plain") == frozenset()
    assert untrusted("doc").is_tainted() is True
    assert trusted("user").is_tainted() is False
    assert combine(trusted("a"), untrusted("b")) == frozenset({"untrusted"})
    assert combine("x", "y") == frozenset()


def test_decide_matrix() -> None:
    read = ToolCap("r", Effect.READ)
    write = ToolCap("w", Effect.WRITE)
    egress = ToolCap("e", Effect.EGRESS)
    clean, dirty = [frozenset()], [frozenset({"untrusted"})]

    assert decide(read, dirty).allowed is True                       # reads always allowed
    assert decide(write, clean).allowed is True                      # trusted -> sink ok
    assert decide(write, dirty).blocked is True                      # lethal trifecta, no approver
    assert decide(write, dirty, approver=lambda *a: True).action == "require_hitl"
    assert decide(egress, dirty, profile="airgap").blocked is True   # airgap blocks egress
    assert decide(egress, clean, profile="airgap").blocked is True   # ...even trusted egress


def test_unknown_tool_default_denies() -> None:
    cap = cap_for("totally_unknown_tool")
    assert cap.effect == Effect.WRITE and cap.accepts_tainted is False
    assert decide(cap, [frozenset({"untrusted"})]).blocked is True


def test_real_manifest_classification() -> None:
    assert cap_for("sophia_wiki_read").effect == Effect.READ
    assert cap_for("sophia_wiki_upsert").effect == Effect.WRITE
    assert cap_for("sophia_openclaw_infer").effect == Effect.EGRESS
    assert cap_for("sophia_web_evidence_search").effect == Effect.EGRESS


def test_firewalled_blocks_lethal_trifecta() -> None:
    calls = []
    raw = lambda content: calls.append(content) or "wrote"  # noqa: E731
    guarded = firewalled(raw, name="sophia_wiki_upsert")
    # untrusted content into a WRITE sink, no approver -> blocked, fn never runs
    try:
        guarded(untrusted("poisoned by injection"))
        assert False, "expected FirewallBlocked"
    except FirewallBlocked:
        pass
    assert calls == []
    # trusted content flows through and the fn sees the unwrapped value
    assert guarded(trusted("legit edit")) == "wrote"
    assert calls == ["legit edit"]


def test_firewalled_hitl_approval_path() -> None:
    raw = lambda content: "ok"  # noqa: E731
    approved = firewalled(raw, name="sophia_wiki_upsert", approver=lambda *a: True)
    denied = firewalled(raw, name="sophia_wiki_upsert", approver=lambda *a: False)
    assert approved(untrusted("x")) == "ok"
    try:
        denied(untrusted("x"))
        assert False, "expected FirewallBlocked"
    except FirewallBlocked:
        pass


def test_guard_call_decisions() -> None:
    assert guard_call("sophia_wiki_read", (untrusted("q"),)).allowed is True
    assert guard_call("sophia_wiki_upsert", (untrusted("p"),)).blocked is True
    assert guard_call("sophia_openclaw_infer", (trusted("p"),), profile="airgap").blocked is True


def test_nested_container_taint_is_caught() -> None:
    # tainted value nested inside a list/dict arg must still be seen (deep walk)
    assert taint_of([trusted("a"), untrusted("b")]) == frozenset({"untrusted"})
    assert taint_of({"k": untrusted("v")}) == frozenset({"untrusted"})
    assert guard_call("sophia_wiki_upsert", ([untrusted("poison")],)).blocked is True
    assert guard_call("sophia_openclaw_infer", ({"q": untrusted("p")},)).blocked is True


def test_approver_fails_closed() -> None:
    raw = lambda x: "ran"  # noqa: E731

    def raises(*a):
        raise RuntimeError("boom")

    # a raising approver and a non-True (merely truthy) approver must NOT approve
    for approver in (raises, lambda *a: "yes", lambda *a: 1):
        g = firewalled(raw, name="sophia_wiki_upsert", approver=approver)
        try:
            g(untrusted("p"))
            assert False, "approver should have failed closed"
        except FirewallBlocked:
            pass
    # only an explicit True approves
    assert firewalled(raw, name="sophia_wiki_upsert", approver=lambda *a: True)(untrusted("p")) == "ran"


def test_model_adapter_airgap_kill_switch() -> None:
    import os

    from agent.model import ModelClient, resolve_config

    os.environ["SOPHIA_PROFILE"] = "airgap"
    try:
        remote = ModelClient(resolve_config("deepseek")).generate("s", "u")
        assert remote.ok is False and "airgap" in (remote.error or "")
        # mock (local, no egress) still works under airgap
        assert ModelClient(resolve_config("mock")).generate("s", "u").ok is True
    finally:
        os.environ.pop("SOPHIA_PROFILE", None)


def test_airgap_profile_blocks_real_egress_tools() -> None:
    # The airgap profile is fail-closed on the actual MCP egress tools.
    import os

    from sophia_mcp.tools_impl import openclaw_infer, web_evidence_search

    os.environ["SOPHIA_PROFILE"] = "airgap"
    try:
        assert openclaw_infer(prompt="hi")["ok"] is False
        assert "airgap" in web_evidence_search("q", online=True)["error"]
        # ...but local (non-egress) retrieval still works under airgap
        assert "error" not in web_evidence_search("q", online=False) or \
            "airgap" not in str(web_evidence_search("q", online=False).get("error", ""))
    finally:
        os.environ.pop("SOPHIA_PROFILE", None)


def main() -> int:
    test_taint_propagation()
    test_decide_matrix()
    test_unknown_tool_default_denies()
    test_real_manifest_classification()
    test_firewalled_blocks_lethal_trifecta()
    test_firewalled_hitl_approval_path()
    test_guard_call_decisions()
    test_nested_container_taint_is_caught()
    test_approver_fails_closed()
    test_model_adapter_airgap_kill_switch()
    test_airgap_profile_blocks_real_egress_tools()
    print("test_dataflow: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
