#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/hooks.py — the lifecycle hook bus (Stage A).

Falsifiable invariants:
  1. PRE_TOOL_USE is a blocking event: a handler returning allow=False blocks,
     and a raising handler ALSO blocks (fail-closed).
  2. POST_TOOL_USE / PRE_COMPACT are observe-only: a raising handler never blocks.
  3. Handlers run in registration order and short-circuit on the first block.
  4. The provenance PreToolUse guard blocks side-effecting calls lacking role/clearance
     and allows read calls.
  5. The gateway interceptor honours an attached bus: a blocking PreToolUse hook
     prevents tool execution; an absent bus is fully backward-compatible.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hooks import (  # noqa: E402
    HookBus,
    HookContext,
    HookDecision,
    HookEvent,
    make_precompact_snapshot,
    make_provenance_pretool_guard,
)


def test_pretool_block_is_failclosed_on_false() -> None:
    bus = HookBus()
    bus.register(HookEvent.PRE_TOOL_USE,
                 lambda ctx: HookDecision(allow=False, reason="nope"), name="denier")
    res = bus.dispatch(HookContext(event=HookEvent.PRE_TOOL_USE, tool_id="t"))
    assert res.blocked is True
    assert res.blocked_by == "denier"
    assert "nope" in res.reason


def test_pretool_raising_handler_blocks() -> None:
    def boom(ctx):
        raise RuntimeError("explode")

    bus = HookBus().register(HookEvent.PRE_TOOL_USE, boom)
    res = bus.dispatch(HookContext(event=HookEvent.PRE_TOOL_USE, tool_id="t"))
    assert res.blocked is True
    assert res.blocked_by == "boom"
    assert "fail-closed" in res.reason


def test_observe_only_raising_handler_does_not_block() -> None:
    def boom(ctx):
        raise RuntimeError("explode")

    for ev in (HookEvent.POST_TOOL_USE, HookEvent.PRE_COMPACT, HookEvent.SESSION_START):
        bus = HookBus().register(ev, boom)
        res = bus.dispatch(HookContext(event=ev))
        assert res.allowed is True, ev


def test_handlers_run_in_order_and_short_circuit() -> None:
    seen: list = []

    def first(ctx):
        seen.append("first")
        return None

    def second(ctx):
        seen.append("second")
        return HookDecision(allow=False, reason="stop")

    def third(ctx):
        seen.append("third")
        return None

    bus = HookBus()
    bus.register(HookEvent.PRE_TOOL_USE, first)
    bus.register(HookEvent.PRE_TOOL_USE, second)
    bus.register(HookEvent.PRE_TOOL_USE, third)
    res = bus.dispatch(HookContext(event=HookEvent.PRE_TOOL_USE, tool_id="t"))
    assert res.blocked is True
    assert seen == ["first", "second"]  # third never runs (short-circuit)


def test_provenance_guard_blocks_side_effect_without_clearance() -> None:
    guard = make_provenance_pretool_guard()
    bus = HookBus().register(HookEvent.PRE_TOOL_USE, guard, name="prov")

    blocked = bus.dispatch(HookContext(
        event=HookEvent.PRE_TOOL_USE, tool_id="writer",
        payload={"side_effects": "write", "role": None, "clearance": None}))
    assert blocked.blocked is True

    ok = bus.dispatch(HookContext(
        event=HookEvent.PRE_TOOL_USE, tool_id="writer",
        payload={"side_effects": "write", "role": "role_09_agents", "clearance": "SECRET"}))
    assert ok.allowed is True

    read = bus.dispatch(HookContext(
        event=HookEvent.PRE_TOOL_USE, tool_id="reader",
        payload={"side_effects": "read"}))
    assert read.allowed is True


def test_precompact_snapshot_sink_receives_payload() -> None:
    captured: list = []
    handler = make_precompact_snapshot(captured.append)
    bus = HookBus().register(HookEvent.PRE_COMPACT, handler, name="snap")
    bus.dispatch(HookContext(event=HookEvent.PRE_COMPACT,
                             payload={"belief_delta": 3, "provenance_ids": ["c1", "c2"]}))
    assert len(captured) == 1
    assert captured[0]["belief_delta"] == 3
    assert captured[0]["provenance_ids"] == ["c1", "c2"]
    assert captured[0]["event"].endswith("PreCompact")


def test_gateway_honours_blocking_pretool_hook() -> None:
    from gateway import Gateway, ToolEntry

    bus = HookBus().register(
        HookEvent.PRE_TOOL_USE,
        lambda ctx: HookDecision(allow=False, reason="policy-block"),
        name="policy")
    gw = Gateway(hook_bus=bus)
    ran: list = []

    def handler(args):
        ran.append(args)
        return {"answer": "ok", "sources": ["kb"]}

    gw.register(ToolEntry(id="kb.lookup", handler=handler, verifier_ref="grounding"))
    resp = gw.call_tool("kb.lookup", {"q": "x"}, role="role_09_agents", clearance="SECRET")
    assert resp["verdict"] == "held"
    assert resp.get("blocked_by") == "policy"
    assert ran == []  # the tool handler never executed (hook blocked first)


def test_gateway_without_bus_is_backward_compatible() -> None:
    from gateway import Gateway, ToolEntry

    gw = Gateway()  # no bus
    gw.register(ToolEntry(id="kb.lookup", handler=lambda a: {"answer": "ok", "sources": ["kb"]},
                          verifier_ref="grounding"))
    resp = gw.call_tool("kb.lookup", {"q": "x"}, role="role_09_agents", clearance="SECRET")
    assert resp["verdict"] in ("accepted", "held", "rejected")  # pipeline still runs


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_hooks: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
