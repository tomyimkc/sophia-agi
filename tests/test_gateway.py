#!/usr/bin/env python3
"""Sophia Gateway P0 acceptance tests — the fail-closed intercept pipeline.
Deterministic, offline (in-process tools, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import Gateway, ToolEntry  # noqa: E402
from sophia_contract import SophiaContract  # noqa: E402

_CLK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


def _kb(args):  # grounding tool: sourced for known queries, unsourced otherwise
    if args.get("q") == "dao":
        return {"answer": "Laozi", "sources": ["wiki://dao-de-jing"]}
    return {"answer": "unknown", "sources": []}


def _calc(args):  # env-verified tool: returns a candidate expression to be executed
    return {"candidate": args["expr"]}


def _gw(**kw):
    gw = Gateway(contract=SophiaContract(clock=_CLK), **kw)
    gw.register(ToolEntry(id="kb.lookup", handler=_kb, verifier_ref="grounding",
                          blp_level="UNCLASSIFIED", side_effects="read"))
    gw.register(ToolEntry(id="calc.eval", handler=_calc, verifier_ref="env:arithmetic",
                          blp_level="UNCLASSIFIED"))
    gw.register(ToolEntry(id="secret.read", handler=_kb, verifier_ref="grounding",
                          blp_level="SECRET"))
    gw.register(ToolEntry(id="agents.only", handler=_kb, verifier_ref="grounding",
                          blp_level="UNCLASSIFIED", allowed_roles={"role_09_agents"}))
    gw.register(ToolEntry(id="conf.tool", handler=_kb, verifier_ref="grounding",
                          blp_level="CONFIDENTIAL"))
    return gw


def test_accepted_surfaces_result_with_provenance() -> None:
    r = _gw().call_tool("kb.lookup", {"q": "dao"})
    assert r["verdict"] == "accepted"
    assert r["result"] == {"answer": "Laozi", "sources": ["wiki://dao-de-jing"]}
    assert r["provenance_id"].startswith("clm_")


def test_ungrounded_output_is_withheld() -> None:
    r = _gw().call_tool("kb.lookup", {"q": "mystery"})   # no sources
    assert r["verdict"] == "held" and r["held_reason"] == "no_source"
    assert r["result"] is None                            # raw output NEVER surfaced


def test_env_verified_pass_and_fail() -> None:
    ok = _gw().call_tool("calc.eval", {"expr": "6*7", "verify_spec": {"expected": 42}})
    assert ok["verdict"] == "accepted" and ok["result"]["candidate"] == "6*7"
    bad = _gw().call_tool("calc.eval", {"expr": "6*7", "verify_spec": {"expected": 41}})
    assert bad["verdict"] == "rejected" and bad["result"] is None


def test_blp_no_read_up_holds_and_withholds() -> None:
    r = _gw().call_tool("secret.read", {"q": "dao"}, clearance="UNCLASSIFIED")
    assert r["verdict"] == "held" and r["held_reason"] == "blp_violation" and r["result"] is None


def test_role_allowlist_denied() -> None:
    r = _gw().call_tool("agents.only", {"q": "dao"}, role="role_02_coding")
    assert r["error"]["code"] == "UNAUTHENTICATED"


def test_role_blp_cap_denied() -> None:
    # content_marketing is capped at UNCLASSIFIED; conf.tool is CONFIDENTIAL
    r = _gw().call_tool("conf.tool", {"q": "dao"}, role="role_06_content_marketing",
                        clearance="CONFIDENTIAL")
    assert r["error"]["code"] == "UNAUTHENTICATED"


def test_kill_switch_blocks() -> None:
    gw = _gw()
    gw.contract.engage_kill_switch("incident")
    r = gw.call_tool("kb.lookup", {"q": "dao"})
    assert r["error"]["code"] == "UNAVAILABLE" and r["error"]["retryable"] is True


def test_dry_run_holds_side_effecting_tool() -> None:
    called = {"n": 0}

    def writer(args):
        called["n"] += 1
        return {"answer": "done", "sources": ["s"]}

    gw = _gw()
    gw.register(ToolEntry(id="fs.write", handler=writer, verifier_ref="grounding",
                          blp_level="UNCLASSIFIED", side_effects="write"))
    r = gw.call_tool("fs.write", {}, dry_run=True)
    assert r["verdict"] == "held" and r["held_reason"] == "needs_human"
    assert r["result"] is None and called["n"] == 0       # not executed


def test_unknown_tool_bad_request() -> None:
    assert _gw().call_tool("nope.tool", {})["error"]["code"] == "BAD_REQUEST"


def test_budget_stop_and_report() -> None:
    gw = _gw(call_budget=0)
    r = gw.call_tool("kb.lookup", {"q": "dao"})
    assert r["verdict"] == "held" and r["held_reason"] == "over_budget"


def test_competence_and_list_tools() -> None:
    gw = _gw()
    gw.call_tool("kb.lookup", {"q": "dao"})               # accepted -> reliability up
    tools = {t["tool_id"]: t for t in gw.list_tools()}
    assert "kb.lookup" in tools and tools["kb.lookup"]["reliability"] >= 0.5
    assert {"gateway_describe", "list_tools", "call_tool"}.issubset(set(gw.describe()["capabilities"]))


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_gateway: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
