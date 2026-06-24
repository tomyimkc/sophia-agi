#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia Gateway P0 demo — federate tools, gate every call, surface only 'accepted'.

    python tools/run_gateway_demo.py

Offline, no key. Shows: a grounded read accepted + provenance-stamped; an ungrounded
read held (output withheld); an execution-verified pass and fail; a SECRET tool blocked
by BLP no-read-up; a role denied by scope; and the kill switch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import Gateway, ToolEntry  # noqa: E402


def _kb(args):
    return ({"answer": "Laozi", "sources": ["wiki://dao-de-jing"]} if args.get("q") == "dao"
            else {"answer": "unknown", "sources": []})


def _calc(args):
    return {"candidate": args["expr"]}


def _show(label, r):
    res = "withheld" if r.get("result") is None else r.get("result")
    extra = f" · held_reason={r['held_reason']}" if r.get("held_reason") else ""
    extra += f" · {r['error']['code']}" if r.get("error") else ""
    print(f"  {label:42s} verdict={r.get('verdict') or 'ERROR'}{extra}")
    print(f"      result={res}  provenance={r.get('provenance_id')}")


def main() -> int:
    gw = Gateway()
    gw.register(ToolEntry(id="kb.lookup", handler=_kb, verifier_ref="grounding", side_effects="read"))
    gw.register(ToolEntry(id="calc.eval", handler=_calc, verifier_ref="env:arithmetic"))
    gw.register(ToolEntry(id="secret.read", handler=_kb, verifier_ref="grounding", blp_level="SECRET"))
    gw.register(ToolEntry(id="agents.only", handler=_kb, verifier_ref="grounding",
                          allowed_roles={"role_09_agents"}))

    print("Sophia Gateway —", gw.describe()["gateway"], "·", len(gw.describe()["tools"]), "tools\n")
    _show("grounded read (q=dao)", gw.call_tool("kb.lookup", {"q": "dao"}))
    _show("ungrounded read (q=mystery)", gw.call_tool("kb.lookup", {"q": "mystery"}))
    _show("execution-verified 6*7==42", gw.call_tool("calc.eval", {"expr": "6*7", "verify_spec": {"expected": 42}}))
    _show("execution-verified 6*7==41", gw.call_tool("calc.eval", {"expr": "6*7", "verify_spec": {"expected": 41}}))
    _show("SECRET tool, UNCLASSIFIED caller", gw.call_tool("secret.read", {"q": "dao"}))
    _show("agents-only tool as role_02_coding", gw.call_tool("agents.only", {"q": "dao"}, role="role_02_coding"))
    gw.contract.engage_kill_switch("demo")
    _show("any call with kill switch engaged", gw.call_tool("kb.lookup", {"q": "dao"}))
    print("\nOnly 'accepted' surfaces a result; everything else fails closed and is withheld.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
