"""Sophia Gateway — the super-MCP / super-skills governance proxy (P0 MVP).

One endpoint that federates tools/skills and gates EVERY call through the contract:
role scope → budget/kill-switch → BLP no-read-up → dispatch → verify output →
provenance-stamp + no-write-down → audit/ROI → competence update. A result is returned
ONLY when verification accepts; raw unverified output is never surfaced. Fail-closed.

Spec: docs/11-Platform/Sophia-Gateway.md.

    from gateway import Gateway, ToolEntry
    gw = Gateway()
    gw.register(ToolEntry(id="kb.lookup", handler=fn, verifier_ref="grounding"))
    gw.call_tool("kb.lookup", {"q": "..."}, role="role_09_agents", clearance="SECRET")
"""

from gateway.registry import Registry, ToolEntry
from gateway.interceptor import Gateway

__all__ = ["Gateway", "Registry", "ToolEntry"]
