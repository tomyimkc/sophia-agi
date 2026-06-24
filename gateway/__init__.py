# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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
from gateway.skills import SkillEntry, eval_skill
from gateway.federation import HttpMcpTransport, StubTransport, register_mcp_server
from gateway.knowledge import register_knowledge_tools
from gateway.skill_flywheel import improve_skill, synthesize_skill
from gateway.consensus import verified_consensus

__all__ = [
    "Gateway", "Registry", "ToolEntry", "SkillEntry", "eval_skill",
    "register_mcp_server", "StubTransport", "HttpMcpTransport",
    "register_knowledge_tools", "improve_skill", "synthesize_skill", "verified_consensus",
]
