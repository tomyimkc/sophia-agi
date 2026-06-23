#!/usr/bin/env python3
"""Sophia Gateway MCP server — the super-MCP front door.

Run:
    python gateway/server.py

This exposes the in-process ``gateway/`` package as a real MCP server. Every
``gateway_call_tool`` invocation goes through the fail-closed Gateway interceptor:
role/scope → firewall → budget/kill switch → BLP → dispatch → verify → provenance
stamp → audit → competence update. Raw tool output is returned only when accepted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import (  # noqa: E402
    Gateway,
    HttpMcpTransport,
    ToolEntry,
    improve_skill,
    register_knowledge_tools,
    register_mcp_server,
    synthesize_skill,
    verified_consensus,
)
from sophia_contract import SophiaContract  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised by MCP users, not unit tests
    raise SystemExit("Install MCP deps: pip install -r requirements-mcp.txt") from exc

mcp = FastMCP(
    "sophia-gateway",
    instructions=(
        "Sophia Gateway: a fail-closed super-MCP that gates downstream tools, "
        "skills, knowledge, verifier synthesis, and verified consensus. Only "
        "accepted outputs surface; rejected/held outputs are withheld."
    ),
)

_GW = Gateway(contract=SophiaContract())
register_knowledge_tools(_GW)


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _loads_obj(raw: str | None, *, default):
    if raw is None or raw == "":
        return default
    return json.loads(raw)


def _pairs(raw: str) -> list[tuple[str, bool]]:
    data = _loads_obj(raw, default=[])
    out: list[tuple[str, bool]] = []
    for item in data:
        if isinstance(item, dict):
            text = item.get("text") or item.get("answer") or item.get("input") or item.get("query") or ""
            label = item.get("label", item.get("accepted", item.get("positive", False)))
            out.append((str(text), bool(label)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), bool(item[1])))
    return out


@mcp.tool()
def gateway_describe() -> str:
    """Describe gateway version, capabilities, and registered tools."""
    return dumps(_GW.describe())


@mcp.tool()
def gateway_list_tools(role: str = "") -> str:
    """List registered gateway tools visible to an optional role."""
    return dumps({"tools": _GW.list_tools(role=role or None)})


@mcp.tool()
def gateway_rank_tools(role: str = "") -> str:
    """List tools sorted by measured reliability."""
    return dumps({"tools": _GW.rank_tools(role=role or None)})


@mcp.tool()
def gateway_call_tool(
    tool_id: str,
    args_json: str = "{}",
    role: str = "",
    clearance: str = "UNCLASSIFIED",
    dry_run: bool = False,
    idempotency_key: str = "",
) -> str:
    """Call a registered tool through the fail-closed gateway.

    Invariant: result is present only when verdict == accepted.
    """
    return dumps(_GW.call_tool(
        tool_id,
        _loads_obj(args_json, default={}),
        role=role or None,
        clearance=clearance,
        dry_run=dry_run,
        idempotency_key=idempotency_key or None,
    ))


@mcp.tool()
def gateway_verify(
    content_json: str,
    verifier_ref: str = "grounding",
    sources_json: str = "[]",
    blp_level: str = "UNCLASSIFIED",
    clearance: str = "UNCLASSIFIED",
) -> str:
    """Universal Verify API: verify arbitrary content/output via a verifier_ref."""
    try:
        content = json.loads(content_json)
    except json.JSONDecodeError:
        content = content_json
    return dumps(_GW.verify(
        content,
        verifier_ref=verifier_ref,
        sources=_loads_obj(sources_json, default=[]),
        blp_level=blp_level,
        clearance=clearance,
    ))


@mcp.tool()
def gateway_register_http_mcp_server(
    server_id: str,
    base_url: str,
    tools_json: str,
    token: str = "",
    timeout_sec: int = 30,
) -> str:
    """Federate a downstream Streamable-HTTP MCP server behind the gateway.

    tools_json: [{id, blp_level?, verifier_ref?, risk_tier?, allowed_roles?,
    side_effects?, description?}]
    """
    transport = HttpMcpTransport(base_url, token=token or None, timeout_sec=timeout_sec)
    entries = register_mcp_server(_GW, server_id, transport, tools=_loads_obj(tools_json, default=[]))
    return dumps({"registered": [e.public(_GW.competence.reliability(e.id)) for e in entries]})


@mcp.tool()
def gateway_register_echo_tool(tool_id: str = "echo", verifier_ref: str = "grounding") -> str:
    """Register a safe local echo/demo tool for smoke testing the gateway."""
    def _echo(args):
        sources = args.get("sources") or ["gateway://echo"]
        return {"answer": args.get("text", ""), "sources": sources}

    entry = _GW.register(ToolEntry(id=tool_id, handler=_echo, verifier_ref=verifier_ref,
                                   side_effects="none", description="gateway echo smoke-test tool"))
    return dumps({"registered": entry.public(_GW.competence.reliability(entry.id))})


@mcp.tool()
def gateway_synthesize_skill(
    domain: str,
    examples_json: str,
    blp_level: str = "UNCLASSIFIED",
    threshold: float = 0.8,
    seed: int = 1,
) -> str:
    """Create/register a meta-verified classifier skill from labelled examples.

    examples_json accepts [{text, label}] or [[text, label], ...].
    """
    return dumps(synthesize_skill(_GW, domain, _pairs(examples_json), blp_level=blp_level,
                                  threshold=threshold, seed=seed))


@mcp.tool()
def gateway_forge_skill(
    spec_json: str,
    threshold: float = 0.8,
    seed: int = 1,
    proposer_model: str = "mock",
    register: bool = True,
) -> str:
    """Skill Forge: task spec -> verifier -> files -> optional gateway registration.

    spec_json: {"task_id", "description"?, "examples":[{text,label}|[text,label]]}
    """
    from tools.sophia_skill_forge import forge_skill

    return dumps(forge_skill(
        _loads_obj(spec_json, default={}),
        threshold=threshold,
        seed=seed,
        proposer_model=proposer_model,
        gateway=_GW if register else None,
    ))


@mcp.tool()
def gateway_improve_skill(skill_id: str, examples_json: str, threshold: float = 0.8, seed: int = 1) -> str:
    """Attach a meta-verified synthesized verifier to an existing skill/tool."""
    return dumps(improve_skill(_GW, skill_id, _pairs(examples_json), threshold=threshold, seed=seed))


@mcp.tool()
def gateway_verified_consensus(
    candidates_json: str,
    verifier_ref: str = "grounding",
    blp_level: str = "UNCLASSIFIED",
    clearance: str = "UNCLASSIFIED",
    topic: str = "consensus",
) -> str:
    """Choose among candidate outputs by verification, not majority vote."""
    return dumps(verified_consensus(
        _GW,
        _loads_obj(candidates_json, default=[]),
        verifier_ref=verifier_ref,
        blp_level=blp_level,
        clearance=clearance,
        topic=topic,
    ))


if __name__ == "__main__":
    mcp.run()
