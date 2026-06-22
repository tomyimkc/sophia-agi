"""Federate a real downstream MCP server's tools behind the gateway (P1).

Each downstream tool is registered as a ``ToolEntry`` whose handler calls a transport.
``StubTransport`` makes the whole path offline-testable; ``HttpMcpTransport`` speaks the
MCP Streamable-HTTP ``tools/call`` over urllib (no new deps) and is used only when a live
server is configured — mirroring the model adapter's offline-stubbable design.

    from gateway.federation import register_mcp_server, StubTransport
    register_mcp_server(gw, "fs", StubTransport({"fs.read": {"text": "...", "sources": ["f"]}}),
                        tools=[{"id": "fs.read", "blp_level": "CONFIDENTIAL", "verifier_ref": "grounding"}])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from gateway.registry import ToolEntry


class StubTransport:
    """Offline transport: maps tool_id -> canned output (callable or value)."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list = []

    def call(self, tool_id: str, args: dict):
        self.calls.append((tool_id, args))
        r = self.responses.get(tool_id)
        return r(args) if callable(r) else r


class HttpMcpTransport:
    """MCP Streamable-HTTP transport: POST tools/call to {base}. Used only live."""

    def __init__(self, base_url: str, *, token: "str | None" = None, timeout_sec: int = 30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec

    def call(self, tool_id: str, args: dict):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": tool_id, "arguments": args}}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base_url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"downstream MCP call failed: {exc!r}") from exc
        return (data.get("result") or {}).get("content", data.get("result"))


def register_mcp_server(gateway, server_id: str, transport, *, tools: "list[dict]") -> "list[ToolEntry]":
    """Register each downstream tool (metadata in ``tools``) as a gated gateway tool.

    Each ``tools`` item: {id, blp_level?, verifier_ref?, risk_tier?, allowed_roles?,
    side_effects?, description?}. The handler routes through ``transport``."""
    registered = []
    for meta in tools:
        tid = meta["id"]

        def _handler(args, _tid=tid):
            return transport.call(_tid, args)

        entry = ToolEntry(
            id=tid, handler=_handler, kind="mcp",
            blp_level=meta.get("blp_level", "UNCLASSIFIED"),
            verifier_ref=meta.get("verifier_ref", "none"),
            risk_tier=meta.get("risk_tier", "medium"),
            side_effects=meta.get("side_effects", "read"),
            allowed_roles=meta.get("allowed_roles"),
            description=meta.get("description", f"federated from {server_id}"),
        )
        registered.append(gateway.register(entry))
    return registered
