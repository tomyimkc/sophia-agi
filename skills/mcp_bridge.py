"""Bridge from the Skills layer to the Sophia MCP surface.

Two transports, auto-detected:

1. **in-process (default)** — calls ``sophia_mcp.tools_impl`` directly. No network,
   deterministic, dependency-free, works in CI. This is the honest default: the
   "MCP tools" are plain Python functions importable without the ``mcp`` package.
2. **HTTP (opt-in)** — when ``SOPHIA_MCP_URL`` is set, POSTs to a running MCP
   server. ``requests`` is imported lazily only in this mode.

Skills call :func:`call`; the ``@sophia_skill`` decorator turns any error here into
a fail-closed ``held`` result.
"""
from __future__ import annotations

import os
from typing import Any


def _tools():
    from sophia_mcp import tools_impl  # imported lazily so import errors fail closed

    return tools_impl


def mcp_status() -> dict:
    """Report which transport is active and whether it is reachable."""
    url = os.environ.get("SOPHIA_MCP_URL")
    if url:
        return {"mode": "http", "url": url}
    try:
        _tools()
        return {"mode": "in-process", "available": True}
    except Exception as e:  # pragma: no cover - only if the package is broken
        return {"mode": "in-process", "available": False, "error": str(e)}


def mcp_is_running() -> bool:
    """True if an MCP transport is usable (HTTP configured, or in-process import OK)."""
    status = mcp_status()
    return status.get("mode") == "http" or bool(status.get("available"))


def call(tool: str, /, **kwargs: Any) -> dict:
    """Invoke an MCP tool by name. Raises on unknown tool / transport error; the
    skill decorator wraps that into a fail-closed result."""
    url = os.environ.get("SOPHIA_MCP_URL")
    if url:
        return _http_call(url, tool, kwargs)
    fn = getattr(_tools(), tool, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"unknown MCP tool: {tool!r}")
    return fn(**kwargs)


def _http_call(url: str, tool: str, payload: dict) -> dict:
    try:
        import requests  # lazy; only needed for HTTP mode
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("SOPHIA_MCP_URL is set but 'requests' is not installed") from e
    resp = requests.post(f"{url.rstrip('/')}/tools/{tool}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_mcp_client():
    """Back-compat shim returning a tiny client object with ``.call`` / ``.status``."""
    return type(
        "MCPClient",
        (),
        {"call": staticmethod(call), "status": staticmethod(mcp_status), "is_running": staticmethod(mcp_is_running)},
    )
