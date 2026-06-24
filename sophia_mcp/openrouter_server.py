#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""OpenRouter MCP server — call any OpenRouter-hosted model as an MCP tool.

There is no official OpenRouter MCP server (OpenRouter is an OpenAI-compatible LLM API,
not an MCP server), so this is a small first-party wrapper: it exposes OpenRouter over
MCP so Claude Code (or any MCP client) can route prompts to models like
anthropic/claude-sonnet-4.5, qwen/qwen-2.5-72b-instruct, deepseek/deepseek-chat, etc.

Auth: reads OPENROUTER_API_KEY from the environment (wired via .mcp.json env). The key is
never hardcoded or logged. Rotate any key shared in plaintext at https://openrouter.ai/keys.

Uses only the stdlib (urllib) for HTTP — no extra deps beyond `mcp` — and respects
HTTPS_PROXY automatically, so it works behind the agent proxy.

Run: python3 sophia_mcp/openrouter_server.py     (stdio MCP server)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/tomyimkc/sophia-agi",
    "X-Title": "sophia-agi openrouter-mcp",
}

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install MCP deps: pip install -r requirements-mcp.txt") from exc

mcp = FastMCP(
    "openrouter",
    instructions=(
        "Call any OpenRouter-hosted model. openrouter_chat routes a prompt to a "
        "vendor/model id and returns its text; openrouter_models lists available models. "
        "Requires the OPENROUTER_API_KEY environment variable."
    ),
)


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env / .mcp.json env).")
    return key


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    req.add_header("Content-Type", "application/json")
    for k, v in ATTRIBUTION_HEADERS.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # respects HTTPS_PROXY
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc


@mcp.tool()
def openrouter_chat(prompt: str, model: str = DEFAULT_MODEL, system: str = "",
                    max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Send a prompt to an OpenRouter model (vendor/model id) and return its text reply."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    resp = _request("POST", "/chat/completions", body)
    try:
        return resp["choices"][0]["message"]["content"] or "(empty response)"
    except (KeyError, IndexError, TypeError):
        return json.dumps(resp)[:2000]


@mcp.tool()
def openrouter_models(filter: str = "") -> str:
    """List available OpenRouter models (id + context length); optional substring filter."""
    resp = _request("GET", "/models")
    rows = resp.get("data", []) if isinstance(resp, dict) else []
    out = []
    for m in rows:
        mid = m.get("id", "")
        if filter and filter.lower() not in mid.lower():
            continue
        out.append({"id": mid, "context": m.get("context_length"),
                    "prompt_price": (m.get("pricing") or {}).get("prompt")})
    return json.dumps({"count": len(out), "models": out[:200]}, indent=2)


if __name__ == "__main__":
    mcp.run()
