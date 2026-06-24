#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bridge an OpenRouter-hosted model to a stdio MCP server (OpenAI-compatible tool loop).

Implements the OpenRouter "MCP servers for coding agents" pattern:
https://openrouter.ai/docs/cookbook/coding-agents/mcp-servers

Connect to a local MCP server, list its tools, convert MCP tool schemas to OpenAI
function specs, and run a stateful tool-calling loop against an OpenRouter model.

Install:  pip install openai mcp
Auth:     export OPENROUTER_API_KEY=sk-or-v1-...   (keep it in .env; never commit it)

Usage:
  python openrouter_mcp_client.py \
      --model anthropic/claude-sonnet-4.5 \
      --server "python -m my_mcp_server" \
      --prompt "What tools do you have? Call one."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from contextlib import AsyncExitStack
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def convert_tool_format(tool: Any) -> dict:
    """MCP tool -> OpenAI function spec. `required` defaults to [] for lax MCP servers."""
    schema = getattr(tool, "inputSchema", None) or {}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        },
    }


async def run(model: str, server_cmd: str, prompt: str, max_turns: int) -> int:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        print(f"Install deps: pip install openai mcp ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY (keep it in .env; never commit it).", file=sys.stderr)
        return 1

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        # OpenRouter attribution headers (optional but recommended):
        default_headers={"HTTP-Referer": "https://github.com/tomyimkc/sophia-agi",
                         "X-Title": "sophia-agi openrouter-mcp"},
    )

    parts = shlex.split(server_cmd)
    params = StdioServerParameters(command=parts[0], args=parts[1:], env=None)

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        listed = await session.list_tools()
        tools = [convert_tool_format(t) for t in listed.tools]
        print(f"[mcp] {len(tools)} tool(s): {', '.join(t['function']['name'] for t in tools)}", flush=True)

        messages: list[dict] = [{"role": "user", "content": prompt}]
        for turn in range(max_turns):
            resp = client.chat.completions.create(model=model, messages=messages, tools=tools or None)
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print(f"\n{msg.content}")
                return 0

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"[tool] {name}({args})", flush=True)
                try:
                    result = await session.call_tool(name, args)
                    content = "\n".join(
                        getattr(c, "text", str(c)) for c in (result.content or [])
                    ) or "(no content)"
                except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
                    content = f"ERROR calling {name}: {type(exc).__name__}: {exc}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        print("[warn] hit --max-turns without a final answer", file=sys.stderr)
        return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.5", help="OpenRouter model id (vendor/model)")
    ap.add_argument("--server", required=True, help="stdio MCP server launch command, e.g. 'python -m my_mcp_server'")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()
    return asyncio.run(run(args.model, args.server, args.prompt, args.max_turns))


if __name__ == "__main__":
    raise SystemExit(main())
