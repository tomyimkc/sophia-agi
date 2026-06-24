---
name: openrouter-mcp
description: >
  Connect any OpenRouter-hosted model to MCP (Model Context Protocol) servers from a
  coding agent. Use when the user wants to drive MCP tools through OpenRouter, route a
  coding/tool-using agent via OpenRouter's OpenAI-compatible API, bridge MCP tool
  definitions to OpenAI function-calling, or asks about "OpenRouter MCP". Covers the
  request shape, the MCP↔OpenAI tool-schema conversion, the stateful tool-calling loop,
  required headers, and a runnable reference client (scripts/openrouter_mcp_client.py).
metadata:
  short-description: "Bridge OpenRouter models to MCP servers (OpenAI-compatible tool loop)"
  source: "https://openrouter.ai/docs/cookbook/coding-agents/mcp-servers"
---

# OpenRouter ↔ MCP for coding agents

OpenRouter exposes an **OpenAI-compatible** chat-completions API, so you can drive any
hosted model (Anthropic, OpenAI, Llama, Qwen, DeepSeek, …) with normal function-calling.
MCP servers, however, define tools in **MCP format**, not OpenAI format, and the MCP
protocol is **stateful** (you hold a session to the server). This skill bridges the two:
spin up/connect an MCP session, convert its tool list to OpenAI function specs, and run a
tool-calling loop against an OpenRouter model.

## Credentials (read this first)

- The key lives **only** in the `OPENROUTER_API_KEY` environment variable. **Never**
  hardcode it in a script, commit it, or paste it into a tracked file — this repo's
  `.gitignore` already ignores `.env`, so put it there:
  ```bash
  echo 'OPENROUTER_API_KEY=sk-or-v1-...' >> .env      # .env is gitignored
  ```
- If a key was ever shared in plaintext (chat, logs, a commit), **rotate it** at
  https://openrouter.ai/keys and replace the env value.

## Request shape

- Base URL: `https://openrouter.ai/api/v1` (OpenAI-compatible)
- Endpoint: `POST /chat/completions`
- Model id format: `vendor/model`, e.g. `anthropic/claude-sonnet-4.5`,
  `qwen/qwen-2.5-72b-instruct`, `deepseek/deepseek-chat`.
- Headers:
  - `Authorization: Bearer $OPENROUTER_API_KEY` (the OpenAI SDK sets this from `api_key`)
  - Optional but recommended for OpenRouter attribution/ranking:
    `HTTP-Referer: <your app/site>` and `X-Title: <your app name>`.

## The bridge, in three parts

1. **Connect the MCP server** and `list_tools()` over a `ClientSession` (stdio transport
   for a local server command; streamable-HTTP for a remote one). Keep the session open —
   MCP is stateful.
2. **Convert** each MCP tool to an OpenAI function spec:
   ```python
   def convert_tool_format(tool):
       return {
           "type": "function",
           "function": {
               "name": tool.name,
               "description": tool.description,
               "parameters": {
                   "type": "object",
                   "properties": tool.inputSchema["properties"],
                   "required": tool.inputSchema.get("required", []),
               },
           },
       }
   ```
3. **Tool-calling loop**: send messages + tools → if the model returns `tool_calls`,
   execute each via `await session.call_tool(name, args)`, append the results as
   `role: "tool"` messages, and call the model again until it answers with no tool calls.

## Installed in this repo

This repo ships a first-party OpenRouter **MCP server** at `sophia_mcp/openrouter_server.py`,
registered in `.mcp.json` as the `openrouter` server. It exposes two tools to any MCP
client (Claude Code included):
- `openrouter_chat(prompt, model, system?, max_tokens?, temperature?)` — route a prompt to
  a `vendor/model` id and get the text reply.
- `openrouter_models(filter?)` — list available models (id, context length, prompt price).

It uses only the stdlib for HTTP (no extra deps beyond `mcp`), respects `HTTPS_PROXY`, and
reads the key from the `OPENROUTER_API_KEY` env var (wired via `.mcp.json` env expansion).
To enable it on a Claude Code cloud server, set `OPENROUTER_API_KEY` in the environment's
secrets — do **not** put the value in `.mcp.json` or any tracked file.

## Runnable reference client (OpenRouter as the MCP *client*)

`scripts/openrouter_mcp_client.py` is a complete, dependency-light implementation
(`pip install openai mcp`). It connects to a **stdio** MCP server you specify, lists its
tools, and runs the loop against an OpenRouter model. Usage:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...            # from .env, never committed
python scripts/openrouter_mcp_client.py \
    --model anthropic/claude-sonnet-4.5 \
    --server "python -m my_mcp_server" \
    --prompt "List the available tools and call one."
```

For this repo's own MCP server, point `--server` at the Sophia MCP launch command
(see `sophia_mcp/server.py` / `.mcp.json`).

## Caveats

- **Stateful**: keep the `ClientSession` alive for the whole conversation; tools can
  depend on prior calls. The reference client uses `AsyncExitStack` for clean teardown.
- **Schema drift**: some MCP tools omit `required` or use nested schemas; the converter
  defaults `required` to `[]` and passes `properties` through verbatim.
- **Not every model tool-calls well**: prefer strong tool-callers (Claude,
  GPT-class, Qwen-2.5-Instruct) for multi-step MCP loops.
- **Cost/rate limits** are OpenRouter-side; set `HTTP-Referer`/`X-Title` for proper
  attribution and watch usage at https://openrouter.ai/activity.
