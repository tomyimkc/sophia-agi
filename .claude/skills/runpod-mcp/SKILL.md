---
name: runpod-mcp
description: >
  Manage RunPod infrastructure through MCP — create/list/stop GPU pods, check pod and
  serverless-endpoint status, manage templates, network volumes, and container
  registries via natural language. Use when the user wants pod running status, to
  launch/terminate a RunPod pod, inspect a serverless endpoint, or asks about "RunPod
  MCP". Wraps the official @runpod/mcp-server (RunPod REST API). Requires a RUNPOD_API_KEY.
metadata:
  short-description: "Official RunPod MCP server — manage pods/endpoints, check status"
  source: "https://docs.runpod.io/get-started/mcp-servers"
  package: "@runpod/mcp-server"
---

# RunPod MCP (official)

RunPod ships an official MCP server (`@runpod/mcp-server`) that exposes the RunPod REST
API as MCP tools, so an MCP client (Claude Code, Cursor, Codex, etc.) can manage GPU
infrastructure in natural language: **create and manage Pods**, **serverless endpoints**,
**templates, network volumes, and container registries** — and read **pod running status**.

## Installed in this repo

`.mcp.json` registers it as the `runpod` server:

```json
"runpod": {
  "command": "npx",
  "args": ["-y", "@runpod/mcp-server@latest"],
  "env": { "RUNPOD_API_KEY": "${RUNPOD_API_KEY}" }
}
```

On session start, Claude Code launches `npx -y @runpod/mcp-server@latest` and surfaces the
tools as `mcp__runpod__*`. The `${RUNPOD_API_KEY}` is expanded from the environment — the
secret is **never** written into `.mcp.json` or any tracked file.

## Prerequisites

- **Node / npx** (the server runs via `npx`). This repo's environments have Node 22.
- A **RunPod API key** from https://www.runpod.io/console/user/settings, set as the
  `RUNPOD_API_KEY` environment variable.

## Enabling it (the one manual step)

Set `RUNPOD_API_KEY` in the environment, then restart the session so the server picks it up:
- **Claude Code cloud server:** add `RUNPOD_API_KEY` to the environment's secrets/env
  (see https://code.claude.com/docs/en/claude-code-on-the-web). This is the *same* key the
  `speedup-runpod` GitHub workflow uses as a repo secret, but the MCP server needs it as an
  **environment variable**, not a GitHub Actions secret.
- **Local:** `echo 'RUNPOD_API_KEY=...' >> .env` (`.env` is gitignored) and export it.

Verify after enabling by asking the agent to list pods (e.g. call `mcp__runpod__*` to list
pods / show status).

## Typical uses

- "What pods are running?" → list pods + status (the **pod running status** ask).
- "Stop pod <id>" / "terminate idle pods" → cleanup.
- "Launch a 4090 pod from template X" → create a pod.
- Inspect serverless endpoints, templates, network volumes, registries.

## Relationship to the speedup benchmark

The `speedup-runpod` workflow + `tools/runpod_speedup.py` rent a pod *programmatically* via
the REST API to run the LoRA benchmark and always delete it. This RunPod MCP server is the
*interactive* counterpart: it lets you (or the agent) inspect/manage pods on demand —
including checking whether a benchmark pod is still up.

## Security

- Key only via `RUNPOD_API_KEY` env; never hardcode or commit it. Rotate any key shared in
  plaintext at https://www.runpod.io/console/user/settings.
- The server has **full RunPod API access** (it can create/terminate billable pods) — treat
  its tools as money-spending and confirm destructive/creating actions before running them.
