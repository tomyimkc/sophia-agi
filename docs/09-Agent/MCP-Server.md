# Sophia AGI MCP server

Local MCP tools for **validate**, **epistemic gate**, and **benchmark** — usable from Cursor, Grok CLI, or any MCP client.

## Tools

| Tool | Description |
|------|-------------|
| `sophia_validate` | Validate `data/attributions.json` + `training/examples/` |
| `sophia_corpus_stats` | Version, counts, benchmark case totals |
| `sophia_gate_check` | Post-generation gate (traps, 中文, discipline) |
| `sophia_benchmark_list` | List case IDs + questions per domain |
| `sophia_benchmark_score` | Score a JSON map of `{case_id: response}` |

## Install

```bash
pip install -r requirements-mcp.txt
python mcp/server.py   # stdio MCP — should hang waiting for client
```

## Cursor / Grok wiring

Edit `sophia-agi/.cursor/mcp.json` (merge with existing `github` entry):

```json
{
  "mcpServers": {
    "sophia-agi": {
      "command": "python",
      "args": ["mcp/server.py"],
      "cwd": "C:/Users/tomyim/Documents/GitHub/sophia-agi"
    },
    "github": {
      "url": "https://api.githubcopilot.com/mcp/x/issues",
      "headers": {
        "Authorization": "Bearer YOUR_GITHUB_PAT"
      }
    }
  }
}
```

Replace `cwd` with your clone path. On macOS/Linux use forward slashes.

**Global fallback:** duplicate the `sophia-agi` block in `~/.cursor/mcp.json` with an absolute `cwd`.

Reload MCP in Cursor: Command Palette → **MCP: List Servers** → restart `sophia-agi`.

## Skill pairing

Project skill: `.grok/skills/sophia-agi/SKILL.md`

- Slash: `/sophia-agi`
- Auto-invokes on: source discipline, provenance, attribution traps, corpus validation
- Prefer MCP tools when the server is connected; CLI fallback documented in the skill

## Example tool calls

**Gate check**

```json
{
  "response": "Confucius wrote the Dao De Jing...",
  "question": "Did Confucius write the Dao De Jing?",
  "mode": "advisor"
}
```

**Benchmark score**

```json
{
  "domain": "philosophy",
  "responses_json": "{\"trap_confucius_ddj\": \"Confucius did not write the Dao De Jing...\"}"
}
```

## Security

- No network calls; reads local repo data only
- Does not execute `sophia_agent.py` or shell commands (read-only eval + validate)
- Keep `.env` out of MCP — gate/validate do not need API keys