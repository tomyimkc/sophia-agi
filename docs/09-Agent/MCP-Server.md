# Sophia AGI MCP server

Local MCP tools for **validate**, **epistemic gate**, **benchmark**, **corpus lookup**, and **dispute notes**.

Package path: `sophia_mcp/` (avoids name clash with pip `mcp`).

## Tools

| Tool | Description |
|------|-------------|
| `sophia_validate` | Validate `data/attributions.json` + `training/examples/` |
| `sophia_corpus_stats` | Version, counts, benchmark case totals |
| `sophia_export_corpus` | Write `training/corpus.jsonl` from examples |
| `sophia_gate_check` | Post-generation gate (traps, 中文, discipline) |
| `sophia_benchmark_list` | List case IDs + questions per domain |
| `sophia_benchmark_score` | Score `{case_id: response}` JSON |
| `sophia_get_attribution` | Lookup `data/attributions.json` by textId |
| `sophia_get_record` | Lookup psychology/history/religion/philosophy record |
| `sophia_list_disputes` | List `docs/04-Disputes/` slugs |
| `sophia_read_dispute` | Read dispute markdown by slug |

## Install

```bash
pip install -r requirements-mcp.txt
python sophia_mcp/server.py   # stdio MCP — waits for client
python tests/test_mcp_tools.py
```

## Cursor / Grok wiring

Edit `sophia-agi/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "sophia-agi": {
      "command": "python",
      "args": ["sophia_mcp/server.py"],
      "cwd": "C:/Users/tomyim/Documents/GitHub/sophia-agi"
    }
  }
}
```

See `.cursor/mcp.json.example`. Reload MCP after edits.

## Skills pairing

| Skill | Install |
|-------|---------|
| Project `/sophia-agi` | `.grok/skills/sophia-agi/` in repo |
| Portable `/sophia-source-discipline` | `python tools/install_skills.py --all` |

Full guide: [Skills-Install.md](Skills-Install.md)

## Security

- Local repo reads only; no API keys required
- `sophia_export_corpus` writes `training/corpus.jsonl` only
- Does not run shell or `sophia_agent.py` automatically