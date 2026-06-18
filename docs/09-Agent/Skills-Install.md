# Sophia skills — install guide

Two skills work together:

| Skill | Scope | Slash command |
|-------|-------|---------------|
| **sophia-source-discipline** | User (any project) | `/sophia-source-discipline` |
| **sophia-agi** | Project (this repo) | `/sophia-agi` |

## Quick install

```bash
python tools/install_skills.py --all --cursor
```

This copies:

- `skills/portable/sophia-source-discipline/` → `~/.grok/skills/`
- `.grok/skills/sophia-agi/` → `~/.grok/skills/sophia-agi`
- Optional: portable skill → `~/.cursor/skills/` (`--cursor`)

## Manual install (portable only)

```bash
# Grok CLI / global
cp -r skills/portable/sophia-source-discipline ~/.grok/skills/

# Windows PowerShell
Copy-Item -Recurse skills\portable\sophia-source-discipline $env:USERPROFILE\.grok\skills\
```

## MCP pairing

Install MCP deps and wire the server (see [MCP-Server.md](MCP-Server.md)):

```bash
pip install -r requirements-mcp.txt
```

Portable skill uses MCP when available; otherwise follows trap rules in `references/trap-patterns.md`.

## When to use which

| Situation | Skill |
|-----------|-------|
| Editing sophia-agi corpus | `/sophia-agi` |
| Attribution question in HVE or any repo | `/sophia-source-discipline` |
| Validate before PR | MCP `sophia_validate` |
| Check draft blog post / answer | MCP `sophia_gate_check` |

## Verify

Restart Grok/Cursor session, then:

```
/sophia-source-discipline Did Confucius write the Dao De Jing?
/sophia-agi validate the corpus
```