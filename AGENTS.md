# Sophia AGI — AI Assistant Guide

Open corpus for **provenance-aware philosophy** and AGI-shaped epistemic reasoning.

## Read first

1. `README.md`
2. `docs/00-Index/Home.md`
3. `data/attributions.json`
4. `docs/04-Disputes/` for contested cases

## Rules

- Never attribute a text to a figure listed in `doNotAttributeTo`.
- Flag `authorConfidence: legendary` or `compiled` as uncertain — do not state as settled fact.
- Keep 儒家 (Confucian) and 道家 (Daoist) lineages distinct unless evidence supports a link.
- Training outputs: bilingual EN + 中文 summary; JSONL-compatible structure.

## Agents

| Agent | Entry | Role |
|-------|-------|------|
| **Sophia Advisor** | `python tools/sophia_agent.py advisor "..."` | Project & epistemic decisions |
| **Sophia Repo** | `python tools/sophia_agent.py repo "..."` | Repo ops + approved tools |
| **Sophia Life** | `python tools/sophia_agent.py life "..."` | General decisions + guardrails |
| `grok-cli-teacher` | `.grok/agents/grok-cli-teacher.md` | Training pair generation |

## Skill + MCP

| Layer | Path |
|-------|------|
| **Grok skill** | `.grok/skills/sophia-agi/SKILL.md` — `/sophia-agi` |
| **MCP server** | `mcp/server.py` — `sophia_validate`, `sophia_gate_check`, `sophia_benchmark_*` |
| **Setup** | `docs/09-Agent/MCP-Server.md`, `.cursor/mcp.json.example` |

Docs: `docs/09-Agent/Sophia-Agent.md`

## Validation

```bash
python tools/validate_attribution.py
python tools/export_training_jsonl.py
```