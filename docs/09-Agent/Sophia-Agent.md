# Sophia Agent — three AI paths

Turn the repo into an agent that **retrieves sources**, **reasons**, **self-checks**, and **acts** (with approval).

## Paths

| Path | Command | Purpose |
|------|---------|---------|
| **Advisor** | `python tools/sophia_agent.py advisor "..."` | Project, corpus, benchmark, growth decisions |
| **Repo operator** | `python tools/sophia_agent.py repo "..."` | Next steps + optional repo tool execution |
| **Life & work** | `python tools/sophia_agent.py life "..."` | General decisions with wisdom framing + guardrails |

## Setup

Same `.env` as benchmarks:

```env
ANTHROPIC_API_KEY=...
ANTHROPIC_BASE_URL=https://api.llmhub.com.cn
ANTHROPIC_MODEL=claude-sonnet-4-6
```

```powershell
pip install anthropic
```

## Examples

```powershell
# Advisor — should I launch?
python tools/sophia_agent.py advisor "Should I post on Hacker News this week?"

# Repo — what next?
python tools/sophia_agent.py repo "What are the top 3 repo tasks after v0.3.1?"

# Repo — run tools (requires approval)
python tools/sophia_agent.py repo "Validate and export corpus" --execute --approve

# Life — personal decision
python tools/sophia_agent.py life "Should I focus on launch or more training examples this month?"

# List repo tools
python tools/sophia_agent.py tools
```

## Architecture

```text
Question → RAG (data/, docs/, examples/) → Claude → Epistemic gate → Memory
                                              ↓
                                    Repo tools (if --execute --approve)
```

## Repo tools (approval required)

| Tool | Risk | Action |
|------|------|--------|
| `validate` | low | `validate_attribution.py` |
| `export_corpus` | low | `export_training_jsonl.py` |
| `build_reference` | low | `build_reference_responses.py` |
| `update_leaderboards` | low | `update_leaderboards.py` |
| `upload_hf` | medium | `upload_huggingface.py` |
| `benchmark_claude` | high | full external benchmark run |

Memory: `agent/memory/decisions.jsonl` (gitignored).

## Roadmap alignment

- Phase 3: Runtime gate → `agent/gate.py`
- Phase 4: Repo tools → `agent/tools.py`
- Phase 5 M5: Planner + memory → this agent CLI