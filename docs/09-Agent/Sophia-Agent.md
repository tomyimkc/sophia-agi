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
Question → RAG (rag/index curated corpus) → Claude → Epistemic gate → Memory
                                              ↓
                                    Repo tools (if --execute --approve)
```

When `rag/index/chunks.jsonl` exists, retrieval uses the **curated index** (holdouts excluded). See [Online-RAG.md](Online-RAG.md) for Gemini/Vertex generation and Cloud Run API.

```powershell
python tools/build_rag_index.py
python tools/sophia_rag.py "Did Confucius write the Dao De Jing?"   # Gemini + gate
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

## Web Evidence And Review Tools

Sophia can now collect local RAG evidence plus optional online evidence:

```powershell
# local-only source context
python tools/sophia_agent.py web_evidence "Buddhist no-self doctrine and modern psychology"

# opt-in online search via Brave, Tavily, or SerpAPI
python tools/sophia_agent.py web_evidence "Buddhist no-self doctrine and modern psychology" --web-evidence --web-provider auto
```

Draft answers can be checked before publishing:

```powershell
python tools/sophia_agent.py rubric_review "Question" \
  --response "Draft answer" \
  --must-include-json "[\"source path\", \"Decision\"]"
```

Hidden evals use the same review pass to build a rubric evidence map. Online
search remains disabled by default for hidden packs to avoid leaking reviewer
prompts to third-party APIs.

## Runtime gate (v0.5.0)

After the LLM answers, `agent/gate.py` runs:

1. Style checks (source-discipline markers, 中文摘要)
2. Question-matched benchmark traps via `agent/benchmark_checks.py` (same logic as `score_benchmark.py`)

Gate output includes `violations` and per-trap `checks`. Philosophy reference responses pass 100% (`tests/test_gate.py`).

## Roadmap alignment

- Phase 3: Runtime gate → `agent/gate.py` ✅ (v0.5.0)
- Phase 4: Repo tools → `agent/tools.py`
- Phase 5 M5: Planner + memory → this agent CLI
