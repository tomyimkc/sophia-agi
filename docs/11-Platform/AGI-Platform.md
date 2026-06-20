# Sophia AGI Platform

The agentic platform that lets a base/frontier LLM (GLM-5.2, Claude, or a local
model) be improved by surrounding infrastructure: a unified model adapter, an
agent harness, a skills library, RAG/memory, verifiers, evals, and
trace/fine-tuning pipelines.

## Architecture

```
              ┌──────────────────────────────────────────────┐
  goal ─────► │ agent/harness.py  plan → act → critic → retry │
              │   planner   executor   critic    reflection   │
              └───┬──────────┬───────────┬──────────┬─────────┘
                  │          │           │          │
        agent/skills.py  agent/tools.py  agent/gate.py   RunStore
        (skill picks)   (approval-gated  + verifiers   (JSONL trace,
                         repo tools)     (score_pack)   checkpoint/resume)
                  │          │           │
                  └────► agent/model.py  ◄────  unified model adapter
                          (anthropic | openai-compatible: GLM-5.2 / vLLM /
                           SGLang / Ollama / llama.cpp / DeepSeek | grok | mock)
                          retry · fallback · streaming · tools · cost/latency
                  │
        agent/retrieval.py + web_evidence.py (RAG) · sophia_mcp/ (typed tools)
                  │
        eval: tools/eval_agent.py · tools/run_benchmark.py · hidden-eval harnesses
                  │
        improve: tools/train_lora.py · claude_teacher.py · correction_loop
```

Everything runs **offline with the `mock` provider**, so the whole stack is
testable without credentials.

## 1. Model adapter (`agent/model.py`)

One interface over every backend. Pick a provider by preset or `provider:model`.

```bash
python tools/agent_harness.py models --provider glm     # show resolved config
```

```python
from agent.model import default_client
client = default_client("glm:glm-5.2")          # or "ollama:llama3.1", "anthropic", "mock"
result = client.generate("system", "user")       # -> ModelResult
print(result.text, result.cost_usd, result.latency_sec, result.tool_calls)
```

- **Presets**: `anthropic`, `openai`, `glm`, `deepseek`, `ollama`, `vllm`,
  `sglang`, `llamacpp`, `grok`, `mock`. Any OpenAI-compatible server works via
  `SOPHIA_MODEL_BASE_URL`.
- **Config** (see `.env.example`): `SOPHIA_MODEL_PROVIDER`, `SOPHIA_MODEL`,
  `SOPHIA_MODEL_FALLBACKS`, `SOPHIA_REASONING_EFFORT`, `SOPHIA_MODEL_RETRIES`.
- Retry with backoff on transient errors, ordered fallback chain, streaming
  (`on_token`), native tool-calling pass-through, and per-call cost/latency.
- `agent.llm.complete` and the hidden runner's `call_model` remain as-is;
  new code should use `agent.model`.

### GLM-5.2 / local serving

| Target | Setup |
|---|---|
| GLM-5.2 (Zhipu) | `SOPHIA_MODEL_PROVIDER=glm SOPHIA_MODEL=glm-5.2 ZHIPUAI_API_KEY=…` |
| vLLM / SGLang | serve OpenAI-compatible, then `SOPHIA_MODEL_PROVIDER=vllm SOPHIA_MODEL_BASE_URL=http://host:8000/v1` |
| Ollama | `ollama serve`, then `SOPHIA_MODEL_PROVIDER=ollama SOPHIA_MODEL=llama3.1` |
| llama.cpp | `SOPHIA_MODEL_PROVIDER=llamacpp SOPHIA_MODEL_BASE_URL=http://host:8080/v1` |

## 2. Agent harness (`agent/harness.py`)

```bash
# offline smoke
python tools/agent_harness.py run "Should we launch on HN this week?" --provider mock
# real, with an auto-selected skill
python tools/agent_harness.py run "Fix the failing auth test" --auto-skill --provider glm
```

Loop: **plan** (model emits a JSON step plan) → **execute** each step (model or
approval-gated tool) → **critic** (epistemic gate + pluggable verifier) →
**reflect & retry** up to `--max-retries` → persist. Every event is appended to
`agent/memory/agent_runs/<task>.jsonl`; `--resume` skips completed steps.

Failure classes: `empty_output`, `model_error`, `tool_error`, `gate_violation`,
`verifier_fail`, `exception`, `max_retries_exhausted`.

## 3. Skills (`agent/skills.py`, `skills/registry/*.json`)

```bash
python tools/agent_harness.py skills          # list
```

Starter skills: coding-debugging, research-rag, terminal-automation,
repo-analysis, long-context-summarization, eval-generation, lora-dataset-creation.

### Add a skill

Create `skills/registry/<name>.json` with: `name`, `whenToUse`, `triggers`,
`requiredTools`, `workflow`, `ioSchema`, `verification`, `commonFailures`,
`examples`. It is validated on load (`agent.skills.validate_skill`) and becomes
auto-selectable (`agent.skills.select`). Add a routing test in
`tests/test_skills.py`.

## 4. Verifiers (`agent/verifiers.py`)

A verifier is `(text, task, step) -> {passed, reasons, detail}`. Built-ins:
`exact_match`, `regex_match`, `keyword`, `unit_test` (grades by a command's exit
code — the strongest signal for code), `score_pack_case` (rubric/operational),
`citation_present`, plus `all_of`/`any_of` combinators. The harness default is
`gate_verifier` (epistemic gate). This is the seam the RL/eval loops use —
quality follows verifiability.

## 5. RAG quality

- `agent/chunking.py` — token-aware recursive chunking with overlap + stable ids
  (replaces the old 4000-char truncation; wired into `agent/retrieval.py`).
- `agent/rerank.py` — `lexical_rerank` (BM25-lite), optional `llm_rerank` via the
  adapter, and `citation_faithfulness` (is each answer sentence grounded in a
  source?).
- `tools/eval_retrieval.py` — golden-query `recall@k` / `MRR`, so retrieval
  changes are measurable.

## 6. Safety

- `agent/untrusted.py` — fences retrieved/web/material text in untrusted-data
  delimiters (prompt-injection defense), wired into the agent prompt boundary;
  flags common injection phrasings.
- `sophia_mcp/audit.py` — append-only audit log for every MCP tool call + a
  risk/approval gate (mutating tools need `SOPHIA_MCP_APPROVE_WRITES=1`).
- `agent/tools.py` — repo-tool execution stays approval-gated.

## 7. Self-improvement flywheel

- `tools/collect_traces.py` — agent run traces → verified SFT + (chosen,rejected)
  DPO jsonl, with benchmark-leakage check.
- `tools/distill_export.py` — teacher model (GLM-5.2/DeepSeek/Claude via the
  adapter) → verifier-gated SFT + rejected set + trajectory.
- `tools/train_lora.py`, `tools/prepare_lora_dataset.py`, `tools/eval_local_model.py`
  — train and grade a local student. See `docs/11-Platform/Inference.md` for the
  distill→serve loop and inference optimization.

## 8. Evaluation (`tools/eval_agent.py`)

```bash
python tools/eval_agent.py --provider mock                 # offline plumbing smoke
python tools/eval_agent.py suite.json --provider glm:glm-5.2 --out eval/results/agent.json
```

Reports pass-rate, failure-class histogram, mean cost, and latency — so every
agent/model change is measurable. Pairs with `tools/run_benchmark.py`,
`tools/run_ablation_sophia.py`, and the hidden-eval harnesses.

## 9. Add an MCP tool

Implement in `sophia_mcp/tools_impl.py` (return a dict, structured errors),
register a thin `@mcp.tool()` wrapper in `sophia_mcp/server.py` (typed params +
docstring schema), and add a smoke test in `tests/test_mcp_tools.py`. See
`sophia_sector_council` for a template.

## 10. Improve the model (pipelines)

- Trace collection: `tools/collect_traces.py` (agent runs → SFT/DPO).
- Distillation: `tools/distill_export.py` (verifier-gated teacher data).
- LoRA/QLoRA: `tools/train_lora.py`, `tools/prepare_lora_dataset.py`.
- Before/after: `tools/eval_local_model.py`. Serving: `docs/11-Platform/Inference.md`.

See the [Roadmap](./Roadmap.md) for what is built vs planned across all 12 areas.
