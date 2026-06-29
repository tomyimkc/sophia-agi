# Thinking logs — capturing how the agent reasons and decides

This repo logs three layers of "how the AI thinks", each with different evidentiary
weight. The point of separating them is honesty: a log of what the model *said it was
thinking* is not the same as a record of what the system *actually did*, and neither is
proof of the model's true decision cause.

| Layer | What it captures | Where | Trust |
|---|---|---|---|
| **Decision trace** | plan → tool calls → gate verdicts → final answer | `agent/memory/agent_runs/<task>.jsonl` (`RunStore`) | Ground truth of what the system did. |
| **Thinking trace** | the model's own reasoning tokens (Claude adaptive/extended thinking, DeepSeek/GLM `reasoning_content`, `<think>` tags) | `agent/memory/thinking/<traceId>.jsonl` (`agent/thinking_trace.py`) | The model's *stated* reasoning — evidence, not cause. |
| **A2A messages** | inter-agent prompts/answers in swarm + networked A2A mode | same thinking-trace files, `kind: "a2a_message"` | The actual messages agents exchanged. |

## Why "stated reasoning ≠ how it decided"

The 2025–26 literature (FaithCoT-Bench `arXiv:2510.04040`; Anthropic's CoT-monitorability
work) shows chain-of-thought is frequently **unfaithful** — a post-hoc rationalization that
happens to be self-consistent, not the real driver of the answer. So the thinking trace is
**evidence to be probed**, never ground truth. `agent/faithfulness_probe.py` is the causal
test: perturb a reasoning step and measure whether the answer flips. A high flip-rate is
positive evidence the reasoning was load-bearing; a low one suggests it was decoration.

## How it works

Every LLM call in the stack flows through one choke point —
`agent.model.ModelClient.generate()`. A `trace_sink` hook there captures **every** call
(planner, step, reflect, swarm synthesis), not just the ones the harness happens to log.
`ModelResult` now carries `reasoning_text` / `reasoning_tokens`, populated per provider:

- **Claude** — adaptive thinking (`{"type": "adaptive", "display": "summarized"}`) on 4.6+,
  legacy budgeted thinking on older models; `thinking` blocks are collected, `redacted_thinking`
  is counted but never decoded.
- **OpenAI-compatible** — `reasoning_content` (DeepSeek/GLM) and inline `<think>…</think>`.

The trace schema is aligned to the OpenTelemetry **GenAI semantic conventions**
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons`),
so it can be exported to any OTel backend later without re-instrumenting.

## Enabling it (opt-in, privacy-aware)

| Env var | Effect |
|---|---|
| `SOPHIA_THINKING_LOG` | `1`/`on` → default dir `agent/memory/thinking/`; or a path. Enables the trace **and** A2A message logging. Off by default. |
| `SOPHIA_CAPTURE_THINKING` | Requests reasoning tokens from the provider **and** stores verbatim text in the trace. When unset, the trace keeps only sizes + SHA-256 hashes (`systemHash`, `reasoningHash`, …) — privacy-light by default. |

```bash
SOPHIA_THINKING_LOG=1 SOPHIA_CAPTURE_THINKING=1 \
  python tools/council_deliberate.py "Review our gacha odds + refund policy" --model anthropic
```

## A2A / swarm: messages → skills → training data

In swarm mode (`agent/subagent.py delegate()`) and networked A2A (`agent/a2a.py`), every
inter-agent leg is logged as an `a2a_message` span: parent→child `delegate`, child→parent
`result`, the `synthesis` reduce, and networked `peer` exchanges (with the gate verdict).

`agent/a2a_distill.py` turns those exchanges into two reusable artifacts — mirroring the
discipline of `agent/trace_distill.py`:

- **SFT training rows** (`a2a_training_rows`) — prompt/completion pairs from *successful,
  gated-accepted* exchanges, so future models internalise the A2A chain and need fewer hops.
- **Skill candidates** (`skill_candidates`) — recurring (intent, receiver-role) routing
  shapes proposed as skills. **Candidates only** — peer output is untrusted; a human/gate
  vets them before promotion. Nothing here auto-registers a skill or trains a model.

Distillation is fail-closed (abstains/blocks/empties never become "good" examples),
offline, provenance-preserving (every row records its trace id, sender, receiver, leg), and
reports hash-only spans it had to skip rather than silently dropping them.

## Tests

`tests/test_thinking_trace.py` — offline, mock-client coverage of reasoning capture, the
hash-only-vs-verbatim trace, the choke-point sink, A2A logging, and fail-closed distillation.
