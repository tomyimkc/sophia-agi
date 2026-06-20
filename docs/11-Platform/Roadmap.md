# AGI Platform Roadmap

Status of the 12-point AGI-platform mandate after the **Stage 1 vertical slice**
(unified model adapter → agent harness → skills → verifier → eval). Derived from
the repo audit in `agi-proof/` and the platform build.

Legend: ✅ built · 🟡 partial (existing pieces, gaps remain) · ⬜ planned.

| # | Area | Status | Where |
|---|------|--------|-------|
| 1 | Agent harness | ✅ | `agent/harness.py` (plan/act/critic/reflect/retry, persistence, checkpoint/resume, failure classes, decision logs + output logging) |
| 2 | Skills system | ✅ | `agent/skills.py` + `skills/registry/*.json` (7 starter skills, typed io, verification, failures, examples) |
| 3 | MCP tool layer | ✅/🟡 | `sophia_mcp/` (13 typed tools) + `sophia_mcp/audit.py` (audit log + risk/approval on mutating tools); gap: sandboxed exec + git/CI tools |
| 4 | RAG + long-context memory | ✅/🟡 | `agent/chunking.py` (token-aware chunking, wired into retrieval) + `agent/rerank.py` (BM25-lite + citation faithfulness) + `tools/eval_retrieval.py` (recall@k/MRR); gap: cross-encoder rerank, working-memory summaries |
| 5 | Model adapter (GLM-5.2/frontier) | ✅ | `agent/model.py` (anthropic + OpenAI-compatible GLM/vLLM/SGLang/Ollama/llama.cpp/DeepSeek + grok + mock; retry/fallback/streaming/tools/cost). Wired into the hidden/ablation runner via `--backend adapter` |
| 6 | LoRA/QLoRA | 🟡 | `tools/train_lora.py`, `prepare_lora_dataset.py` + `tools/collect_traces.py` (traces→SFT/DPO with leakage check); gap: assistant-loss masking, holdout early-stop, adapter versioning |
| 7 | Distillation | ✅/🟡 | `tools/distill_export.py` (teacher→verify→SFT + rejected set + trajectory) + `claude_teacher.py`; gap: best-of-N teacher sampling |
| 8 | Verifier-based RL | ✅/🟡 | `agent/verifiers.py` (unit-test/exact/regex/keyword/score_pack/citation + combinators) as the harness/eval seam + `collect_traces` DPO export; gap: GRPO/online loop |
| 9 | Evaluation harness | ✅ | `tools/eval_agent.py` (pass-rate/failure-hist/cost/latency, **run on DeepSeek**) + `tools/eval_retrieval.py` + benchmark/ablation harnesses; gap: executable coding eval |
| 10 | Inference optimization | ✅ | `docs/11-Platform/Inference.md` (vLLM/SGLang/Ollama/llama.cpp serving, quantization, KV-cache, speculative decoding, batching, long-context) via the adapter |
| 11 | Safety & reliability | ✅/🟡 | `agent/untrusted.py` (prompt-injection delimiters, wired into agent prompts) + MCP audit/approval + tool approval gate + epistemic gate + reproducible JSONL traces; gap: OS-level sandbox |
| 12 | Developer experience | ✅ | `docs/11-Platform/` (AGI-Platform + Inference + Roadmap), `.env.example`, per-module tests |

## Stages

**Stage 1 — Adapter seam + vertical slice (DONE)**
- `agent/model.py` unified adapter; `agent/harness.py` plan/act/critic/retry loop;
  `agent/skills.py` + 7 skills; `tools/agent_harness.py` + `tools/eval_agent.py`;
  tests for all; docs + env config.

**Stage 2 — Provider breadth + safety/audit substrate (DONE)**
- ✅ `sophia_mcp/audit.py`: audit JSONL + risk/approval on mutating tools.
- ✅ `agent/untrusted.py`: prompt-injection delimiters, wired into the agent prompt boundary.
- ✅ adapter wired into the hidden/ablation runner (`--backend adapter`) for GLM/DeepSeek/local.
- ⬜ remaining: OS-level sandbox + git/CI external MCP tools.

**Stage 3 — Measurable retrieval (DONE)**
- ✅ `agent/chunking.py`: token-aware recursive chunking with overlap + stable ids (wired into retrieval).
- ✅ `agent/rerank.py`: BM25-lite + optional LLM rerank + citation-faithfulness/groundedness.
- ✅ `tools/eval_retrieval.py`: golden-query recall@k / MRR.
- ⬜ remaining: cross-encoder reranker; working-memory summaries for long runs.

**Stage 4 — Self-improvement flywheel (DONE core)**
- ✅ `agent/verifiers.py`: unit-test/exact/regex/keyword/score_pack/citation verifiers + combinators.
- ✅ `tools/collect_traces.py`: agent traces → SFT + (chosen,rejected) DPO with leakage check.
- ✅ `tools/distill_export.py`: teacher → verifier-gated SFT + rejected set + trajectory.
- ⬜ remaining: assistant-loss masking + holdout early-stop in `train_lora.py`; executable coding-eval verifier; online GRPO loop.

## Tracked TODOs (specific, justified)

- [ ] `train_lora.py`: assistant-only loss masking + consume `holdout.jsonl` for early-stop (currently trains on full sequence).
- [ ] Executable coding-eval lane: wire `agent/verifiers.unit_test` into a sandboxed coding pack so code cases are graded by `pytest` exit code.
- [ ] OS-level sandbox (container/seccomp) for `agent/tools.py` + an external git/CI MCP tool through `sophia_mcp/audit.py`.
- [ ] Cross-encoder reranker option in `agent/rerank.py` (LLM rerank already available via the adapter).
