# AGI Platform Roadmap

Status of the 12-point AGI-platform mandate after the **Stage 1 vertical slice**
(unified model adapter → agent harness → skills → verifier → eval). Derived from
the repo audit in `agi-proof/` and the platform build.

Legend: ✅ built · 🟡 partial (existing pieces, gaps remain) · ⬜ planned.

| # | Area | Status | Where |
|---|------|--------|-------|
| 1 | Agent harness | ✅ | `agent/harness.py` (plan/act/critic/reflect/retry, persistence, checkpoint/resume, failure classes, decision logs) |
| 2 | Skills system | ✅ | `agent/skills.py` + `skills/registry/*.json` (7 starter skills, typed io, verification, failures, examples) |
| 3 | MCP tool layer | 🟡 | `sophia_mcp/` (13 typed tools, structured errors); gaps: audited/permissioned wrapper, sandboxed exec + git/CI tools |
| 4 | RAG + long-context memory | 🟡 | `agent/retrieval.py`, `vector_store.py`, `web_evidence.py`; gaps: token-aware chunking, reranking, retrieval-eval |
| 5 | Model adapter (GLM-5.2/frontier) | ✅ | `agent/model.py` (anthropic + OpenAI-compatible GLM/vLLM/SGLang/Ollama/llama.cpp/DeepSeek + grok + mock; retry/fallback/streaming/tools/cost) |
| 6 | LoRA/QLoRA | 🟡 | `tools/train_lora.py`, `prepare_lora_dataset.py`; gaps: assistant-loss masking, holdout early-stop, adapter versioning/rollback |
| 7 | Distillation | 🟡 | `tools/claude_teacher.py`, `claude_model_lab.py`; gap: verifier-gated solution filtering + rejected-set capture |
| 8 | Verifier-based RL | 🟡 | verifier seam in `agent/harness.py` + `hidden_eval_protocol.score_pack`; gap: best-of-N sampling → chosen/rejected DPO export |
| 9 | Evaluation harness | ✅/🟡 | `tools/eval_agent.py` (pass-rate/failure-hist/cost/latency) + existing benchmark/ablation harnesses; gap: executable coding eval, RAG-faithfulness |
| 10 | Inference optimization | 🟡 | adapter targets vLLM/SGLang/Ollama/llama.cpp; gap: quantization/spec-decoding/draft-model docs + batching |
| 11 | Safety & reliability | 🟡 | tool approval gate, secrets hygiene (`config.is_real_secret`), epistemic gate, reproducible JSONL traces; gap: sandbox, prompt-injection delimiters, MCP audit log |
| 12 | Developer experience | ✅ | `docs/11-Platform/AGI-Platform.md` (architecture + how-tos), this roadmap, `.env.example` |

## Stages

**Stage 1 — Adapter seam + vertical slice (DONE)**
- `agent/model.py` unified adapter; `agent/harness.py` plan/act/critic/retry loop;
  `agent/skills.py` + 7 skills; `tools/agent_harness.py` + `tools/eval_agent.py`;
  tests for all; docs + env config.

**Stage 2 — Provider breadth + safety/audit substrate**
- Permissioned + audited `@mcp.tool` wrapper (append-only audit JSONL, risk/approval).
- Prompt-injection boundary: wrap retrieved/web/material text in untrusted-data delimiters.
- First external MCP tool through the substrate: sandboxed read-only repo + code-search + test-runner.

**Stage 3 — Measurable retrieval + unified eval/regression gate**
- Token-aware recursive chunking with overlap (replace truncate-to-N-chars).
- Reranking (BM25 hybrid + cross-encoder / LLM reranker via the adapter).
- Golden-query retrieval-eval (recall@k / nDCG / MRR) excluding benchmark holdouts.
- RAG-faithfulness / groundedness metric in `eval_rag_benchmark`.

**Stage 4 — Executable coding eval + verifier-driven self-improvement**
- Executable coding/repo eval lane (sandboxed pytest/build verifier).
- `train_lora.py`: assistant-only loss masking + holdout early-stop + per-epoch checkpoints.
- `eval_local_model.py --baseline-adapter` → single base-vs-adapter delta artifact.
- Verified distillation: `score_case` + gate filter before promotion; emit accepted-SFT + (chosen,rejected) DPO jsonl.

## Tracked TODOs (specific, justified)

- [ ] Stage 2: MCP audit/permission wrapper (`sophia_mcp/`) — needed before any write/exec external tool.
- [ ] Stage 2: untrusted-data delimiters around retrieved/web content in `agent/web_evidence.py` + harness prompts (prompt-injection defense).
- [ ] Stage 3: replace `_iter_markdown` 4000-char truncation in `agent/retrieval.py` with token-aware chunking + stable chunk ids.
- [ ] Stage 4: assistant-only loss masking in `tools/train_lora.py` (currently trains on full sequence).
