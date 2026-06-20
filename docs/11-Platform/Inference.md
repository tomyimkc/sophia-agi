# Inference & Serving

How to serve a local/student model behind the unified adapter (`agent/model.py`)
and optimize latency, throughput, and cost. The adapter speaks the
OpenAI-compatible `/chat/completions` API, so every server below is a drop-in:
set `SOPHIA_MODEL_PROVIDER` + `SOPHIA_MODEL_BASE_URL`.

## Serving backends

| Backend | Start | Adapter config |
|---|---|---|
| **vLLM** | `vllm serve <model> --port 8000` | `SOPHIA_MODEL_PROVIDER=vllm SOPHIA_MODEL=<model>` |
| **SGLang** | `python -m sglang.launch_server --model <model> --port 30000` | `SOPHIA_MODEL_PROVIDER=sglang SOPHIA_MODEL=<model>` |
| **Ollama** | `ollama serve` | `SOPHIA_MODEL_PROVIDER=ollama SOPHIA_MODEL=llama3.1` |
| **llama.cpp** | `llama-server -m model.gguf --port 8080` | `SOPHIA_MODEL_PROVIDER=llamacpp SOPHIA_MODEL_BASE_URL=http://localhost:8080/v1` |
| **GLM-5.2 (hosted)** | — | `SOPHIA_MODEL_PROVIDER=glm SOPHIA_MODEL=glm-5.2 ZHIPUAI_API_KEY=…` |

Verify any of them: `python tools/agent_harness.py models --provider <preset>` then
`python tools/agent_harness.py run "hello" --provider <preset>`.

## Optimization checklist

- **Quantization** — serve AWQ/GPTQ/INT4 (vLLM `--quantization awq`) or GGUF
  Q4_K_M (llama.cpp) to fit a 7-14B student on one GPU / a laptop. Re-run
  `tools/eval_agent.py` and `tools/run_ablation_sophia.py` to confirm the quant
  did not regress quality (the eval harness makes this measurable).
- **KV-cache** — vLLM/SGLang use PagedAttention/RadixAttention automatically;
  keep prompts stable-prefixed (system prompt first, untrusted data last) so the
  cache is reused across the agent loop's repeated calls.
- **Speculative decoding / draft model** — pair the student with a tiny draft
  model: vLLM `--speculative-model <small> --num-speculative-tokens 5`. Best for
  the harness's many short tool/critic calls.
- **Batching / concurrency** — vLLM/SGLang continuous-batch; run eval suites and
  ablation sweeps concurrently to amortize. The adapter is stateless per call, so
  parallelism is safe.
- **Long-context serving** — enable the server's long-context mode and rely on the
  RAG layer (chunking + rerank in `agent/chunking.py`/`agent/rerank.py`) to keep
  packed context small and cited rather than stuffing the window.
- **Cost/latency tracking** — every call returns `cost_usd`/`latency_sec`
  (`ModelResult`); `tools/eval_agent.py` aggregates them, so you can compare
  serving configs on price-per-passed-task.

## Distill → serve loop

1. Generate verified teacher data: `tools/distill_export.py --provider glm:glm-5.2`.
2. Collect agent traces: `tools/collect_traces.py` → SFT + DPO jsonl.
3. Train a LoRA student: `tools/train_lora.py` (+ holdout eval).
4. Serve the merged student via vLLM/Ollama and point the adapter at it.
5. Grade: `tools/eval_agent.py --provider vllm` and `tools/run_ablation_sophia.py`
   to confirm the local student approaches the teacher on your tasks.
