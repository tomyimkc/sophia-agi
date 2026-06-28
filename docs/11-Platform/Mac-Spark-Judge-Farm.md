# Mac + Spark judge farm — the ≥2-family certification gate, locally and for free

**Status:** config + wiring doc. Config: `config/inference.local.mac-judge.json`. Test:
`tests/test_mac_judge_config.py`. No capability claim.

> The no-overclaim gate (`RESULTS.md`) needs **≥2 independent judge families** (κ ≥ 0.40,
> ≥3 seeds, 95% CIs excluding zero). This wires two **local** judges — a Qwen judge on the
> **DGX Spark** (vLLM/CUDA) and a Llama judge on the **Mac Studio** (MLX/Metal) — so you clear
> the family bar with **no metered cloud**, using hardware you already own.

## How the gate actually counts families (get this right)

`provenance_bench.aggregate._distinct_families` keys judges like this:
- **vLLM / sglang / llamacpp** are treated as **aggregators** → family = the **model vendor**
  (`vllm:Qwen/...` → `qwen`, `vllm:meta-llama/...` → `meta-llama`).
- **mlx / ollama** are not aggregators → family = the **engine** (`mlx:anything` → `mlx`).

Consequences (both matter):
- ✅ Two **different-vendor** models even on **one** vLLM port already count as 2 families
  (`qwen` + `meta-llama`). So a second box is **not required** to clear the bar.
- ❌ Two **same-vendor** judges collapse to **1** family (`vllm:Qwen-7B` + `vllm:Qwen-14B` → just
  `qwen`). This is the real pitfall — the test guards it.

This config keys to **`qwen`** (Spark vLLM) + **`mlx`** (Mac) = 2 families.

## So why the second box at all? (honest)

Not for the family count — for **grader quality**:
1. **Independence.** A Metal/MLX runtime is a different numerical + serving stack from CUDA/vLLM.
   Two graders on *different hardware and engines* have **less-correlated errors** than two
   models on the same vLLM — which is the whole point of a multi-family gate.
2. **Offload.** The Spark is busy *training* + *NVFP4-serving the subject under test*. Running a
   judge there too contends for its 128 GB / compute; the Mac runs the second judge in parallel.
3. **Free.** The Mac Studio is otherwise idle.

## Launch (fill in `SPARK_HOST` / `MAC_HOST`)

```bash
# 0. See the ready plan (print-only, CI-safe):
python tools/run_local_judge_eval.py --config config/inference.local.mac-judge.json

# 1. On the Spark (CUDA, vLLM):
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 2. On the Mac Studio (Apple Silicon, MLX):
mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8080

# 3. Run a judged eval with the two-box farm (>=2 families, no cloud):
python tools/judge_pilot_answers.py --judges \
  vllm:Qwen/Qwen2.5-7B-Instruct@http://SPARK_HOST:8000/v1,mlx:mlx-community/Meta-Llama-3.1-8B-Instruct-4bit@http://MAC_HOST:8080/v1 \
  ...
```

The `provider:model@http://host/v1` suffix sets each judge's base URL, so the two judges hit
their own boxes. The Mac is also your **control plane** — SSH / VS Code Remote-SSH into the Spark
over **Cat 6** (don't try to *cluster* the two: different architectures, no shared collective
backend — see the chat notes).

## judge ≠ subject

The subject under test is OLMoE / Sophia-V1 (lineage `allenai`/`olmoe`) — distinct from both judge
lineages (`qwen`, `meta-llama`). The config lists `subject_lineages_to_avoid: [qwen, meta-llama]`
so you never accidentally judge a Qwen/Llama subject with these judges. The test enforces it.
