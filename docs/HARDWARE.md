# Hardware — author's local cluster

Single source of truth: [`config/devices.local.json`](../config/devices.local.json).
Per-device serving topology: [`config/inference.local.spark.json`](../config/inference.local.spark.json)
(Spark) and [`config/inference.local.example.json`](../config/inference.local.example.json) (Apple Silicon).

## The two machines

| | **DGX Spark** | **Mac Studio M3 Ultra** |
|---|---|---|
| Chip | GB10 Grace Blackwell, aarch64 | Apple M3 Ultra, arm64 |
| Accelerator | CUDA (single integrated Blackwell GPU) | Metal / MLX |
| Unified memory | 128 GB LPDDR5x | 96 GB |
| Memory bandwidth | ~273 GB/s | **~819 GB/s** |
| Peak compute | ~1 PFLOP FP4 (sparse) | lower FP4, far higher bandwidth |
| In-repo backend | `hf` (transformers/PEFT, NVFP4 kernels) | `mlx` (mlx-lm) |

## Roles are assigned by the roofline, not by vibes

The two bottlenecks are different, and the assignment follows from that:

- **Training is compute-bound** at FP4 → the **Spark** owns the optimizer/training step
  (QLoRA, GRPO, merge/export). Use the in-repo NVFP4 GEMM kernels
  (`kernels/src/nvfp4_gemm.py`) for the frozen base; LoRA in bf16.
- **Token generation is bandwidth-bound** → the **Mac**, with ~3× the memory bandwidth,
  is *faster per token* and owns generation: the async **teacher farm** (distillation
  trace-gen), **evals**, and **judge** passes via MLX.

This is why the wisdom-internalization pipeline is split the way it is — see the role map
in `config/devices.local.json` (`pipeline_roles`).

## How the wisdom-internalization pipeline maps onto the cluster

```
  ┌────────────────────────── Mac Studio M3 Ultra (mlx) ──────────────────────────┐
  │  1. tools/gen_distill_traces.py  --backend mlx --model Qwen3-4B                │
  │       teacher = Sophia's OWN gated pipeline  →  verified, passport-stamped      │
  │       cited-abstention traces  →  training/council/distill_traces.jsonl         │
  └────────────────────────────────────────┬───────────────────────────────────────┘
                                            │  (rsync traces)
  ┌─────────────────────────────────────────▼──────────── DGX Spark (hf/cuda) ─────┐
  │  2. tools/train_lora.py  --backend peft --base Qwen3-4B --guard --scaffold      │
  │       --distill training/council/distill_traces.jsonl  (NVFP4 base + bf16 LoRA) │
  │       → checkpoints  models/sophia-4b-internalized/checkpoint-*                 │
  └────────────────────────────────────────┬───────────────────────────────────────┘
                                            │  (rsync adapter checkpoints)
  ┌─────────────────────────────────────────▼─── Mac Studio M3 Ultra (mlx) ────────┐
  │  3. tools/run_wisdom_ablation.py  --backend mlx --base Qwen3-4B                 │
  │       --student-adapter <ckpt>  --checkpoints ...   (sealed held-out)           │
  │       → intrinsic-wisdom drop + ECE/Brier + fabrication-vs-compute curve        │
  └────────────────────────────────────────────────────────────────────────────────┘
```

One-command driver: [`scripts/wisdom_internalization.sh`](../scripts/wisdom_internalization.sh).
Runbook: [`docs/WISDOM-INTERNALIZATION.md`](WISDOM-INTERNALIZATION.md).

## Setup notes

- **Spark**: CUDA/aarch64. `pip install -r requirements-lora.txt` (torch<2.9 for flash-attn
  wheels; bitsandbytes installs on non-Darwin). vLLM/SGLang serving per
  `config/inference.local.spark.json`.
- **Mac**: `pip install "mlx-lm>=0.20"`. No bitsandbytes (Darwin-excluded in requirements).
- **Networking**: traces and adapter checkpoints move by `rsync`/`scp` between the two;
  `agent/cluster/ssh_provider.py` holds the SSH plumbing for an async producer/consumer.
