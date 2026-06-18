# Claude Model Lab — build your local Sophia LLM

Use **Claude API** as the factory; train **open weights** offline (Qwen/Llama + LoRA).

## Pipeline

```text
Claude teacher/distill/review/judge  →  corpus quality
prepare_lora_dataset (holdout)       →  train.jsonl
train_lora.py (QLoRA 4-bit)          →  adapter weights
eval_local_model.py + gate           →  benchmark proof
write-modelfile                      →  Ollama sophia-7b
```

## Commands

```bash
# Full lab (review + distill + modelfile; judge if local reports exist)
python tools/claude_model_lab.py run-all --dry-run
python tools/claude_model_lab.py run-all --review-limit 10 --distill-limit 15

# Individual steps
python tools/claude_model_lab.py review-batch --limit 20
python tools/claude_model_lab.py distill --limit 30 --promote
python tools/claude_model_lab.py judge --generate-corrections
python tools/claude_model_lab.py write-modelfile --adapter training/lora/checkpoints/sophia-v1

# Train + eval (GPU)
pip install -r requirements-lora.txt
python tools/train_lora.py --4bit --epochs 3
python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1 --with-gate

# Offline package
ollama create sophia-7b -f models/ollama/Modelfile
```

## Recommended parameters

| VRAM | Base model | Quant |
|------|------------|-------|
| 8 GB | Qwen2.5-3B-Instruct | 4-bit |
| 12–16 GB | Qwen2.5-7B-Instruct | 4-bit |
| 24 GB | Qwen2.5-14B-Instruct | 4-bit |

## Outputs

| Path | Content |
|------|---------|
| `training/lab/reviews/` | Claude QA on teacher batches |
| `training/lab/distill/` | Gold distilled examples |
| `training/lab/judgements/` | Claude judge on benchmark failures |
| `models/ollama/Modelfile` | Ollama create spec |
| `models/hf-model-card/README.md` | Adapter model card template |
| `models/manifest.json` | Build metadata |

## Google Colab (no local GPU)

Open [`notebooks/Sophia-LoRA-Colab.ipynb`](../../notebooks/Sophia-LoRA-Colab.ipynb) in Colab:

1. Runtime → **GPU**
2. Run all cells
3. Download `sophia-lora-v1.zip` → unzip into `training/lora/checkpoints/sophia-v1`

## Always use the gate

Weights alone ≠ trustworthy Sophia answers. Run `sophia_gate_check` (MCP) or `--with-gate` on eval.

See also: [LoRA-Experiment.md](LoRA-Experiment.md), [MCP-Server.md](MCP-Server.md).