---
language:
- en
- zh
license: mit
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- sophia-agi
- provenance
- source-discipline
- lora
---

# Sophia-7B (Sophia AGI adapter)

**Wisdom before intelligence.** LoRA adapter for provenance-aware instruction on `Qwen/Qwen2.5-7B-Instruct`.

- **Project:** [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi)
- **Version:** 0.5.4
- **Training examples:** 500
- **Benchmark total:** 23

## Usage

Train locally:

```bash
pip install -r requirements-lora.txt
python tools/prepare_lora_dataset.py
python tools/train_lora.py --4bit --epochs 3
```

Evaluate:

```bash
python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1 --with-gate
```

Ollama:

```bash
ollama create sophia-7b -f models/ollama/Modelfile
```

## Always pair with runtime gate

`sophia_gate_check` (MCP) or `agent/gate.py` — weights alone do not guarantee trap safety.
