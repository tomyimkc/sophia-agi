---
language:
- en
- zh
license: mit
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- sophia-agi
- provenance
- source-discipline
- lora
---

# Sophia-3B (Sophia AGI LoRA adapter)

**Wisdom before intelligence.** LoRA adapter for provenance-aware instruction on `Qwen/Qwen2.5-3B-Instruct`.

- **Project:** [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi)
- **Dataset:** [tomyimkc/sophia-agi-corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus)
- **Version:** 0.6.0
- **Train split:** 436 examples (benchmark cases held out)
- **Benchmark score:** 20/23 (87%) on sophia-v1 harness (philosophy 9/9, history 5/5, psychology 3/4, religion 3/5)

## Load adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-3B-Instruct"
adapter = "tomyimkc/sophia-agi-lora-v1"

tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)
```

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
