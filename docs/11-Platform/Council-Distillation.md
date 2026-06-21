# Council distillation — internalising the discipline into a small model

Implements [Council-Distillation-Spec.md](./Council-Distillation-Spec.md). Teaches a
small **student** (Qwen2.5-7B) to emit the council's discipline (decompose → cite →
gate-clean → abstain) in a single pass, so it behaves like `council+gate` **without**
the scaffold at inference.

Pipeline: **generate → gate-filter → train → eval**.

## 1. Generate gate-filtered traces (teacher ≠ student family)

```bash
# real traces (teacher = DeepSeek via OpenRouter; OPENROUTER_API_KEY in .env). Rotate the key after.
python tools/distill_council_traces.py --teacher openrouter:deepseek/deepseek-chat \
    --out training/council/traces.jsonl
# offline plumbing
python tools/distill_council_traces.py --teacher mock --limit 4 --out training/council/traces.jsonl
```
Every output is gate-checked; a fabricated citation / false arithmetic / forbidden
attribution is **dropped, never distilled** (the anti-circularity firewall).
Abstention traces are capped so the student isn't taught to always abstain. Output
is `{messages, metadata}` JSONL, ready for `train_lora.py`.

## 2. Train the student (GPU step)

LoRA fine-tune Qwen2.5-7B on the traces. **Requires a CUDA GPU** (≈16GB for 4-bit);
this repo's sandbox has none, so run on a GPU box / Colab.

```bash
# validate the data pipeline anywhere (no GPU, no torch needed):
python tools/train_lora.py --model Qwen/Qwen2.5-7B-Instruct \
    --train training/council/traces.jsonl --dry-run

# real run on a GPU (or notebooks/Sophia-LoRA-Colab.ipynb):
pip install -r requirements-lora.txt
python tools/train_lora.py --model Qwen/Qwen2.5-7B-Instruct \
    --train training/council/traces.jsonl --4bit --epochs 3 \
    --output training/lora/checkpoints/sophia-council-7b
ollama create sophia-council-7b -f models/ollama/Modelfile   # to serve it locally
```

## 3. Evaluate the uplift (held-out, honest)

Use the existing uplift harness — the **distilled-alone** condition is just the
student served as `--model`:

```bash
# base student vs distilled student, single pass each:
python tools/run_council_uplift.py --model ollama:qwen2.5:7b          # base-alone + base+council(+gate)
python tools/run_council_uplift.py --model ollama:sophia-council-7b   # distilled-alone
```
Report base-alone vs base+council+gate vs distilled-alone `cleanRate` + abstention
calibration on **held-out** tasks. Pre-registered criteria and guardrails are in the
spec. Single-run numbers are **illustrative**; a validated number needs ≥2
eval families + runs + CIs (repo no-overclaim policy).

## Honest status

- **v0 (offline plumbing):** done — generator + gate-filter + tests + CI.
- **v1 dataset:** generated here via OpenRouter (gate-filtered).
- **v1 training + distilled-uplift:** the **GPU step** — validated by `--dry-run`
  here; the actual fine-tune and distilled eval run on a GPU box. No GPU = no
  trained adapter committed (weights are gitignored anyway).
