# LoRA experiment (M1 ablation)

Minimal workflow to test whether the Sophia corpus teaches provenance behavior better than prompting alone.

## Hypothesis

| Condition | Expected |
|-----------|----------|
| Base model + prompt | Moderate benchmark pass rate |
| LoRA fine-tune | Higher pass rate on held-out traps |
| LoRA + runtime gate | Highest trustworthy deployment |

Benchmark cases are **held out** of training (`tools/prepare_lora_dataset.py`).

## Steps

```bash
python tools/claude_model_lab.py run-all   # Claude review + distill + Modelfile
pip install -r requirements-lora.txt
python tools/prepare_lora_dataset.py
python tools/train_lora.py --4bit --epochs 3
python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1
python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1 --with-gate
python tools/update_leaderboards.py
```

Compare against Claude Sonnet rows in `benchmark/model_runs/`.

## Colab notebooks

| Notebook | Purpose |
|----------|---------|
| [Sophia-LoRA-Colab.ipynb](../../notebooks/Sophia-LoRA-Colab.ipynb) | Train QLoRA adapter |
| [Sophia-LoRA-Eval-Colab.ipynb](../../notebooks/Sophia-LoRA-Eval-Colab.ipynb) | Benchmark eval + gate |

## Hugging Face model

```bash
python tools/upload_huggingface_adapter.py --approve
```

Repo: `tomyimkc/sophia-agi-lora-v1`

## Hardware

- **QLoRA (`--4bit`)**: ~8 GB VRAM (Qwen2.5-3B)
- **CPU**: prepare + dry-run only; training/eval need GPU

## Publish results

Add a row to the thesis site leaderboards with `local-sophia-v1` scores. Document base vs LoRA vs LoRA+gate in `CHANGELOG.md`.