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

Leaderboards updated with `sophia-v1` rows; thesis site manifest includes LoRA score (v0.6.0). See also [Online-RAG.md](Online-RAG.md) for cloud retrieval path.

## v1 eval (2026-06-18)

| Domain | sophia-v1 | Notes |
|--------|-----------|-------|
| History | 5/5 | Perfect |
| Philosophy | 9/9 | After `DENY_PATTERNS` fix, matches model substance |
| Psychology | 3/4 | `stockholm` needs `pop_myth` label |
| Religion | 3/5 | Council-panel format missing on 2 cases |
| **Total** | **20/23 (87%)** | Claude Sonnet: 23/23 |

Re-score without re-run: `python tools/rescore_model_runs.py`

## v2 training (sophia-v2)

### Google Colab (recommended)

Windows local training can fail on HF `Trainer` import; Colab + manual SFT loop is stable.

1. Open [Sophia-LoRA-Colab.ipynb](../../notebooks/Sophia-LoRA-Colab.ipynb) → **Runtime → T4 GPU**
2. Clone repo (gets v0.6.1+ with `516–518` and `--resume-adapter`)
3. Run **sophia-v2** cells: pull `tomyimkc/sophia-agi-lora-v1` from HF → train → download `sophia-lora-v2.zip`
4. Eval: [Sophia-LoRA-Eval-Colab.ipynb](../../notebooks/Sophia-LoRA-Eval-Colab.ipynb) (upload v2 zip, set adapter path to `sophia-v2`)

**Important:** v2 must use **`Qwen/Qwen2.5-3B-Instruct`** (same base as v1). Do not switch to 7B when resuming.

~30–45 min on Colab T4 for 2 epochs / 439 rows.

### Local (RTX 3080+)

```bash
python tools/prepare_lora_dataset.py
python tools/train_lora.py --4bit --epochs 2 \
  --resume-adapter training/lora/checkpoints/sophia-v1 \
  --output training/lora/checkpoints/sophia-v2
python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v2 --with-gate
```

Or: `.\tools\run_v2_pipeline.ps1`

Trainable paraphrases (not holdouts): `516–518`. Bench-aligned gold references `511–515` stay held out.

## v2 training seed (bench-aligned)

Examples `511–515` in `training/examples/` — held out of LoRA train via `benchmarkCase` metadata:

- `511` trap_confucius_ddj — explicit "did not write"
- `512` symposium_not_autograph — Socrates wrote nothing
- `513` stockholm — clinical + pop_myth
- `514` ancestor veneration — council panel
- `515` buddha nirvana — council panel + myth label