# Hugging Face upload

## One-time setup

```powershell
pip install huggingface_hub
# Create token: https://huggingface.co/settings/tokens (Write access)
$env:HF_TOKEN = "hf_..."
```

## Upload corpus

```powershell
cd C:\Users\tomyim\Documents\GitHub\sophia-agi
python tools/export_training_jsonl.py
python tools/upload_huggingface.py
```

Dataset URL: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus

## Upload LoRA adapter (model repo)

After training (`training/lora/checkpoints/sophia-v1`):

```powershell
python tools/upload_huggingface_adapter.py --dry-run
python tools/upload_huggingface_adapter.py --approve
```

Model URL: https://huggingface.co/tomyimkc/sophia-agi-lora-v1

Override repo: `$env:HF_ADAPTER_REPO_ID = "your-user/sophia-lora-v1"`

## Colab benchmark eval

Open: [Sophia-LoRA-Eval-Colab.ipynb](https://colab.research.google.com/github/tomyimkc/sophia-agi/blob/main/notebooks/Sophia-LoRA-Eval-Colab.ipynb)

Upload `sophia-lora-v1.zip`, run 23 held-out cases + gate, download `sophia-eval-reports.zip`.

## External model benchmark

**Option A (`.env` — set once):** copy `.env.example` → `.env`, add your Claude key:

```powershell
pip install anthropic
# .env:
#   ANTHROPIC_API_KEY=your-llmhub-key
#   ANTHROPIC_BASE_URL=https://api.llmhub.com.cn
#   ANTHROPIC_MODEL=claude-sonnet-4-6

python tools/run_external_models.py --all
python tools/update_leaderboards.py
```

Only configured keys run (Claude-only is fine). `CLAUDE_API_KEY` works as an alias.

Optional: `MONICA_API_KEY` (all models), `OPENAI_API_KEY`, `XAI_API_KEY`.