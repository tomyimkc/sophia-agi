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

## External model benchmark

```powershell
pip install openai anthropic
$env:OPENAI_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
$env:XAI_API_KEY = "..."
python tools/run_external_models.py --all
python tools/update_leaderboards.py
```