# Sophia LoRA v2 pipeline — run from repo root on RTX 3080+
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

Write-Host "[1/5] Prepare dataset..."
& $Py tools/prepare_lora_dataset.py

Write-Host "[2/5] Train sophia-v2 (resume sophia-v1)..."
& $Py tools/train_lora.py --4bit --epochs 2 `
    --resume-adapter training/lora/checkpoints/sophia-v1 `
    --output training/lora/checkpoints/sophia-v2

Write-Host "[3/5] Eval benchmark..."
& $Py tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v2 --with-gate

Write-Host "[4/5] Update leaderboards..."
& $Py tools/update_leaderboards.py
& $Py tools/build_web_data.py

Write-Host "[5/5] Done. Target: 23/23 on held-out benchmark."