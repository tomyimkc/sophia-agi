"""Tests for LoRA adapter upload helper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "training" / "lora" / "checkpoints" / "sophia-v1"
UPLOAD = ROOT / "tools" / "upload_huggingface_adapter.py"


def test_adapter_checkpoint_present() -> None:
    assert (ADAPTER / "adapter_config.json").exists()
    assert (ADAPTER / "adapter_model.safetensors").exists()
    meta = json.loads((ADAPTER / "sophia_lora_config.json").read_text(encoding="utf-8"))
    assert meta["baseModel"] == "Qwen/Qwen2.5-3B-Instruct"


def test_upload_dry_run() -> None:
    proc = subprocess.run(
        [sys.executable, str(UPLOAD), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "tomyimkc/sophia-agi-lora-v1" in proc.stdout