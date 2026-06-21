"""Tests for LoRA adapter upload helper.

The adapter checkpoint is a TRAINING ARTIFACT (model weights), gitignored under
training/lora/checkpoints/. On a fresh clone / CI it does not exist, so these
tests skip gracefully rather than fail — they validate the upload tool only when
an adapter has actually been built locally.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "training" / "lora" / "checkpoints" / "sophia-v1"
UPLOAD = ROOT / "tools" / "upload_huggingface_adapter.py"

_REASON = "LoRA adapter checkpoint not built (training artifact, gitignored)"


def test_adapter_checkpoint_present() -> None:
    if not (ADAPTER / "adapter_config.json").exists():
        pytest.skip(_REASON)
    assert (ADAPTER / "adapter_model.safetensors").exists()
    meta = json.loads((ADAPTER / "sophia_lora_config.json").read_text(encoding="utf-8"))
    assert meta["baseModel"] == "Qwen/Qwen2.5-3B-Instruct"


def test_upload_dry_run() -> None:
    if not (ADAPTER / "adapter_config.json").exists():
        pytest.skip(_REASON)
    proc = subprocess.run(
        [sys.executable, str(UPLOAD), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "tomyimkc/sophia-agi-lora-v1" in proc.stdout
