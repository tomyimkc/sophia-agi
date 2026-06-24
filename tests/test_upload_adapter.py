# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for LoRA adapter upload helper.

The adapter checkpoint is a TRAINING ARTIFACT (model weights), gitignored under
training/lora/checkpoints/. On a fresh clone / CI it does not exist, so these
tests skip gracefully rather than fail — they validate the upload tool only when
an adapter has actually been built locally.

No top-level pytest dependency: the repo runs tests as plain scripts when pytest
is unavailable (CI + tools/run_replication_check.py), so pytest is imported lazily
only to signal a skip when running under pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "training" / "lora" / "checkpoints" / "sophia-v1"
UPLOAD = ROOT / "tools" / "upload_huggingface_adapter.py"

_REASON = "LoRA adapter checkpoint not built (training artifact, gitignored)"


def _missing_adapter() -> bool:
    """True if the checkpoint is absent. Skips only under an active pytest run;
    prints a note and returns otherwise (so importing+calling these outside pytest,
    even where pytest happens to be installed, never raises Skipped)."""
    if (ADAPTER / "adapter_config.json").exists():
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest

        pytest.skip(_REASON)
    print(f"SKIP: {_REASON}")
    return True


def test_adapter_checkpoint_present() -> None:
    if _missing_adapter():
        return
    assert (ADAPTER / "adapter_model.safetensors").exists()
    meta = json.loads((ADAPTER / "sophia_lora_config.json").read_text(encoding="utf-8"))
    assert meta["baseModel"] == "Qwen/Qwen2.5-3B-Instruct"


def test_upload_dry_run() -> None:
    if _missing_adapter():
        return
    proc = subprocess.run(
        [sys.executable, str(UPLOAD), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "tomyimkc/sophia-agi-lora-v1" in proc.stdout
