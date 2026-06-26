# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Guard for the anchor-KL anti-forgetting term in tools/train_lora.py.

train_lora's torch path is hardware code (no GPU in CI), so this test stays dependency-
free: it asserts the wiring (run_manual_train accepts the anchor kwargs; masked_kl is
exposed) and, only when torch happens to be importable, checks the KL numerics. Importing
tools.train_lora is safe with no torch — its torch/peft imports are all lazy inside funcs.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_lora


def test_run_manual_train_accepts_anchor_kwargs() -> None:
    params = inspect.signature(train_lora.run_manual_train).parameters
    assert "anchor_kl" in params and "anchor_records" in params
    assert params["anchor_kl"].default == 0.0          # off by default — opt-in
    assert callable(train_lora.masked_kl)
    assert train_lora.DEFAULT_ANCHOR.name == "moral_gate_sft.jsonl"


def test_masked_kl_numerics_when_torch_present() -> None:
    try:
        import torch
    except ImportError:
        print("torch absent — skipping numeric KL check (hardware path)")
        return
    B, T, V = 2, 4, 7
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100)
    labels[:, 2:] = torch.randint(0, V, (B, 2))         # only last 2 positions count

    # identical distributions -> KL 0
    assert abs(train_lora.masked_kl(logits, logits.clone(), labels).item()) < 1e-5
    # different distributions -> KL strictly positive
    other = torch.randn(B, T, V)
    assert train_lora.masked_kl(logits, other, labels).item() > 0.0
    # all-masked batch -> exactly 0 (and no NaN)
    allmask = torch.full((B, T), -100)
    assert train_lora.masked_kl(logits, other, allmask).item() == 0.0


def main() -> int:
    import inspect as _inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and _inspect.isfunction(fn):
            fn()
    print("test_anchor_kl: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
