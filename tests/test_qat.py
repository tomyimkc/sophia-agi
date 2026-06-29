# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the QAT-on-known-floor study."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.nano.model import NanoLM  # noqa: E402
from pretraining.qat import study  # noqa: E402


def test_qat_offline_invariants() -> None:
    ok, detail = study.offline_invariants()
    assert ok, detail["checks"]


def test_ternary_quantize_in_grid() -> None:
    s = 2.0
    for v in (-3.0, -0.3, 0.0, 0.4, 0.6, 3.0):
        q = study.ternary_quantize_value(v, s)
        assert abs(q) in (0.0, s)


def test_regularizer_zero_on_grid() -> None:
    # Use per-layer scales (the honest form — BitNet uses one scale per weight matrix),
    # passed as a dict so the regularizer evaluates at the exact grid weights were snapped to.
    m = NanoLM(4, 1, 4, seed=0)
    scales = {}
    for key, W in (("W1", m.W1), ("W2", m.W2)):
        flat = [abs(x) for row in W for x in row]
        ss = sum(flat) / max(1, len(flat))
        scales[key] = ss
        for r in range(len(W)):
            for c in range(len(W[r])):
                W[r][c] = study.ternary_quantize_value(W[r][c], ss)
    assert study.ternary_regularizer(m, target_scale=scales) < 1e-12


def test_quantize_is_copy_not_inplace() -> None:
    import copy
    m = NanoLM(4, 1, 6, seed=2)
    before = copy.deepcopy(m.W1)
    _ = study.ternary_quantize_model(m)
    assert m.W1 == before  # original untouched


def test_run_study_reports_floor_and_gap() -> None:
    rep = study.run_study(vocab=8, context=2, hidden=20, n_train=150,
                          n_eval=60, epochs=4, lr=0.05, lam=0.3, seed=0)
    assert rep["E"] > 0
    assert "gap_qat" in rep and "gap_control" in rep
    assert "Nano-scale methodology result only" in rep["honest_scope"]


# ---------------------------------------------------------------------------
# Regression: QAT must not bypass the PEFT LoRA adapter (the v2 zero-B bug).
#
# PEFT wraps a target nn.Linear in a ``lora.Linear`` that is ALSO class-named
# "Linear" and exposes ``.weight`` (a property onto ``base_layer.weight``).
# attach_qat used to wrap THAT module, replacing its forward with a base-only
# ``F.linear(x, fake_quant(weight))`` and silently dropping the lora_A/lora_B
# path — so the adapter got no gradient and the released checkpoint was an
# untrained no-op (every OLMoE NVFP4 certify came back bit-identical to base).
# The string-only ``offline_invariants`` could not see this: it is a *runtime*
# wrapping bug, not a name-classification bug. This test exercises the real
# peft wrapper + a few optimizer steps and asserts the adapter actually moves.
# CPU-only and tiny (no GPU / no triton), but torch+peft are optional → skip.
# ---------------------------------------------------------------------------

def test_attach_qat_does_not_bypass_lora_adapter() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    from training.qat import attach_qat

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # name it like an expert projection so is_qat_target_name matches
            self.down_proj = nn.Linear(16, 16, bias=False)

        def forward(self, x):  # noqa: ANN001
            return self.down_proj(x)

    torch.manual_seed(0)
    model = get_peft_model(
        Tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=["down_proj"], lora_dropout=0.0))

    n_wrapped = attach_qat(model, scheme="nvfp4")

    # The LoRA WRAPPER (exposes lora_A) must be skipped; only its inner base_layer is wrapped.
    wrapped = {name for name, m in model.named_modules() if getattr(m, "_qat_wrapped", False)}
    assert any(name.endswith(".base_layer") for name in wrapped), wrapped
    assert not any(hasattr(m, "lora_A") and getattr(m, "_qat_wrapped", False)
                   for _, m in model.named_modules()), "QAT wrapped the LoRA wrapper (bypass bug)"
    # No double-count: one fake-quant per real base weight (the doubled count was the v2 tell).
    assert n_wrapped == sum(1 for n in wrapped if n.endswith(".base_layer"))

    # lora_B starts at zero init; after a few steps under fake-quant it must MOVE (got gradient).
    def lora_b_norm() -> float:
        return next(p.detach().norm().item()
                    for name, p in model.named_parameters() if "lora_B" in name)

    assert lora_b_norm() == 0.0
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(20):
        x = torch.randn(8, 16)
        y = torch.randn(8, 16)
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()          # would raise "does not require grad" under the bypass bug
        opt.step()
    assert lora_b_norm() > 0.0, "lora_B never moved — QAT is still bypassing the adapter"
