"""Spec B — toy-decoder steering-hook tests (skip-guarded; torch-only)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip the whole module under pytest when torch is unavailable (the dependency-free
# CI `validate` job runs this file as a plain script, where main() guards instead).
if importlib.util.find_spec("torch") is None:
    try:
        import pytest

        pytestmark = pytest.mark.skip(reason="torch unavailable (Spec B hook tests)")
    except ModuleNotFoundError:  # script run without pytest installed
        pytestmark = []


def _build():
    import torch
    import torch.nn as nn

    class ToyDecoderLayer(nn.Module):
        def __init__(self, d_model, seed):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.proj = nn.Linear(d_model, d_model)
            with torch.no_grad():
                self.proj.weight.copy_(torch.empty(d_model, d_model).normal_(generator=g))
                self.proj.bias.copy_(torch.empty(d_model).normal_(generator=g))

        def forward(self, hidden, *args, **kwargs):
            return (hidden + torch.tanh(self.proj(hidden)),)

    class ToyDecoder(nn.Module):
        def __init__(self, d_model=16, n_layers=3, seed=0):
            super().__init__()
            self.d_model = d_model
            # name the attribute `layers` AND nest under `.model` so attach_hooks'
            # `model.model.layers[L]` path works against the toy too.
            inner = nn.Module()
            inner.layers = nn.ModuleList([ToyDecoderLayer(d_model, seed + i) for i in range(n_layers)])
            self.model = inner

        def forward(self, hidden):
            per_layer = []
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
                per_layer.append(hidden)
            return hidden, per_layer

    return torch, ToyDecoder


def test_hook_adds_alpha_v_surgically_and_removes() -> None:
    torch, ToyDecoder = _build()
    from agent.steering.hooks import attach_hooks

    model = ToyDecoder().eval()
    g = torch.Generator().manual_seed(123)
    x = torch.empty(1, 4, model.d_model).normal_(generator=g)
    with torch.no_grad():
        clean_out, clean_layers = model(x)

    L, alpha = 1, 2.5
    v = [1.0] * model.d_model
    with attach_hooks(model, v, alpha, [L]):
        with torch.no_grad():
            _, steered_layers = model(x)
    vt = torch.tensor(v)
    assert torch.allclose(steered_layers[L], clean_layers[L] + alpha * vt, atol=1e-5)
    for i in range(L):  # earlier layers unchanged
        assert torch.allclose(steered_layers[i], clean_layers[i], atol=1e-5)
    assert not torch.allclose(steered_layers[L + 1], clean_layers[L + 1], atol=1e-5)  # propagates
    # context manager removed the hook → clean forward restored
    with torch.no_grad():
        restored_out, _ = model(x)
    assert torch.allclose(restored_out, clean_out, atol=1e-5)


def test_capture_residual_on_toy() -> None:
    torch, ToyDecoder = _build()
    from agent.steering.hooks import capture_residual

    model = ToyDecoder().eval()
    g = torch.Generator().manual_seed(1)
    x1 = torch.empty(1, 4, model.d_model).normal_(generator=g)
    x2 = x1 + 1.0
    v1 = capture_residual(model, 1, lambda: model(x1))
    v2 = capture_residual(model, 1, lambda: model(x2))
    assert len(v1) == model.d_model and len(v2) == model.d_model
    assert v1 != v2                                           # input-sensitive
    assert capture_residual(model, 1, lambda: model(x1)) == v1  # deterministic


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception:
        print("SKIP test_personality_steering (torch unavailable)")
        return 0
    tests = [test_hook_adds_alpha_v_surgically_and_removes, test_capture_residual_on_toy]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} hook tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
