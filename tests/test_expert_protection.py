"""Unit tests for tools/expert_protection.py (the opt-in --keep-top-experts cert lever).

Exercises the per-expert-slice skip logic in isolation (the risky part) with an injected
`is_served` so the test does not depend on certify_lowram's suffix matcher or a real model.
The end-to-end reproduction (cert --keep-top-experts 8 -> raw 0.9414 / coverage 0.9297) is
verified on the Spark GPU, not here.
"""
import pytest

torch = pytest.importorskip("torch")  # torch-only test; skip cleanly on a torch-less CI runner
import torch.nn as nn

from tools.expert_protection import protected_quantize_served, layer_of, layers_to_hold


def test_layer_of():
    assert layer_of("model.layers.7.mlp.experts.down_proj") == 7
    assert layer_of("lm_head.weight") == -1


class _FakeModel:
    """Duck-typed model exposing only named_parameters() (all protected_quantize_served needs)."""

    def __init__(self, params):
        self._params = params  # list[(name, nn.Parameter)]

    def named_parameters(self):
        return iter(self._params)


def test_protected_keeps_top_experts_bf16_quantizes_rest():
    torch.manual_seed(0)
    experts = nn.Parameter(torch.randn(4, 6, 6))     # fused 3-D expert tensor at layer 0
    embed = nn.Parameter(torch.randn(10, 6))         # NOT served -> kept, untouched
    m = _FakeModel([("layers.0.mlp.experts.down_proj", experts),
                    ("model.embed_tokens.weight", embed)])
    served = lambda name, suffixes=(): name.endswith("down_proj")
    before = experts.detach().clone()
    embed_before = embed.detach().clone()

    info = protected_quantize_served(m, scheme="nvfp4", suffixes=("down_proj",),
                                     keep_experts={0: {1, 2}}, is_served=served)

    # kept experts 1,2 are byte-identical (bf16); 0,3 were quantized (changed)
    assert torch.equal(experts[1], before[1]) and torch.equal(experts[2], before[2])
    assert not torch.equal(experts[0], before[0])
    assert not torch.equal(experts[3], before[3])
    # non-served param untouched + counted as kept
    assert torch.equal(embed, embed_before)
    assert info["protected_experts"] == 2
    assert info["quantized_modules"] == 1
    assert info["kept_params"] >= embed.numel() + 2 * experts[0].numel()


def test_no_keep_quantizes_all_slices():
    torch.manual_seed(1)
    experts = nn.Parameter(torch.randn(3, 5, 5))
    m = _FakeModel([("layers.2.mlp.experts.gate_up_proj", experts)])
    served = lambda name, suffixes=(): name.endswith("gate_up_proj")
    before = experts.detach().clone()

    info = protected_quantize_served(m, scheme="nvfp4", suffixes=("gate_up_proj",),
                                     keep_experts={}, is_served=served)

    assert info["protected_experts"] == 0
    assert not torch.equal(experts, before)  # every slice quantized


def test_layers_to_hold_returns_last_n():
    m = _FakeModel([(f"model.layers.{L}.mlp.down_proj", nn.Parameter(torch.zeros(2, 2)))
                    for L in range(4)] + [("model.embed_tokens.weight", nn.Parameter(torch.zeros(2, 2)))])
    assert layers_to_hold(m, 2) == {2, 3}
    assert layers_to_hold(m, 1) == {3}
    assert layers_to_hold(m, 0) == set()
    assert layers_to_hold(m, 10) == {0, 1, 2, 3}   # clamps to the layers present


def test_protected_keep_layers_holds_last_blocks_bf16():
    """Depth-based hold (QAT-v7 Lever D): a served param in a kept layer stays bf16, others quantize."""
    torch.manual_seed(2)
    p0 = nn.Parameter(torch.randn(5, 5))   # layer 0 served -> quantized
    p2 = nn.Parameter(torch.randn(5, 5))   # layer 2 served -> HELD bf16 (in keep_layers)
    m = _FakeModel([("model.layers.0.mlp.down_proj", p0),
                    ("model.layers.2.mlp.down_proj", p2)])
    served = lambda name, suffixes=(): name.endswith("down_proj")
    b0, b2 = p0.detach().clone(), p2.detach().clone()

    info = protected_quantize_served(m, scheme="nvfp4", suffixes=("down_proj",),
                                     keep_experts={}, keep_layers={2}, is_served=served)

    assert not torch.equal(p0, b0)         # layer 0 quantized
    assert torch.equal(p2, b2)             # layer 2 held bf16 (byte-identical)
    assert info["protected_layer_params"] == 1
    assert info["quantized_modules"] == 1  # only layer 0 was quantized
    assert info["kept_params"] >= p2.numel()
