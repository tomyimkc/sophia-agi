# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the NVFP4 v5 mixed-precision keep-list lever (certify_lowram).

The v5 recipe lets us hold the most KL-sensitive served projection(s) in bf16 to push the
NVFP4 top-1 agreement past the 0.97 gate without over-training (v4's failure mode). These
tests pin the suffix-resolution + served-param logic GPU-free; the actual fidelity gain is a
measured, on-device result and is NOT asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import certify_lowram  # noqa: E402


def test_empty_keep_is_full_served_set() -> None:
    # default / empty keep == unchanged v3/v4 behaviour
    assert certify_lowram.resolve_served_suffixes("") == certify_lowram.SERVED_LINEAR_SUFFIXES
    assert certify_lowram.resolve_served_suffixes("   ") == certify_lowram.SERVED_LINEAR_SUFFIXES


def test_keep_down_proj_drops_only_that_suffix() -> None:
    served = certify_lowram.resolve_served_suffixes("down_proj")
    assert "down_proj" not in served
    # every other served suffix is preserved, order-stable
    for s in certify_lowram.SERVED_LINEAR_SUFFIXES:
        if s != "down_proj":
            assert s in served
    assert len(served) == len(certify_lowram.SERVED_LINEAR_SUFFIXES) - 1


def test_keep_multiple_and_whitespace_and_unknown() -> None:
    served = certify_lowram.resolve_served_suffixes(" down_proj , gate_proj , bogus_suffix ")
    assert "down_proj" not in served and "gate_proj" not in served
    assert "q_proj" in served and "up_proj" in served
    # unknown suffix is a harmless no-op (does not remove anything real)
    assert set(served) == set(certify_lowram.SERVED_LINEAR_SUFFIXES) - {"down_proj", "gate_proj"}


def test_is_served_param_honours_reduced_suffixes() -> None:
    name = "model.layers.0.mlp.experts.7.down_proj.weight"
    # full set: down_proj IS served (quantized)
    assert certify_lowram.is_served_param(name) is True
    # v5 keep-list holds down_proj bf16 -> NOT served
    served = certify_lowram.resolve_served_suffixes("down_proj")
    assert certify_lowram.is_served_param(name, suffixes=served) is False
    # an untouched projection stays served under the reduced set
    qn = "model.layers.0.self_attn.q_proj.weight"
    assert certify_lowram.is_served_param(qn, suffixes=served) is True


def test_router_embed_lmhead_never_served_regardless_of_keep() -> None:
    # the already-correct hard exclusions must hold under any keep-list
    full = certify_lowram.SERVED_LINEAR_SUFFIXES
    reduced = certify_lowram.resolve_served_suffixes("down_proj")
    for name in ("model.layers.0.mlp.gate.weight",          # MoE router gate
                 "model.embed_tokens.weight",               # embeddings
                 "lm_head.weight",                          # output head
                 "model.layers.0.input_layernorm.weight"):  # norm
        assert certify_lowram.is_served_param(name, suffixes=full) is False
        assert certify_lowram.is_served_param(name, suffixes=reduced) is False
