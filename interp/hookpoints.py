# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hook-point name resolution for SAE harvesting (pure stdlib, no torch).

A thin adapter mapping TransformerLens-style hook names ↔ HuggingFace module
paths for the repo's standardized Qwen2.5 models, so the harvesting code (M1) and
the steering hook layer (`agent/steering/hooks.py`) agree on *where* the residual
stream is captured. Importing this module never loads torch or a model.

Convention: ``resid_post`` at layer L is the residual stream *after* decoder
block L — i.e. the output tensor of ``model.model.layers[L]`` (the module
``agent.steering.hooks`` already registers a forward hook on). In TransformerLens
that is ``blocks.{L}.hook_resid_post``.
"""
from __future__ import annotations

# Standardized base models (see models/manifest.json). d_model / n_layers from
# the published HF configs; used only to validate a requested layer offline.
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": {"n_layers": 28, "d_model": 3584},
    "Qwen/Qwen2.5-3B-Instruct": {"n_layers": 36, "d_model": 2048},
}

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HOOK_POINTS = ("resid_post", "resid_pre", "mlp_out", "attn_out")


def model_config(model: str) -> dict:
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(MODELS)}")
    return MODELS[model]


def n_layers(model: str = DEFAULT_MODEL) -> int:
    return model_config(model)["n_layers"]


def d_model(model: str = DEFAULT_MODEL) -> int:
    return model_config(model)["d_model"]


def default_layer(model: str = DEFAULT_MODEL) -> int:
    """A sensible mid-stack layer to harvest first (≈ 0.55 depth; plan M1: L≈14–18/28)."""
    return round(n_layers(model) * 0.55)


def validate_layer(layer: int, model: str = DEFAULT_MODEL) -> int:
    nl = n_layers(model)
    if not isinstance(layer, int) or not (0 <= layer < nl):
        raise ValueError(f"layer {layer} out of range [0,{nl}) for {model}")
    return layer


def tl_hook_name(layer: int, point: str = "resid_post", model: str = DEFAULT_MODEL) -> str:
    """TransformerLens-style hook name, e.g. 'blocks.14.hook_resid_post'."""
    validate_layer(layer, model)
    if point not in HOOK_POINTS:
        raise ValueError(f"unknown hook point {point!r}; known: {HOOK_POINTS}")
    return f"blocks.{layer}.hook_{point}"


def hf_module_path(layer: int, point: str = "resid_post", model: str = DEFAULT_MODEL) -> str:
    """HF module path whose forward output carries the activation.

    ``resid_post`` → the decoder block output (``model.model.layers.L``), which is
    exactly what ``agent.steering.hooks.attach_hooks`` hooks. ``mlp_out`` /
    ``attn_out`` point at the submodules for finer captures (M5 patching).
    """
    validate_layer(layer, model)
    base = f"model.model.layers.{layer}"
    if point in ("resid_post", "resid_pre"):
        return base
    if point == "mlp_out":
        return f"{base}.mlp"
    if point == "attn_out":
        return f"{base}.self_attn"
    raise ValueError(f"unknown hook point {point!r}")


def resolve(layer: int, point: str = "resid_post", model: str = DEFAULT_MODEL) -> dict:
    """Both names + dims for a (model, layer, point), validated. No torch."""
    return {
        "model": model,
        "layer": validate_layer(layer, model),
        "point": point,
        "tlName": tl_hook_name(layer, point, model),
        "hfModulePath": hf_module_path(layer, point, model),
        "dModel": d_model(model),
        "nLayers": n_layers(model),
    }
