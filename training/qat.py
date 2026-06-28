# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Quantization-aware training for the real trainer — make the *released* artifact low-bit-safe.

*Why this exists.* ``tools/train_lora.py`` already loads in 4-bit for *training memory*
(QLoRA's bitsandbytes path). That shrinks the *training* footprint; it does **not** make the
*released* checkpoint robust to being *served* quantized. The gap this closes: a model that
was never shown its own quantization error at train time can degrade sharply when you later
crush it to INT8/NVFP4 for low-RAM serving (the ``serving/layer_stream.py`` + ``moe/quant.py``
path). QAT removes that surprise — the model *co-adapts* to its deployment quantization during
training, so the byte savings you take at serving time cost little measured quality (which
``serving/lowram_eval.py`` then certifies).

*The two levers, both built on the repo's existing quantizers.*
  1. **Fake-quant via straight-through estimator (STE).** On the forward pass a weight is
     replaced by ``dequant(quant(W))`` using the *exact* scheme it will be served with
     (``moe.quant``: INT8 per-channel or NVFP4). On the backward pass the gradient flows
     through unchanged (the quant step is non-differentiable; STE treats it as identity).
     The model therefore trains *as if* quantized while still optimizing full-precision
     master weights — the standard QAT recipe.
  2. **Quant-pushing penalty.** A regularizer ``λ · mean((W − fake_quant(W))²)`` that pulls
     weights toward values their quantizer reproduces exactly. This is the
     ``moe.quant`` analog of the ternary regularizer in ``pretraining/qat/study.py`` —
     same idea (concentrate importance into grid-friendly weights), real deployment grid.

*What is CI-tested vs deployment.* The numerics here (fake-quant round-trip, STE passthrough,
penalty behavior, and a small gradient-descent demonstration that the penalty lowers
post-quant error) are pure-numpy with deterministic ``offline_invariants()`` — the repo's
reference discipline. The torch glue (``attach_qat`` forward hooks, ``qat_penalty`` as a loss
term) is a thin wrapper guarded behind a torch import; it runs inside ``tools/train_lora.py``
on GPU, out of scope for the CI reference exactly like the bitsandbytes path.

Honest scope: QAT makes the *released* artifact cheaper to serve *at quality*, but the claim
"capability retained at N bits" is only earned once the QAT checkpoint clears
``serving/lowram_eval.py`` against the FP16 reference on a held-out, decontaminated set
(``docs/11-Platform/Cheap-Compute-Boundary.md`` Boundary 3). This module is the training
mechanism, not the measurement.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

SCHEMES = ("int8", "nvfp4")

# Served low-bit set: attention + MLP/expert projection name suffixes. Fused MoE experts may
# expose a combined ``gate_up_proj``. These mirror tools/certify_lowram.SERVED_LINEAR_SUFFIXES so
# the QAT *training* set and the NVFP4 *serving/certify* set are the same weights.
_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
_EXPERT_SUFFIXES = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
# Never fake-quant these (kept high-precision in real serving): embeddings, norms, router gate,
# the LM head. Co-adapting a layer you will NOT quantize at serve time only adds noise.
_QAT_EXCLUDE = ("embed", "lm_head", "norm")


def is_qat_target_name(name: str) -> bool:
    """True if parameter/module ``name`` is in the served low-bit set QAT should fake-quant.

    Matches the attn + MLP/expert projections (per-expert ``...experts.7.down_proj`` AND fused
    ``...experts.gate_up_proj``); excludes embeddings, norms, the MoE router gate (``mlp.gate``),
    ``lm_head``, and any LoRA tensors. Pure/string-only so it is CI-testable without torch."""
    low = name.lower()
    if any(x in low for x in _QAT_EXCLUDE) or "lora" in low:
        return False
    base = name.replace(".base_layer", "")
    if base.endswith(".weight"):
        base = base[: -len(".weight")]
    if base.endswith("mlp.gate"):              # MoE router gate stays high-precision
        return False
    return any(base.endswith(s) for s in (_ATTN_SUFFIXES + _EXPERT_SUFFIXES))


def summarize_qat_coverage(names: "list[str]") -> dict:
    """Break a list of QAT-covered module/param names into attn / expert / other counts.

    The number that mattered and was invisible for four certify rounds: how many MoE *experts*
    QAT actually reached. ``expert`` counts names under an ``experts`` container; ``attn`` counts
    the attention projections; ``other`` is anything else covered."""
    names = list(names)
    expert = sum(1 for n in names if "experts" in n or "expert." in n)
    attn = sum(1 for n in names if "experts" not in n and "expert." not in n
               and any((n[:-7] if n.endswith(".weight") else n).endswith(s) for s in _ATTN_SUFFIXES))
    return {"total": len(names), "attn": attn, "expert": expert,
            "other": len(names) - attn - expert}


# ---------------------------------------------------------------------------
# 1. Fake-quant — the deployment grid the model trains against (built on moe.quant)
# ---------------------------------------------------------------------------

def fake_quant(W, scheme: str = "int8"):
    """Return ``dequant(quant(W))`` for ``W`` under the named deployment scheme.

    ``int8``  : symmetric per-channel INT8 (``moe.quant.quantize_int8``, axis=0).
    ``nvfp4`` : Blackwell NVFP4, E2M1 + per-block FP8 micro-scale (``moe.quant.nvfp4_roundtrip``).

    This is the value the forward pass sees under QAT — the model's activations are computed
    from the quantized weight, so it learns to be robust to exactly this error.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    from moe.quant import dequantize_int8, nvfp4_roundtrip, quantize_int8

    W = np.asarray(W, dtype=np.float64)
    if scheme == "int8":
        q, scale = quantize_int8(W, per_channel=True, axis=0)
        return dequantize_int8(q, scale)
    return nvfp4_roundtrip(W)


def ste_forward(W, scheme: str = "int8"):
    """STE forward: the quantized value the layer computes with (an alias of fake_quant)."""
    return fake_quant(W, scheme)


def ste_backward(grad_out):
    """STE backward: identity. The non-differentiable quant step passes gradient through.

    This is the whole trick — ``d/dW dequant(quant(W)) ≈ 1`` inside the representable range,
    so the master weights receive the same gradient they would full-precision, while the
    forward error teaches robustness. Returned as-is for the numpy reference; the torch
    autograd Function below implements the same rule.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    return np.asarray(grad_out, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Quant-pushing penalty — pull weights toward their own quantization grid
# ---------------------------------------------------------------------------

def quant_push_penalty(W, scheme: str = "int8") -> float:
    """Mean squared distance of ``W`` from its quantized reconstruction.

    ~0 for weights already on the grid; positive and proportional to how far they sit from
    it. Adding ``λ · penalty`` to the loss co-adapts the model to its deployment quantization
    (the ``moe.quant`` analog of ``pretraining/qat/study.ternary_regularizer``).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    W = np.asarray(W, dtype=np.float64)
    dq = fake_quant(W, scheme)
    return float(np.mean((W - dq) ** 2))


def post_quant_matmul_error(W, x, scheme: str = "int8") -> float:
    """Relative error of ``x @ fake_quant(W)`` vs ``x @ W`` — the served-inference damage.

    This is the number QAT is trying to shrink: how much the quantized weight distorts the
    layer's output. ``serving/lowram_eval.py`` measures the model-level analog (output KL).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    ref = x @ W
    approx = x @ fake_quant(W, scheme)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(approx - ref) / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# 3. Torch glue (deployment path; guarded — not run in CI)
# ---------------------------------------------------------------------------

def _torch_ste_quant():  # pragma: no cover - exercised only with torch present
    """Build a torch autograd Function implementing fake-quant with STE backward."""
    import torch

    class _STEQuant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w: "torch.Tensor", scheme: str) -> "torch.Tensor":
            # Per-channel symmetric INT8 / NVFP4 emulation on-device. Forward = dequant(quant).
            if scheme == "int8":
                amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
                scale = amax / 127.0
                q = torch.clamp(torch.round(w / scale), -127, 127)
                return q * scale
            # nvfp4: per-block (16) absmax scale + E2M1 snap, emulated in fp.
            return _torch_nvfp4(w)

        @staticmethod
        def backward(ctx, g: "torch.Tensor"):
            return g, None   # identity passthrough; no grad to the scheme string

    return _STEQuant


def _torch_nvfp4(w):  # pragma: no cover - torch-only
    import torch

    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          device=w.device, dtype=w.dtype)
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 16
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, 16)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / 6.0
    scaled = (blocks / scale).abs().unsqueeze(-1)
    idx = (scaled - levels).abs().argmin(dim=-1)
    mag = levels[idx]
    dq = torch.sign(blocks) * mag * scale
    return dq.reshape(-1)[: w.numel()].reshape(w.shape)


def attach_qat(model: Any, *, scheme: str = "int8",
               module_types: "tuple[str, ...]" = ("Linear",)) -> int:  # pragma: no cover - torch-only
    """Wrap each target module's ``forward`` to compute with the STE-fake-quantized weight.

    The master weight (the optimizer's parameter) is untouched, so training optimizes
    full-precision weights; only the *value used in the forward* is quantized, with the STE
    backward routing gradient straight back to the master weight. Idempotent per module
    (skips already-wrapped ones). Returns the number of modules wrapped. Deployment path:
    called from ``tools/train_lora.py`` when ``--qat`` is set; call :func:`detach_qat` (or
    just save before wrapping) to restore fp forward.
    """
    import warnings

    import torch.nn.functional as F
    ste = _torch_ste_quant()
    wrapped_names: list[str] = []
    for name, m in model.named_modules():
        if type(m).__name__ not in module_types or not hasattr(m, "weight"):
            continue
        if getattr(m, "_qat_wrapped", False):
            continue
        # Only fake-quant the served set (attn + MLP/experts). Skipping lm_head / router / norms /
        # embeddings aligns the QAT training grid with what NVFP4 serving actually quantizes
        # (serving/lowram_eval.py + tools/certify_lowram.is_served_param), so we never co-adapt a
        # layer that stays bf16 at serve time.
        if not is_qat_target_name(name):
            continue
        orig_forward = m.forward

        def qat_forward(x, _m=m, _orig=orig_forward):
            if not _m.training:
                return _orig(x)
            w_q = ste.apply(_m.weight, scheme)            # fake-quant with STE backward
            return F.linear(x, w_q, _m.bias)

        m._qat_orig_forward = orig_forward
        m.forward = qat_forward
        m._qat_wrapped = True
        wrapped_names.append(name)

    cov = summarize_qat_coverage(wrapped_names)
    # If QAT reached NO experts but the model HAS expert weights, the forward-STE can't see them
    # (fused expert Parameters, or experts simply absent from module_types). Co-adapting nothing
    # there means NVFP4-serving the experts later will degrade — warn loudly rather than silently.
    has_expert_params = any(
        ("experts" in n) and getattr(p, "ndim", 0) >= 2
        for n, p in model.named_parameters())
    cov["model_has_expert_params"] = bool(has_expert_params)
    model._qat_coverage = cov
    if has_expert_params and cov["expert"] == 0:
        warnings.warn(
            "QAT attached to 0 expert modules but the model has expert weights: the MoE experts "
            "will NOT be co-adapted to "
            f"{scheme} and are expected to degrade when served quantized. Ensure experts load as "
            "per-expert nn.Linear (so they match module_types) and are LoRA-trainable.",
            RuntimeWarning, stacklevel=2)
    return len(wrapped_names)


def detach_qat(model: Any) -> int:  # pragma: no cover - torch-only
    """Restore original forwards wrapped by :func:`attach_qat`. Returns count restored."""
    restored = 0
    for m in model.modules():
        if getattr(m, "_qat_wrapped", False):
            m.forward = m._qat_orig_forward
            m._qat_wrapped = False
            restored += 1
    return restored


def qat_penalty(model: Any, *, scheme: str = "int8", lam: float = 1e-3,
                module_types: "tuple[str, ...]" = ("Linear",)):
    """Sum the quant-pushing penalty over target modules as a torch loss term (× ``lam``).

    Add to the task loss in the training loop: ``loss = loss + qat_penalty(model, ...)``.
    """
    import torch  # pragma: no cover - torch-only path
    ste = _torch_ste_quant()                      # build the autograd Function once per call
    total = None
    device = None
    for name, m in model.named_modules():
        if type(m).__name__ in module_types and hasattr(m, "weight") and is_qat_target_name(name):
            w = m.weight
            if device is None:
                device = w.device
            dq = ste.apply(w, scheme)
            term = torch.mean((w - dq) ** 2)
            total = term if total is None else total + term
    if total is None:
        # No matching modules: a zero on the model's own device, so adding it to a GPU
        # loss never triggers a device mismatch.
        dev = device or next(model.parameters()).device
        return torch.zeros((), requires_grad=True, device=dev)
    return lam * total


# ---------------------------------------------------------------------------
# 4. Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    W = rng.standard_normal((64, 48))

    # 1. INT8 fake-quant round-trips within the quant tolerance; NVFP4 looser but bounded.
    rel_int8 = np.linalg.norm(fake_quant(W, "int8") - W) / np.linalg.norm(W)
    rel_nvfp4 = np.linalg.norm(fake_quant(W, "nvfp4") - W) / np.linalg.norm(W)
    checks["int8_roundtrip_tight"] = rel_int8 < 0.02
    checks["nvfp4_roundtrip_bounded"] = rel_nvfp4 < 0.20
    detail["rel_int8"] = round(float(rel_int8), 5)
    detail["rel_nvfp4"] = round(float(rel_nvfp4), 5)

    # 2. STE backward is identity (gradient passes through unchanged).
    g = rng.standard_normal((64, 48))
    checks["ste_passthrough"] = bool(np.allclose(ste_backward(g), g))

    # 3. Penalty is ~0 on an already-quantized weight, positive on a continuous one.
    Wq = fake_quant(W, "int8")
    checks["penalty_low_on_grid"] = quant_push_penalty(Wq, "int8") <= quant_push_penalty(W, "int8")
    checks["penalty_positive_off_grid"] = quant_push_penalty(W, "int8") > 0.0
    detail["penalty_off_grid"] = round(quant_push_penalty(W, "int8"), 6)
    detail["penalty_on_grid"] = round(quant_push_penalty(Wq, "int8"), 8)

    # 4. The QAT claim, in miniature: pushing weights toward their NVFP4 grid lowers the
    #    post-quant matmul error. Run a few penalty-gradient steps and confirm the served
    #    error drops vs the un-pushed weight. (NVFP4 — the 4-bit path with real headroom.)
    x = rng.standard_normal((16, 64))
    Wc = W.copy()
    err_before = post_quant_matmul_error(Wc, x, "nvfp4")
    for _ in range(40):
        dq = fake_quant(Wc, "nvfp4")
        Wc = Wc - 0.5 * (Wc - dq)        # gradient of mean((W-dq)^2) ≈ (W-dq); step toward grid
    err_after = post_quant_matmul_error(Wc, x, "nvfp4")
    checks["qat_lowers_post_quant_error"] = err_after < err_before
    detail["post_quant_err_before"] = round(err_before, 5)
    detail["post_quant_err_after"] = round(err_after, 5)

    # 5. Fail-closed on an unknown scheme.
    try:
        fake_quant(W, "ternary"); checks["unknown_scheme_rejected"] = False
    except ValueError:
        checks["unknown_scheme_rejected"] = True

    # 6. QAT target/coverage accounting (the visibility that was missing): the served set covers
    #    attention AND per-expert/fused experts; it excludes head/embeddings/norms/router/LoRA. And
    #    summarize_qat_coverage must SEE the experts (the 4-round blind spot) — a run that reaches 0
    #    experts is the failure signature.
    targets = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.experts.7.down_proj",
               "model.layers.0.mlp.experts.gate_up_proj"]
    nontargets = ["lm_head", "model.embed_tokens", "model.layers.0.input_layernorm",
                  "model.layers.0.mlp.gate",   # router
                  "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    checks["qat_targets_attn_and_experts"] = all(is_qat_target_name(n) for n in targets)
    checks["qat_skips_head_norm_router_lora"] = not any(is_qat_target_name(n) for n in nontargets)
    cov_full = summarize_qat_coverage([
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.down_proj"])
    checks["coverage_counts_experts"] = cov_full["expert"] == 2 and cov_full["attn"] == 2
    cov_attn_only = summarize_qat_coverage(["model.layers.0.self_attn.q_proj"])
    checks["coverage_flags_zero_experts"] = cov_attn_only["expert"] == 0
    detail["coverage_full"] = cov_full

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    # Script execution only: put the repo root on the path so the lazy ``moe.quant`` import
    # resolves. As a library (pytest / installed package) this side effect never runs — the
    # importer is responsible for the path, matching serving/ and moe/.
    import sys
    from pathlib import Path

    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    ok, detail = offline_invariants()
    print("QAT-training offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  fake-quant rel err int8={detail.get('rel_int8')} nvfp4={detail.get('rel_nvfp4')}")
    print(f"  post-quant matmul err {detail.get('post_quant_err_before')} -> "
          f"{detail.get('post_quant_err_after')} (QAT push)")
    raise SystemExit(0 if ok else 1)
