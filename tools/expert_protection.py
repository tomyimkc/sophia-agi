"""Expert-protection mixed precision for the OLMoE NVFP4 low-RAM cert (opt-in, no-train lever).

Holds the top-k MOST-ROUTED experts per layer in bf16 and NVFP4-quantizes the rest — a no-train
mixed-precision lever that lifts the quant top1 / abstention-coverage frontier at a small memory
cost. Measured on real v5 (n=256; harness-verified — the all-NVFP4 baseline reproduces the cert's
raw top1 0.9219 @ coverage 0.8594 exactly): top-8/64 experts/layer bf16 -> raw top1 0.9219->0.9414,
shippable coverage 0.8594->0.9297 (abstention 14%->7%), still clearing the 0.97-answered target.

Kept ISOLATED from ``certify_lowram.quantize_served_params`` so the default cert path is
byte-identical (this only runs when ``--keep-top-experts N`` with N>0 is passed). See
``docs/06-Roadmap/2026-07-01-nvfp4-lever-portfolio-methodology.md``. canClaimAGI=false.
"""
from __future__ import annotations

import re
from typing import Any, Callable

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def layer_of(name: str) -> int:
    """Transformer layer index parsed from a parameter/module name, or -1 if none."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def layers_to_hold(model: Any, n: int) -> "set[int]":
    """The set of the LAST ``n`` transformer-layer indices present in ``model`` — the depth-based
    bf16 hold (QAT-v7 Lever D: the argmax is decided by the final layers, so hold them full-precision
    at serve time). ``n<=0`` => empty set. Uses the layer indices actually present, not a config
    field, so it is robust to name schemes; returns e.g. {14, 15} for a 16-layer model and n=2."""
    if n <= 0:
        return set()
    layers = {layer_of(name) for name, _ in model.named_parameters() if layer_of(name) >= 0}
    # the last n indices PRESENT, robust to gaps (e.g. {0,2,4,6}, n=2 -> {4,6}), not `top-n`
    return set(sorted(layers)[-n:])


def top_routed_experts(model: Any, tok: Any, rows: "list[dict]", *, k: int,
                       n_eval: int, max_seq_len: int, device: str) -> "dict[int, set[int]]":
    """Return ``{layer: {top-k most-routed expert ids}}`` from a bf16 calib forward.

    Router-gate forward hooks count how often each expert is selected (top-``num_experts_per_tok``)
    over the same calib rows the cert scores; the per-layer top-``k`` are the experts worth holding
    bf16. Measured on the *unquantized* model (call before quantizing). No grad, hooks removed after.
    """
    import numpy as np
    import torch
    from tools.certify_lowram import collect_next_token_probs

    ne = int(model.config.num_experts)
    tk = int(model.config.num_experts_per_tok)
    counts: "dict[int, Any]" = {}

    def _mk(layer: int):
        def _hook(_module, _inp, out):
            logits = (out[0] if isinstance(out, tuple) else out).reshape(-1, ne).float()
            _, sel = torch.topk(logits, tk, dim=-1)
            c = counts.setdefault(layer, np.zeros(ne))
            for e in sel.reshape(-1).tolist():
                c[e] += 1
        return _hook

    handles = [mod.register_forward_hook(_mk(layer_of(name)))
               for name, mod in model.named_modules() if name.endswith(".mlp.gate")]
    try:
        with torch.no_grad():
            collect_next_token_probs(model, tok, rows, n_eval=n_eval,
                                     max_seq_len=max_seq_len, device=device)
    finally:
        for h in handles:
            h.remove()
    return {layer: set(np.argsort(-c)[:k].tolist()) for layer, c in counts.items()}


def protected_quantize_served(model: Any, *, scheme: str, suffixes,
                              keep_experts: "dict[int, set[int]]",
                              keep_layers: "set[int] | None" = None,
                              is_served: Callable[..., bool] | None = None) -> dict:
    """Quantize served weights to ``scheme`` in place, but hold two kinds of slices bf16:
    ``keep_experts[layer]`` (per-expert-slice skip on the fused 3-D expert tensors) AND any served
    param whose transformer layer is in ``keep_layers`` (whole-param skip — depth-based mixed
    precision, QAT-v7 Lever D). ``keep_layers`` takes precedence (a held layer's experts are held
    too). Kept slices/params count as ``kept_params`` so the memory ratio stays honest. Returns the
    ``certify_lowram.quantize_served_params`` dict shape plus ``protected_experts`` +
    ``protected_layer_params``.

    ``is_served`` is injectable for tests; default = ``certify_lowram.is_served_param``. Uses the
    exact fake-quant the model trained against (``training.qat`` torch NVFP4 / INT8 per-channel).
    """
    import torch
    from training.qat import _torch_nvfp4
    if is_served is None:
        from tools.certify_lowram import is_served_param as is_served

    def fq(w: "torch.Tensor") -> "torch.Tensor":
        if scheme == "nvfp4":
            return _torch_nvfp4(w)
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = amax / 127.0
        return torch.clamp(torch.round(w / scale), -127, 127) * scale

    q_params = kept_params = q_tensors = protected = protected_layers = 0
    served_names: "list[str]" = []
    for name, p in model.named_parameters():
        n = p.numel()
        if not (p.dim() >= 2 and is_served(name, suffixes=suffixes)):
            kept_params += n
            continue
        if keep_layers and layer_of(name) in keep_layers:
            # depth-based hold: keep the whole served param bf16 (the final layers decide the argmax)
            kept_params += n
            protected_layers += 1
            continue
        keep = keep_experts.get(layer_of(name)) if keep_experts else None
        with torch.no_grad():
            if p.dim() == 3 and keep:
                for e in range(p.shape[0]):
                    if e in keep:
                        kept_params += p[e].numel()
                        protected += 1
                    else:
                        p[e].copy_(fq(p[e].float()).to(p.dtype))
                        q_params += p[e].numel()
            else:
                p.copy_(fq(p.float()).to(p.dtype))
                q_params += n
        q_tensors += 1
        if len(served_names) < 8:
            served_names.append(name)
    return {"quantized_modules": q_tensors, "quantized_params": q_params,
            "kept_params": kept_params, "total_params": q_params + kept_params,
            "quantized_sample": served_names, "protected_experts": protected,
            "protected_layer_params": protected_layers}
