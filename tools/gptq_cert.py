#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""GPTQ quantization of the served params for the low-RAM cert (--round-mode gptq).

Drop-in for ``certify_lowram.quantize_served_params`` that replaces round-to-nearest with
Hessian-aware GPTQ on the NVFP4 group-16 SERVED grid (``moe.gptq.gptq_quantize_grouped``),
so it minimizes OUTPUT error where the RTN snap throws away cross-column structure. One
calibration pass captures per-served-param input activations; a final ``_torch_nvfp4`` snap
makes every weight EXACTLY the served representation.

- Attention q/k/v/o_proj: per-layer Hessian ``H = XᵀX`` from the linear's input.
- Fused MoE experts (``gate_up_proj``/``down_proj``): PER-EXPERT ROUTED Hessian — a hook on
  ``OlmoeExperts`` captures ``(hidden_states, top_k_index)``; expert ``e`` sees only its routed
  tokens (gate_up) and its recomputed intermediate ``act(gate)·up`` (down_proj).

GRID-IDENTITY is VERIFIED on a canary expert: the group-16 quantizer's output is asserted
``torch.equal`` to the canonical ``training.qat._torch_nvfp4`` — served-reproducible, not
asserted. canClaimAGI false; MSE→top1 is nonlinear — this produces a weight, not a verdict.
"""
from __future__ import annotations

from typing import Any


def gptq_quantize_served_params(model: Any, tok: Any, rows: "list[dict]", *,
                                scheme: str, suffixes: "tuple[str, ...]",
                                n_calib: int = 512, max_seq_len: int = 1024,
                                device: str = "cuda", group_size: int = 16,
                                percdamp: float = 0.01) -> dict:
    import torch
    import torch.nn as nn

    from training.qat import _torch_nvfp4
    from moe.gptq import gptq_quantize_grouped, nvfp4_group_quantize
    from tools.certify_lowram import is_served_param, _split_prompt_completion

    if scheme != "nvfp4":
        raise ValueError("--round-mode gptq supports scheme=nvfp4 only")

    attn_lin = {n: m for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and is_served_param(n + ".weight", suffixes=suffixes)}
    expert_mods = {n: m for n, m in model.named_modules() if m.__class__.__name__ == "OlmoeExperts"}

    attn_H: dict = {n: None for n in attn_lin}
    exp_Hgu: dict = {n: [None] * m.num_experts for n, m in expert_mods.items()}
    exp_Hdp: dict = {n: [None] * m.num_experts for n, m in expert_mods.items()}
    hooks = []

    for name, mod in attn_lin.items():
        def mk(nm):
            def pre(_m, args):
                x = args[0].detach().float().reshape(-1, args[0].shape[-1])
                h = x.t() @ x
                attn_H[nm] = h if attn_H[nm] is None else attn_H[nm] + h
            return pre
        hooks.append(mod.register_forward_pre_hook(mk(name)))

    for name, mod in expert_mods.items():
        def mke(nm, m):
            def pre(_mm, args):
                hs = args[0].detach().float()      # [T, hidden]
                idx = args[1].detach()             # [T, top_k]
                gate_up = m.gate_up_proj.data.float()  # [E, 2*inter, hidden]
                for e in range(m.num_experts):
                    sel = (idx == e).any(dim=1)
                    if not sel.any():
                        continue
                    Xe = hs[sel]                                       # [ne, hidden]
                    hgu = Xe.t() @ Xe
                    exp_Hgu[nm][e] = hgu if exp_Hgu[nm][e] is None else exp_Hgu[nm][e] + hgu
                    gu = nn.functional.linear(Xe, gate_up[e])         # [ne, 2*inter]
                    g_, u_ = gu.chunk(2, dim=-1)
                    inter = m.act_fn(g_) * u_                          # [ne, inter]
                    hdp = inter.t() @ inter
                    exp_Hdp[nm][e] = hdp if exp_Hdp[nm][e] is None else exp_Hdp[nm][e] + hdp
            return pre
        hooks.append(mod.register_forward_pre_hook(mke(name, mod)))

    seen = 0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for row in rows:
            if seen >= n_calib:
                break
            prompt, completion = _split_prompt_completion(row.get("messages") or [])
            if not completion:
                continue
            full = tok(prompt + completion, add_special_tokens=False)["input_ids"][:max_seq_len]
            if len(full) < 2:
                continue
            model(torch.tensor([full], dtype=torch.long, device=dev))
            seen += len(full)
    for h in hooks:
        h.remove()

    def served_snap(W, H):
        Wq = gptq_quantize_grouped(W.float(), H, group_size=group_size, percdamp=percdamp)
        return _torch_nvfp4(Wq.to(W.dtype))     # final canonical SERVED snap

    grid_identity = {"verified": False, "canary": None, "maxAbsDiff": None}
    q_params = kept_params = q_tensors = 0
    served_names: list[str] = []

    for name, mod in attn_lin.items():
        W = mod.weight.data
        H = attn_H[name]
        with torch.no_grad():
            W.copy_(served_snap(W, H) if H is not None else _torch_nvfp4(W.float()).to(W.dtype))
        q_params += W.numel(); q_tensors += 1
        if len(served_names) < 8:
            served_names.append(name)

    for name, mod in expert_mods.items():
        gate_up = mod.gate_up_proj.data
        down = mod.down_proj.data
        for e in range(mod.num_experts):
            with torch.no_grad():
                if exp_Hgu[name][e] is not None:
                    Wq = gptq_quantize_grouped(gate_up[e].float(), exp_Hgu[name][e],
                                               group_size=group_size, percdamp=percdamp)
                    if not grid_identity["verified"]:  # VERIFY grid-identity on the first real expert
                        a = _torch_nvfp4(Wq.to(gate_up.dtype))          # canonical served grid
                        b = nvfp4_group_quantize(Wq.to(gate_up.dtype), group_size=group_size)
                        grid_identity = {"verified": bool(torch.equal(a, b)),
                                         "canary": f"{name}.gate_up_proj[{e}]",
                                         "maxAbsDiff": float((a.float() - b.float()).abs().max())}
                    gate_up[e].copy_(_torch_nvfp4(Wq.to(gate_up.dtype)))
                else:
                    gate_up[e].copy_(_torch_nvfp4(gate_up[e].float()).to(gate_up.dtype))
                if exp_Hdp[name][e] is not None:
                    down[e].copy_(served_snap(down[e], exp_Hdp[name][e]))
                else:
                    down[e].copy_(_torch_nvfp4(down[e].float()).to(down.dtype))
        q_params += gate_up.numel() + down.numel(); q_tensors += 2
        if len(served_names) < 8:
            served_names.append(name + ".gate_up_proj")

    for pname, p in model.named_parameters():
        if not (p.dim() >= 2 and is_served_param(pname, suffixes=suffixes)):
            kept_params += p.numel()

    return {"quantized_modules": q_tensors, "quantized_params": q_params,
            "kept_params": kept_params, "total_params": q_params + kept_params,
            "quantized_sample": served_names, "round_mode": "gptq",
            "gridIdentity": grid_identity, "nCalibTokens": seen}
