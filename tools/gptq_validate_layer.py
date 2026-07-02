#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the GPTQ output-error lever on REAL model layers (reproducibility harness).

Does GPTQ (``moe.gptq``, Hessian-aware) beat round-to-nearest on the SAME NVFP4 grid the
low-RAM cert serves (``training.qat._torch_nvfp4``)? Captures a target linear's input
activations over a small calibration set, builds ``H = X Xᵀ``, quantizes with RTN vs GPTQ,
and reports mean output-MSE ``mean(((W-Wq)X)²)`` for each. A ratio < 1 means GPTQ reduces the
served output error. Runs on the DGX Spark (bf16, CUDA). ``canClaimAGI`` false — an output-MSE
diagnostic, not a top1/capability claim; a certifiable number needs the served-path wiring.

    ~/sophia-agi/.venv/bin/python tools/gptq_validate_layer.py \
        --model allenai/OLMoE-1B-7B-0924-Instruct --proj o_proj --n-rows 32

The dominant top1 mass is the fused MoE experts (3-D ``[E,in,out]`` tensors, per-expert routed
activations) — a harder follow-on than the 2-D attention linears this harness targets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_text(r: dict) -> str:
    for k in ("text", "prompt", "content", "instruction"):
        if isinstance(r.get(k), str):
            return r[k]
    return json.dumps(r)[:1000]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPTQ-vs-RTN output-MSE probe on real layers (NVFP4 grid).")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--proj", default="o_proj", help="self_attn projection to probe (2-D linear)")
    ap.add_argument("--n-rows", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--calib", type=Path, default=ROOT / "training" / "lora" / "train.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from moe.gptq import gptq_quantize, output_mse, rtn_quantize
    from training.qat import _torch_nvfp4

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()

    n_layers = len(model.model.layers)
    depths = sorted({0, n_layers // 2, n_layers - 1})
    targets = {f"L{L}.{args.proj}": getattr(model.model.layers[L].self_attn, args.proj) for L in depths}
    caps: dict = {name: [] for name in targets}
    hooks = []
    for name, lin in targets.items():
        def mk(nm):
            def pre(_mod, a):
                x = a[0]
                caps[nm].append(x.reshape(-1, x.shape[-1]).detach().float().cpu())
            return pre
        hooks.append(lin.register_forward_pre_hook(mk(name)))

    rows = [json.loads(x) for x in args.calib.read_text().splitlines() if x.strip()][: args.n_rows]
    with torch.no_grad():
        for r in rows:
            ids = tok(_as_text(r), return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
            model(ids)
    for h in hooks:
        h.remove()

    def nvfp4_col(col):
        return _torch_nvfp4(col.to(dev).to(torch.bfloat16)).float()

    results = []
    for name, lin in targets.items():
        W = lin.weight.data.float().to(dev)
        X = torch.cat(caps[name], 0)
        if X.shape[0] > args.max_tokens:
            g = torch.Generator().manual_seed(0)
            X = X[torch.randperm(X.shape[0], generator=g)[: args.max_tokens]]
        Xm = X.t().contiguous().to(dev)
        H = Xm @ Xm.t()
        rtn = rtn_quantize(W, nvfp4_col)
        gptq = gptq_quantize(W, H, nvfp4_col)
        mr, mg = output_mse(W, rtn, Xm), output_mse(W, gptq, Xm)
        results.append({"layer": name, "in": W.shape[1], "out": W.shape[0], "tokens": Xm.shape[1],
                        "mse_rtn": mr, "mse_gptq": mg, "ratio_gptq_over_rtn": (mg / mr) if mr else None})

    report = {"schema": "sophia.gptq_layer_probe.v1", "candidateOnly": True, "level3Evidence": False,
              "model": args.model, "grid": "nvfp4", "proj": args.proj, "results": results,
              "note": "output-MSE diagnostic on 2-D attention linears; NOT a top1 claim. The top1 "
                      "mass is the fused MoE experts (per-expert routed GPTQ = follow-on). A "
                      "certifiable number needs served-path block-boundary matching. canClaimAGI false."}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
