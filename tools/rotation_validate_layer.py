#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Upper-bound probe: does a random-Hadamard rotation cut NVFP4 output-MSE on real layers?

Measures plain-NVFP4 vs (paired) rotated-NVFP4 output-MSE on OLMoE ``o_proj`` at a few depths.
HONEST caveat: the rotated number is what the PAIRED-ABSORBED served model would have IF the
residual-stream rotation is folded into the graph correctly — a feasibility upper bound, NOT a
certifiable number (a rotation applied only here is not reproducible by a stock ``--quantization
nvfp4`` server). ``canClaimAGI`` false; a run is not a result.

    ~/sophia-agi/.venv/bin/python tools/rotation_validate_layer.py --proj o_proj --n-rows 32
"""
from __future__ import annotations

import argparse
import json
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


def _output_mse(W, Wq, X) -> float:
    d = (W.float() - Wq.float()) @ X.float()
    return float((d * d).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="random-Hadamard rotation NVFP4 output-MSE probe.")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--proj", default="o_proj")
    ap.add_argument("--n-rows", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--calib", type=Path, default=ROOT / "training" / "lora" / "train.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from moe.rotation import apply_input_rotation, random_hadamard_matrix, rotate_activations
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
            def pre(_m, a):
                caps[nm].append(a[0].reshape(-1, a[0].shape[-1]).detach().float().cpu())
            return pre
        hooks.append(lin.register_forward_pre_hook(mk(name)))
    rows = [json.loads(x) for x in args.calib.read_text().splitlines() if x.strip()][: args.n_rows]
    with torch.no_grad():
        for r in rows:
            ids = tok(_as_text(r), return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
            model(ids)
    for h in hooks:
        h.remove()

    results = []
    for name, lin in targets.items():
        W = lin.weight.data.float().to(dev)
        X = torch.cat(caps[name], 0)
        if X.shape[0] > args.max_tokens:
            g = torch.Generator().manual_seed(0)
            X = X[torch.randperm(X.shape[0], generator=g)[: args.max_tokens]]
        Xm = X.t().contiguous().to(dev)
        R = random_hadamard_matrix(W.shape[1], seed=0, device=dev, dtype=torch.float64)
        mse_plain = _output_mse(W, _torch_nvfp4(W.to(torch.bfloat16)).float(), Xm)
        Wr = apply_input_rotation(W.double(), R).float()
        Xr = rotate_activations(Xm.double(), R).float()
        mse_rot = _output_mse(Wr, _torch_nvfp4(Wr.to(torch.bfloat16)).float(), Xr)
        results.append({"layer": name, "in": W.shape[1], "mse_nvfp4_plain": mse_plain,
                        "mse_nvfp4_rotated": mse_rot,
                        "ratio_rot_over_plain": (mse_rot / mse_plain) if mse_plain else None})

    report = {"schema": "sophia.rotation_layer_probe.v1", "candidateOnly": True, "level3Evidence": False,
              "model": args.model, "grid": "nvfp4", "lever": "random-hadamard paired rotation",
              "results": results,
              "note": "UPPER BOUND: served-reproducible only under paired residual-stream absorption "
                      "(the follow-on); a rotation applied only here is not reproducible by a stock "
                      "nvfp4 server. NOT a top1 claim. canClaimAGI false."}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
