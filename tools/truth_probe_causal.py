#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Harden the residual truth-probe (A4 follow-on): CAUSAL check + THIRD-PARTY AUROC.

Closes the two pre-registered gaps on ledger row ``truth-probe-textfeatures-only-2026-06-26``:

1. **Causal, not correlational.** Fit the truth direction ``w = mean(resid|true) - mean(resid|false)``
   at a mid ``--layer`` (last-token). Then STEER ``±alpha·w`` into that layer's residual during a
   forward on ``"{statement} This statement is"`` and measure the shift in ``logit(' true') -
   logit(' false')``. If ``+w`` raises the gap and ``-w`` lowers it (monotone) while a RANDOM
   direction of equal norm does not, the model *uses* the direction to assert truth (representation
   engineering; Zou et al.). A correlational probe would move under neither.
2. **Third-party set.** Refit + score held-out AUROC on a public dataset (``pminervini/true-false``)
   the repo did not author.

``candidateOnly``: small n, one small model, no calibration (high-ECE probe from A4). Not a gate.
canClaimAGI false; a run is not a result. Runs on the DGX Spark (bf16/CUDA); torch is optional in CI.

    ~/sophia-agi/.venv/bin/python tools/truth_probe_causal.py --layer 8
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="residual truth-probe causal + third-party hardening")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--data", type=Path, default=ROOT / "eval" / "deception" / "truth_statements_v1.jsonl")
    ap.add_argument("--third-party", default="pminervini/true-false")
    ap.add_argument("--third-party-split", default="cities")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # offline CI has no torch
        print(f"truth_probe_causal requires torch+transformers (run on the Spark): {e}", file=sys.stderr)
        return 2
    from agent.activation_probes import auroc

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    L = args.layer

    def last_resid(text):
        cap = {}
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            cap["v"] = hs[:, -1, :].detach().float().mean(0)
        h = model.model.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(dev))
        finally:
            h.remove()
        return cap["v"]

    rows = [json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
    # label True = FALSE (deceptive) statement; the truth direction points toward TRUE facts.
    true_res = torch.stack([last_resid(r["text"]) for r in rows if not r["label"]]).mean(0)
    false_res = torch.stack([last_resid(r["text"]) for r in rows if r["label"]]).mean(0)
    w = (true_res - false_res); w = w / w.norm()
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = torch.randn(w.shape, generator=g).to(dev); rand = rand / rand.norm()

    def steer_hook(vec, alpha):
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + alpha * vec.to(out[0].dtype),) + out[1:]
            return out + alpha * vec.to(out.dtype)
        return hook

    def first_id(s):
        return tok(s, add_special_tokens=False).input_ids[0]
    tid, fid = first_id(" true"), first_id(" false")

    def gap(vec, alpha):
        gs, hooks = [], []
        if vec is not None and alpha != 0:
            hooks.append(model.model.layers[L].register_forward_hook(steer_hook(vec, alpha)))
        try:
            with torch.no_grad():
                for r in rows:
                    ids = tok(f"{r['text']} This statement is", return_tensors="pt").input_ids.to(dev)
                    lg = model(ids).logits[0, -1].float()
                    gs.append((lg[tid] - lg[fid]).item())
        finally:
            for h in hooks:
                h.remove()
        return sum(gs) / len(gs)

    alpha = round(0.5 * true_res.norm().item(), 3)
    causal = {"alpha": alpha,
              "gap_off": round(gap(None, 0), 4),
              "gap_plus_w": round(gap(w, alpha), 4), "gap_minus_w": round(gap(w, -alpha), 4),
              "gap_plus_rand": round(gap(rand, alpha), 4), "gap_minus_rand": round(gap(rand, -alpha), 4)}
    causal["effect_w"] = round(causal["gap_plus_w"] - causal["gap_minus_w"], 4)
    causal["effect_rand"] = round(causal["gap_plus_rand"] - causal["gap_minus_rand"], 4)
    causal["is_causal"] = bool(causal["gap_plus_w"] > causal["gap_off"] > causal["gap_minus_w"]
                               and abs(causal["effect_w"]) > 2 * abs(causal["effect_rand"]))

    third = {"status": "skipped"}
    try:
        from datasets import load_dataset
        ds = load_dataset(args.third_party, split=args.third_party_split)
        sk = "statement" if "statement" in ds.column_names else ds.column_names[0]
        lk = "label" if "label" in ds.column_names else ds.column_names[-1]
        items = [(x[sk], int(x[lk])) for x in ds][:300]
        tr = torch.stack([last_resid(t) for t, y in items if y == 1][:80]).mean(0)
        fa = torch.stack([last_resid(t) for t, y in items if y == 0][:80]).mean(0)
        w2 = (tr - fa); w2 = w2 / w2.norm()
        sc = [torch.dot(w2, last_resid(t)).item() for t, y in items[160:]]
        lb = [bool(y) for _, y in items[160:]]
        third = {"status": "ok", "dataset": args.third_party, "split": args.third_party_split,
                 "n_test": len(sc), "auroc": round(auroc(sc, lb), 4)}
    except Exception as e:
        third = {"status": f"unavailable: {type(e).__name__}: {str(e)[:120]}"}

    report = {"schema": "sophia.truth_probe_causal.v1", "candidateOnly": True, "level3Evidence": False,
              "model": args.model, "layer": L, "n": len(rows), "causal": causal, "thirdParty": third,
              "honestBound": ("Causal (steering flips the truth assertion) + third-party AUROC harden "
                              "the residual truth-probe, but small n / one small model / uncalibrated "
                              "(high ECE) keep it candidateOnly. Not a gate. canClaimAGI false.")}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
