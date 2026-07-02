#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the residual truth-probe vs the text-feature baseline (the ledger's next step).

On a true/false factual-statement set (deception = confidently-stated FALSE fact — semantic,
NOT keyword-flagged, so regex text-features can't trivially catch it), fit the SAME
diff-in-means probe on (a) the 8 transparent text features and (b) the model's LAST-TOKEN
residual stream, SWEEPING every layer, split by fact-pair, and compare held-out AUROC + ECE.
A residual AUROC well above the text baseline at some layer is evidence the model linearly
encodes truth (Marks & Tegmark); a residual AUROC near chance at every layer on this small
model is a first-class HONEST NEGATIVE.

``candidateOnly``; the set is FIRST-PARTY + small (a third-party labeled set and a causal
patch/ablate check remain the pre-registered gaps). No claim flips. canClaimAGI false.

    ~/sophia-agi/.venv/bin/python tools/residual_probe_eval.py --data eval/deception/truth_statements_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.activation_probes import (  # noqa: E402
    auroc,
    ece,
    featurize_text,
    load_jsonl,
    train_vector_probe,
)


def _pair_split(rows):
    pairs = sorted({r.get("pair", r["id"]) for r in rows})
    train_pairs = set(pairs[: len(pairs) // 2])
    train = [r for r in rows if r.get("pair", r["id"]) in train_pairs]
    test = [r for r in rows if r.get("pair", r["id"]) not in train_pairs]
    return train, test


def _fit_eval(feats, rows_train, rows_test):
    # `feats` maps row id -> feature vector; adapt to train_vector_probe's (rows, featurizer(text))
    id_by_text = {r["text"]: r["id"] for r in rows_train + rows_test}
    feat = lambda t: feats[id_by_text[t]]  # noqa: E731
    probe = train_vector_probe(rows_train, feat, name="residual_truth")
    scores = [probe.score_vector(feat(r["text"])) for r in rows_test]
    labels = [bool(r.get("label")) for r in rows_test]
    return {"auroc": round(auroc(scores, labels), 4), "ece": round(ece(scores, labels), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="residual truth-probe (last-token, layer sweep) vs text baseline")
    ap.add_argument("--data", type=Path, default=ROOT / "eval" / "deception" / "truth_statements_v1.jsonl")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.data)
    train, test = _pair_split(rows)

    text_feats = {r["id"]: featurize_text(r["text"]) for r in rows}
    text_res = _fit_eval(text_feats, train, test)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    layers = model.model.layers
    n_layers = len(layers)

    # One forward per statement captures the LAST-TOKEN residual at EVERY layer.
    res_by_layer: dict = {L: {} for L in range(n_layers)}
    for r in rows:
        ids = tok(r["text"], return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
        caught: dict = {}
        handles = []
        for L in range(n_layers):
            def mk(idx):
                def hook(_m, _i, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    caught[idx] = hs[:, -1, :].detach().float().mean(dim=0).cpu().tolist()
                return hook
            handles.append(layers[L].register_forward_hook(mk(L)))
        with torch.no_grad():
            model(ids)
        for h in handles:
            h.remove()
        for L in range(n_layers):
            res_by_layer[L][r["id"]] = caught[L]

    sweep = []
    for L in range(n_layers):
        e = _fit_eval(res_by_layer[L], train, test)
        sweep.append({"layer": L, **e})
    best = max(sweep, key=lambda s: (s["auroc"] if s["auroc"] == s["auroc"] else -1))  # skip NaN

    report = {"schema": "sophia.residual_truth_probe_eval.v1", "candidateOnly": True,
              "level3Evidence": False, "syntheticData": True, "validated": False,
              "model": args.model, "pool": "last", "hiddenDim": int(model.config.hidden_size),
              "n": len(rows), "nTrain": len(train), "nTest": len(test),
              "textBaseline": text_res,
              "residualBestLayer": best,
              "residualSweep": sweep,
              "residualBeatsText": bool(best["auroc"] > text_res["auroc"]),
              "honestBound": ("FIRST-PARTY small true/false set; mechanism demonstration, not a "
                              "validated capability. A third-party labeled set + a causal "
                              "patch/ablate check are the pre-registered gaps. canClaimAGI false.")}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
