#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data mixture-ratio (配比) sweep: find the training mix that minimizes target loss.

The data direction's "持续优化数据配比与筛选策略" in miniature. Two distinct sources (e.g.
"web" and "code") generate text from different Markov sources. We hold the training token
budget FIXED and sweep the mixing ratio, training a proxy nano LM at each ratio and
measuring held-out loss on a TARGET distribution. The curve has an interior optimum — the
best mix is rarely 100/0 — which is the whole point of mixture-ratio research.

Three target regimes are reported so the lesson is visible:
  * target = source A only  -> optimum skews toward A
  * target = source B only  -> optimum skews toward B
  * target = 50/50 blend     -> optimum is interior

Honest: a toy that demonstrates the methodology (fixed budget, proxy model, target-driven
optimum), not a production data recipe. Pure stdlib.

    python -m pretraining.data_mixing.run_mixing --quick
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pretraining.nano import (
    NanoLM, eval_loss, make_source, mixed_corpus, sample_stream, to_examples, train,
)

HERE = Path(__file__).resolve().parent


def _held_for_target(srcA, srcB, target, context, n=1000):
    if target == "A":
        return to_examples(sample_stream(srcA, n, seed=777), context)
    if target == "B":
        return to_examples(sample_stream(srcB, n, seed=778), context)
    # blended target: half from each
    h = to_examples(sample_stream(srcA, n // 2, seed=777), context)
    h += to_examples(sample_stream(srcB, n // 2, seed=778), context)
    return h


def run(*, quick: bool = False, out: Path | None = None) -> dict:
    vocab, order, context, hidden = 8, 2, 2, 16
    budget = 1600           # FIXED total training examples across the mix
    epochs = 10 if quick else 14
    seeds = 1 if quick else 2
    # two genuinely different sources (different seeds => different transition tables)
    srcA = make_source(vocab=vocab, order=order, seed=1, peak=3.0)
    srcB = make_source(vocab=vocab, order=order, seed=2, peak=3.0)
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0] if quick else [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    targets = {}
    for target in ("A", "B", "blend"):
        held = _held_for_target(srcA, srcB, target, context)
        curve = []
        for wA in ratios:
            vals = []
            for s in range(seeds):
                ex = mixed_corpus([srcA, srcB], [wA, 1 - wA], budget,
                                  context=context, seed=10 + s)
                m = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=s)
                train(m, ex, epochs=epochs, optimizer="adam", lr=0.03, seed=s)
                vals.append(eval_loss(m, held))
            curve.append({"weight_A": wA, "loss": round(statistics.mean(vals), 5),
                          "sd": round(statistics.pstdev(vals), 5) if seeds > 1 else 0.0})
        best = min(curve, key=lambda c: c["loss"])
        targets[target] = {
            "curve": curve,
            "best_weight_A": best["weight_A"],
            "best_loss": best["loss"],
            "interior_optimum": 0.0 < best["weight_A"] < 1.0,
        }

    report = {
        "study": "data mixture-ratio (配比) sweep at fixed token budget",
        "honesty_note": ("Toy two-source mix; demonstrates fixed-budget, target-driven "
                         "optimum methodology, not a production data recipe."),
        "config": {"vocab": vocab, "context": context, "hidden": hidden,
                   "budget": budget, "epochs": epochs, "seeds": seeds, "ratios": ratios},
        "targets": targets,
        "lesson": ("Best mix tracks the target distribution and the blended target's "
                   "optimum is interior — pure single-source data is suboptimal when the "
                   "deployment distribution is mixed."),
    }
    out = out or (HERE / "mixing-curve-latest.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = run(quick=args.quick, out=args.out)
    for t, blk in r["targets"].items():
        print(f"target={t:5s} best_weight_A={blk['best_weight_A']} "
              f"loss={blk['best_loss']} interior={blk['interior_optimum']}")


if __name__ == "__main__":
    main()
