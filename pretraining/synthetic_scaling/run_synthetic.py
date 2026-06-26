#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Synthetic-data scaling & collapse study (研究数据的合成与 scaling 行为).

Question: as you add synthetic data to a fixed real-data budget, when does it help, when
does it saturate, and when does it *hurt*? We hold a small real budget from the target
source fixed, then add increasing amounts of SYNTHETIC data produced by a drifted
generator (``drifted_source`` — a stand-in for an imperfect synthesizer), and measure
held-out loss on the REAL target.

Two generator fidelities make the lesson explicit:
  * high-fidelity (low drift): adding synthetic keeps helping, then saturates near the floor.
  * low-fidelity (high drift): adding synthetic helps briefly, then loss RISES — model
    collapse, the failure the recent synthetic-data literature warns about.

The takeaway mirrors the data direction's reality: synthetic data scales only as far as
its fidelity allows; quantity cannot substitute for distributional faithfulness. Honest
toy; pure stdlib.

    python -m pretraining.synthetic_scaling.run_synthetic --quick
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pretraining.nano import (
    NanoLM, drifted_source, eval_loss, make_source, sample_stream,
    source_entropy, to_examples, train,
)

HERE = Path(__file__).resolve().parent


def run(*, quick: bool = False, out: Path | None = None) -> dict:
    vocab, order, context, hidden = 8, 2, 2, 16
    real_budget = 300
    epochs = 10 if quick else 14
    seeds = 1 if quick else 2
    synth_multiples = [0, 1, 2, 4] if quick else [0, 1, 2, 4, 8, 16]

    real_src = make_source(vocab=vocab, order=order, seed=1, peak=3.0)
    E = source_entropy(real_src)
    held = to_examples(sample_stream(real_src, 1200, seed=99), context=context)
    real_ex = to_examples(sample_stream(real_src, real_budget + context, seed=2), context=context)

    fidelities = {"high_fidelity": 0.15, "low_fidelity": 0.6}
    results = {}
    for name, drift in fidelities.items():
        synth_src = drifted_source(real_src, drift, seed=42)
        curve = []
        for mult in synth_multiples:
            vals = []
            for s in range(seeds):
                ex = list(real_ex)
                if mult > 0:
                    n_syn = real_budget * mult
                    ex += to_examples(sample_stream(synth_src, n_syn + context, seed=50 + s),
                                      context=context)
                m = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=s)
                train(m, ex, epochs=epochs, optimizer="adam", lr=0.03, seed=s)
                vals.append(eval_loss(m, held))
            curve.append({"synth_multiple": mult,
                          "total_examples": real_budget * (1 + mult),
                          "real_held_loss": round(statistics.mean(vals), 5),
                          "sd": round(statistics.pstdev(vals), 5) if seeds > 1 else 0.0})
        best = min(curve, key=lambda c: c["real_held_loss"])
        last = curve[-1]
        results[name] = {
            "drift": drift,
            "curve": curve,
            "best_multiple": best["synth_multiple"],
            "best_loss": best["real_held_loss"],
            "collapsed": last["real_held_loss"] > best["real_held_loss"] + 0.02,
            "collapse_note": ("loss rose after the optimum -> synthetic past best hurts"
                              if last["real_held_loss"] > best["real_held_loss"] + 0.02
                              else "monotone-or-saturating in tested range"),
        }

    report = {
        "study": "synthetic-data scaling & collapse",
        "honesty_note": ("Toy. Drifted Markov source stands in for an imperfect synthesizer; "
                         "demonstrates the fidelity-bounded scaling lesson, not a recipe."),
        "analytic_floor_E": round(E, 5),
        "config": {"real_budget": real_budget, "synth_multiples": synth_multiples,
                   "epochs": epochs, "seeds": seeds},
        "results": results,
        "lesson": ("High-fidelity synthetic data scales and saturates near the floor; "
                   "low-fidelity synthetic data collapses the model once it dominates the "
                   "mix. Quantity cannot substitute for distributional fidelity."),
    }
    out = out or (HERE / "synthetic-scaling-latest.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = run(quick=args.quick, out=args.out)
    print(f"floor E = {r['analytic_floor_E']}")
    for name, blk in r["results"].items():
        print(f"  {name:14s} drift={blk['drift']} best_mult={blk['best_multiple']} "
              f"best_loss={blk['best_loss']} collapsed={blk['collapsed']}")


if __name__ == "__main__":
    main()
