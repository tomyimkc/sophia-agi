#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the nano data-scaling law: measure L(D), fit it, and test a pre-registered prediction.

The scientific loop DeepSeek's algorithm direction calls "建立科学的 scaling law … 进行
合理的预测和规划":

  1. Measure held-out loss across a sweep of training-set sizes D (averaged over seeds).
  2. Fit L(D) = E + A·D^-p with the floor UNKNOWN, then check the recovered floor E
     against the analytic source entropy (the known irreducible loss). Recovering it is
     the honesty test — the curve isn't trusted unless the floor lands right.
  3. PRE-REGISTERED PREDICTION: fit only on the small/mid sizes, extrapolate to a larger
     held-out size, then run that size and report predicted-vs-measured. Planning works
     only if extrapolation does.

Pure stdlib. Writes a JSON report to pretraining/scaling/ (or --out). Run:

    python -m pretraining.scaling.run_scaling --quick      # fast, fewer seeds
    python -m pretraining.scaling.run_scaling              # default study
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pretraining.nano import (
    NanoLM,
    eval_loss,
    make_source,
    sample_stream,
    source_entropy,
    to_examples,
    train,
)
from pretraining.scaling.fit import fit_free_floor, fit_with_floor, predict

HERE = Path(__file__).resolve().parent


def measure_D(src: dict, D: int, *, context: int, hidden: int, epochs: int,
              lr: float, seeds: int, held) -> "dict":
    vals = []
    for s in range(seeds):
        ex = to_examples(sample_stream(src, D + context, seed=10 + s), context=context)
        m = NanoLM(vocab=src["vocab"], context=context, hidden=hidden, seed=s)
        train(m, ex, epochs=epochs, optimizer="adam", lr=lr, seed=s)
        vals.append(eval_loss(m, held))
    mean = statistics.mean(vals)
    return {"D": D, "loss": round(mean, 5),
            "sd": round(statistics.pstdev(vals), 5) if seeds > 1 else 0.0,
            "seeds": seeds}


def run(*, quick: bool = False, out: Path | None = None) -> dict:
    vocab, order, context, hidden = 8, 2, 2, 16
    epochs = 12 if quick else 15
    seeds = 2 if quick else 3
    lr = 0.03
    sizes = [200, 400, 800, 1600] + ([] if quick else [3200])
    holdout_size = 1600

    src = make_source(vocab=vocab, order=order, seed=1, peak=3.0)
    E_true = source_entropy(src)
    held = to_examples(sample_stream(src, holdout_size, seed=99), context=context)

    points = [measure_D(src, D, context=context, hidden=hidden, epochs=epochs,
                        lr=lr, seeds=seeds, held=held) for D in sizes]

    xs = [p["D"] for p in points]
    ls = [p["loss"] for p in points]

    # Fit on ALL points, floor free → check recovered E against analytic entropy.
    free = fit_free_floor(xs, ls)
    floor_err = abs(free["E"] - E_true)
    # Identifiability diagnostic: the irreducible loss E cannot be recovered from data
    # that never approaches saturation. We flag this honestly via the smallest measured
    # excess-loss ratio — if the curve is still far above the floor everywhere, a free
    # 3-parameter fit trades E off against (A, p) and collapses E toward 0. This is a
    # real property of scaling-law fitting, not a bug: labs pin E by fitting jointly
    # across runs that DO reach saturation.
    min_excess_ratio = (min(ls) - E_true) / E_true if E_true else float("nan")
    floor_identified = floor_err <= 0.15 * E_true

    # Pre-registered prediction: fit on all-but-largest, predict the largest.
    pred_block = None
    if len(xs) >= 4:
        fit_small = fit_with_floor(xs[:-1], ls[:-1], E_true)
        target_D = xs[-1]
        predicted = predict(fit_small, target_D)
        measured = ls[-1]
        rel_err = abs(predicted - measured) / measured if measured else float("nan")
        pred_block = {
            "fit_on_D": xs[:-1],
            "target_D": target_D,
            "predicted_loss": round(predicted, 5),
            "measured_loss": round(measured, 5),
            "relative_error": round(rel_err, 5),
            "passes_10pct_gate": rel_err <= 0.10,
            "fit_small": {k: round(v, 5) if isinstance(v, float) else v
                          for k, v in fit_small.items()},
        }

    report = {
        "study": "nano data-scaling law L(D) = E + A·D^-p",
        "honesty_note": ("Toy pure-Python LM on a synthetic order-2 Markov source. The "
                         "point is verifiable METHODOLOGY (known floor, pre-registered "
                         "prediction), not a frontier scaling law."),
        "config": {"vocab": vocab, "order": order, "context": context,
                   "hidden": hidden, "epochs": epochs, "seeds": seeds, "lr": lr,
                   "optimizer": "adam", "holdout_size": holdout_size},
        "analytic_floor_E": round(E_true, 5),
        "points": points,
        "fit_free_floor": {k: round(v, 5) if isinstance(v, float) else v
                           for k, v in free.items()},
        "recovered_floor_error": round(floor_err, 5),
        "recovered_floor_within_15pct": floor_identified,
        "min_excess_ratio": round(min_excess_ratio, 5),
        "floor_identifiability": (
            "identified" if floor_identified else
            "UNDER-IDENTIFIED: data has not approached saturation (min excess/floor = "
            f"{round(min_excess_ratio, 3)}); free-floor fit collapses E toward 0. "
            "Use the known-floor fit + the pre-registered prediction gate as the result. "
            "This is the headline lesson: you cannot read off the irreducible loss "
            "without runs near the floor."),
        "prediction": pred_block,
    }

    out = out or (HERE / "scaling-curve-latest.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer seeds/sizes for speed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = run(quick=args.quick, out=args.out)
    print(f"analytic floor E      = {rep['analytic_floor_E']}")
    print(f"recovered floor E     = {rep['fit_free_floor']['E']} "
          f"(err {rep['recovered_floor_error']}, "
          f"within15%={rep['recovered_floor_within_15pct']})")
    print(f"fitted exponent p     = {rep['fit_free_floor']['p']}  "
          f"r2(logspace)={rep['fit_free_floor']['r2_logspace']}")
    if rep["prediction"]:
        pr = rep["prediction"]
        print(f"pre-registered predict D={pr['target_D']}: "
              f"pred={pr['predicted_loss']} measured={pr['measured_loss']} "
              f"relerr={pr['relative_error']} pass10%={pr['passes_10pct_gate']}")


if __name__ == "__main__":
    main()
