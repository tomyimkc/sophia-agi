#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Optimizer dynamics & stability probe on the nano LM.

The algorithm direction's "设计高性能和鲁棒的优化器，研究模型训练过程中的动力学和稳定性问题"
in miniature, but with real gradients. Trains the *same* model/data under SGD, momentum,
and Adam across a learning-rate grid, and reports — per (optimizer, lr):

  * final held-out loss (performance),
  * whether training diverged (robustness),
  * max gradient-norm and a spike count (stability of the dynamics),
  * the loss curve (so the descent shape is inspectable).

The headline is the *stability frontier*: the largest lr each optimizer tolerates before
diverging, and how close to the entropy floor it gets at its best lr. Pure stdlib.

    python -m pretraining.optimizer_probe.run_probe --quick
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pretraining.nano import (
    NanoLM, eval_loss, make_source, sample_stream, source_entropy, to_examples, train,
)

HERE = Path(__file__).resolve().parent


def _spikes(grad_norms: "list[float]", factor: float = 3.0) -> int:
    """Count steps where the grad norm jumps > factor× the running median — a proxy
    for instability / loss spikes during training."""
    if len(grad_norms) < 10:
        return 0
    med = statistics.median(grad_norms)
    if med <= 0:
        return 0
    return sum(1 for g in grad_norms if g > factor * med)


def run(*, quick: bool = False, out: Path | None = None) -> dict:
    vocab, order, context, hidden = 8, 2, 2, 16
    epochs = 8 if quick else 12
    src = make_source(vocab=vocab, order=order, seed=1, peak=3.0)
    E = source_entropy(src)
    ex = to_examples(sample_stream(src, 1600 + context, seed=2), context=context)
    held = to_examples(sample_stream(src, 1200, seed=99), context=context)

    lrs = [0.01, 0.05, 0.2] if quick else [0.005, 0.01, 0.05, 0.1, 0.3]
    optimizers = ["sgd", "momentum", "adam"]

    cells = []
    for opt in optimizers:
        best = None
        for lr in lrs:
            m = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=0)
            hist = train(m, ex, epochs=epochs, optimizer=opt, lr=lr, seed=0)
            held_loss = eval_loss(m, held)
            cell = {
                "optimizer": opt, "lr": lr,
                "final_held_loss": round(held_loss, 5),
                "excess_over_floor": round(held_loss - E, 5),
                "diverged": hist["diverged"] or held_loss != held_loss,  # NaN check
                "max_grad_norm": round(hist["max_grad_norm"], 4),
                "grad_spikes": _spikes(hist["grad_norms"]),
                "loss_curve": [round(x, 4) for x in hist["epoch_loss"]],
            }
            cells.append(cell)
            if not cell["diverged"] and (best is None or held_loss < best["final_held_loss"]):
                best = cell
        # annotate the stability frontier for this optimizer
        stable_lrs = [c["lr"] for c in cells if c["optimizer"] == opt and not c["diverged"]]
        if best is not None:
            best["is_best_for_optimizer"] = True
            best["max_stable_lr"] = max(stable_lrs) if stable_lrs else None

    report = {
        "study": "optimizer dynamics & stability probe (nano LM)",
        "honesty_note": ("Toy scale. Demonstrates a stability/performance frontier with "
                         "real measured gradients, not a claim about frontier optimizers."),
        "analytic_floor_E": round(E, 5),
        "config": {"vocab": vocab, "context": context, "hidden": hidden,
                   "epochs": epochs, "lrs": lrs},
        "cells": cells,
        "summary": [
            {"optimizer": opt,
             "best_lr": next((c["lr"] for c in cells
                              if c["optimizer"] == opt and c.get("is_best_for_optimizer")), None),
             "best_held_loss": next((c["final_held_loss"] for c in cells
                                     if c["optimizer"] == opt and c.get("is_best_for_optimizer")), None),
             "max_stable_lr": max([c["lr"] for c in cells
                                   if c["optimizer"] == opt and not c["diverged"]] or [None])}
            for opt in optimizers
        ],
    }
    out = out or (HERE / "optimizer-probe-latest.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = run(quick=args.quick, out=args.out)
    print(f"floor E = {rep['analytic_floor_E']}")
    for s in rep["summary"]:
        print(f"  {s['optimizer']:9s} best_lr={s['best_lr']} "
              f"best_held={s['best_held_loss']} max_stable_lr={s['max_stable_lr']}")


if __name__ == "__main__":
    main()
