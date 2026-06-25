#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Architecture probe: top-1 MoE vs dense at matched ACTIVE compute (nano LM).

The sparse-scaling question — "探索开拓性的新型模型结构" — at toy scale: does routing tokens
to specialized experts beat a single dense block that uses the *same* per-token compute?
We compare a dense model of width ``h`` against a ``k``-expert MoE whose experts are each
width ``h`` (so active params per token are comparable, total params are ~k×). We also
report routing load-balance, since collapse (all tokens to one expert) is the classic
MoE failure mode this probe is meant to surface.

Honest: this is a reference experiment about routing *behavior*, not a SOTA claim. The
companion ``ARCHITECTURE.md`` documents the real DeepSeek MLA + fine-grained-MoE design
this toy gestures at. Pure stdlib.

    python -m pretraining.architecture.run_arch --quick
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pretraining.architecture.moe import MoELM
from pretraining.nano import (
    NanoLM, eval_loss, make_source, sample_stream, source_entropy, to_examples, train,
)

HERE = Path(__file__).resolve().parent


def _train_moe(m: MoELM, examples, epochs, lr, seed):
    rng = random.Random(seed)
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for j in order:
            ctx, t = examples[j]
            m.train_step(ctx, t, lr)


def _eval_moe(m: MoELM, examples):
    return sum(m.nll(c, t) for c, t in examples) / max(1, len(examples))


def run(*, quick: bool = False, out: Path | None = None) -> dict:
    # Use an order-2 source so multiple sub-structures exist for experts to specialize on.
    vocab, order, context, hidden = 8, 2, 2, 8
    epochs = 8 if quick else 14
    n_experts = 3
    src = make_source(vocab=vocab, order=order, seed=1, peak=3.0)
    E = source_entropy(src)
    ex = to_examples(sample_stream(src, 2000 + context, seed=2), context=context)
    held = to_examples(sample_stream(src, 1200, seed=99), context=context)

    # Dense baseline (SGD to match MoE's SGD router updates fairly).
    dense = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=0)
    train(dense, ex, epochs=epochs, optimizer="sgd", lr=0.1, seed=0)
    dense_loss = eval_loss(dense, held)

    # MoE with k experts of the same width.
    moe = MoELM(vocab=vocab, context=context, hidden=hidden, n_experts=n_experts, seed=0)
    _train_moe(moe, ex, epochs, 0.1, 0)
    moe_loss = _eval_moe(moe, held)

    report = {
        "study": "architecture probe — top-1 MoE vs dense (nano LM)",
        "honesty_note": ("Toy. Studies routing behavior/load-balance, not a SOTA claim. "
                         "See ARCHITECTURE.md for the real DeepSeek MLA + MoE design."),
        "analytic_floor_E": round(E, 5),
        "config": {"vocab": vocab, "context": context, "hidden": hidden,
                   "n_experts": n_experts, "epochs": epochs},
        "dense": {"params": dense.num_params(),
                  "active_params": dense.num_params(),
                  "held_loss": round(dense_loss, 5),
                  "excess_over_floor": round(dense_loss - E, 5)},
        "moe": {"total_params": moe.num_params(),
                "active_params_per_token": moe.active_params(),
                "held_loss": round(moe_loss, 5),
                "excess_over_floor": round(moe_loss - E, 5),
                "load_balance_max_share": round(moe.load_balance(), 4),
                "route_counts": moe.route_counts,
                "balanced": moe.load_balance() < 0.6},
        "verdict": ("moe_better" if moe_loss < dense_loss - 1e-3
                    else "dense_better" if dense_loss < moe_loss - 1e-3 else "tie"),
    }
    out = out or (HERE / "arch-probe-latest.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = run(quick=args.quick, out=args.out)
    print(f"floor E = {r['analytic_floor_E']}")
    print(f"dense: held={r['dense']['held_loss']} params={r['dense']['params']}")
    print(f"moe  : held={r['moe']['held_loss']} total={r['moe']['total_params']} "
          f"active/token={r['moe']['active_params_per_token']} "
          f"balance(max share)={r['moe']['load_balance_max_share']}")
    print(f"verdict: {r['verdict']}")


if __name__ == "__main__":
    main()
