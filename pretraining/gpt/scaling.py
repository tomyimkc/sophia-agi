# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reproduce the data-scaling law on the REAL from-scratch GPT.

``pretraining/scaling`` fits ``L(D) = E + A·D^-p`` on the 1-hidden-layer ``nano``
model, whose irreducible floor ``E`` is known in closed form. This module runs the
*same* pre-registered methodology on the actual GPT (idea: "reproduce the scaling
law on the real model"):

  1. train fresh GPTs on increasing token budgets ``D``,
  2. fit the law (floor = the uniform-prediction cross-entropy, a real upper bound),
  3. **pre-registered extrapolation** — fit on the smaller budgets, predict the
     largest held-out budget, and check predicted-vs-measured.

The schedule + fit reuse are dependency-free (tested in CI). Running it needs
torch (`run_scaling`), so it is gated and stamps ``canClaimAGI: false``. Honest
boundary: a real GPT on a tiny corpus is a *methodology* reproduction, never a
frontier-scaling claim — exactly the ``pretraining/`` charter.

    python -m pretraining.gpt.scaling --quick
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pretraining.gpt.tokenizer import ByteProvenanceTokenizer
from pretraining.scaling.fit import fit_with_floor, predict

HERE = Path(__file__).resolve().parent


def data_size_schedule(total: int, points: int = 5, min_frac: float = 0.15) -> "list[int]":
    """Geometric token-budget schedule from ``min_frac*total`` up to ``total``.

    Dependency-free and deterministic — the ``D`` axis of the scaling sweep.
    """
    points = max(2, points)
    lo = max(1, int(total * min_frac))
    if lo >= total:
        return [total]
    ratio = (total / lo) ** (1.0 / (points - 1))
    sizes = sorted({min(total, int(lo * ratio ** i)) for i in range(points)})
    if sizes[-1] != total:
        sizes.append(total)
    return sizes


def run_scaling(*, quick: bool = False, points: int = 5, steps: int = 400,
                seed: int = 0, born_gated: bool = False) -> dict:
    """Train at each budget, fit the law, run the pre-registered extrapolation."""
    from pretraining.gpt.model import estimate_loss_floor  # noqa: PLC0415 — torch-gated
    from pretraining.gpt.train import token_stream, train  # noqa: PLC0415

    tok = ByteProvenanceTokenizer()
    if born_gated:
        from pretraining.gpt.born_gated import born_gated_token_stream
        stream = born_gated_token_stream(tok)
    else:
        stream = token_stream(tok)

    if quick:
        points, steps = 3, 40

    sizes = data_size_schedule(len(stream), points=points)
    floor = estimate_loss_floor(tok.vocab_size)

    measured: list[dict] = []
    for d in sizes:
        rep = train(quick=quick, steps=steps, prefer="cpu", seed=seed,
                    ids=stream[:d])
        measured.append({"D": d, "val_loss": rep["val_loss"]})

    xs = [m["D"] for m in measured]
    ys = [m["val_loss"] for m in measured]

    full_fit = fit_with_floor(xs, ys, floor)
    # Pre-registered: fit on all-but-largest, predict the largest, check error.
    held = xs[-1]
    part_fit = fit_with_floor(xs[:-1], ys[:-1], floor)
    pred = predict(part_fit, held)
    measured_held = ys[-1]
    rel_err = abs(pred - measured_held) / max(abs(measured_held), 1e-9)

    return {
        "canClaimAGI": False,
        "boundary": "scaling-law METHODOLOGY reproduced on a real GPT over a tiny "
                    "corpus — not a frontier-scaling claim; floor is the uniform "
                    "upper bound, not an analytic irreducible loss.",
        "born_gated": born_gated,
        "vocab_size": tok.vocab_size,
        "uniform_loss_floor_nats": round(floor, 4),
        "points": measured,
        "fit_all": full_fit,
        "extrapolation": {
            "fit_on": xs[:-1], "held_out_D": held,
            "predicted": round(pred, 4), "measured": round(measured_held, 4),
            "rel_error": round(rel_err, 4), "passes_10pct_gate": rel_err <= 0.10,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce the scaling law on the real GPT.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--born-gated", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    try:
        report = run_scaling(quick=args.quick, points=args.points, steps=args.steps,
                             seed=args.seed, born_gated=args.born_gated)
    except ImportError as exc:
        print(f"[gpt.scaling] {exc}")
        return 2

    print(json.dumps({"points": report["points"],
                      "extrapolation": report["extrapolation"]},
                     indent=2, ensure_ascii=False))
    if args.report:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (HERE / "gpt-scaling-latest.json").write_text(
            json.dumps({**report, "generatedAt": stamp}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"[gpt.scaling] wrote {HERE / 'gpt-scaling-latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
