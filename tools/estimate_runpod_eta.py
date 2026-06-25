#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Estimate wall-clock for a RunPod SFT run so you know how long to wait.

This is a transparent HEURISTIC, not a guarantee. The train-time rate is
calibrated on the two real Qwen2.5-7B QLoRA-4bit SFT runs observed on 2026-06-25
(seed1/seed2: ~27-30 min total job each, incl. ~6 min provisioning + ~5 min eval
ladder on ~754 train rows x 2 epochs). GPU provisioning time and RunPod capacity
dominate the variance, so the output is a RANGE, not a point.

Wall-clock per execution mode (N = number of seeds):
  separate-pods         provision + train + eval           (N pods in parallel; capacity permitting)
  on-pod-parallel       provision + train                  (1 multi-GPU pod, one seed per GPU; no on-pod eval)
  on-pod-sequential     provision + N * train              (1 pod, seeds back-to-back; no on-pod eval)
  all-seeds-sequential  N * (provision + train + eval)     (the per-seed loop: one pod at a time)
  single                provision + train + eval           (one seed, one pod)

    python3 tools/estimate_runpod_eta.py --model-params 7 --epochs 2 \\
      --train-data training/local_sophia_7b/mlx/train.jsonl --seeds 0,1,2 \\
      --mode on-pod-parallel --markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Calibration (minutes). Heuristic — see module docstring.
PROVISION_MIN = 6.0          # apt/pip/model-pull + SSH bring-up before training
EVAL_MIN = 5.0               # eval_ladder + promote_adapter (skipped by --no-eval / on-pod modes)
# Train minutes per (1e9 params) x (1000 rows) x epoch. 7B x 0.754k x 2ep ~= 16 train min -> ~1.5.
TRAIN_RATE_MIN = 1.5
LOW_FACTOR = 0.7             # optimistic (warm cache, instant GPU)
HIGH_FACTOR = 1.8            # pessimistic (cold pull, slow/scarce GPU, retries)

MODES = ("separate-pods", "on-pod-parallel", "on-pod-sequential",
         "all-seeds-sequential", "single")
_ON_POD = {"on-pod-parallel", "on-pod-sequential"}


def _count_rows(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def estimate(
    *,
    model_params_b: float,
    seeds: int,
    mode: str,
    epochs: int,
    rows: int,
    include_eval: bool = True,
    provision_min: float = PROVISION_MIN,
    eval_min: float = EVAL_MIN,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    seeds = max(1, int(seeds))
    train_one = TRAIN_RATE_MIN * model_params_b * (rows / 1000.0) * max(1, epochs)
    # on-pod modes return the adapter only (no on-pod eval/promote)
    eval_part = eval_min if (include_eval and mode not in _ON_POD) else 0.0

    if mode in ("separate-pods", "single"):
        point = provision_min + train_one + eval_part
    elif mode == "on-pod-parallel":
        point = provision_min + train_one
    elif mode == "on-pod-sequential":
        point = provision_min + seeds * train_one
    elif mode == "all-seeds-sequential":
        point = seeds * (provision_min + train_one + eval_part)
    else:  # pragma: no cover - guarded above
        raise ValueError(mode)

    low, high = point * LOW_FACTOR, point * HIGH_FACTOR
    return {
        "modelParamsB": model_params_b,
        "seeds": seeds,
        "mode": mode,
        "epochs": epochs,
        "rows": rows,
        "trainOneMin": round(train_one, 1),
        "provisionMin": provision_min,
        "evalMin": eval_part,
        "wallMin": {"point": round(point, 1), "low": round(low, 1), "high": round(high, 1)},
        "note": "heuristic ETA (calibrated on 2 observed 7B QLoRA runs); GPU provisioning dominates variance — treat as a range, not a guarantee",
    }


def _fmt(minutes: float) -> str:
    m = int(round(minutes))
    h, mm = divmod(m, 60)
    return f"{h}h{mm:02d}m" if h else f"{mm}m"


def to_markdown(est: dict, start: datetime | None) -> str:
    w = est["wallMin"]
    lines = [
        "## ⏱️ RunPod SFT — estimated wall-clock",
        "",
        f"- **Mode:** `{est['mode']}` · **seeds:** {est['seeds']} · "
        f"**model:** {est['modelParamsB']}B · **epochs:** {est['epochs']} · **rows:** {est['rows']}",
        f"- **Estimate:** **~{_fmt(w['point'])}**  (range {_fmt(w['low'])} – {_fmt(w['high'])})",
        f"- Per-seed train ~{_fmt(est['trainOneMin'])}; provisioning ~{_fmt(est['provisionMin'])}"
        + (f"; eval ~{_fmt(est['evalMin'])}" if est["evalMin"] else "; on-pod eval skipped"),
    ]
    if start is not None:
        eta = start + timedelta(minutes=w["point"])
        eta_lo = start + timedelta(minutes=w["low"])
        eta_hi = start + timedelta(minutes=w["high"])
        lines += [
            f"- **Expected finish:** ~{eta:%Y-%m-%d %H:%M UTC} "
            f"(between {eta_lo:%H:%M} and {eta_hi:%H:%M} UTC)",
        ]
    lines += ["", f"> {est['note']}", ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-params", type=float, default=7.0, help="model size in billions (e.g. 7)")
    ap.add_argument("--seeds", default="0", help="comma seeds (count is what matters), e.g. 0,1,2")
    ap.add_argument("--mode", choices=MODES, default="separate-pods")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rows", type=int, default=0, help="train rows; 0 = count from --train-data")
    ap.add_argument("--train-data", type=Path, default=None, help="JSONL to count rows from when --rows 0")
    ap.add_argument("--no-eval", action="store_true", help="train-only run (no on-pod eval/promote)")
    ap.add_argument("--start-iso", default="", help="ISO start time for expected-finish (default: now UTC)")
    ap.add_argument("--markdown", action="store_true", help="emit GitHub-friendly markdown (else JSON)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = args.rows
    if rows <= 0 and args.train_data and args.train_data.exists():
        rows = _count_rows(args.train_data)
    rows = max(1, rows)
    seeds = len([s for s in args.seeds.split(",") if s.strip() != ""]) or 1
    est = estimate(
        model_params_b=args.model_params,
        seeds=seeds,
        mode=args.mode,
        epochs=args.epochs,
        rows=rows,
        include_eval=not args.no_eval,
    )
    if args.markdown:
        if args.start_iso:
            start = datetime.fromisoformat(args.start_iso)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        else:
            start = datetime.now(timezone.utc)
        print(to_markdown(est, start))
    else:
        print(json.dumps(est, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
