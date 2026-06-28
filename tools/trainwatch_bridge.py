#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bridge a sophia ``tools/train_lora.py`` log into TrainWatch (~/.trainwatch).

``train_lora.py`` emits a custom progress format (``epoch E/N step S/T (..%) loss=.. lr=..``
and ``[eval] step S val_loss=.. train_loss=..``) that TrainWatch's tqdm-oriented parser does
not read cleanly (it would latch onto the checkpoint-shard ``3/3`` bar instead of the real
training steps). This bridge parses the sophia format directly and feeds TrainWatch's
``init`` / ``run.log`` API so every training gets a real step/ETA/loss/val_loss dashboard.

Usage (point it at any sophia train log; backfills then live-follows):

    python tools/trainwatch_bridge.py training/lora/train_v3.full.log --name olmoe-qat-v3

Then view: ``trainwatch serve`` → http://<host>:8420 (over Tailscale too). Reusable for every
run — one bridge per training log. Zero deps beyond TrainWatch itself.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# train_lora step line:  "epoch 2/2 step 200/220 (90.9%) loss=0.5418 lr=1.93e-06"
_STEP = re.compile(r"step\s+(\d+)/(\d+).*?loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                   r"(?:.*?lr=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?")
# eval line:  "  [eval] step 150 val_loss=1.5012 train_loss=0.5373 val/train=2.79"
_EVAL = re.compile(r"\[eval\]\s+step\s+(\d+)\s+val_loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                   r"(?:\s+train_loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?")
# completion markers train_lora / certify chains write
_DONE = re.compile(r"\.train_complete|saved adapter|VERDICT:|=== certify done|training finished")


def _emit(run, step, metrics):
    if step > 0:
        run.log(metrics, step=step)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("logfile")
    ap.add_argument("--name", required=True, help="TrainWatch run name (e.g. olmoe-qat-v3)")
    ap.add_argument("--description", default="")
    ap.add_argument("--total", type=int, default=None, help="Total steps (else auto from S/T)")
    ap.add_argument("--idle-exit", type=float, default=180.0,
                    help="Stop following after this many seconds with no new step + a done marker")
    args = ap.parse_args(argv)

    import trainwatch
    from trainwatch import db

    path = Path(args.logfile)
    for _ in range(60):                       # tolerate being launched slightly before the log
        if path.exists():
            break
        time.sleep(1)
    if not path.exists():
        print(f"no such log: {path}", file=sys.stderr)
        return 2

    # Pre-scan for the total so the dashboard progress % is correct from creation (the DB row's
    # total_steps is set at create_run; assigning run.total_steps later only affects in-proc ETA).
    total = args.total
    if total is None:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _STEP.search(line)
            if m:
                total = int(m.group(2))
                break

    run = trainwatch.init(name=args.name, description=args.description or f"sophia: {path.name}",
                          total_steps=total)
    last_step = 0
    saw_done = False
    last_change = time.time()
    pos = 0
    while True:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        for line in chunk.splitlines():
            m = _STEP.search(line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                if run.total_steps != total:
                    run.total_steps = total
                    db.update_run(run.id, total_steps=total)   # persist for dashboard progress
                metrics = {"loss": float(m.group(3))}
                if m.group(4):
                    metrics["lr"] = float(m.group(4))
                _emit(run, step, metrics)
                last_step = max(last_step, step)
                last_change = time.time()
                continue
            e = _EVAL.search(line)
            if e:
                step = int(e.group(1))
                metrics = {"val_loss": float(e.group(2))}
                if e.group(3):
                    metrics["train_loss"] = float(e.group(3))
                _emit(run, step, metrics)
                last_change = time.time()
                continue
            if _DONE.search(line):
                saw_done = True
                last_change = time.time()
        if saw_done and (time.time() - last_change) > args.idle_exit:
            run.finish("completed")
            print(f"[trainwatch_bridge] {args.name}: done at step {last_step}")
            return 0
        time.sleep(3)


if __name__ == "__main__":
    raise SystemExit(main())
