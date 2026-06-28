#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fire the Path A world-model experiment: mine harness traces -> train a
DreamerV3-style discrete-latent predictor -> run the shift-degeneracy canary.

The load-bearing question: does a neural dynamics model generalize to held-out AND
distribution-shifted (state, action) pairs, or collapse to memorization? See
docs/06-Roadmap/Two-Paths-To-Novelty.md.

    python tools/run_world_model.py                                  # CI/dry: mines traces, trains if torch present, runs canary
    python tools/run_world_model.py --runs-dir agent/memory/agent_runs --out agi-proof/world-model/dreamer.json
    python tools/run_world_model.py --epochs 120 --shift-states novel-task   # GPU run with more capacity

Fail-closed: without torch, the predictor abstains and the report records it (the
canary is the experiment; this driver only assembles its inputs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import trace_mining, verified_world_model as vwm  # noqa: E402
from agent.world_model_dreamer import DreamerConfig, train_dreamer_report, write_dreamer_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "agent" / "memory" / "agent_runs")
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "world-model" / "dreamer.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--classes", type=int, default=16, help="discrete-latent categories per stochastic dim")
    ap.add_argument("--stoch", type=int, default=16, help="number of stochastic dims (DreamerV3 uses 32)")
    ap.add_argument("--val-bar", type=float, default=0.65, help="held-out accuracy bar for promote")
    ap.add_argument("--max-shift-deg", type=float, default=0.15, help="max held-out - shift degradation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1. Mine the corpus.
    pairs = trace_mining.mine_dir(args.runs_dir)
    info = trace_mining.corpus_report(pairs)
    print(json.dumps({"mined": info}, indent=2, ensure_ascii=False))
    if len(pairs) < 20:
        print(f"\nOnly {len(pairs)} traces mined — too few to train. Run the harness on more tasks first.")
        return 1

    # 2. Split (train / val / shift). shift = traces whose state-bucket wasn't in train.
    splits = vwm.make_splits(pairs, seed=args.seed)
    print(f"splits: train={len(splits.train)} val={len(splits.val)} shift={len(splits.shift)}")

    # 3. Train the DreamerV3-style predictor (abstains if no torch).
    cfg = DreamerConfig(epochs=args.epochs, hidden=args.hidden, classes=args.classes,
                        stoch=args.stoch, seed=args.seed)
    pred, dreamer_rep = train_dreamer_report(splits.train, val_traces=splits.val, cfg=cfg)
    write_dreamer_report(dreamer_rep, args.out)
    print(f"\nDreamer training report -> {args.out}")
    print(json.dumps({k: dreamer_rep.to_dict()[k] for k in ("torchAvailable", "cudaAvailable", "trained", "valAccuracy", "trainLoss")}, indent=2))

    if not dreamer_rep.trained:
        print("\ntorch not available (CPU/CUDA) — predictor abstained. Install torch to fire the experiment.")
        return 0

    # 4. THE CANARY: promote only on held-out gain + bounded shift-degradation.
    report = vwm.train_verified_world_model(
        pairs, predictor_factory=lambda: pred, splits=splits,
        val_bar=args.val_bar, max_shift_degradation=args.max_shift_deg, seed=args.seed,
    )
    print(f"\n{'='*60}\nVERIFIED-WORLD-MODEL CANARY\n{'='*60}")
    print(f"verdict      : {report.verdict}")
    print(f"val accuracy : {report.val_accuracy:.4f}  (bar {args.val_bar})")
    print(f"shift accuracy: {report.shift_accuracy:.4f}")
    print(f"shift degrad. : {report.shift_degradation:.4f}  (max {args.max_shift_deg})")
    print(f"reason       : {report.reason}")
    print(f"\n{report.to_dict()['interpretation']}")
    # Non-zero for any non-promote verdict: every non-promote outcome (hold-below-bar,
    # hold-shift-degenerate) is a NOT-promoted canary, so a failed canary must look like
    # failure in scripts/CI (exit 0 only when the model is actually promoted).
    return 0 if report.verdict == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
