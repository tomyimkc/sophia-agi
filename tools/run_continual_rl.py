#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drive the Continual-Governed-RL loop (Phase 1 wiring).

Falsifiable claim (OFFLINE, CI-gated, runs anywhere incl. Apple Silicon):
  the loop composes the real seams end-to-end — model generation
  (agent.model, mock backend offline) → verifier-as-reward
  (provenance_bench.rl_reward) → fail-closed admission (gate + OKF grounding +
  staleness) → trainer step — and ONLY verified, grounded, fresh trajectories are
  admitted toward an update. A purely fabricating policy is admitted 0 times.

This does NOT assert a real model eval gain — the optimizer is a synthetic skill
proxy and no weights are trained. The live GPU GRPO update stays gated (see
tools/run_rlvr.py / docs/11-Platform/Continual-Governed-RL.md).

    python tools/run_continual_rl.py --check              # offline invariants (CI)
    python tools/run_continual_rl.py --demo --rounds 60   # show admit-rate rising
    SOPHIA_MODEL_PROVIDER=deepseek python tools/run_continual_rl.py --demo --live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import continual_rl  # noqa: E402
from provenance_bench.continual_rl import (  # noqa: E402
    ContinualGovernedLoop,
    model_generate_fn,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="run offline invariants")
    ap.add_argument("--demo", action="store_true", help="run the loop and print a report")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--live", action="store_true",
                    help="use the real model adapter (provider from env; mock otherwise)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if not (args.check or args.demo):
        args.check = True

    rc = 0
    if args.check:
        ok, detail = continual_rl.offline_invariants()
        print("continual-RL offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        rc |= 0 if ok else 1

    if args.demo:
        gen = None
        if args.live:
            provider = os.environ.get("SOPHIA_MODEL_PROVIDER", "mock")
            print(f"\n[live] generating via agent.model provider={provider!r} "
                  f"(mock if no keys configured)")
            gen = model_generate_fn()
        loop = ContinualGovernedLoop(
            continual_rl._CASES, continual_rl._RECORDS,
            generate_fn=gen, group_size=6, batch_size=12, seed=args.seed)
        rep = loop.run(args.rounds)
        print(f"\nContinual-Governed-RL over {args.rounds} rounds:")
        s = rep.admit_stats
        print(f"  generated        {rep.generated}")
        print(f"  admitted         {rep.admitted}")
        print(f"  rejected         ungated={s['ungated']} "
              f"ungrounded={s['ungrounded']} low_reward={s['low_reward']}")
        print(f"  train steps      {rep.train_steps}")
        print(f"  admit-rate       {rep.early_admit_rate} → {rep.late_admit_rate} "
              f"(early → late)")
        print(f"  policy skill     → {rep.final_skill}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
