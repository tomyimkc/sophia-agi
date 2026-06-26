#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drive the async-RL scheduling simulation and print the async-vs-sync report.

Falsifiable claim (OFFLINE, CI-gated, runs anywhere incl. Apple Silicon):
  the async (decoupled generation/training) loop sustains strictly higher
  trainer throughput than the synchronous barrier loop over the same horizon,
  while never training on a trajectory whose off-policy staleness exceeds the
  configured bound — and the GRPO group-advantages stay zero-mean throughout.

This does NOT assert that async RL improves a real model's eval score — the
policy here is a synthetic improvement proxy (see provenance_bench/async_rl.py).
The real GPU GRPO step lives in tools/run_rlvr.py behind the gated live path;
this tool measures the *scheduler*, which is the part that's hardware-independent.

    python tools/run_async_rl.py --check          # offline invariants (CI)
    python tools/run_async_rl.py --compare        # async vs sync table
    python tools/run_async_rl.py --reward provenance   # wire the real reward seam
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import async_rl  # noqa: E402


def _real_reward_smoke(group_size: int = 4) -> dict:
    """Prove the async actor composes with the repo's real verifier-as-reward.

    Uses provenance_bench.rl_reward (the same gate the synchronous RLVR run uses)
    as the actor's reward_fn, so the async scaffolding is shown to be a drop-in
    over the existing reward seam — not a parallel universe.
    """
    from provenance_bench import rl_reward

    grpo = rl_reward.make_grpo_reward(records={})

    def reward_fn(completion: str, case: dict) -> float:
        out = grpo(
            [case.get("prompt", "p")],
            [completion],
            label=case.get("label", "false"),
            gold_author=case.get("gold_author", ""),
            claimed_author=case.get("claimed_author"),
        )
        return out[0]

    def gen(prompt: str, version: int, rng: random.Random) -> str:
        # Two candidate behaviours; the real gate scores them deterministically.
        return rng.choice(
            ["The author is undocumented; I will not assert one.",
             "It was definitely written by Homer."]
        )

    actor = async_rl.RolloutActor(gen, reward_fn, group_size=group_size)
    case = {"label": "false", "gold_author": "", "claimed_author": "Homer",
            "prompt": "Who wrote it?"}
    grp = actor.rollout(0, case["prompt"], case, 0, random.Random(0))
    return {
        "rewards": [round(t.reward, 3) for t in grp],
        "advantages": [round(t.advantage, 3) for t in grp],
        "advantages_zero_mean": abs(sum(t.advantage for t in grp)) < 1e-6,
        "reward_seam": "provenance_bench.rl_reward.make_grpo_reward",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="run offline invariants")
    ap.add_argument("--compare", action="store_true", help="print async vs sync report")
    ap.add_argument("--reward", choices=["scripted", "provenance"], default="scripted")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--max-staleness", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not (args.check or args.compare) and args.reward == "scripted":
        args.check = args.compare = True  # default: do both

    rc = 0
    if args.check:
        ok, detail = async_rl.offline_invariants()
        print("async-RL offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        rc |= 0 if ok else 1

    if args.reward == "provenance":
        print("\nreal reward-seam smoke (provenance_faithful gate):")
        print(json.dumps(_real_reward_smoke(), indent=2))

    if args.compare:
        a = async_rl.simulate(mode="async", ticks=args.ticks,
                              max_staleness=args.max_staleness, seed=args.seed)
        s = async_rl.simulate(mode="sync", ticks=args.ticks,
                              max_staleness=args.max_staleness, seed=args.seed)
        print(f"\nasync vs sync over {args.ticks} ticks "
              f"(max_staleness={args.max_staleness}):")
        cols = ["train_steps", "trained_trajectories", "trainer_idle_ticks",
                "max_staleness_trained", "final_skill"]
        print(f"  {'metric':<24}{'async':>12}{'sync':>12}")
        for c in cols:
            print(f"  {c:<24}{getattr(a, c):>12}{getattr(s, c):>12}")
        speedup = (a.trained_trajectories / s.trained_trajectories
                   if s.trained_trajectories else float('inf'))
        print(f"  {'throughput x':<24}{speedup:>12.2f}{'1.00':>12}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
