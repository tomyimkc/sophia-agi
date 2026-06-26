#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the multimodal RLVR (visual-grounding-as-reward) GRPO experiment.

Falsifiable claim, in two tiers (mirrors tools/run_rlvr.py):

  OFFLINE (asserted here, CI-gated, any machine): the visual reward machinery is
    sound — deterministic, bounded in [-1, 1], the honesty ordering
    correct > abstain > wrong holds, the judge-free verifier seam is invoked on
    every call, and the train/eval split is contamination-free (family-disjoint).

  LIVE (pre-registered OPEN in agi-proof/failure-ledger.md
    'visual-rlvr-live-run-not-yet-gated'): on the held-out family-disjoint split, a
    VLM trained with the visual-grounding reward raises grounding / lowers
    hallucination vs the untrained base, under the no-overclaim gate (>=2 judge
    families, kappa>=0.40, >=3 runs, 95% bootstrap CI excludes 0). This run does
    NOT assert that claim.

Hardware & scope: GRPO over a VLM is CUDA-only AND needs a vision-capable GRPO
trainer (the dense-LM GRPO path in run_rlvr.py does not ingest images). This tool
proves the reward wiring offline and *prepares* the live run — it builds the
family-disjoint dataset with rendered images and writes a config-only report —
but the VLM-GRPO trainer is intentionally not bundled; the live run stays OPEN in
the ledger until a gated VLM run lands.

    python tools/run_visual_rlvr.py                 # offline invariants (CI / Mac)
    python tools/run_visual_rlvr.py --prepare       # build the live dataset + report
    python tools/run_visual_rlvr.py --gpu --model <vlm>   # refuses: needs CUDA + VLM-GRPO
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import visual_dataset, visual_reward  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "visual-rlvr.public-report.json"


def _prepare(args) -> dict:
    """Build the family-disjoint RL dataset and (optionally) render its images."""
    data = visual_dataset.build_visual_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    rendered, render_blocker = 0, None
    if args.render:
        try:
            from multimodal_bench.render import render_png
            for row in data["train_rows"]:
                render_png(row["trap"]["scene"])  # exercise the rasteriser
                rendered += 1
        except Exception as exc:  # Pillow absent -> blocker, not a failure
            render_blocker = f"{type(exc).__name__}:{exc}"
    return {
        "benchmark": "visual-rlvr",
        "visibility": "public-aggregate",
        "claimStatus": ("Open — capability claim requires a gated VLM-GRPO run "
                        "(>=2 judge families, kappa>=0.40, >=3 runs, CI excludes 0); "
                        "this artifact records the prepared config only"),
        "ledgerItem": "visual-rlvr-live-run-not-yet-gated",
        "rewardSurface": visual_reward.DEFAULT_WEIGHTS,
        "trainRows": len(data["train_rows"]),
        "evalRows": len(data["eval_rows"]),
        "trainFamilies": data["train_families"],
        "evalFamilies": data["eval_families"],
        "familyIntersection": data["family_intersection"],
        "imagesRendered": rendered,
        "renderBlocker": render_blocker,
    }


def _gpu_refusal(args) -> int:
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception as exc:
        print(f"Install RL deps: pip install -r requirements-rl.txt ({type(exc).__name__}: {exc})")
        return 1
    if not cuda:
        print("CUDA GPU not detected. Visual GRPO is CUDA-only; use the offline "
              "invariants on this machine. See the ledger item visual-rlvr-live-run-not-yet-gated.")
        return 1
    print("REFUSED: a VLM-capable GRPO trainer is not bundled. The dense-LM GRPO path "
          "(tools/run_rlvr.py) does not ingest images, and wiring a full VLM-GRPO stack "
          "is the gated live step (ledger: visual-rlvr-live-run-not-yet-gated). Run "
          "--prepare to build the family-disjoint dataset + config report for that run.")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="build the live dataset + config report (no training)")
    ap.add_argument("--gpu", action="store_true", help="attempt the live VLM-GRPO run (currently refuses; see ledger)")
    ap.add_argument("--render", action="store_true", help="under --prepare, also rasterise train images (needs Pillow)")
    ap.add_argument("--model", default="mock", help="VLM id for the live run")
    ap.add_argument("--eval-frac", type=float, default=0.34)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    if args.gpu:
        return _gpu_refusal(args)

    ok, detail = visual_reward.offline_invariants()
    print("=== offline reward-machinery invariants ===")
    print(json.dumps(detail["checks"], indent=2))
    print(f"ALL INVARIANTS HOLD: {ok}")

    if args.prepare:
        report = _prepare(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
        print(f"prepared: train={report['trainRows']} eval={report['evalRows']} "
              f"families train={report['trainFamilies']} eval={report['evalFamilies']} "
              f"(intersection {report['familyIntersection']})")
        print(f"claimStatus: {report['claimStatus']}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
