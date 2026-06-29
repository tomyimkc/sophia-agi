#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Open-invention pack generator + instrument-validity report (novelty pillar).

Emits a depth-``k`` compositional-generalization pack (``provenance_bench.
invention_dataset``) whose EVAL compositions are absent from TRAIN while every
primitive is still seen in TRAIN — so solving eval requires *composing seen pieces
in an unseen way* (derivation), not recall. Prints the recall-vs-derivation
discrimination that proves the eval pass-rate measures invention, and verifies the
instrument's offline invariants.

    python tools/gen_invention_pack.py                      # report + GO/NO-GO
    python tools/gen_invention_pack.py --depth 3 --emit pack.json
    python tools/gen_invention_pack.py --check             # CI: invariants only

The instrument is reported as a CANDIDATE measurement device, not a model result —
no model has been run on it. canClaimAGI:false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import invention_dataset as inv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open-invention pack + discrimination report.")
    ap.add_argument("--depth", type=int, default=2, help="composition depth (1=recall, >=2=invention)")
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", default=None, help="write the pack JSON to this path")
    ap.add_argument("--check", action="store_true", help="CI: assert invariants, no output pack")
    args = ap.parse_args(argv)

    ok, detail = inv.offline_invariants(depth=args.depth, eval_frac=args.eval_frac, seed=args.seed)
    disc = detail["discrimination"]

    print(f"OPEN-INVENTION INSTRUMENT  depth={args.depth} seed={args.seed} "
          f"eval_frac={args.eval_frac}")
    print(f"  compositions: {detail['nTrain']} train / {detail['nEval']} eval (disjoint, "
          f"primitives all seen in train)")
    print(f"  recall-vs-derivation discrimination:")
    print(f"    memorizer: recall={disc['memorizer']['recall']:.2f}  "
          f"derivation={disc['memorizer']['derivation']:.2f}  "
          f"(gap={disc['memorizer']['recall_minus_derivation']:.2f})")
    print(f"    deriver:   recall={disc['deriver']['recall']:.2f}  "
          f"derivation={disc['deriver']['derivation']:.2f}")
    print(f"  -> eval pass-rate measures INVENTION, not recall: {disc['discriminates']}")
    for k, v in detail["checks"].items():
        print(f"     {'ok ' if v else 'FAIL'} {k}")

    if args.emit and not args.check:
        data = inv.build_invention_dataset(depth=args.depth, eval_frac=args.eval_frac, seed=args.seed)
        payload = {
            "instrument": "open-invention", "depth": args.depth, "seed": args.seed,
            "eval_frac": args.eval_frac, "discrimination": disc,
            "train_tasks": data["train_tasks"], "eval_tasks": data["eval_tasks"],
            "note": "EVAL compositions absent from TRAIN; primitives all seen. "
                    "reference_solution is generator-only — do NOT place in a training row.",
            "canClaimAGI": False,
        }
        Path(args.emit).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.emit}  ({len(data['eval_tasks'])} eval tasks)")

    print(f"\nINVENTION INSTRUMENT: {'GO' if ok else 'NO-GO'} — "
          + ("sound (disjoint, covered, discriminates)" if ok else "invariants failed"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
