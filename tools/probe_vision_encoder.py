#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Probe a vision encoder on image->text retrieval over the trap suite.

Mirrors tools/eval_ladder.py's discipline: the wiring runs offline against a
deterministic hashing stand-in; a real encoder (clip:/siglip:) needs weights and
deps, and when they are absent the rung is recorded as a BLOCKER, not a result.
recall@1 ships with a bootstrap 95% CI and the chance baseline, and every line is
labelled with whether it measured a real encoder or the caption stand-in.

    python tools/probe_vision_encoder.py
    python tools/probe_vision_encoder.py --encoder clip:openai/clip-vit-base-patch32
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import encoder_probe  # noqa: E402

DEFAULT_LADDER = ["hashing", "clip:openai/clip-vit-base-patch32", "siglip:google/siglip-base-patch16-224"]


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    as_json = "--json" in argv
    enc = None
    if "--encoder" in argv:
        enc = argv[argv.index("--encoder") + 1]
    specs = [enc] if enc else DEFAULT_LADDER

    results = [encoder_probe.retrieval_probe(s) for s in specs]
    if as_json:
        print(json.dumps(results, indent=2))
        return 0

    print("\nVision-encoder retrieval probe (image->text recall@1, 95% CI):")
    for r in results:
        if r.get("blocked"):
            print(f"  {r['backend']:<42} BLOCKED ({r['blocker']})")
            continue
        real = "real encoder" if r["isRealEncoder"] else "stand-in"
        print(f"  {r['backend']:<42} recall@1={r['recallAt1']:.3f} CI{r['ci95']} "
              f"chance={r['chance']:.3f} [{real}]")
    print("  note: hashing measures harness/caption separability, NOT pixels; only")
    print("        clip/siglip rungs (real weights on rendered PNGs) measure perception.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
