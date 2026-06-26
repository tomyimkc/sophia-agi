#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Print the multimodal RLVR reward invariants (offline, no torch/GPU).

The visual-grounding reward (``multimodal_bench/visual_reward.py``) is the
multimodal analogue of the math/code RLVR rewards: the judge-free scene verifier
IS the reward signal, shaped to prefer honest abstention over confident
hallucination (correct > abstain > wrong). This tool asserts the reward
invariants and shows the contamination-free train/eval family split — the same
honesty bar as ``tools/run_rlvr.py``'s ``--offline-invariants``.

    python tools/run_multimodal_reward.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import visual_reward  # noqa: E402


def main(argv=None) -> int:
    ok, detail = visual_reward.offline_invariants()
    print(json.dumps({"allInvariantsHold": ok, **detail}, indent=2))
    print(f"\nrewards: correct={detail['rewards']['correct']['reward']:+.2f} "
          f"abstain={detail['rewards']['abstain']['reward']:+.2f} "
          f"wrong={detail['rewards']['wrong']['reward']:+.2f}  "
          f"(ordering correct > abstain > wrong)")
    print(f"train families: {detail['trainFamilies']}")
    print(f"eval  families: {detail['evalFamilies']}")
    print(f"family intersection (must be empty): {detail['familyIntersection']}")
    print(f"\nALL INVARIANTS HOLD: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
