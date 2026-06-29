#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Demo the fail-closed metric grounding gate (offline).

Every physical claim a VLM makes about an image — "the cup is in front of the
laptop", "the car is bigger than the cup", "they are 30 units apart" — is a
hypothesis. The gate re-checks each against the depth-grounded scene
(``multimodal_bench/verifiers.py`` over a depth source): the cited region must
contain the subject, and the claimed relation/measure must match the verifier.
Mismatch, ungroundable region, or an unavailable depth source -> block + escalate.

Shows a grounded claim stream (all accepted) vs a hallucinated one (reversed depth
order, the apparent-size illusion, a region that misses its subject) — all blocked.

    python tools/run_metric_gate.py                       # authored depth (offline)
    python tools/run_metric_gate.py --depth depth-anything  # pixel depth (blocker w/o weights)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import depth_backend, metric_gate  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    spec = "authored"
    if "--depth" in argv:
        spec = argv[argv.index("--depth") + 1]

    src, label, blocker = depth_backend.make_depth_source(spec)
    if blocker:
        # eval_ladder discipline: a real backend with no weights is a blocker, not a result.
        report = {"depthSource": label, "blocked": True, "blocker": blocker,
                  "note": "recorded as a blocker, not a result — wire weights to measure pixel depth"}
        print(json.dumps(report, indent=2) if "--json" in argv else
              f"depth source '{label}' unavailable -> blocker: {blocker}\n"
              f"  (the gate fails closed: with no depth it blocks + escalates every metric claim)")
        return 0

    good = metric_gate.MetricGate(metric_gate.DEMO_SCENE, depth_source=src)
    good.run(metric_gate.GROUNDED_CLAIMS)
    bad = metric_gate.MetricGate(metric_gate.DEMO_SCENE, depth_source=src)
    bad.run(metric_gate.HALLUCINATED_CLAIMS)
    out = {"depthSource": label, "grounded": good.summary(), "hallucinated": bad.summary()}

    if "--json" in argv:
        print(json.dumps(out, indent=2))
        return 0
    print(f"\nFail-closed metric grounding gate (depth source: {label}):")
    for name in ("grounded", "hallucinated"):
        s = out[name]
        print(f"  {name:<13} proposed={s['proposed']} accepted={s['accepted']} "
              f"blocked={s['blocked']} (rate {s['blockRate']:.2f}) escalations={s['escalations']}")
        if s["reasons"]:
            print(f"                block reasons: {', '.join(s['reasons'])}")
    print(f"\n  grounded claims all accepted : {out['grounded']['blocked'] == 0}")
    print(f"  hallucinated claims all blocked: {out['hallucinated']['accepted'] == 0}")
    print("  -> a confident-but-wrong metric claim never passes; it escalates to a human.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
