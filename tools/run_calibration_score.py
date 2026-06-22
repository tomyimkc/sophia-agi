#!/usr/bin/env python3
"""Re-score an ablation private dump on epistemic calibration instead of keywords.

The keyword/regex scorer cannot see Sophia's value (correct abstention, refusing to
fabricate). This re-scores the per-mode responses captured by
``run_ablation_sophia.py --private-out`` on the calibration axis and reports the
per-mode calibration score, fabrication rate, and over-abstention rate, plus the
sophia-full-minus-raw deltas.

    python tools/run_calibration_score.py <pack.json> <private-dump.json> [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.calibration_score import score_pack_calibration  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path)
    ap.add_argument("private_dump", type=Path, help="run_ablation_sophia --private-out JSON")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    dump = json.loads(args.private_dump.read_text(encoding="utf-8"))

    per_mode = {}
    for mode, blob in dump.items():
        responses = blob.get("responses", {}) if isinstance(blob, dict) else {}
        per_mode[mode] = score_pack_calibration(pack, responses)

    full = per_mode.get("sophia-full", {}).get("calibrationScore")
    deltas = {}
    for mode, r in per_mode.items():
        if mode != "sophia-full" and full is not None:
            deltas[mode] = round(full - r["calibrationScore"], 4)

    out = {
        "pack": str(args.pack),
        "perMode": {m: {k: v for k, v in r.items() if k != "perCase"} for m, r in per_mode.items()},
        "calibrationDeltaFullMinusMode": deltas,
    }
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"{'mode':24s} {'calib':>7s} {'fabRate':>8s} {'overAbst':>9s}")
    for mode, r in per_mode.items():
        fab = r["fabricationRate"]; ov = r["overAbstentionRate"]
        print(f"  {mode:22s} {r['calibrationScore']:>7.3f} "
              f"{('—' if fab is None else f'{fab:.2f}'):>8s} {('—' if ov is None else f'{ov:.2f}'):>9s}")
    print("\ncalibration delta (sophia-full minus mode):")
    for mode, d in deltas.items():
        print(f"  {mode:22s} {d:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
