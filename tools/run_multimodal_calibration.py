#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Risk-coverage / calibration report for the visual-trap suite (offline demo).

Reuses ``agent.calibration`` (ECE, risk-coverage, selective risk, AURC) over
visual-trap outcomes to ask the falsifiable question: does answering only above a
confidence threshold lower the error rate? Compares a *calibrated* synthetic VLM
(confidence tracks correctness) against an *overconfident* one (uniformly high
confidence) — the calibrated model should show selective risk < base risk at
partial coverage and a lower AURC; the overconfident model should not.

    python tools/run_multimodal_calibration.py

For a real VLM, plug a confidence-returning answer function (e.g. self-consistency
across sampled answers, ``agent.calibration.self_consistency``) into
``multimodal_bench.calibration.run_with_confidence``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import calibration  # noqa: E402


def _line(name: str, rep: dict) -> None:
    sr = rep["selectiveRisk"]
    print(f"  {name:<14} ECE={rep['ece']:.3f}  baseRisk={rep['baseRisk']:.3f}  "
          f"AURC={rep['aurc']:.3f}  selRisk@0.5={sr['0.5']:.3f}  selRisk@1.0={sr['1.0']:.3f}")


def main(argv=None) -> int:
    as_json = "--json" in (argv or sys.argv[1:])
    out = calibration.demo()
    if as_json:
        print(json.dumps(out, indent=2))
        return 0
    print("\nCalibrated abstention over the visual-trap suite (synthetic confidence A/B):")
    _line("calibrated", out["calibrated"])
    _line("overconfident", out["overconfident"])
    c = out["calibrated"]
    improves = c["selectiveRisk"]["0.5"] < c["baseRisk"]
    print(f"\n  calibrated selective-risk@0.5 < baseRisk : {improves}  "
          f"({c['selectiveRisk']['0.5']:.3f} < {c['baseRisk']:.3f})")
    print("  -> a calibrated VLM lowers error by abstaining on low-confidence cases;")
    print("     the overconfident VLM cannot separate right from wrong (high ECE, flat risk).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
