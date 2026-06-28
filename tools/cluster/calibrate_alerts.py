#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fit calibrated alert thresholds from labelled history (R3/R4).

Reads a JSONL file of observations — one ``{"signal","value","fault"}`` per line — and
fits a fail-closed alert threshold per signal that maximises recall under a false-alert
budget, using agent/calibration.py to audit the signal's discriminative power. Small-N
fits are reported but NOT adopted (see RESULTS.md no-overclaim discipline).

    python3 tools/cluster/calibrate_alerts.py --obs data/cluster/alert_history.jsonl
    python3 tools/cluster/calibrate_alerts.py --demo   # synthetic, illustrates mechanism
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster.calibrate import fit_threshold  # noqa: E402


def _demo_observations() -> list[dict]:
    # Synthetic gpu_temp_c history: hotter nodes fault more often (separable signal).
    obs = []
    for i in range(40):
        temp = 60 + i  # 60..99
        fault = temp >= 85  # ground-truth-ish boundary with a little overlap below
        if temp in (83, 84):
            fault = True  # inject overlap so the fit isn't trivially perfect
        obs.append({"signal": "gpu_temp_c", "value": float(temp), "fault": bool(fault)})
    return obs


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit calibrated alert thresholds (R3/R4).")
    ap.add_argument("--obs", default=None, help="JSONL of {signal,value,fault}")
    ap.add_argument("--demo", action="store_true", help="use synthetic demo observations")
    ap.add_argument("--max-far", type=float, default=0.10, help="false-alert-rate budget")
    ap.add_argument("--min-faults", type=int, default=8, help="min positives to adopt a fit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.demo:
        rows = _demo_observations()
    elif args.obs:
        rows = _load(Path(args.obs))
    else:
        ap.error("pass --obs <file> or --demo")
        return 2

    by_signal: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_signal[r["signal"]].append(r)

    fits = []
    for signal, items in sorted(by_signal.items()):
        values = [float(it["value"]) for it in items]
        labels = [bool(it["fault"]) for it in items]
        fit = fit_threshold(signal, values, labels,
                            max_false_alert_rate=args.max_far, min_faults=args.min_faults)
        fits.append(fit.to_dict())

    if args.json:
        print(json.dumps({"fits": fits}, ensure_ascii=False, indent=2))
    else:
        for f in fits:
            adopt = "ADOPTED" if f["adopted"] else "not adopted"
            print(f"  {f['signal']}: alert ≥ {f['threshold']}  "
                  f"(precision {f['precision']}, recall {f['recall']}, FAR {f['false_alert_rate']}) "
                  f"[{adopt}]")
            print(f"      {f['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
