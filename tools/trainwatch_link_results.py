#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Link past NON-training results (cert / bench / judge JSONs) into TrainWatch (:8420).

TrainWatch only models TRAINING runs (it parses ``step S/T loss=..`` curves), so cert verdicts
(top1 / mean_kl), judge κ, and benchmark pass-rates were logged to ``agi-proof/benchmark-results/``
+ the failure ledger — the repo's authoritative record — NOT to TrainWatch. This tool registers a
chosen result JSON as a single-point TrainWatch run (its headline metrics as the final step), so it
also shows on the :8420 dashboard alongside the trainings. It does NOT move or alter the
authoritative JSON; it only mirrors a view into TrainWatch.

Metric extraction is pure + offline-testable; the actual ``trainwatch.init`` registration runs
only where TrainWatch is installed (the Spark). ``--dry-run`` prints what WOULD be registered with
no TrainWatch import, so you can preview from anywhere.

Usage (on the Spark, TrainWatch installed):
  python tools/trainwatch_link_results.py agi-proof/benchmark-results/nvfp4-v3-downproj-cert.json
  python tools/trainwatch_link_results.py --glob 'agi-proof/benchmark-results/*uplift*.json'
  python tools/trainwatch_link_results.py --dry-run agi-proof/benchmark-results/baseline-math.json
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Headline metrics worth surfacing first, across the repo's result schemas (cert / uplift / bench).
_PREFERRED = (
    "top1_agreement", "mean_kl", "protected_mean_kl",      # certify_lowram
    "meanDelta", "meanPairwiseKappa", "interJudgeKappa",   # uplift / judge
    "passAt1", "accuracy", "score", "adapter_winrate", "n",  # benchmarks
)
# Boolean verdict-ish fields -> rendered as 1.0 / 0.0 so the dashboard shows the GO/NO-GO.
_VERDICT_BOOL = ("passed", "validated", "canClaimAGI")
_MAX_METRICS = 8


def _num(v) -> "float | None":
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def extract_run(result: dict, name: str) -> dict:
    """Pure: pick a name + a small set of numeric headline metrics for one result JSON.

    Order: verdict booleans first (passed/validated), then preferred metric keys present, then any
    remaining top-level numeric fields — capped at _MAX_METRICS. Returns the TrainWatch-run shape
    {name, status, metrics}. Deterministic (dict iteration order is insertion order)."""
    metrics: dict[str, float] = {}
    for k in _VERDICT_BOOL:
        if k in result and _num(result[k]) is not None:
            metrics[k] = _num(result[k])
    for k in _PREFERRED:
        if k in result and (val := _num(result[k])) is not None and k not in metrics:
            metrics[k] = val
    if len(metrics) < _MAX_METRICS:
        for k, v in result.items():
            if len(metrics) >= _MAX_METRICS:
                break
            val = _num(v)
            if val is not None and k not in metrics:
                metrics[k] = val
    return {"name": name, "status": "completed", "metrics": dict(list(metrics.items())[:_MAX_METRICS])}


def runs_from_paths(paths: list, prefix: str) -> list:
    runs = []
    for p in paths:
        path = Path(p)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"  skip {p}: not a JSON object", file=sys.stderr)
            continue
        runs.append(extract_run(data, f"{prefix}{path.stem}"))
    return runs


def link_to_trainwatch(runs: list) -> int:
    """Register each run in TrainWatch (Spark-only; trainwatch must be importable)."""
    try:
        import trainwatch
    except Exception as e:  # noqa: BLE001
        print(f"trainwatch not importable here ({e}). Run this on the Spark, or use --dry-run.",
              file=sys.stderr)
        return 2
    n = 0
    for r in runs:
        run = trainwatch.init(name=r["name"], description="linked result (non-training)",
                              total_steps=1)
        if r["metrics"]:
            run.log(r["metrics"], step=1)
        run.finish(r["status"])
        n += 1
        print(f"  linked {r['name']}  {r['metrics']}")
    print(f"linked {n} result(s) into TrainWatch (:8420)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="result JSON file(s) to link")
    ap.add_argument("--glob", default=None, help="glob of result JSONs (e.g. 'agi-proof/benchmark-results/*cert*.json')")
    ap.add_argument("--name-prefix", default="result:", help="run-name prefix (distinguishes from trainings)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be registered; no trainwatch import")
    ap.add_argument("--selftest", action="store_true", help="offline self-test of the extractor")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    paths = list(args.paths)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    if not paths:
        ap.error("give result JSON path(s) or --glob (or --selftest)")
    runs = runs_from_paths(paths, args.name_prefix)
    if args.dry_run:
        for r in runs:
            print(f"WOULD link {r['name']}  status={r['status']}  metrics={r['metrics']}")
        print(f"(dry-run) {len(runs)} run(s) — re-run without --dry-run on the Spark to register")
        return 0
    return link_to_trainwatch(runs)


def _selftest() -> int:
    # cert-shaped
    cert = {"passed": True, "mean_kl": 0.045, "top1_agreement": 0.953, "protected_mean_kl": 0.02,
            "scheme": "nvfp4", "keep_suffixes": ["down_proj"]}
    r = extract_run(cert, "result:nvfp4-cert")
    assert r["name"] == "result:nvfp4-cert" and r["status"] == "completed"
    assert r["metrics"]["passed"] == 1.0 and r["metrics"]["top1_agreement"] == 0.953
    assert "scheme" not in r["metrics"]  # non-numeric dropped
    # uplift-shaped
    up = {"validated": False, "meanDelta": 0.341, "meanPairwiseKappa": 0.394, "canClaimAGI": False}
    ru = extract_run(up, "result:uplift")
    assert ru["metrics"]["validated"] == 0.0 and ru["metrics"]["meanDelta"] == 0.341
    assert ru["metrics"]["canClaimAGI"] == 0.0
    # bench-shaped + the _MAX_METRICS cap
    bench = {f"m{i}": i for i in range(20)}
    rb = extract_run(bench, "result:bench")
    assert len(rb["metrics"]) == _MAX_METRICS
    print("ok trainwatch_link_results selftest (extractor verdicts+metrics+cap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
