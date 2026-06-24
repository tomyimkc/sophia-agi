#!/usr/bin/env python3
"""Ingest a real RLVR adapter-eval report and run it through the SSIL Layer-1 gate.

Turns a trained weight delta's held-out eval into an SSIL verdict — the single
command that makes Layer 1 real instead of mock. It reads the JSON produced by

    python3 tools/eval_rlvr_adapter.py --mode real --adapter <ckpt> --out <report>

and maps it onto `agent/ssil_layer1.adapter_candidate`, then runs the IDENTICAL
orchestrator (G2/G4/G5/G6) that gates skills:

  before/after capability  <- base.meanReward / adapterScore.meanReward   (G4 headline)
  protected integrity      <- 1 - trueFalsePositiveRate  (lower FP = higher integrity;
                              a rise in false-positives shows up as a protected regression)
  contamination            <- entityIntersection non-empty -> G4 reject

Fail-closed: if the report lacks a before/after capability pair, this errors with a
clear message instead of inventing numbers. Output carries candidateOnly /
canClaimAGI=false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_layer1 import adapter_candidate, run_layer1  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-layer1-real.public-report.json"


def _dig(report: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first present value among dotted key paths; None if none found."""
    for path in paths:
        cur: Any = report
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def map_report(report: dict[str, Any], *, adapter_id: str | None = None) -> dict[str, Any]:
    before = _dig(report, ("base", "meanReward"), ("baseScore", "meanReward"))
    after = _dig(report, ("adapterScore", "meanReward"), ("adapter", "meanReward"))
    base_fp = _dig(report, ("base", "trueFalsePositiveRate"), ("baseScore", "trueFalsePositiveRate"))
    adapter_fp = _dig(report, ("adapterScore", "trueFalsePositiveRate"), ("adapter", "trueFalsePositiveRate"))
    if before is None or after is None:
        raise SystemExit(
            "ERROR: report has no before/after capability pair "
            "(expected base.meanReward + adapterScore.meanReward). "
            "Produce it with: python3 tools/eval_rlvr_adapter.py --mode real --adapter <ckpt>"
        )
    # False-positive rate: lower is better -> invert to an integrity metric (higher better)
    # so a rise in FP rate registers as a protected regression. Fail-closed: a MISSING FP
    # rate is unverified integrity, not perfect integrity — refuse rather than assume 1.0.
    if base_fp is None or adapter_fp is None:
        raise SystemExit(
            "ERROR: report is missing trueFalsePositiveRate on base and/or adapter. "
            "The Layer-1 gate is fail-closed: unverified integrity cannot promote. "
            "Re-run tools/eval_rlvr_adapter.py so the eval reports false-positive rates."
        )
    prot_before = round(1.0 - float(base_fp), 4)
    prot_after = round(1.0 - float(adapter_fp), 4)
    entity_intersection = _dig(report, ("entityIntersection",), ("split", "entityIntersection")) or []
    contaminated = bool(entity_intersection)
    aid = adapter_id or _dig(report, ("adapter",)) or report.get("model") or "rlvr-adapter"
    aid = str(aid).rsplit("/", 1)[-1]
    return {
        "id": aid, "before": float(before), "after": float(after),
        "protected_before": prot_before, "protected_after": prot_after,
        "contaminated": contaminated, "entityIntersection": entity_intersection,
    }


def ingest(report_path: str | Path, *, adapter_id: str | None = None) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    m = map_report(report, adapter_id=adapter_id)
    candidate = adapter_candidate(
        m["id"], before=m["before"], after=m["after"],
        protected_before=m["protected_before"], protected_after=m["protected_after"],
        contaminated=m["contaminated"],
    )
    record = run_layer1(candidate)
    record["sourceReport"] = str(report_path)
    record["mappedMetrics"] = m
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest an RLVR adapter-eval report into the SSIL Layer-1 gate")
    ap.add_argument("report", help="path to *.rlvr.adapter-eval*.json (from tools/eval_rlvr_adapter.py)")
    ap.add_argument("--adapter-id", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args(argv)

    record = ingest(args.report, adapter_id=args.adapter_id)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    m = record["mappedMetrics"]
    if args.print:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"adapter={m['id']} capability {m['before']}->{m['after']} "
          f"integrity {m['protected_before']}->{m['protected_after']} contaminated={m['contaminated']}")
    print(f"SSIL VERDICT={record['verdict']} blocking={record['blockingGates']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
