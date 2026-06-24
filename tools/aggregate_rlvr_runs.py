#!/usr/bin/env python3
"""Aggregate multi-seed RLVR adapter runs into a no-overclaim result + optionally
record the replicated adapter into the compounding registry as canonical.

Two input modes:
  --reports <json...>   : raw eval_rlvr_adapter report JSON files (from the artifact)
  --runs-json '<json>'  : inline list of per-seed numbers (for the orchestrator, which
                          reads before/after/integrity from the ingest-step logs), e.g.
        '[{"seed":0,"before":0.531,"after":0.7149,"protected_before":0.7917,
           "protected_after":0.7917,"contaminated":false}, ...]'

Writes agi-proof/benchmark-results/rlvr-replication/aggregate.public-report.json and,
with --registry, records each seed run so the config canonicalizes after N replications.
Exits non-zero if the result is not claim-ready (so CI/automation can detect it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_aggregate import AdapterAggregate, SeedRun, runs_from_eval_reports  # noqa: E402
from agent.ssil_registry import Registry  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "rlvr-replication" / "aggregate.public-report.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", nargs="*", default=[], help="eval_rlvr_adapter report JSON files")
    ap.add_argument("--runs-json", default=None, help="inline JSON list of per-seed numbers")
    ap.add_argument("--adapter-id", default="sophia-rlvr-v1")
    ap.add_argument("--canonical-n", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=0.03)
    ap.add_argument("--registry", default=None, help="append-only registry path (records canonical)")
    ap.add_argument("--baseline-after", type=float, default=None, help="canonical mean to beat (compounding)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args(argv)

    if args.runs_json:
        rows = json.loads(args.runs_json)
        runs = [SeedRun(seed=int(r["seed"]), before=float(r["before"]), after=float(r["after"]),
                        protected_before=float(r["protected_before"]), protected_after=float(r["protected_after"]),
                        contaminated=bool(r.get("contaminated", False))) for r in rows]
        agg = AdapterAggregate(adapter_id=args.adapter_id, config={"adapter": args.adapter_id, "kind": "lora_adapter"},
                               runs=runs, canonical_n=args.canonical_n, min_delta=args.min_delta)
    elif args.reports:
        reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
        agg = runs_from_eval_reports(reports, adapter_id=args.adapter_id)
        agg.canonical_n, agg.min_delta = args.canonical_n, args.min_delta
    else:
        raise SystemExit("ERROR: provide --reports <files> or --runs-json '<json>'")

    summary = agg.summary(baseline_after=args.baseline_after)
    if args.registry:
        reg = Registry(path=Path(args.registry), canonical_n=args.canonical_n)
        summary["registry"] = agg.record_to_registry(reg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    cap = summary["capability"]
    print(f"adapter={summary['adapterId']} n={summary['n']} promotes={summary['promotes']}/{summary['n']} "
          f"meanDelta={cap['meanDelta']} (min {cap['minDelta']} / max {cap['maxDelta']}) "
          f"claimReady={summary['capabilityClaimReady']}")
    print(f"-> {out}")
    return 0 if summary["capabilityClaimReady"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
