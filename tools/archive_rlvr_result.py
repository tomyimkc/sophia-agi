#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Archive a multi-seed RLVR replication as durable evidence.

GitHub Actions artifacts expire (~90 days). This preserves the small, durable pieces:
  - per-seed eval/gate JSON records (reconstructed from logs or copied from the artifact)
    into agi-proof/benchmark-results/rlvr-replication/;
  - the aggregate report;
  - an appended RESULTS.md entry (no-overclaim wording), tied to commit + run URLs.

It does NOT publish weights to Hugging Face — that is an outward-facing publish and is
left to an explicit, separate step (the adapter tarball lives in the run artifact).

Usage:
  python3 tools/archive_rlvr_result.py \
    --aggregate agi-proof/benchmark-results/rlvr-replication/aggregate.public-report.json \
    --seed-report 0=<json> --seed-report 1=<json> --seed-report 2=<json> \
    --run-urls "https://github.com/.../runs/...,..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "agi-proof" / "benchmark-results" / "rlvr-replication"
# NOTE: RESULTS.md is GENERATED from published-results.json by tools/build_results_page.py
# and CI fails if it is hand-edited (staleness gate). So the archive writes a STANDALONE
# evidence file here instead of touching RESULTS.md. Curating a number into the public
# RESULTS.md is the maintainer's gated process (edit published-results.json + rebuild).
SUMMARY_MD = ARCHIVE_DIR / "SUMMARY.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregate", required=True, help="aggregate.public-report.json path")
    ap.add_argument("--seed-report", action="append", default=[], help="SEED=path/to/eval.json (repeatable)")
    ap.add_argument("--run-urls", default="", help="comma-separated GitHub run URLs")
    ap.add_argument("--summary-path", default=str(SUMMARY_MD), help="standalone evidence summary path")
    args = ap.parse_args(argv)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    agg = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))

    # Copy per-seed reports into the durable archive dir.
    for spec in args.seed_report:
        seed, _, path = spec.partition("=")
        dest = ARCHIVE_DIR / f"seed{seed}.adapter-eval.json"
        dest.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"archived {dest}")

    cap = agg["capability"]
    summary = (
        f"# RLVR adapter — multi-seed replication (candidate evidence)\n\n"
        f"_Standalone evidence — NOT the gated public RESULTS.md (which is generated from "
        f"published-results.json)._\n\n"
        f"- Adapter: `{agg['adapterId']}`; seeds: {agg['seeds']}; n={agg['n']}\n"
        f"- Held-out capability (meanReward): {cap['meanBefore']} → {cap['meanAfter']} "
        f"(mean Δ {cap['meanDelta']}, range {cap['minDelta']}…{cap['maxDelta']}, σ {cap['stdevDelta']})\n"
        f"- SSIL gate promotes: {agg['promotes']}/{agg['n']}; "
        f"protected regression: {agg['anyProtectedRegression']}; contaminated: {agg['anyContaminated']}\n"
        f"- capabilityClaimReady: {agg['capabilityClaimReady']}\n"
        f"- Runs: {args.run_urls or '(see Actions)'}\n\n"
        f"**Boundary:** aggregated gate result under the no-overclaim measurement gate; n is small; "
        f"`candidateOnly: true`, `canClaimAGI: false`. Not a validated capability claim.\n"
    )
    Path(args.summary_path).write_text(summary, encoding="utf-8")
    print(f"wrote {args.summary_path}")
    print(f"archive dir: {ARCHIVE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
