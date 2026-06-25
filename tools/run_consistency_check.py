#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run OKF local-global syntactic consistency check over wiki pages.

Escalates undeclared cross-context disagreements; does not decide truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_store  # noqa: E402
from okf.consistency_check import (  # noqa: E402
    consistency_report,
    sync_epistemic_holes,
)

DEFAULT_OUT = ROOT / "agi-proof" / "okf-consistency" / "consistency.public-report.json"


def _load_dnm() -> dict:
    path = ROOT / "data" / "traditions.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v.get("doNotMergeWith", []) for k, v in raw.items()}


def run_check(
    *,
    roots: list[Path] | None = None,
    partition_key: str = "referent",
    write_holes: bool = True,
    holes_path: Path | None = None,
) -> dict:
    if roots:
        from okf.page import load_pages

        pages = load_pages(*roots)
    else:
        pages = wiki_store.belief_graph_pages()

    report = consistency_report(pages, partition_key=partition_key, dnm_by_tradition=_load_dnm())
    if write_holes:
        sync_epistemic_holes(report["epistemicHoles"], path=holes_path)

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", help="OKF page root (repeatable); default wiki_store")
    ap.add_argument("--partition-key", default="referent", help="referent-attribution (default) or tradition")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-write-holes", action="store_true")
    ap.add_argument("--holes-path", default=None)
    args = ap.parse_args(argv)

    roots = [Path(r) for r in args.root] if args.root else None
    report = run_check(
        roots=roots,
        partition_key=args.partition_key,
        write_holes=not args.no_write_holes,
        holes_path=Path(args.holes_path) if args.holes_path else None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "out": str(out),
        "sharedReferents": report.get("sharedReferents", 0),
        "epistemicHoleCount": report["epistemicHoleCount"],
        "declaredContradictionsDeferred": report["declaredContradictionsDeferred"],
        "gatePass": report.get("gatePass", report["epistemicHoleCount"] == 0),
        "candidateOnly": report["candidateOnly"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
