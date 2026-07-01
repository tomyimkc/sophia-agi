#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Consolidate verifier-certified beliefs into a reversible weight-update candidate.

This is the SLOW loop. It reads the ``ingested`` rows the fast loop
(tools/run_realtime_grounding.py) wrote to the belief store, emits habit-shaped
GRPO/DPO training rows (self-decontaminated against the eval surface), and appends a
reversibility-ledger entry so the batch can be un-merged later.

  --dry-run (default; CI-safe, changes no weights)
    Produces training rows + a ledger entry only.

  --live (seam only; does NOT train locally)
    Prints the exact tools/run_rlvr.py command that must run on RunPod via GitHub
    Actions (never local SSH). This driver never launches a GPU job itself.

    python3 tools/run_realtime_consolidation.py --dry-run
    python3 tools/run_realtime_consolidation.py --live --based-on-spec zai-org/glm-4-9b-chat-hf
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import realtime_consolidation as rc  # noqa: E402

DEFAULT_STORE = ROOT / "agent" / "memory" / "realtime" / "belief_store.jsonl"
DEFAULT_OUT_DIR = ROOT / "training" / "realtime"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", action="store_true", help="produce rows + ledger only (default)")
    grp.add_argument("--live", action="store_true", help="print the RunPod/Actions GPU seam command; never trains locally")
    ap.add_argument("--belief-store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--based-on-spec", default="zai-org/glm-4-9b-chat-hf")
    ap.add_argument("--delta-id", default="")
    args = ap.parse_args()

    delta_id = args.delta_id or ("rt-delta-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    report = rc.consolidate(
        args.belief_store,
        out_dir=args.out_dir,
        based_on_spec=args.based_on_spec,
        delta_id=delta_id,
        root=ROOT,
    )

    print(f"REAL-TIME CONSOLIDATION (dryRun={report['dryRun']})  delta={delta_id}")
    print(f"  ingested={report['nIngested']} -> trainingRows={report['nRows']} dropped(decontam)={report['nDropped']}")
    print(f"  rows:   {report['rowsPath']}")
    print(f"  ledger: {report['ledgerPath']}  (mergeState={report['ledgerEntry']['mergeState']}, canRevert={report['ledgerEntry']['canRevert']})")
    print(f"  note:   {report['note']}")

    if args.live:
        # The seam only — GPU jobs go through GitHub Actions -> RunPod, never here.
        cmd = [
            "python", "tools/run_rlvr.py",
            "--task", "provenance",
            "--model", args.based_on_spec,
            "--output", f"training/closed-loop/{delta_id}",
        ]
        print("\n--live is a seam: run this on RunPod via GitHub Actions (NOT locally):")
        print("  " + " ".join(cmd))
        print("  then set the resulting ledger entry mergeState=merged (or revert to un-merge).")

    out_report = args.out_dir / "consolidation-report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
