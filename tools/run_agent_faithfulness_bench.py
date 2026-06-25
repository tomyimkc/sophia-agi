#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the agent-faithfulness benchmark and (optionally) write the public report.

Deterministic: the trajectory evaluator's default support judge is lexical, so this
benchmark needs no model and no multi-family gate — it is reproducible bit-for-bit
in CI. The remaining honesty caveat is label provenance (first-party seed pack);
that is stamped into the report, not hidden.

    python tools/run_agent_faithfulness_bench.py            # human-readable
    python tools/run_agent_faithfulness_bench.py --json     # machine-readable
    python tools/run_agent_faithfulness_bench.py --write    # write agi-proof artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.agent_faithfulness import DEFAULT_PACK, load_pack, score_pack  # noqa: E402

ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "agent-faithfulness.public-report.json"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", default=str(DEFAULT_PACK), help="benchmark pack JSON")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--write", action="store_true", help="write the artifact under agi-proof/")
    args = ap.parse_args(argv)

    pack = load_pack(args.pack)
    report = score_pack(pack)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        acc = report["verdictAccuracy"]
        ci = report["verdictAccuracyCI95"]
        det = report["detection"]
        print(
            f"agent-faithfulness benchmark — N={report['n']} · "
            "deterministic (no judge) · label provenance: first-party seed"
        )
        print(f"  verdict accuracy {acc * 100:.1f}%  Wilson-95 {ci}")
        print(
            f"  detection (should-not-certify)  P={det['precision']} R={det['recall']} "
            f"F1={det['f1']}  (tp={det['tp']} fp={det['fp']} fn={det['fn']} tn={det['tn']})"
        )
        print(
            f"  localization {report['localizationAccuracy']} over {report['localizationN']} "
            "culprit-bearing cases"
        )
        print("  by category (verdict accuracy):")
        for cat, s in report["byCategory"].items():
            print(f"    {cat:22} n={s['n']:<2} {s['verdictAccuracy'] * 100:.0f}%")

    if args.write:
        ARTIFACT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
