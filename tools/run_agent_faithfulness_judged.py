#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the JUDGED agent-faithfulness benchmark on the sealed held-out pack.

This is the model-judged tier: it measures whether an entailment judge can settle
the grounding decisions the deterministic lexical judge cannot (paraphrase,
multi-hop, negation/scope distractors). It is validated ONLY under the no-overclaim
gate (>=2 judge families + kappa>=0.40 + >=3 runs + CI above chance).

    # offline default: mock judge -> abstains -> validated=False (illustrative)
    python tools/run_agent_faithfulness_judged.py --json

    # real measured run (needs API access for >=2 distinct vendor families):
    python tools/run_agent_faithfulness_judged.py \
        --judges openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct \
        --runs 3 --write

The committed artifact is the OFFLINE mock run (validated=False), so CI stays
network-free; a real validated run is OPEN (see agi-proof/failure-ledger.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.agent_faithfulness_judged import (  # noqa: E402
    DEFAULT_PACK,
    aggregate,
    build_judges,
    load_pack,
    run_once,
    verify_seal,
)

ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "agent-faithfulness-judged.public-report.json"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", default=str(DEFAULT_PACK))
    ap.add_argument("--judges", default="mock",
                    help="comma-separated judge specs (e.g. openrouter:deepseek/..,openrouter:meta-llama/..)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true", help="write artifact under agi-proof/")
    args = ap.parse_args(argv)

    pack = load_pack(args.pack)
    if not verify_seal(pack):
        print("SEAL MISMATCH: the held-out pack does not match its committed hash.", file=sys.stderr)
        return 2

    cases = pack["cases"]
    specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    judges = build_judges(specs)
    runs = [run_once(cases, judges) for _ in range(max(1, args.runs))]
    report = aggregate(runs, judge_specs=specs, cases=cases)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        tier = "VALIDATED" if report["validated"] else "ILLUSTRATIVE (not headline-grade)"
        print(
            f"judged agent-faithfulness — N={report['n']} · judges={specs} · "
            f"runs={report['runs']}  [{tier}]"
        )
        print(f"  consensus certify-accuracy {report['consensusAccuracy'] * 100:.1f}%  CI {report['ci']}")
        print(f"  lexical baseline {report['lexicalBaselineAccuracy'] * 100:.1f}%  "
              f"-> judge value-add {report['judgeValueAdd'] * 100:+.1f} pts")
        print(f"  mean pairwise kappa {report['meanPairwiseKappa']}")
        print("  by failure type (accuracy | kappa):")
        for lab, s in report["byFailureType"].items():
            acc = f"{s['accuracy'] * 100:.0f}%" if s["accuracy"] is not None else "—"
            print(f"    {lab:22} n={s['n']:<2} {acc:>4} | k={s['meanPairwiseKappa']}")
        for k, ok in report["validatedChecks"].items():
            print(f"  [{'x' if ok else ' '}] {k}")

    if args.write:
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
