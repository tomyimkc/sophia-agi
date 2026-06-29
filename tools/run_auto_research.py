#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the auto-research loop: the agent proposes + runs its own experiments, judged
by the same trust gates so it cannot overclaim.

    python tools/run_auto_research.py [--n N] [--out PATH] [--json]
    python tools/run_auto_research.py --corpus wiki        # real OKF domains

Offline, deterministic. Confirms a hypothesis only when the self-evolving agent
COMMITTED the update (cleared reward-hack + plasticity + no-forgetting gates); logs
refutations too. With --corpus, experiments are drawn from the real OKF wiki (one per
domain: is a claim's grounding predictable from its text?) and the agent runs in
verifier mode (the gain is held-out verifier generalization, not selection).
Default ledger: agi-proof/auto-research/ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.auto_research import AutoResearcher  # noqa: E402
from agent.self_evolving_agent import SelfEvolvingAgent  # noqa: E402

OUT = ROOT / "agi-proof" / "auto-research" / "ledger.jsonl"
CORPUS_OUT = ROOT / "agi-proof" / "auto-research" / "ledger-okf.jsonl"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=6, help="number of synthetic hypotheses (ignored with --corpus)")
    ap.add_argument("--corpus", default=None, help="OKF wiki root (e.g. 'wiki') for real-domain experiments")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.corpus:
        from agent.okf_research_source import okf_experiments
        researcher = AutoResearcher(SelfEvolvingAgent(evolve_mode="verifier"))
        researcher.run_experiments(okf_experiments(args.corpus))
        default_out = CORPUS_OUT
    else:
        researcher = AutoResearcher()
        researcher.run(n=args.n)
        default_out = OUT
    if args.out is None:
        args.out = str(default_out)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    report = researcher.write_ledger(out)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if all(report["invariants"].values()) else 1

    print("Auto-research loop:")
    print(f"  hypotheses : {report['hypotheses']}")
    print(f"  confirmed  : {report['confirmed']}  (cleared all trust gates)")
    print(f"  refuted    : {report['refuted']}")
    print("  ledger:")
    for e in report["ledger"]:
        print(f"    [{e['verdict']:>9}] {e['hypothesis']['id']} "
              f"{e['hypothesis']['domain']:<18} committed={e['committed']} "
              f"improvement=+{e['improvement']}")
    print("\n  invariants:")
    for k, v in report["invariants"].items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\n  ledger written -> {shown}")
    return 0 if all(report["invariants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
