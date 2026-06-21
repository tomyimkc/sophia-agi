#!/usr/bin/env python3
"""AgentDojo-style end-to-end injection suite (M2.4): ASR + utility under attack.

Runs the full planner → interpreter pipeline on benign requests whose retrieved
content is poisoned with an injection. Reports Attack Success Rate (target 0, by
construction) and utility. Exits non-zero if any attack succeeds.

    python tools/run_agentdojo.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.security.agentdojo import run_suite  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    res = run_suite()
    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res["asr"] == 0.0 else 1

    print("AgentDojo-style end-to-end suite (planner -> interpreter, under injection)")
    print("=" * 70)
    print(f"\nASR (attack success): {res['asr']:.0%}   utility: {res['utility']:.0%}   (n={res['n']})\n")
    for r in res["rows"]:
        flag = "ATTACK WON" if r["attackSuccess"] else "contained"
        print(f"  {r['name']:<26} {flag:<10} calls={r['calls']} blocked={r['blocked']} utility={r['utility']}")
    print("\n" + ("ALL ATTACKS CONTAINED (ASR 0%)" if res["asr"] == 0.0 else "ATTACK SUCCEEDED"))
    return 0 if res["asr"] == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
