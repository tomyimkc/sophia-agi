#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Route a query through a Sophia sector council (law / financial / economy).

    python tools/sector_council.py law "Review our gacha odds disclosure for Hong Kong and the EU"
    python tools/sector_council.py financial "Model runway under base/bear and flag AML for our Stripe payouts"
    python tools/sector_council.py economy "Simulate a minimum-wage rise: who gains and who loses?"
    python tools/sector_council.py --list

Output is decision-support scaffolding, not licensed legal/financial advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.sector_council import available_councils, format_council, load_council, route_council  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Route a query through a Sophia sector council")
    parser.add_argument("council", nargs="?", choices=available_councils(), help="council id")
    parser.add_argument("query", nargs="?", default="", help="the question or task")
    parser.add_argument("--material", action="append", default=[], help="extra context line (repeatable)")
    parser.add_argument("--list", action="store_true", help="list available councils and exit")
    parser.add_argument("--json", action="store_true", help="emit the routed council as JSON")
    args = parser.parse_args()

    if args.list or not args.council:
        for council_id in available_councils():
            council = load_council(council_id)
            seat_count = sum(len(g.get("seats", {})) for g in council.get("seatGroups", {}).values())
            print(f"{council_id:10s} {council.get('displayName')} — {seat_count} seats")
        return 0

    if not args.query:
        parser.error("provide a query, or use --list")

    council = load_council(args.council)
    route = route_council(council, args.query, args.material)
    if args.json:
        print(json.dumps(route, ensure_ascii=False, indent=2))
    else:
        print(format_council(route))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
