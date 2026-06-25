#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deliberate a query across a sector council's seats (map-reduce + per-seat gate).

Decomposes into a few narrow seat passes (small-model friendly), gates each, then
synthesises under the guardian seats. Output is decision-support scaffolding, not
licensed professional advice.

    python tools/council_deliberate.py "Review our gacha odds + refund policy for a HK + EU launch"
    python tools/council_deliberate.py "Model 18-month runway and flag AML for Stripe" --model mock --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import deliberate  # noqa: E402
from agent.team_agents import deliberate_team  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--model", default="mock", help="model spec (mock, ollama:.., anthropic:.., openrouter:..)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (sets SOPHIA_MLX_ADAPTER)")
    ap.add_argument("--team-mode", action="store_true", help="use deliberate_team() orchestrator")
    ap.add_argument("--max-seats", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true", help="disable per-seat gating")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from agent.model import default_client

    if args.adapter:
        os.environ["SOPHIA_MLX_ADAPTER"] = args.adapter
    client = default_client(args.model)
    fn = deliberate_team if args.team_mode else deliberate
    d = fn(args.query, client=client, max_seats=args.max_seats, gate=not args.no_gate)

    if args.json:
        print(json.dumps(d.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"council: {d.councilId or '(none)'} · {d.note}")
        for s in d.seats:
            mark = "OK " if s.gatePassed else "GATED"
            print(f"  [{mark}] {s.displayName}: {s.answer[:120]}")
            for v in s.violations:
                print(f"          ! {v}")
        if d.guardians:
            print(f"  guardians: {', '.join(d.guardians)}")
        print("\n=== Decision ===\n" + d.synthesis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
