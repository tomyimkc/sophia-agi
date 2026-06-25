#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run deliberate_team() with an optional Sophia LoRA adapter.

    python tools/team_agents_deliberate.py "Model runway and flag AML" --model mock --json
    python tools/team_agents_deliberate.py "..." --model mlx:Qwen/Qwen2.5-3B-Instruct \\
        --adapter training/mlx_adapters/sophia-v3 --json
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

from agent.team_agents import deliberate_team  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (sets SOPHIA_MLX_ADAPTER)")
    ap.add_argument("--max-seats", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from agent.model import default_client

    if args.adapter:
        os.environ["SOPHIA_MLX_ADAPTER"] = args.adapter
    d = deliberate_team(
        args.query,
        client=default_client(args.model),
        max_seats=args.max_seats,
        gate=not args.no_gate,
    )
    if args.json:
        out = d.to_dict()
        out.update(canClaimAGI=False, candidateOnly=True)
        if args.adapter:
            out["adapterPath"] = args.adapter
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(d.synthesis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
