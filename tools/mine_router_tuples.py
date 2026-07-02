#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mine router-training tuples from harness run traces.

The trained router head (``agent/router_head.py``) learns from
``(task signals → v1 plan → run outcome)`` tuples. The signals and the v1 plan are
recomputed deterministically here (the v1 policy is a pure function of the task);
the outcome comes from the trace's ``task_end`` event. Runs without both a
``task_start.goal`` and a ``task_end.ok`` are skipped — no outcome is ever guessed.

Run:  python tools/mine_router_tuples.py [--runs-dir agent/memory/agent_runs]
                                         [--out training/swarm_router/router_tuples.jsonl]

Deterministic given the trace files (sorted order, no timestamps in output rows).
The output is a *training input*, not a result artifact — nothing here feeds
RESULTS.md or the claim gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.swarm_router import SwarmRouter  # noqa: E402

SCHEMA = "sophia.router_tuple.v1"
DEFAULT_RUNS_DIR = ROOT / "agent" / "memory" / "agent_runs"
DEFAULT_OUT = ROOT / "training" / "swarm_router" / "router_tuples.jsonl"


def mine_trace(path: Path) -> "dict | None":
    """Extract {goal, ok} from one run-trace JSONL, or None when incomplete."""
    goal: "str | None" = None
    ok: "bool | None" = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "task_start" and isinstance(event.get("goal"), str):
            goal = goal or event["goal"]
        elif event.get("type") == "task_end" and isinstance(event.get("ok"), bool):
            ok = event["ok"]  # last task_end wins (resumed runs)
    if not goal or ok is None:
        return None
    return {"goal": goal, "ok": ok}


def mine(runs_dir: Path) -> "list[dict]":
    router = SwarmRouter()
    rows: list[dict] = []
    seen: set = set()
    for path in sorted(runs_dir.glob("*.jsonl")):
        item = mine_trace(path)
        if item is None:
            continue
        key = (item["goal"], item["ok"])
        if key in seen:
            continue  # identical (goal, outcome) adds no information
        seen.add(key)
        plan = router.decide(item["goal"])
        rows.append({
            "schema": SCHEMA,
            "task": item["goal"],
            "ok": item["ok"],
            "signals": plan.signals.to_dict(),
            "v1Plan": {"mode": plan.mode,
                       "teams": sorted({a.team for a in plan.assignments})},
            "source": path.name,
        })
    return rows


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if not args.runs_dir.is_dir():
        print(f"no runs dir at {args.runs_dir}; nothing to mine (this is not an error "
              f"— traces accumulate as the harness runs)")
        return 0
    rows = mine(args.runs_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"mined {len(rows)} tuple(s) ({n_ok} ok / {len(rows) - n_ok} failed) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
