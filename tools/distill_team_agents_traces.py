#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate externally-verified team-agents SFT traces (no self-distillation).

Source tasks: ``data/council_tasks.json`` (disjoint from sealed benchmark).
Each trace must pass BOTH the intrinsic gate AND external verification — gate-only
rows are dropped fail-closed.

    python tools/distill_team_agents_traces.py --teacher mock --limit 6 \\
        --out training/team_agents/sft_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_format import TEAM_AGENTS_SYSTEM, render_team_target  # noqa: E402
from agent.gate import check_response  # noqa: E402
from agent.team_agents import deliberate_team, verify_trace_external  # noqa: E402
from provenance_bench import dataset_guard  # noqa: E402

TASKS = ROOT / "data" / "council_tasks.json"
OUT = ROOT / "training" / "team_agents" / "sft_traces.jsonl"

# Lightweight external oracle per task — NOT the intrinsic gate.
_EXTERNAL_GOLD = {
    "fin_runway": {"forbiddenSynthesisPatterns": ["unanimous", "consensus reached"]},
    "law_lease_forfeit": {"forbiddenSynthesisPatterns": ["unanimous"]},
    "econ_minwage": {"forbiddenSynthesisPatterns": ["consensus reached"]},
}


def _external_gold(task: dict) -> dict:
    tid = task.get("id", "")
    base = _EXTERNAL_GOLD.get(tid, {"forbiddenSynthesisPatterns": ["unanimous", "all seats agree"]})
    return {**base, "caseKind": "trace_gen"}


def _prompt_disjoint(prompt: str, forbidden: set[str]) -> bool:
    return dataset_guard.normalize(prompt) not in forbidden


def generate_traces(
    tasks: list[dict],
    client,
    *,
    gate: bool = True,
    max_seats: int = 4,
    forbidden: set[str] | None = None,
) -> tuple[list[dict], dict]:
    forbidden = forbidden if forbidden is not None else dataset_guard.eval_prompt_set(root=ROOT)
    rows: list[dict] = []
    kept = dropped_overlap = dropped_external = dropped_dirty = dropped_empty = 0
    for task in tasks:
        prompt = task["prompt"]
        if not _prompt_disjoint(prompt, forbidden):
            dropped_overlap += 1
            continue
        gold = _external_gold(task)
        d = deliberate_team(prompt, client=client, gate=gate, max_seats=max_seats, gold=gold)
        if not verify_trace_external(d, gold):
            dropped_external += 1
            continue
        kind, target = render_team_target(d)
        if not target.strip():
            dropped_empty += 1
            continue
        if check_response(target, mode="advisor", question=prompt)["violations"]:
            dropped_dirty += 1
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": TEAM_AGENTS_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "metadata": {
                "taskId": task.get("id"),
                "councilId": d.councilId,
                "kind": kind,
                "verifiedBy": "external",
                "sourceCitation": "data/council_tasks.json",
                "gatePassed": True,
            },
        })
        kept += 1
    stats = {
        "tasks": len(tasks),
        "kept": kept,
        "droppedOverlap": dropped_overlap,
        "droppedExternal": dropped_external,
        "droppedDirty": dropped_dirty,
        "droppedEmpty": dropped_empty,
    }
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="mock")
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seats", type=int, default=4)
    args = ap.parse_args(argv)

    from agent.model import default_client

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    rows, stats = generate_traces(tasks, default_client(args.teacher), max_seats=args.max_seats)
    stats["teacher"] = args.teacher

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} traces -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
