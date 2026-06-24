#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council-distillation stage 1-2: generate teacher traces, then GATE-FILTER them.

For each seed task, the strong teacher answers via the map-reduce council
(`deliberate`, gate on). We keep ONLY gate-clean outputs and render them into
chat-format training targets that show the discipline we want a small student to
internalise: per-seat findings → a synthesised decision (or an explicit
abstention). No gate-violating output (fabricated citation / false arithmetic /
forbidden attribution) ever enters the dataset — the anti-circularity firewall.

Output is `{"messages":[...], "metadata":{...}}` JSONL, consumable directly by
tools/train_lora.py.

    # offline plumbing (mock teacher)
    python tools/distill_council_traces.py --teacher mock --limit 4 --out training/council/traces.jsonl
    # real traces (teacher != student family); rotate the key afterwards
    python tools/distill_council_traces.py --teacher openrouter:deepseek/deepseek-chat \
        --out training/council/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import deliberate  # noqa: E402
from agent.gate import check_response  # noqa: E402

TASKS = ROOT / "data" / "council_tasks.json"
OUT = ROOT / "training" / "council" / "traces.jsonl"

SYSTEM = (
    "You are a source-disciplined council advisor. Decompose the question across the "
    "relevant expert seats, state each seat's finding with a source where one is "
    "relied on, then give one synthesised decision. If you cannot verify a needed "
    "authority or figure, ABSTAIN and route to a human rather than guess. Label "
    "clearly as not professional advice; end with a 中文摘要."
)

ABSTAIN_MARK = "insufficient"


def _render_target(d) -> "tuple[str, str]":
    """Return (kind, assistant_text) from a Deliberation. kind: trace|abstention."""
    clean = [s for s in d.seats if s.ok and s.gatePassed]
    if not clean and ABSTAIN_MARK in (d.synthesis or "").lower():
        return "abstention", d.synthesis.strip()
    perspectives = "\n".join(f"- {s.displayName}: {s.answer}" for s in clean)
    body = (f"Perspectives:\n{perspectives}\n\nDecision: {d.synthesis.strip()}"
            if perspectives else d.synthesis.strip())
    return "trace", body


def generate_traces(tasks: list[dict], client, *, gate: bool = True, max_seats: int = 4,
                    abstain_cap: float = 0.25) -> "tuple[list[dict], dict]":
    """Generate gate-filtered chat traces. Returns (rows, stats)."""
    rows: list[dict] = []
    kept = dropped_dirty = dropped_empty = abstentions = 0
    for task in tasks:
        prompt = task["prompt"]
        d = deliberate(prompt, client=client, gate=gate, max_seats=max_seats)
        kind, target = _render_target(d)
        if not target.strip():
            dropped_empty += 1
            continue
        # FIREWALL: the rendered target must itself be gate-clean (no violations).
        if check_response(target, mode="advisor", question=prompt)["violations"]:
            dropped_dirty += 1
            continue
        if kind == "abstention":
            if abstentions >= max(1, int(abstain_cap * (len(tasks) or 1))):
                continue  # cap abstention share so the student isn't taught to always abstain
            abstentions += 1
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "metadata": {"taskId": task.get("id"), "councilId": d.councilId, "kind": kind,
                         "gatePassed": True, "labelStatus": "teacher-trace"},
        })
        kept += 1
    stats = {"tasks": len(tasks), "kept": kept, "abstentions": abstentions,
             "droppedDirty": dropped_dirty, "droppedEmpty": dropped_empty}
    return rows, stats


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="mock", help="teacher model spec (must differ from the student family)")
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="cap number of tasks (0 = all)")
    ap.add_argument("--max-seats", type=int, default=4)
    args = ap.parse_args(argv)

    from agent.model import default_client
    tasks = json.loads(Path(args.tasks).read_text("utf-8"))["tasks"]
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
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out  # output path outside the repo (e.g. /tmp) — show it absolute
    print(f"wrote {len(rows)} traces -> {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
