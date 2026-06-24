#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rejection-sampling fine-tuning (STaR / RFT) gated by the provenance gate.

The repo's unique speed advantage: the deterministic gate is the *reward model*.
For each seed task we sample N candidate answers from a model client, then run the
INTRINSIC fail-closed gate on each candidate -- ``check_response(text,
mode="advisor")["violations"]`` WITHOUT a question. A candidate is kept as an SFT
target iff it is gate-CLEAN (no fabricated citation / false arithmetic /
forbidden-lineage merge). No trap-grader is invoked here: passing a question would
run the attribution *positive-expectation* grader, which would wrongly reject
correctly-disciplined answers phrased differently -- so we never filter curated
targets with it (mirrors tools/train_lora.py:guard_filter).

CRITICAL fail-closed property: a gate-clean ABSTENTION (an answer that says it
cannot verify / routes to a human) is a CORRECT, reward-positive output and is
KEPT. RFT must never train abstention out of the student -- abstaining when an
authority cannot be verified is exactly the discipline we are distilling. We tag
each kept target as kind=answer|abstention and never drop a clean abstention.

Output is ``{"messages":[...], "metadata":{...}}`` JSONL, consumable directly by
tools/train_lora.py (same schema as tools/distill_council_traces.py). Identical
kept targets (same prompt+answer) are de-duplicated.

    # offline plumbing (mock client, writes nothing)
    python tools/run_rft.py --model mock --dry-run
    # real RFT pass
    python tools/run_rft.py --model openrouter:deepseek/deepseek-chat --samples 6 \
        --out training/rft/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402

TASKS = ROOT / "data" / "council_tasks.json"
OUT = ROOT / "training" / "rft" / "traces.jsonl"

SYSTEM = (
    "You are a source-disciplined advisor. Answer the question, stating each finding with "
    "a source where one is relied on, then give one decision. If you cannot verify a needed "
    "authority or figure, ABSTAIN and route to a human rather than guess. Label clearly as "
    "not professional advice; end with a 中文摘要."
)

# An answer that abstains is reward-POSITIVE. Mark it so it is never trained out.
ABSTAIN_MARKERS = (
    "abstain", "cannot verify", "can't verify", "insufficient", "unable to verify",
    "consult a", "seek a", "refer to a", "route to a", "需要核實", "無法核實", "諮詢專業",
)


def _is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in ABSTAIN_MARKERS)


def _gen(client, system: str, user: str) -> str:
    """Single completion as plain text; broken/failed client yields '' (mirrors
    agent.council_deliberate._gen)."""
    try:
        res = client.generate(system, user)
    except Exception:  # noqa: BLE001 - a broken client yields no content, not a crash
        return ""
    return (getattr(res, "text", "") or "").strip() if getattr(res, "ok", False) else ""


def sample_task(task: dict, client, *, samples: int) -> "tuple[list[dict], dict]":
    """Sample N candidates for one task, keep only gate-CLEAN ones (intrinsic check,
    no question). Returns (kept_rows, stats). Clean abstentions are kept and tagged."""
    prompt = task["prompt"]
    rows: list[dict] = []
    clean = dirty = empty = abstentions = 0
    for _ in range(samples):
        cand = _gen(client, SYSTEM, prompt)
        if not cand.strip():
            empty += 1
            continue
        # INTRINSIC fail-closed gate: NO question -> no trap grader.
        if check_response(cand, mode="advisor")["violations"]:
            dirty += 1
            continue
        clean += 1
        kind = "abstention" if _is_abstention(cand) else "answer"
        if kind == "abstention":
            abstentions += 1  # KEEP: a clean abstention is a correct, reward-positive target
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": cand},
            ],
            "metadata": {"taskId": task.get("id"), "kind": kind, "gatePassed": True,
                         "source": "rft", "labelStatus": "rft-sample"},
        })
    stats = {"clean": clean, "dirty": dirty, "empty": empty, "abstentions": abstentions}
    return rows, stats


def generate_rft(tasks: list[dict], client, *, samples: int = 4,
                 max_keep: int = 0) -> "tuple[list[dict], dict]":
    """Run rejection sampling over every task. De-dups identical kept targets
    (same prompt + assistant text). ``max_keep`` (0 = unlimited) caps the dataset."""
    rows: list[dict] = []
    seen: set = set()
    clean = dirty = empty = abstentions = deduped = 0
    for task in tasks:
        task_rows, st = sample_task(task, client, samples=samples)
        clean += st["clean"]
        dirty += st["dirty"]
        empty += st["empty"]
        abstentions += st["abstentions"]
        for row in task_rows:
            key = (row["metadata"]["taskId"], row["messages"][-1]["content"])
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            rows.append(row)
            if max_keep and len(rows) >= max_keep:
                stats = {"tasks": len(tasks), "samples": samples, "kept": len(rows),
                         "clean": clean, "dirty": dirty, "empty": empty,
                         "abstentions": abstentions, "deduped": deduped,
                         "cappedAt": max_keep}
                return rows, stats
    stats = {"tasks": len(tasks), "samples": samples, "kept": len(rows),
             "clean": clean, "dirty": dirty, "empty": empty,
             "abstentions": abstentions, "deduped": deduped}
    return rows, stats


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock", help="model spec (default mock = offline plumbing)")
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--samples", type=int, default=4, help="candidate answers sampled per task")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-keep", type=int, default=0, help="cap total kept targets (0 = unlimited)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + task count; write nothing")
    args = ap.parse_args(argv)

    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = ROOT / tasks_path
    tasks = json.loads(tasks_path.read_text("utf-8"))["tasks"]

    if args.dry_run:
        plan = {
            "model": args.model,
            "tasks": len(tasks),
            "samplesPerTask": args.samples,
            "maxCandidates": len(tasks) * args.samples,
            "maxKeep": args.max_keep or "unlimited",
            "gate": "intrinsic (mode=advisor, no question) — fail-closed",
            "abstentionPolicy": "gate-clean abstentions KEPT (reward-positive)",
            "out": args.out,
        }
        print("RFT plan (dry-run, nothing written):", flush=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    from agent.model import default_client
    rows, stats = generate_rft(tasks, default_client(args.model),
                               samples=args.samples, max_keep=args.max_keep)
    stats["model"] = args.model

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out  # output path outside the repo (e.g. /tmp) — show it absolute
    print(f"wrote {len(rows)} RFT target(s) -> {shown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
