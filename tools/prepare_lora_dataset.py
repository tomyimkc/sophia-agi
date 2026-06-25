#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prepare LoRA train split with benchmark holdout.

Excludes examples linked to benchmark case IDs so eval measures generalization.

Usage:
  python tools/prepare_lora_dataset.py --dry-run
  python tools/prepare_lora_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES = ROOT / "training" / "examples"
BENCH_DIR = ROOT / "tests"
OUT_DIR = ROOT / "training" / "lora"
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")


def load_benchmark_ids() -> set[str]:
    ids: set[str] = set()
    questions: set[str] = set()
    for domain in DOMAINS:
        bench = json.loads((BENCH_DIR / f"benchmark-{domain}.json").read_text(encoding="utf-8"))
        for case in bench.get("cases", []):
            ids.add(case["id"])
            questions.add(case["question"].strip().lower())
    return ids, questions


def is_holdout(example: dict, bench_ids: set[str], bench_questions: set[str]) -> str | None:
    meta = example.get("metadata") or {}
    if meta.get("benchmarkCase") in bench_ids:
        return f"benchmarkCase={meta['benchmarkCase']}"
    trap = str(meta.get("trap") or "")
    base_trap = re.sub(r"-r\d+$", "", trap)
    if base_trap in bench_ids:
        return f"trap={trap}"
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            q = str(msg.get("content", "")).strip().lower()
            if q in bench_questions:
                return "exact_benchmark_question"
    return None


def presplit_rows(rows: list[dict], max_tokens: int) -> tuple[list[dict], dict]:
    """Guarantee every row fits the token budget BEFORE it reaches a trainer.

    Reuses tools.split_long_training_rows.fit_rows (offline conservative heuristic),
    splitting over-long multi-turn rows at turn boundaries and re-deriving id/text.
    Prevents the silent mid-example truncation seen in the v2 MLX run.
    """
    from tools.split_long_training_rows import fit_rows

    src = [
        {"messages": r["messages"], "metadata": {**(r.get("metadata") or {}), "_srcId": r["id"]}}
        for r in rows
    ]
    fitted, report = fit_rows(src, max_tokens=max_tokens)
    seen: dict[str, int] = {}
    out: list[dict] = []
    for fr in fitted:
        meta = dict(fr.get("metadata") or {})
        src_id = meta.pop("_srcId", "row")
        msgs = fr["messages"]
        if meta.get("split"):
            seen[src_id] = seen.get(src_id, 0) + 1
            row_id = f"{src_id}-p{seen[src_id]}"
        else:
            row_id = src_id
        out.append({"id": row_id, "messages": msgs, "text": format_chat(msgs), "metadata": meta})
    return out, report


def format_chat(messages: list[dict]) -> str:
    """Simple chat template compatible with most instruct models."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LoRA dataset with benchmark holdout")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Pre-split rows over this token budget (offline heuristic)")
    parser.add_argument("--no-presplit", action="store_true",
                        help="Disable pre-split enforcement (rows may then truncate at train time)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from agent.training_safety import unsafe_reasons  # LoRA leakage guard (#7)

    bench_ids, bench_questions = load_benchmark_ids()
    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    dropped_unsafe = 0

    for path in sorted(EXAMPLES.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if unsafe_reasons(payload):           # never let confidential/PII reach the trainer
            dropped_unsafe += 1
            continue
        reason = is_holdout(payload, bench_ids, bench_questions)
        row = {
            "id": path.stem,
            "messages": payload.get("messages", []),
            "text": format_chat(payload.get("messages", [])),
            "metadata": payload.get("metadata", {}),
        }
        if reason:
            row["holdoutReason"] = reason
            holdout_rows.append(row)
        else:
            train_rows.append(row)

    # Aggregate holdout reasons from the source examples BEFORE pre-splitting (reasons
    # are per-source-example; splitting changes row counts, not why a row was held out).
    holdout_reasons: dict[str, int] = {}
    for row in holdout_rows:
        holdout_reasons[row["holdoutReason"]] = holdout_reasons.get(row["holdoutReason"], 0) + 1

    presplit_reports: dict = {}
    if not args.no_presplit:
        train_rows, presplit_reports["train"] = presplit_rows(train_rows, args.max_tokens)
        holdout_rows, presplit_reports["holdout"] = presplit_rows(holdout_rows, args.max_tokens)

    manifest = {
        "totalExamples": len(train_rows) + len(holdout_rows),
        "trainCount": len(train_rows),
        "holdoutCount": len(holdout_rows),
        "benchmarkCaseCount": len(bench_ids),
        "droppedUnsafe": dropped_unsafe,
        "presplit": {"enabled": not args.no_presplit, "maxTokens": args.max_tokens, **presplit_reports},
        "holdoutReasons": holdout_reasons,
    }

    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    holdout_path = args.out_dir / "holdout.jsonl"
    manifest_path = args.out_dir / "manifest.json"

    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with holdout_path.open("w", encoding="utf-8") as handle:
        for row in holdout_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {train_path} ({len(train_rows)} rows)")
    print(f"Wrote {holdout_path} ({len(holdout_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())