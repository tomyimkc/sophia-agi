#!/usr/bin/env python3
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "training" / "examples"
BENCH_DIR = ROOT / "tests"
OUT_DIR = ROOT / "training" / "lora"
DOMAINS = ("philosophy", "psychology", "history", "religion")


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bench_ids, bench_questions = load_benchmark_ids()
    train_rows: list[dict] = []
    holdout_rows: list[dict] = []

    for path in sorted(EXAMPLES.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
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

    manifest = {
        "totalExamples": len(train_rows) + len(holdout_rows),
        "trainCount": len(train_rows),
        "holdoutCount": len(holdout_rows),
        "benchmarkCaseCount": len(bench_ids),
        "holdoutReasons": {},
    }
    for row in holdout_rows:
        reason = row["holdoutReason"]
        manifest["holdoutReasons"][reason] = manifest["holdoutReasons"].get(reason, 0) + 1

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