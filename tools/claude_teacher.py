#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Claude teacher loop — generate training examples toward 500+ pairs.

Usage:
  python tools/claude_teacher.py --dry-run          # count specs
  python tools/claude_teacher.py --target 500       # generate until 500 examples
  python tools/claude_teacher.py --batch-size 8 --limit 40
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

from agent.config import DATA_DIR, TRAINING_DIR, load_dotenv, normalize_api_keys  # noqa: E402
from agent.llm import complete  # noqa: E402

ATTRIBUTIONS = DATA_DIR / "attributions.json"
QUEUE_PATH = ROOT / "training" / "teacher_queue.json"
SYSTEM_PHIL = (
    "You are a Sophia AGI teacher generating ONE training example as JSON only. "
    "Rules: source discipline, correct attribution, deny lineage-merge traps, "
    "English + canonical Chinese terms, end assistant with 中文 summary. "
    "Output valid JSON object with keys: user, assistant, metadata (object)."
)

AUTHOR_LABELS = {
    "confucius": "Confucius",
    "laozi": "Laozi",
    "plato": "Plato",
    "socrates": "Socrates",
    "mencius": "Mencius",
    "zhuangzi": "Zhuangzi",
    "aristotle": "Aristotle",
    "mozi": "Mozi",
    "xunzi": "Xunzi",
    "han_feizi": "Han Fei",
    "sunzi": "Sun Tzu",
    "sigmund_freud": "Sigmund Freud",
    "festinger": "Leon Festinger",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_count() -> int:
    return len(list(TRAINING_DIR.glob("*.json")))


def next_index() -> int:
    nums = []
    for path in TRAINING_DIR.glob("*.json"):
        match = re.match(r"^(\d+)", path.name)
        if match:
            nums.append(int(match.group(1)))
    return (max(nums) if nums else 0) + 1


def label_author(author_id: str) -> str:
    return AUTHOR_LABELS.get(author_id, author_id.replace("_", " ").title())


def enumerate_specs(attributions: dict) -> list[dict]:
    specs: list[dict] = []
    for text_id, record in attributions.items():
        title_en = record.get("canonicalTitleEn") or text_id
        title_zh = record.get("canonicalTitleZh") or ""
        author = record.get("attributedAuthor", "")
        tradition = record.get("tradition", "")
        confidence = record.get("authorConfidence", "")
        for forbidden in record.get("doNotAttributeTo", []):
            specs.append({
                "kind": "deny_attribution",
                "domain": record.get("domain", "philosophy"),
                "user": f"Did {label_author(forbidden)} write {title_en}?",
                "textIds": [text_id],
                "traditions": [tradition] if tradition else [],
                "trap": f"deny {forbidden} -> {text_id}",
            })
        specs.append({
            "kind": "affirm_author",
            "domain": record.get("domain", "philosophy"),
            "user": f"Who is the traditionally attributed author of {title_en}?",
            "textIds": [text_id],
            "traditions": [tradition] if tradition else [],
            "trap": f"affirm {author} for {text_id}",
        })
        if confidence in ("compiled", "legendary", "none_extant"):
            specs.append({
                "kind": "signal_confidence",
                "domain": record.get("domain", "philosophy"),
                "user": f"How certain is the authorship of {title_en} ({title_zh})?",
                "textIds": [text_id],
                "traditions": [tradition] if tradition else [],
                "trap": f"confidence {confidence} for {text_id}",
            })

    for domain in ("philosophy", "psychology", "history", "religion", "personality"):
        bench_path = ROOT / "tests" / f"benchmark-{domain}.json"
        if not bench_path.exists():
            continue
        bench = load_json(bench_path)
        for case in bench.get("cases", []):
            specs.append({
                "kind": "benchmark",
                "domain": domain,
                "user": case["question"],
                "textIds": [],
                "recordIds": [],
                "traditions": [],
                "trap": case["id"],
                "benchmarkCase": case["id"],
            })

    domain_files = {
        "psychology": DATA_DIR / "psychology_concepts.json",
        "history": DATA_DIR / "history_events.json",
        "religion": DATA_DIR / "religion_concepts.json",
    }
    for domain, path in domain_files.items():
        if not path.exists():
            continue
        records = load_json(path)
        for record_id, record in records.items():
            title = record.get("canonicalTitleEn", record_id)
            for forbidden in record.get("doNotAttributeTo", []):
                specs.append({
                    "kind": "deny_domain",
                    "domain": domain,
                    "user": f"Did {label_author(forbidden)} originate {title}?",
                    "recordIds": [record_id],
                    "traditions": [record.get("tradition", "")] if record.get("tradition") else [],
                    "trap": f"{domain} deny {forbidden} -> {record_id}",
                })
            specs.append({
                "kind": "domain_record",
                "domain": domain,
                "user": f"Explain provenance for: {title}",
                "recordIds": [record_id],
                "traditions": [record.get("tradition", "")] if record.get("tradition") else [],
                "trap": f"{domain} explain {record_id}",
            })

    return specs


ROUND_HINTS = [
    "",
    "Round 2: paraphrase each user question naturally; keep the same attribution trap.",
    "Round 3: use a skeptical student tone ('I heard that...', 'Reddit says...').",
    "Round 4: ask in comparative form ('X vs Y — who actually wrote this?').",
    "Round 5: include one canonical Chinese term in the user question.",
    "Round 6: frame as exam-style short-answer prompts.",
    "Round 7: ask follow-up style ('If not X, then who and why?').",
    "Round 8: use pop-culture or classroom misconception framing.",
]


def build_batch_prompt(specs: list[dict], attributions: dict, *, round_idx: int = 0) -> str:
    context = json.dumps(
        {k: {
            "titleEn": v.get("canonicalTitleEn"),
            "titleZh": v.get("canonicalTitleZh"),
            "author": v.get("attributedAuthor"),
            "confidence": v.get("authorConfidence"),
            "doNotAttributeTo": v.get("doNotAttributeTo", []),
        } for k, v in list(attributions.items())[:15]},
        ensure_ascii=False,
    )
    round_hint = ROUND_HINTS[round_idx % len(ROUND_HINTS)]
    lines = [
        f"Generate {len(specs)} training examples. Return JSON array only.",
        f"Attribution sample context: {context}",
    ]
    if round_hint:
        lines.append(round_hint)
    lines.append("")
    for idx, spec in enumerate(specs):
        lines.append(f"Example {idx + 1}: domain={spec['domain']}, question={spec['user']}")
        if spec.get("textIds"):
            lines.append(f"  textIds: {spec['textIds']}")
        if spec.get("recordIds"):
            lines.append(f"  recordIds: {spec['recordIds']}")
        if spec.get("trap"):
            lines.append(f"  trap: {spec['trap']}")
    lines.append(
        "Each item: {user, assistant, metadata:{domain, textIds?, recordIds?, traditions?, source:'claude-teacher', trap?}}"
    )
    return "\n".join(lines)


def parse_batch_response(raw: str) -> list[dict]:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    return data


def write_example(index: int, item: dict, spec: dict) -> Path:
    domain = item.get("metadata", {}).get("domain") or spec.get("domain", "philosophy")
    metadata = {
        "source": "claude-teacher",
        "project": "sophia-agi",
        "domain": domain,
        "trap": spec.get("trap"),
    }
    if spec.get("textIds"):
        metadata["textIds"] = spec["textIds"]
    if spec.get("recordIds"):
        metadata["recordIds"] = spec["recordIds"]
    if spec.get("traditions"):
        metadata["traditions"] = [t for t in spec["traditions"] if t]
    if spec.get("benchmarkCase"):
        metadata["benchmarkCase"] = spec["benchmarkCase"]
    item_meta = item.get("metadata") or {}
    for key in ("textIds", "recordIds", "traditions"):
        if key in item_meta and item_meta[key]:
            metadata[key] = item_meta[key]

    system = (
        SYSTEM_PHIL
        if domain == "philosophy"
        else f"You are a {domain} instructor using source discipline. Include 中文摘要."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "metadata": metadata,
    }
    slug = re.sub(r"[^a-z0-9]+", "-", (spec.get("trap") or "example"))[:40].strip("-")
    path = TRAINING_DIR / f"{index:03d}-claude-{slug}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def export_corpus() -> int:
    corpus = ROOT / "training" / "corpus.jsonl"
    lines = [p.read_text(encoding="utf-8").strip() for p in sorted(TRAINING_DIR.glob("*.json"))]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> int:
    load_dotenv()
    normalize_api_keys()

    parser = argparse.ArgumentParser(description="Claude teacher loop for Sophia AGI")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="Max new examples this run (0=all needed)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    attributions = load_json(ATTRIBUTIONS)
    specs = enumerate_specs(attributions)
    current = existing_count()
    needed = max(0, args.target - current)
    if args.limit:
        needed = min(needed, args.limit)

    print(f"Specs available: {len(specs)} | examples now: {current} | target: {args.target} | to generate: {needed}")
    if args.dry_run or needed == 0:
        return 0

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    start_idx = next_index()
    written = 0
    cursor = 0
    round_idx = 0
    max_rounds = max(1, (needed // max(len(specs), 1)) + 2)
    while written < needed and round_idx < max_rounds:
        if cursor >= len(specs):
            cursor = 0
            round_idx += 1
            if round_idx >= max_rounds:
                break
            print(f"--- round {round_idx + 1} (paraphrase cycle) ---")
            continue
        batch_specs = specs[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        prompt = build_batch_prompt(batch_specs, attributions, round_idx=round_idx)
        try:
            raw = complete(
                "You output only valid JSON arrays for Sophia AGI training data. No markdown outside JSON.",
                prompt,
                max_tokens=6000,
            )
            items = parse_batch_response(raw)
        except Exception as exc:
            print(f"Batch failed at cursor {cursor} round {round_idx}: {exc}")
            continue

        for item, spec in zip(items, batch_specs):
            if written >= needed:
                break
            if not item.get("user") or not item.get("assistant"):
                continue
            spec_round = {**spec, "trap": f"{spec.get('trap', 'example')}-r{round_idx}"}
            path = write_example(start_idx + written, item, spec_round)
            written += 1
            print(f"  wrote {path.name}")

    export_corpus()
    print(f"Done: +{written} examples (total {existing_count()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())