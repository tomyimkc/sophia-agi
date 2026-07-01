# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase 4: failed benchmark eval -> draft training example via Claude."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent.benchmark_checks import DOMAIN_BENCH, load_json
from agent.config import DATA_DIR, ROOT, TRAINING_DIR
from agent.llm import complete

CORRECTIONS_DIR = ROOT / "training" / "corrections_pending"


def load_traditions() -> dict:
    return load_json(DATA_DIR / "traditions.json")


def find_failures(report_path: Path) -> list[dict]:
    report = load_json(report_path)
    domain = report.get("domain", "philosophy")
    bench = load_json(DOMAIN_BENCH[domain])
    case_map = {c["id"]: c for c in bench.get("cases", [])}
    failures = []
    for result in report.get("results", []):
        if result.get("passed"):
            continue
        case_id = result["id"]
        failures.append({
            "case_id": case_id,
            "domain": domain,
            "reasons": result.get("reasons", []),
            "case": case_map.get(case_id, {}),
            "model": report.get("model", "unknown"),
        })
    return failures


def draft_correction(failure: dict, bad_response: str) -> dict:
    case = failure.get("case", {})
    prompt = f"""A model failed Sophia AGI benchmark case '{failure['case_id']}'.

Question: {case.get('question', '')}
Failure reasons: {failure.get('reasons', [])}
Bad model answer excerpt: {bad_response[:1200]}

Write a corrected training example JSON with keys: user, assistant, metadata.
metadata must include domain='{failure['domain']}', source='correction-loop', benchmarkCase='{failure['case_id']}'.
Assistant must fix attribution traps and end with 中文 summary."""
    raw = complete(
        "Output one JSON object only for a corrected Sophia training example.",
        prompt,
        max_tokens=2000,
    )
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    item = json.loads(text)
    domain = failure["domain"]
    metadata = item.get("metadata") or {}
    metadata.setdefault("source", "correction-loop")
    metadata.setdefault("domain", domain)
    metadata.setdefault("benchmarkCase", failure["case_id"])
    metadata.setdefault("correctedFromModel", failure.get("model"))
    system = (
        "You are a precise philosophy instructor specializing in source discipline."
        if domain == "philosophy"
        else f"You are a {domain} instructor using source discipline."
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "metadata": metadata,
    }


def write_pending(example: dict, case_id: str) -> Path:
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-z0-9_-]+", "-", case_id.lower())
    path = CORRECTIONS_DIR / f"correction-{safe}.json"
    path.write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def promote_corrections() -> list[Path]:
    """Move approved pending corrections into training/examples."""
    promoted: list[Path] = []
    if not CORRECTIONS_DIR.exists():
        return promoted
    nums = [int(p.name[:3]) for p in TRAINING_DIR.glob("*.json") if re.match(r"^\d{3}", p.name)]
    index = (max(nums) if nums else 0) + 1
    for path in sorted(CORRECTIONS_DIR.glob("correction-*.json")):
        dest = TRAINING_DIR / f"{index:03d}-{path.stem}.json"
        dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.unlink()
        promoted.append(dest)
        index += 1
    return promoted