# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shared helpers for Claude model lab (no API calls)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training" / "examples"
LAB_DIR = ROOT / "training" / "lab"
REVIEWS_DIR = LAB_DIR / "reviews"
DISTILL_DIR = LAB_DIR / "distill"
JUDGE_DIR = LAB_DIR / "judgements"
MODELS_DIR = ROOT / "models"
LAB_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_DIR = MODELS_DIR / "ollama"
HF_MODEL_DIR = MODELS_DIR / "hf-model-card"
BENCH_DIR = ROOT / "tests"
RUNS_DIR = ROOT / "benchmark" / "model_runs"
ATTRIBUTIONS = ROOT / "data" / "attributions.json"

SYSTEM_PROMPT = (
    "You are a Sophia AGI instructor using source discipline across philosophy, psychology, "
    "history, religion, and personality. Rules: deny false attributions, signal uncertainty for compiled "
    "or legendary authorship, keep traditions separate, label pop myths, end with concise 中文 summary."
)

DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_response(raw: str) -> dict | list:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        start = raw.find(start_char)
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(raw[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return json.loads(raw)


def example_to_text(path: Path) -> dict:
    payload = load_json(path)
    messages = payload.get("messages", [])
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    return {
        "file": path.name,
        "metadata": payload.get("metadata", {}),
        "user": user,
        "assistant": assistant[:2000],
    }


def sample_teacher_examples(limit: int, *, seed: int = 42) -> list[Path]:
    paths = sorted(TRAINING_DIR.glob("*claude*.json"))
    if not paths:
        paths = sorted(TRAINING_DIR.glob("*.json"))
    rng = random.Random(seed)
    if len(paths) <= limit:
        return paths
    return rng.sample(paths, limit)


def distill_specs_from_attributions(limit: int) -> list[dict]:
    records = load_json(ATTRIBUTIONS)
    specs: list[dict] = []
    for text_id, record in records.items():
        title = record.get("canonicalTitleEn", text_id)
        author = record.get("attributedAuthor", "")
        for forbidden in record.get("doNotAttributeTo", [])[:2]:
            specs.append({
                "domain": record.get("domain", "philosophy"),
                "user": f"Did {forbidden.replace('_', ' ').title()} write {title}?",
                "textIds": [text_id],
                "trap": f"deny-{forbidden}-{text_id}",
            })
        specs.append({
            "domain": record.get("domain", "philosophy"),
            "user": f"Who is the attributed author of {title}, and how certain is that?",
            "textIds": [text_id],
            "trap": f"author-{text_id}",
        })
    return specs[:limit]


def find_failed_cases(report_path: Path) -> list[dict]:
    report = load_json(report_path)
    domain = report.get("domain", "philosophy")
    bench = load_json(BENCH_DIR / f"benchmark-{domain}.json")
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
            "report": report_path.name,
        })
    return failures


def find_local_run_responses(report_path: Path) -> dict[str, str]:
    name = report_path.name.replace(".report.json", ".json")
    run_path = report_path.parent / name
    if not run_path.exists():
        return {}
    return load_json(run_path).get("responses", {})


def adapter_config(adapter_dir: Path | None) -> dict:
    base = {
        "baseModel": "Qwen/Qwen2.5-7B-Instruct",
        "adapterPath": str(adapter_dir) if adapter_dir else "",
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
    }
    if adapter_dir:
        cfg = adapter_dir / "sophia_lora_config.json"
        if cfg.exists():
            base.update(load_json(cfg))
    return base


def build_modelfile(cfg: dict) -> str:
    version = cfg.get("version", "0.0.0")
    base = cfg.get("baseModel", "Qwen/Qwen2.5-7B-Instruct")
    adapter = cfg.get("adapterPath", "")
    lines = [
        f"# Sophia AGI local model — v{version}",
        f"# Base: {base}",
        f"# Adapter: {adapter or '(train with tools/train_lora.py)'}",
        f"FROM {base}",
        "",
        "PARAMETER temperature 0.2",
        "PARAMETER top_p 0.9",
        "PARAMETER num_ctx 4096",
        "",
        f'SYSTEM """{SYSTEM_PROMPT}"""',
        "",
        "# Create:",
        "#   ollama create sophia-7b -f models/ollama/Modelfile",
        "# Run with post-gate:",
        "#   python sophia_mcp/server.py  # sophia_gate_check on outputs",
    ]
    return "\n".join(lines) + "\n"


def build_hf_model_card(cfg: dict, stats: dict) -> str:
    version = cfg.get("version", "0.0.0")
    base = cfg.get("baseModel", "Qwen/Qwen2.5-7B-Instruct")
    return f"""---
language:
- en
- zh
license: mit
base_model: {base}
tags:
- sophia-agi
- provenance
- source-discipline
- lora
---

# Sophia-7B (Sophia AGI adapter)

**Wisdom before intelligence.** LoRA adapter for provenance-aware instruction on `{base}`.

- **Project:** [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi)
- **Version:** {version}
- **Training examples:** {stats.get('trainingExamples', 'n/a')}
- **Benchmark total:** {stats.get('benchmarkTotal', 'n/a')}

## Always pair with runtime gate

`sophia_gate_check` (MCP) or `agent/gate.py` — weights alone do not guarantee trap safety.
"""