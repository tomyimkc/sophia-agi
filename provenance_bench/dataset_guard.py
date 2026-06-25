# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Train/eval contamination guard for local-Sophia training packs.

The #1 way to fake training "uplift" is to leak eval prompts into the training set.
This module extracts the user-facing prompt from every training row and asserts it is
DISJOINT from the held-out eval/benchmark prompts (exact + normalized match). The
dataset build fails closed if any overlap is found.

Pure stdlib, deterministic, offline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Held-out sets that training must never overlap.
EVAL_GLOBS = ["eval/**/*.jsonl"]
EVAL_PACKS = ["agi-proof/baseline-ablation/abstain-pack-2026-06-22.json"]
TEAM_AGENTS_MANIFEST = ROOT / "data" / "team_agents_benchmark" / "manifest.json"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def prompt_of(row: dict) -> str | None:
    """User-facing prompt from an SFT (messages) or DPO (prompt) row, or a pack case."""
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content")
    # eval pack cases
    for k in ("question", "input", "text"):
        if isinstance(row.get(k), str):
            return row[k]
    return None


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def tool_use_benchmark_prompt_set(*, root: Path = ROOT) -> set[str]:
    """Normalized prompts from sealed tool-use benchmark (NOT in eval_prompt_set)."""
    out: set[str] = set()
    bench = root / "data" / "tool_use_benchmark" / "heldout_v1.jsonl"
    if bench.exists():
        for row in _load_jsonl(bench):
            pr = prompt_of(row)
            if pr:
                out.add(normalize(pr))
    return out


def hk_advisor_benchmark_prompt_set(*, root: Path = ROOT) -> set[str]:
    """Normalized prompts from sealed HK advisor benchmark (NOT in eval_prompt_set)."""
    out: set[str] = set()
    bench = root / "data" / "hk_advisor_benchmark" / "heldout_v1.jsonl"
    if bench.exists():
        for row in _load_jsonl(bench):
            pr = prompt_of(row)
            if pr:
                out.add(normalize(pr))
    return out


def eval_prompt_set(*, root: Path = ROOT) -> set[str]:
    """All normalized prompts from the held-out eval/benchmark surfaces."""
    out: set[str] = set()
    for g in EVAL_GLOBS:
        for p in root.glob(g):
            for row in _load_jsonl(p):
                pr = prompt_of(row)
                if pr:
                    out.add(normalize(pr))
    for rel in EVAL_PACKS:
        p = root / rel
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for case in data.get("cases", data if isinstance(data, list) else []):
                pr = prompt_of(case) if isinstance(case, dict) else None
                if pr:
                    out.add(normalize(pr))
    manifest_path = root / "data" / "team_agents_benchmark" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bench_dir = manifest_path.parent
        for key in ("heldout", "probe"):
            rel = manifest.get("files", {}).get(key)
            if rel:
                for row in _load_jsonl(bench_dir / rel):
                    pr = prompt_of(row)
                    if pr:
                        out.add(normalize(pr))
    return out


def check_contamination(train_rows: list[dict], eval_prompts: set[str] | None = None,
                        *, root: Path = ROOT) -> dict:
    """Return {clean, nTrain, nEval, overlap:[...]}. clean=False if any train prompt
    matches a held-out eval prompt."""
    evalset = eval_prompts if eval_prompts is not None else eval_prompt_set(root=root)
    overlap = []
    for row in train_rows:
        pr = prompt_of(row)
        if pr and normalize(pr) in evalset:
            overlap.append(pr)
    return {"clean": not overlap, "nTrain": len(train_rows), "nEval": len(evalset),
            "overlap": overlap[:20], "overlapCount": len(overlap)}
