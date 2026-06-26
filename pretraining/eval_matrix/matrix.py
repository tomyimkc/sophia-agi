# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multi-dimensional evaluation coverage matrix (general + vertical, auto + human).

The data direction's "构建多维度模型评估体系（自动+人工），覆盖通用能力及垂直场景". An eval
suite is only trustworthy if you can *see its coverage*: which capability dimensions ×
which domains are tested, how many cases each cell has, and whether scoring is automatic
or human. This builds that matrix from the eval packs already in the repo, so gaps become
visible (empty cells = untested capability/domain combinations).

It classifies each discovered pack into a (dimension, domain, scoring) cell using a small,
declared keyword taxonomy (auditable, not magic), counts cases, and reports coverage plus
an explicit list of uncovered cells. Pure stdlib, offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Declared taxonomy — capability dimensions and domains we expect a pretraining-grade
# eval suite to cover. Keep explicit so coverage claims are auditable.
DIMENSIONS = ["knowledge", "reasoning", "provenance", "calibration", "safety",
              "code", "math", "agentic", "multilingual"]
DOMAINS = ["general", "philosophy", "psychology", "history", "religion",
           "law", "finance", "code", "math", "multimodal"]

# keyword -> dimension / domain (path + filename heuristics, declared and inspectable)
_DIM_KEYWORDS = {
    "fact_check": "provenance", "provenance": "provenance", "attribution": "provenance",
    "calibrat": "calibration", "conscience": "safety", "deception": "safety",
    "security": "safety", "redteam": "safety", "moral": "safety",
    "reason": "reasoning", "belief_revision": "reasoning", "gpqa": "reasoning",
    "coding": "code", "code": "code", "math": "math", "arithmetic": "math",
    "agent": "agentic", "team_agents": "agentic", "continual": "reasoning",
    "personality": "calibration", "constitution": "safety",
}
_DOMAIN_KEYWORDS = {
    "philosophy": "philosophy", "psychology": "psychology", "history": "history",
    "religion": "religion", "law": "law", "legal": "law", "finance": "finance",
    "economy": "finance", "code": "code", "coding": "code", "math": "math",
    "multimodal": "multimodal", "hk_advisor": "law",
}
# packs whose scoring involves human/LLM judgment vs purely automatic checkers
_HUMAN_HINTS = ("conscience", "moral", "arena", "personality", "deception")


def _classify(path: Path) -> "tuple[str, str, str]":
    s = str(path).lower()
    dim = next((v for k, v in _DIM_KEYWORDS.items() if k in s), "knowledge")
    dom = next((v for k, v in _DOMAIN_KEYWORDS.items() if k in s), "general")
    scoring = "human_or_judge" if any(h in s for h in _HUMAN_HINTS) else "automatic"
    return dim, dom, scoring


def _count_cases(path: Path) -> int:
    try:
        if path.suffix == ".jsonl":
            return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        for k in ("cases", "tasks", "items", "questions"):
            if isinstance(data.get(k), list):
                return len(data[k])
        return 1
    except Exception:  # noqa: BLE001
        return 0


def build_matrix(*, root: Path = ROOT, eval_dir: str = "eval") -> "dict[str, Any]":
    """Scan ``eval/`` packs and bucket them into the dimension×domain coverage matrix."""
    base = root / eval_dir
    packs = []
    for path in sorted(base.rglob("*")):
        if path.suffix not in (".json", ".jsonl") or not path.is_file():
            continue
        if "result" in str(path).lower() or "report" in str(path).lower():
            continue  # skip output artifacts, count input packs only
        dim, dom, scoring = _classify(path)
        n = _count_cases(path)
        if n <= 0:
            continue
        packs.append({"path": str(path.relative_to(root)), "dimension": dim,
                      "domain": dom, "scoring": scoring, "cases": n})

    # aggregate into cells
    cells: dict[tuple, dict] = {}
    for p in packs:
        key = (p["dimension"], p["domain"])
        cell = cells.setdefault(key, {"cases": 0, "packs": 0,
                                      "automatic": 0, "human_or_judge": 0})
        cell["cases"] += p["cases"]
        cell["packs"] += 1
        cell[p["scoring"]] += p["cases"]

    covered = {f"{d}|{dm}": cells[(d, dm)] for (d, dm) in cells}
    uncovered = [f"{d}|{dm}" for d in DIMENSIONS for dm in DOMAINS
                 if (d, dm) not in cells]
    total_cells = len(DIMENSIONS) * len(DOMAINS)

    return {
        "study": "multi-dimensional eval coverage matrix",
        "dimensions": DIMENSIONS,
        "domains": DOMAINS,
        "n_packs": len(packs),
        "total_cases": sum(p["cases"] for p in packs),
        "covered_cells": len(cells),
        "total_cells": total_cells,
        "coverage_fraction": round(len(cells) / total_cells, 4),
        "covered": covered,
        "uncovered_cells": uncovered,
        "packs": packs,
        "honesty_note": ("Classification uses a declared keyword taxonomy; coverage means a "
                         "pack exists for the cell, NOT that the cell is well-tested. Empty "
                         "cells are genuine gaps (e.g. multimodal is entirely uncovered)."),
    }


__all__ = ["build_matrix", "DIMENSIONS", "DOMAINS"]
