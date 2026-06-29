# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shared decontamination check for an external virtue battery (pillar 6).

Factored out of tools/assert_andreia_decontam.py so the Sophrosyne / Dikaiosyne
external batteries can assert the SAME contract — their prompts disjoint from every
training corpus in the repo — without duplicating the scan. Two layers, identical to
assert_decontam.py: exact/normalized-prompt disjointness + content-shingle
near-duplicate scan (Jaccard over word k-shingles). Conservative: scans ALL string
fields of each training row.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset_guard import _load_jsonl, normalize  # noqa: E402
from tools.assert_decontam import _jaccard, _shingles  # noqa: E402

# Every training/distillation corpus committed in the repo.
TRAIN_GLOBS = [
    "training/corpus.jsonl",
    "training/moral_gate_sft.jsonl",
    "training/lora/*.jsonl",
    "training/tool_use/*.jsonl",
    "training/council/*.jsonl",
    "training/hk_advisor/*.jsonl",
    "training/local_sophia_v3/**/*.jsonl",
    "training/**/*.jsonl",
    "data/**/*.jsonl",
]


def _row_texts(row: dict) -> list[str]:
    out: list[str] = []
    stack = [row]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            out.append(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


def assert_battery_decontam(prompts: list[str], *, label: str, jaccard: float = 0.6,
                            shingle: int = 5) -> int:
    """Return 0 if every prompt is disjoint from all training corpora, else 1."""
    eval_norm = {normalize(p): p for p in prompts}
    eval_sh = [(p, _shingles(p, shingle)) for p in prompts]

    seen_files: list[str] = []
    train_texts: list[str] = []
    for g in TRAIN_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ROOT))
            if rel in seen_files:
                continue
            seen_files.append(rel)
            for row in _load_jsonl(p):
                if isinstance(row, dict):
                    train_texts.extend(_row_texts(row))

    exact: list[str] = []
    near: list[tuple] = []
    seen_train: set[str] = set()
    for t in train_texts:
        nt = normalize(t)
        if not nt or nt in seen_train:
            continue
        seen_train.add(nt)
        if nt in eval_norm:
            exact.append(eval_norm[nt])
            continue
        tsh = _shingles(t, shingle)
        if not tsh:
            continue
        for e, esh in eval_sh:
            if _jaccard(tsh, esh) >= jaccard and normalize(e) != nt:
                near.append((round(_jaccard(tsh, esh), 3), t[:80], e[:80]))
                break

    clean = not exact and not near
    print(f"{label} DECONTAM: nBattery={len(prompts)} nTrainFiles={len(seen_files)} "
          f"nTrainText(unique)={len(seen_train)} | exact-overlap={len(exact)} "
          f"near-dup(J>={jaccard})={len(near)}")
    for pr in exact[:15]:
        print(f"  EXACT LEAK: «{pr[:90]}»")
    for j, t, e in near[:15]:
        print(f"  NEAR-DUP J={j}: train«{t}»  ~  battery«{e}»")
    if clean:
        print(f"OK — {label} external battery prompts are disjoint from all training corpora.")
        return 0
    print("FAIL — contamination found. Re-build the battery; do not re-weight (pillar 6).")
    return 1


__all__ = ["assert_battery_decontam", "TRAIN_GLOBS"]
