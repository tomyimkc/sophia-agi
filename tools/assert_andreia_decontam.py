#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decontamination assertion for the Andreia EXTERNAL courage battery (pillar 6).

The shared tools/assert_decontam.py guards the local-Sophia training packs against the
EVAL_GLOBS surfaces; the Andreia external battery lives under agi-proof/benchmark-results/
and is not in EVAL_GLOBS, so this asserts ITS prompts are disjoint from every training
corpus present in the repo. Two layers, identical contract to assert_decontam.py:
  * exact/normalized-prompt disjointness;
  * content-shingle near-duplicate scan (Jaccard over word k-shingles).

Conservative: it scans ALL text fields of each training row (not just a 'prompt' key),
since a battery prompt could leak via any field. Exit 0 = clean, 1 = contamination.

    python3 tools/assert_andreia_decontam.py
    python3 tools/assert_andreia_decontam.py --jaccard 0.6 --shingle 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance_bench.dataset_guard import _load_jsonl, normalize  # noqa: E402
from tools.assert_decontam import _jaccard, _shingles  # noqa: E402

BATTERY = ROOT / "agi-proof" / "benchmark-results" / "andreia" / "andreia_external_battery.json"

# Every training/distillation corpus committed in the repo (a battery prompt must not
# appear in any of them). Globs are best-effort; missing files are skipped.
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
    """All string values in a row (recursively), so a leak via any field is caught."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", type=float, default=0.6, help="near-duplicate Jaccard threshold (strict)")
    ap.add_argument("--shingle", type=int, default=5, help="word k-shingle size")
    args = ap.parse_args()

    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    prompts = [c["text"] for c in battery["cases"]]
    eval_norm = {normalize(p): p for p in prompts}
    eval_sh = [(p, _shingles(p, args.shingle)) for p in prompts]

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
        tsh = _shingles(t, args.shingle)
        if not tsh:
            continue
        for e, esh in eval_sh:
            j = _jaccard(tsh, esh)
            if j >= args.jaccard and normalize(e) != nt:
                near.append((round(j, 3), t[:80], e[:80]))
                break

    clean = not exact and not near
    print(f"ANDREIA DECONTAM: nBattery={len(prompts)} nTrainFiles={len(seen_files)} "
          f"nTrainText(unique)={len(seen_train)} | exact-overlap={len(exact)} "
          f"near-dup(J>={args.jaccard})={len(near)}")
    for pr in exact[:15]:
        print(f"  EXACT LEAK: «{pr[:90]}»")
    for j, t, e in near[:15]:
        print(f"  NEAR-DUP J={j}: train«{t}»  ~  battery«{e}»")
    if clean:
        print("OK — Andreia external battery prompts are disjoint from all training corpora.")
        return 0
    print("FAIL — contamination found. Re-build the battery; do not re-weight (pillar 6).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
