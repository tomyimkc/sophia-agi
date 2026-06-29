#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent decontamination assertion for the semantic-grounding benchmark.

Re-checks the COMMITTED semantic-grounding eval prompts against the committed
training packs from scratch (exact/normalized overlap + content-shingle Jaccard
near-duplicates), reusing the same helpers as ``tools/assert_decontam.py`` so the
contract is identical. A train prompt that paraphrases an eval prompt is caught
here, not in the build's own report.

    python3 tools/assert_semantic_grounding_decontam.py
    python3 tools/assert_semantic_grounding_decontam.py --jaccard 0.9 --shingle 5
Exit 0 = clean, 1 = contamination.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance_bench.dataset_guard import _load_jsonl, normalize, prompt_of  # noqa: E402
from tools.assert_decontam import TRAIN_GLOBS, _jaccard, _shingles  # noqa: E402

EVAL_GLOBS = [
    "eval/semantic_grounding/data/d1_definition_faithfulness.jsonl",
    "eval/semantic_grounding/data/d2_compositional_derivation.jsonl",
]


def _rows() -> list[dict]:
    out: list[dict] = []
    for g in EVAL_GLOBS:
        for p in sorted(ROOT.glob(g)):
            out.extend(_load_jsonl(p))
    return out


def _eval_prompts() -> list[str]:
    return [r["prompt"] for r in _rows() if r.get("prompt")]


def _fold_disjointness() -> tuple[int, list[str]]:
    """Internal check: no train-fold prompt may equal an eval-fold prompt."""
    train = {normalize(r["prompt"]) for r in _rows() if r.get("fold") == "train" and r.get("prompt")}
    evalp = {normalize(r["prompt"]) for r in _rows() if r.get("fold") == "eval" and r.get("prompt")}
    return len(train) + len(evalp), sorted(train & evalp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", type=float, default=0.9)
    ap.add_argument("--shingle", type=int, default=5)
    args = ap.parse_args()

    eval_prompts = _eval_prompts()
    eval_norm = {normalize(e): e for e in eval_prompts}

    train_prompts: list[str] = []
    for g in TRAIN_GLOBS:
        for p in sorted(ROOT.glob(g)):
            for row in _load_jsonl(p):
                pr = prompt_of(row)
                if pr:
                    train_prompts.append(pr)

    exact = sorted({pr for pr in train_prompts if normalize(pr) in eval_norm})

    eval_sh = [(e, _shingles(e, args.shingle)) for e in eval_prompts]
    near = []
    seen = set()
    for pr in train_prompts:
        npr = normalize(pr)
        if npr in seen:
            continue
        seen.add(npr)
        tsh = _shingles(pr, args.shingle)
        if not tsh:
            continue
        for e, esh in eval_sh:
            j = _jaccard(tsh, esh)
            if j >= args.jaccard and npr != normalize(e):
                near.append((round(j, 3), pr[:80], e[:80]))
                break

    n_fold, fold_overlap = _fold_disjointness()
    clean = not exact and not near and not fold_overlap
    print(f"SEMANTIC-GROUNDING DECONTAM: nEval={len(eval_prompts)} nTrain(unique)={len(seen)} "
          f"| exact={len(exact)} near-dup(J>={args.jaccard})={len(near)} "
          f"| internal train/eval-fold overlap={len(fold_overlap)} (of {n_fold} prompts)")
    for pr in exact[:15]:
        print(f"  EXACT LEAK: «{pr[:90]}»")
    for j, t, e in near[:15]:
        print(f"  NEAR-DUP J={j}: train«{t}»  ~  eval«{e}»")
    for pr in fold_overlap[:15]:
        print(f"  FOLD OVERLAP: «{pr[:90]}»")
    if clean:
        print("OK — eval prompts disjoint from training packs, and the train/eval folds are disjoint.")
        return 0
    print("FAIL — contamination found. Fix before training/claiming.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
