#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent train/eval decontamination assertion over the COMMITTED artifacts.

The dataset build already decontaminates in-memory, but that trusts the build. This re-checks the
*committed* training packs against the eval surfaces from scratch — so a silent drift (an edited
EVAL_GLOBS, a hand-added row, a new eval file) is caught in CI, not the build's own report. Two
layers:
  * EXACT/normalized-prompt disjointness (same contract as provenance_bench.dataset_guard).
  * CONTENT-SHINGLE near-duplicate scan (Jaccard over word k-shingles) — catches a train prompt
    that paraphrases an eval prompt without being an exact match (the open upgrade noted in the
    measurement spec's pillar-6 line).

    python3 tools/assert_decontam.py            # exact + shingle, fail-closed
    python3 tools/assert_decontam.py --jaccard 0.9 --shingle 5
Exit 0 = clean, 1 = contamination (exact or near-duplicate above threshold).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance_bench.dataset_guard import (  # noqa: E402
    eval_prompt_set, normalize, prompt_of, _load_jsonl)

# Committed training surfaces the assertion guards (globs, relative to repo root).
TRAIN_GLOBS = [
    "training/local_sophia_v3/mlx/train.jsonl",
    "training/local_sophia_v3/mlx/valid.jsonl",
    "training/local_sophia_v3/sft_*.jsonl",
    "training/local_sophia_v3/preference_pairs.jsonl",
]


def _shingles(text: str, k: int) -> set:
    toks = normalize(text).split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", type=float, default=0.9, help="near-duplicate Jaccard threshold")
    ap.add_argument("--shingle", type=int, default=5, help="word k-shingle size")
    ap.add_argument("--max-eval-shingle", type=int, default=4000,
                    help="cap eval prompts scanned for shingles (perf); exact check is always full. "
                         "IGNORED under --accel (full coverage).")
    ap.add_argument("--accel", action="store_true",
                    help="use the sophia-lex Rust scanner for the near-dup layer if built — fast "
                         "enough to scan the FULL eval surface (no --max-eval-shingle cap). Auto "
                         "falls back to the capped Python scan if the binary is unavailable.")
    args = ap.parse_args()

    evalset = eval_prompt_set(root=ROOT)
    train_prompts: list[str] = []
    for g in TRAIN_GLOBS:
        for p in sorted(ROOT.glob(g)):
            for row in _load_jsonl(p):
                pr = prompt_of(row)
                if pr:
                    train_prompts.append(pr)

    # Layer 1: exact/normalized overlap.
    exact = sorted({pr for pr in train_prompts if normalize(pr) in evalset})

    # Layer 2: content-shingle near-duplicates.
    seen_train = set()
    near = []
    coverage = "python-capped"
    eval_list_full = list(evalset)

    accel_done = False
    if args.accel:
        try:
            import sys as _sys
            _sys.path.insert(0, str((ROOT / "tools").resolve()))
            from _lex_accel import decontam_near  # type: ignore
            # de-dup train by normalized prompt (mirror the Python loop) but keep
            # one representative original per unique prompt for reporting
            uniq_train: list[str] = []
            for pr in train_prompts:
                npr = normalize(pr)
                if npr in seen_train:
                    continue
                seen_train.add(npr)
                uniq_train.append(pr)
            near = decontam_near(uniq_train, eval_list_full,
                                 k=args.shingle, jaccard=args.jaccard)
            coverage = f"rust-full({len(eval_list_full)})"
            accel_done = True
        except Exception as exc:  # any bridge error -> Python oracle
            print(f"(decontam: accel unavailable, using capped Python scan — {exc})")
            seen_train = set()
            near = []

    if not accel_done:
        eval_list = eval_list_full[: args.max_eval_shingle]
        eval_sh = [(e, _shingles(e, args.shingle)) for e in eval_list]
        for pr in train_prompts:
            npr = normalize(pr)
            if npr in seen_train:
                continue
            seen_train.add(npr)
            tsh = _shingles(pr, args.shingle)
            if not tsh:
                continue
            for e, esh in eval_sh:
                j = _jaccard(tsh, esh)
                if j >= args.jaccard and npr != e:   # exact handled above
                    near.append((round(j, 3), pr[:80], e[:80]))
                    break

    clean = not exact and not near
    print(f"DECONTAM ASSERT: nTrain(unique-prompt)={len(seen_train)} nEval={len(evalset)} "
          f"coverage={coverage} | exact-overlap={len(exact)} near-dup(J>={args.jaccard})={len(near)}")
    for pr in exact[:15]:
        print(f"  EXACT LEAK: «{pr[:90]}»")
    for j, t, e in near[:15]:
        print(f"  NEAR-DUP J={j}: train«{t}»  ~  eval«{e}»")
    if clean:
        print("OK — committed training packs are disjoint from eval (exact + content-shingle).")
        return 0
    print("FAIL — contamination found. Fix the corpus/decontam config before training/claiming.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
