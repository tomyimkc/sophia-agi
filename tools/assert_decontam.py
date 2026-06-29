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
    eval_prompt_set, jaccard as _jaccard, normalize, prompt_of, shingles as _shingles,
    _load_jsonl)

# Committed training surfaces the assertion guards (globs, relative to repo root).
# Includes verifier-gated DISTILLATION outputs (tools/distill_export.py): a frontier
# teacher may have memorised public benchmarks, so its "verified" traces are decontaminated
# against the same held-out eval as any hand-curated pack before they can become SFT data.
TRAIN_GLOBS = [
    "training/local_sophia_v3/mlx/train.jsonl",
    "training/local_sophia_v3/mlx/valid.jsonl",
    "training/local_sophia_v3/sft_*.jsonl",
    "training/local_sophia_v3/preference_pairs.jsonl",
    "training/distill_sft.jsonl",
    "training/distill_dpo.jsonl",
    "training/**/distill_sft.jsonl",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", type=float, default=0.9, help="near-duplicate Jaccard threshold")
    ap.add_argument("--shingle", type=int, default=5, help="word k-shingle size")
    ap.add_argument("--max-eval-shingle", type=int, default=4000,
                    help="cap eval prompts scanned for shingles (perf); exact check is always full")
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

    # Layer 2: content-shingle near-duplicates (only on the eval prompts, capped for perf).
    eval_list = list(evalset)[: args.max_eval_shingle]
    eval_sh = [(e, _shingles(e, args.shingle)) for e in eval_list]
    near = []
    seen_train = set()
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
          f"| exact-overlap={len(exact)} near-dup(J>={args.jaccard})={len(near)}")
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
