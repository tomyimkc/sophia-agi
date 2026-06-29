#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build a decontaminated NATURALISTIC slice for the Focus-Efficiency-Frontier (thesis §5).

The synthetic battery is single-axis BY CONSTRUCTION (the goal-relevant key is always
placed non-recently), so its GO is construct-bounded. This builds the honest contrast: a
slice of REAL multi-document QA (LongBench v1 hotpotqa / 2wikimqa) where the gold-bearing
passage sits at a NATURAL, varying position among distractors — so recency baselines
sometimes win and a pass would actually generalise.

Each source item -> a task in our schema: the question is the goal, the context's
``Passage N:`` blocks become the competing segments, and the dataset's GOLD answers are
the objective grading anchor (no synthetic key-survival proxy). Decontaminated vs the
committed training corpus and split public/private. Only a SMALL derived slice is committed
(question + passages + gold), with the upstream source + license recorded for provenance.

Source: LongBench (zai-org/LongBench, formerly THUDM) — data.zip; tasks hotpotqa/2wikimqa
derive from HotpotQA (CC BY-SA 4.0) and 2WikiMultihopQA (Apache-2.0/MIT upstream). LongBench
itself is MIT-licensed. The raw 113MB zip is NOT committed; point --zip at a local copy.

    LONGBENCH_ZIP=/path/data.zip python tools/build_longbench_focus_slice.py --public 60 --private 20
    python tools/build_longbench_focus_slice.py --check     # verify committed slice + decontam
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.context_manager import estimate_tokens  # noqa: E402
from tools.assert_decontam import _jaccard, _shingles  # noqa: E402
from tools.build_focus_battery import _train_shingles  # noqa: E402 — reuse the training-corpus shingles
from tools.eval_stats import mde_at_n  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "prosoche"
SLICE_PATH = RESULTS_DIR / "focus-longbench-slice.json"
DECONTAM_PATH = RESULTS_DIR / "focus-longbench-decontam.json"

TASKS = ["hotpotqa", "2wikimqa"]   # English multi-doc QA (gold passage at varying positions)
DECONTAM_JACCARD = 0.5
DECONTAM_SHINGLE = 5
MAX_PASSAGES = 12          # cap competing segments per task (keep budgets sane)
PACK_BUDGET = 300          # token budget the 3 arms compete for (fits ~2-3 passages of ~10)
_PASSAGE = re.compile(r"Passage\s+\d+:\s*\n")


def _passages(context: str) -> list[str]:
    parts = [p.strip() for p in _PASSAGE.split(context) if p.strip()]
    return parts[:MAX_PASSAGES]


def _iter_source(zip_path: Path, task: str):
    with zipfile.ZipFile(zip_path) as z:
        name = next((n for n in z.namelist() if n.endswith(f"{task}.jsonl") and "_e." not in n), None)
        if not name:
            return
        with z.open(name) as f:
            for ln in f:
                yield json.loads(ln)


def build_slice(zip_path: Path, *, public: int, private: int) -> list[dict]:
    want = public + private
    per_task = (want // len(TASKS)) + 1
    tasks: list[dict] = []
    for t in TASKS:
        taken = 0
        for row in _iter_source(zip_path, t):
            if taken >= per_task:
                break
            passages = _passages(row.get("context", ""))
            gold = row.get("answers") or []
            q = (row.get("input") or "").strip()
            # Keep items that genuinely exercise packing: multiple passages, a short gold,
            # and a context that does NOT fit the budget (so an arm must choose).
            if len(passages) < 4 or not gold or not q:
                continue
            if sum(estimate_tokens(p) for p in passages) <= PACK_BUDGET:
                continue
            tasks.append({
                "id": f"{t}-{row.get('_id', taken)[:12]}",
                "question": q,
                "goldAnswers": [str(a) for a in gold],
                "passages": passages,
                "dataset": t,
                "budgetTokens": PACK_BUDGET,
            })
            taken += 1
    # Deterministic order, then split: interleave tasks so both datasets appear in each split.
    tasks.sort(key=lambda x: x["id"])
    for i, x in enumerate(tasks):
        x["split"] = "public" if i < public else "private"
    return tasks[:want]


def decontam_receipt(tasks: list[dict]) -> dict:
    train = _train_shingles(DECONTAM_SHINGLE)
    worst, worst_id = 0.0, None
    for t in tasks:
        text = t["question"] + " " + " ".join(t["passages"]) + " " + " ".join(t["goldAnswers"])
        j = _jaccard(_shingles(text, DECONTAM_SHINGLE), train) if train else 0.0
        if j > worst:
            worst, worst_id = j, t["id"]
    return {
        "schema": "sophia.focus_longbench_decontam.v1",
        "shingle": DECONTAM_SHINGLE, "jaccardThreshold": DECONTAM_JACCARD,
        "maxJaccardVsTrain": round(worst, 4), "worstTaskId": worst_id,
        "clean": worst < DECONTAM_JACCARD,
        "note": "Max word-5-shingle Jaccard of any task text vs the committed TRAINING corpus. "
                "NB: this guards train/eval leakage in THIS repo; it does NOT bound the subject "
                "model's parametric familiarity with HotpotQA (the closed-book control measures that).",
    }


def build_payload(zip_path: Path, *, public: int, private: int) -> dict:
    tasks = build_slice(zip_path, public=public, private=private)
    pub = [t for t in tasks if t["split"] == "public"]
    priv = [t for t in tasks if t["split"] == "private"]
    return {
        "schema": "sophia.focus_longbench_slice.v1",
        "source": "LongBench v1 (zai-org/LongBench, MIT); tasks hotpotqa (HotpotQA, CC BY-SA 4.0) "
                  "+ 2wikimqa (2WikiMultihopQA). Derived eval slice only; raw data.zip not redistributed.",
        "naturalistic": True,
        "n": len(tasks), "publicN": len(pub), "privateN": len(priv),
        "mdeAtPublicN": round(mde_at_n(len(pub), p0=0.5), 4) if pub else None,
        "packBudgetTokens": PACK_BUDGET, "maxPassages": MAX_PASSAGES,
        "note": "REAL multi-doc QA: gold passage at a NATURAL varying position. Solved is graded "
                "against the dataset's GOLD answer (EM/F1 + judges), not a synthetic key-survival "
                "proxy. A closed-book control measures parametric leakage. canClaimAGI:false.",
        "tasks": tasks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=os.environ.get("LONGBENCH_ZIP", ""), help="path to LongBench data.zip")
    ap.add_argument("--public", type=int, default=60)
    ap.add_argument("--private", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        if not SLICE_PATH.exists() or not DECONTAM_PATH.exists():
            print("FOCUS LONGBENCH: slice/decontam not committed", file=sys.stderr)
            return 1
        rec = json.loads(DECONTAM_PATH.read_text(encoding="utf-8"))
        sl = json.loads(SLICE_PATH.read_text(encoding="utf-8"))
        if not rec.get("clean"):
            print(f"FOCUS LONGBENCH: CONTAMINATED maxJaccard {rec['maxJaccardVsTrain']}", file=sys.stderr)
            return 1
        print(f"FOCUS LONGBENCH: OK — N={sl['n']} (public {sl['publicN']}, private {sl['privateN']}), "
              f"maxJaccardVsTrain={rec['maxJaccardVsTrain']} (clean)")
        return 0

    zp = Path(args.zip)
    if not zp.exists():
        print(f"::error:: LongBench data.zip not found at {zp!r}. Set LONGBENCH_ZIP or --zip.", file=sys.stderr)
        return 2
    payload = build_payload(zp, public=args.public, private=args.private)
    receipt = decontam_receipt(payload["tasks"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    DECONTAM_PATH.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SLICE_PATH.relative_to(ROOT)} (N={payload['n']}, public={payload['publicN']}) + decontam "
          f"(maxJaccard={receipt['maxJaccardVsTrain']}, clean={receipt['clean']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
