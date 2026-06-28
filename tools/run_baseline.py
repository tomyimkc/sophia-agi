# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase 0 baseline: pass@1 on the sealed eval split, judge-free.

The pre-registered "before" for the math / physics / code RLVR arc. For each held-out
(FAMILY-disjoint) eval problem it generates one answer from the subject model and
scores it with the SAME verifiable reward the RLVR run optimizes — symbolic
equivalence (math), dimensional + numeric equivalence (physics), or hidden-tests-pass
(code). No LLM judge; the oracle decides.

The eval split is content-hashed (``*_dataset.sealed_hash``) and written into the
report, so the Phase 3 "after" run is provably measured on the identical, unpeeked
held-out set. A Wilson 95% interval is reported because a point pass-rate over a few
dozen problems is not a claim.

**Honesty:** with ``--model mock`` (the default, offline) the answers are gibberish,
so pass@1 is the *harness floor* (~0), NOT a capability number — it proves the
measurement path end-to-end and seals the split. A real subject model
(``--model deepseek`` / a vLLM endpoint / a trained adapter) produces the comparable
figure. The code task additionally needs ``SOPHIA_ALLOW_CODE_EXEC=1`` (sandboxed);
without it every code item scores 0 and the report says so.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import (  # noqa: E402
    code_dataset,
    code_reward,
    math_dataset,
    math_reward,
    physics_dataset,
    physics_reward,
)

OUT_DIR = ROOT / "agi-proof" / "benchmark-results"

# task -> (dataset builder, target column, scorer(answer, target) -> score in [-1,1]).
# All three builders expose train_rows/eval_rows + train_sealed/eval_sealed +
# family_intersection, so the runner is dataset-shape agnostic.
TASKS = {
    "math": (
        math_dataset.build_math_rl_dataset, "gold",
        lambda ans, tgt: math_reward.reward_for_problem(ans, tgt)[0],
    ),
    "physics": (
        physics_dataset.build_physics_rl_dataset, "gold",
        lambda ans, tgt: physics_reward.reward_for_problem(ans, tgt)[0],
    ),
    "code": (
        code_dataset.build_code_rl_dataset, "test",
        lambda ans, tgt: code_reward.reward_for_task(ans, tgt)[0],
    ),
}


def wilson_interval(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score 95% interval for k successes in n trials (pure-Python)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def run(task: str, *, model: str, seed: int, max_items: int | None) -> dict:
    builder, col, scorer = TASKS[task]
    data = builder(seed=seed)
    rows = data["eval_rows"]
    if max_items:
        rows = rows[:max_items]

    from agent.model import default_client

    client = default_client(model)
    system = "Solve the problem. Show concise working and give the final answer."

    passed = 0
    per_item = []
    for row in rows:
        res = client.generate(system, row["prompt"])
        answer = getattr(res, "text", "") or ""
        score = scorer(answer, row[col])
        ok = score > 0
        passed += int(ok)
        per_item.append({"id": row.get("problem_id"), "family": row.get("family"),
                         "passed": ok})

    n = len(rows)
    lo, hi = wilson_interval(passed, n)
    note = "mock model => harness floor, not a capability number" if model == "mock" else "subject-model pass@1"
    if task == "code" and not code_reward.exec_enabled():
        note += "; SOPHIA_ALLOW_CODE_EXEC unset => code items cannot pass"
    return {
        "benchmark": f"baseline-{task}",
        "task": task,
        "model": model,
        "split": "eval",
        "n": n,
        "passed": passed,
        "passAt1": round(passed / n, 4) if n else None,
        "ci95Wilson": [round(lo, 4), round(hi, 4)],
        "evalSealed": data["eval_sealed"],
        "trainSealed": data["train_sealed"],
        "evalFamilies": sorted({r.get("family") for r in data["eval_rows"] if r.get("family")}),
        "contaminationFree": data["family_intersection"] == [],
        "claim": "pre-registered baseline (NOT a capability claim)",
        "note": note,
        "perItem": per_item,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 0 sealed-eval baseline (pass@1).")
    ap.add_argument("--task", choices=sorted(TASKS), default="physics")
    ap.add_argument("--model", default="mock", help='subject model spec (default "mock", offline)')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=None, help="cap eval items (smoke test)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = run(args.task, model=args.model, seed=args.seed, max_items=args.max_items)
    out = args.out or (OUT_DIR / f"baseline-{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"BASELINE {args.task} [{args.model}] "
          f"pass@1={report['passAt1']} CI95={report['ci95Wilson']} "
          f"n={report['n']} sealed={report['evalSealed']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
