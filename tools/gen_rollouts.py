# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate gate-filtered SFT traces with the cache-stable rollout factory.

The Phase-1 data pump in CLI form. For each training problem it runs cache-shared
best-of-N (``pipeline.rollout``), keeps ONLY the verifier-passing solution (reward
== 1.0), and writes it as a validated ``AgentTrajectory`` JSONL row. These survivors
are the R1-style cold-start SFT set — long reasoning traces, every one machine-checked
by the same judge-free oracle the RLVR reward uses (sympy / dimensional / tests-pass),
so no hallucinated or contaminated trace leaks in.

Offline by default (``--model mock`` → ~0 kept, proves the pipeline). Point ``--model``
at a real provider (``vllm`` / ``deepseek`` / a served adapter — see
``agent/model.py`` presets and ``docs/06-Roadmap/DGX-Spark-Smoke-Run-Runbook.md``)
to produce real data. The append-only prefix + branch sharing is what makes this cheap
enough to run at scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.rollout import RolloutFactory  # noqa: E402
from pretraining.vertical_data.schemas import validate_agent_trajectory  # noqa: E402
from provenance_bench import (  # noqa: E402
    code_dataset,
    code_reward,
    math_dataset,
    math_reward,
    physics_dataset,
    physics_reward,
)

OUT_DIR = ROOT / "training" / "rollouts"

# task -> (dataset builder, target column, reward_for(answer, target) -> (score, detail))
TASKS = {
    "math": (math_dataset.build_math_rl_dataset, "gold",
             math_reward.reward_for_problem),
    "physics": (physics_dataset.build_physics_rl_dataset, "gold",
                physics_reward.reward_for_problem),
    "code": (code_dataset.build_code_rl_dataset, "test",
             code_reward.reward_for_task),
}


def run(task: str, *, model: str, n: int, seed: int, max_problems: int | None) -> dict:
    builder, col, reward_for = TASKS[task]
    data = builder(seed=seed)
    rows = data["train_rows"]
    if max_problems:
        rows = rows[:max_problems]

    factory = RolloutFactory(client=None if model == "mock" else _client(model))
    kept: list[dict] = []
    invalid = 0
    for row in rows:
        goal = row.get("prompt") or row.get("goal") or ""
        target = row[col]
        bon = factory.best_of_n(goal, gold=target, reward_for=lambda a, _t=target: reward_for(a, _t), n=n)
        if bon["reward"] != 1.0:
            continue  # gate: keep only verifier-passing traces
        rec = {
            "goal": bon["goal"],
            "steps": bon["steps"],
            "outcome": bon["outcome"],
            "reward": 1.0,
            "source": f"rollout-factory:{task}:{model}",
            "license": "Apache-2.0",
            "problem_id": row.get("problem_id"),
            "family": row.get("family"),
        }
        check = validate_agent_trajectory(rec)
        if not check["ok"]:
            invalid += 1
            continue
        kept.append(rec)

    return {
        "task": task, "model": model, "n": n,
        "problems": len(rows), "kept": len(kept), "invalid": invalid,
        "keepRate": round(len(kept) / len(rows), 4) if rows else 0.0,
        "note": ("mock model => ~0 kept (pipeline proof, not data)" if model == "mock"
                 else "verifier-gated SFT traces"),
        "traces": kept,
    }


def _client(model: str):
    from agent.model import default_client
    return default_client(model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate verifier-gated SFT traces.")
    ap.add_argument("--task", choices=sorted(TASKS), default="physics")
    ap.add_argument("--model", default="mock", help='provider spec (default "mock", offline)')
    ap.add_argument("--n", type=int, default=4, help="best-of-N samples per problem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = run(args.task, model=args.model, n=args.n, seed=args.seed,
                 max_problems=args.max_problems)
    out = args.out or (OUT_DIR / f"{args.task}-{args.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in report["traces"]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"GEN-ROLLOUTS {args.task} [{args.model}] kept={report['kept']}/{report['problems']} "
          f"(keepRate={report['keepRate']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
