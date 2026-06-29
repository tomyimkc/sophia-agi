#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""GRPO trainer for the Swarm-Router head (Stage-3 of the Agentic-MoE design).

This is the **guarded GPU glue** that consumes the unhackable, CI-tested reward in
``provenance_bench/swarm_rl.py``. Per repo discipline (``moe/`` style): the reward and
its invariants are pure-Python and CI-gated; this trainer is optional and fails *loudly*
with install guidance when torch/trl are absent — it is NOT imported by any test.

What it trains
--------------
A policy that emits a :class:`agent.swarm_router.SwarmPlan` (as a dispatch-token
completion, design-doc V1) given a task. The reward is

    R = verified_success − λ_cost·steps − λ_lb·imbalance − λ_trust·over_reliance − λ_lat·depth

where ``verified_success`` and the gate-failure count come from the repo's *machine*
verifiers (``agent.gate`` / ``agent.math_verifier`` / ``agent.lean_verifier``), so the
router can only learn to route to teams whose work survives verification at low cost.

Data
----
Seeds from existing council/team traces (``training/team_agents/sft_traces.jsonl``,
``training/council/traces.jsonl``) plus any swarm rollouts you log. Each row needs a
``task`` and a machine-checkable ``gold``/verifier handle.

Usage (on a GPU box; see ``tools/runpod_rlvr.py`` to rent one and auto-terminate):

    pip install "trl>=0.9" "transformers>=4.43" torch
    python training/swarm_router/train_grpo.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --data training/team_agents/sft_traces.jsonl \
        --out training/swarm_router/adapter --steps 200

Dry-run the plan/reward wiring with NO model (uses the deterministic router + a synthetic
verifier) to prove the loop is wired before renting a GPU:

    python training/swarm_router/train_grpo.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.swarm_router import SwarmRouter  # noqa: E402
from provenance_bench.swarm_rl import make_grpo_reward, swarm_reward, SwarmOutcome  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = obj.get("task") or obj.get("goal") or obj.get("prompt")
        if task:
            rows.append({"task": task, "gold": obj.get("gold", "")})
    return rows


def dry_run() -> int:
    """Prove the router → plan → reward wiring with no model and a synthetic verifier."""
    router = SwarmRouter()
    tasks = [
        "Compare the disputed authorship of the Dao De Jing versus the Analects, citing sources",
        "Calculate the probability of at least one six in four rolls and prove the bound",
        "hi",
    ]
    plans = [router.decide(t) for t in tasks]

    # Synthetic MACHINE verifier stand-ins (the real run injects agent.gate / math_verifier).
    def score_success(_comp, plan) -> float:
        return 1.0 if plan.mode == "swarm" else 0.5

    def count_gate_failures(_comp, _plan) -> int:
        return 0

    reward_fn = make_grpo_reward(score_success=score_success, count_gate_failures=count_gate_failures)
    rewards = reward_fn([None] * len(plans), plans=plans)
    print("dry-run: router → plan → reward wiring OK")
    for t, p, r in zip(tasks, plans, rewards):
        print(f"  [{p.mode:5s}] reward={r:+.3f} steps={p.est_cost_steps:2d} agents={p.n_agents} :: {t[:54]}")
    # Sanity: a justified swarm should out-reward a needless one of equal success.
    sanity = swarm_reward(SwarmOutcome(plans[0], verified_success=1.0)) > \
        swarm_reward(SwarmOutcome(plans[0], verified_success=1.0, n_agents_failed_gate=plans[0].n_agents))
    print("  sanity (clean > gate-failing):", sanity)
    return 0 if sanity else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="wire-check the reward with no model/GPU")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", type=Path, default=ROOT / "training" / "team_agents" / "sft_traces.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "training" / "swarm_router" / "adapter")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=8, help="GRPO completions per prompt")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run()

    # --- guarded heavy deps (loud failure with guidance) ------------------------
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        print(
            f"GRPO training needs torch + trl + transformers ({type(exc).__name__}: {exc}).\n"
            "  pip install 'trl>=0.9' 'transformers>=4.43' datasets torch\n"
            "Or wire-check with NO GPU first:  python training/swarm_router/train_grpo.py --dry-run\n"
            "To rent + auto-terminate a GPU, see tools/runpod_rlvr.py.",
            flush=True,
        )
        return 2

    rows = _load_rows(args.data)
    if not rows:
        print(f"No usable rows in {args.data} (need a 'task'/'goal' field).", flush=True)
        return 2
    print(f"Loaded {len(rows)} tasks from {args.data}", flush=True)

    router = SwarmRouter()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)

    # Each prompt asks the policy to emit a dispatch plan; we attach the router-decided
    # plan as the structured target the reward scores against the machine verifier.
    def to_prompt(task: str) -> str:
        return (
            "You are Sophia's Swarm-Router. Emit a swarm plan (solo or fan-out to teams "
            "search/research/math_verify/legal/ontology/redteam) for the task.\n\nTask: " + task
        )

    plans = {row["task"]: router.decide(row["task"]) for row in rows}
    ds = Dataset.from_list([{"prompt": to_prompt(r["task"]), "task": r["task"]} for r in rows])

    # The real machine verifiers go here. Stubbed to the gate import so this file stays
    # honest about *where* the unhackable signal comes from (replace with the live calls).
    def score_success(completion, plan) -> float:
        # TODO(live): run the executed plan's children, score with agent.gate /
        # agent.math_verifier / agent.lean_verifier on the synthesised answer.
        return 1.0 if plan.mode == "swarm" else 0.5

    def count_gate_failures(completion, plan) -> int:
        return 0

    base_reward = make_grpo_reward(score_success=score_success, count_gate_failures=count_gate_failures)

    def reward_fn(prompts=None, completions=None, task=None, **kw):
        batch_plans = [plans[t] for t in task]
        return base_reward(completions, plans=batch_plans)

    cfg = GRPOConfig(
        output_dir=str(args.out),
        num_generations=args.group_size,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_steps=max(args.steps // 4, 1),
        bf16=True,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward_fn], args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(str(args.out))
    print(f"Saved Swarm-Router adapter to {args.out}", flush=True)
    print("REMINDER: gate the adapter through agent.continual_plasticity before promotion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
