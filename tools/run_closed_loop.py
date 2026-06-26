#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drive the model↔harness closed-loop co-evolution.

Two modes:

  --mock (default; CI-safe, no GPU, no network)
    Exercises the full loop end-to-end: uplift -> distill -> (noop train) ->
    plasticity gate -> re-measure, asserting NON-DEGENERACY. The train step is
    a no-op so every cycle reports `no-candidate` (nothing to gate) — this
    proves the plumbing, not a model advance.

  --live (GPU recipe; not run in CI)
    Each cycle's distilled preference pairs are written to JSONL and a candidate
    is produced by shelling out to `tools/run_rlvr.py` on a CUDA pod (DPO/GRPO
    over the traces), gated through continual_plasticity, and re-measured. The
    live trainer is the seam; this driver wires the cycle around it.

Honest scope: this driver orchestrates an existing, independently-tested stack
(agent.uplift, agent.trace_distill, agent.continual_plasticity, tools/run_rlvr).
It trains nothing itself and changes no weights in --mock. A closing loop on a
real hidden suite is Level-3 evidence; this reports candidateOnly/level3Evidence
until such a gated run exists.

    python tools/run_closed_loop.py --mock
    python tools/run_closed_loop.py --mock --cycles 3 --out agi-proof/closed-loop/smoke.json
    python tools/run_closed_loop.py --live --base-model zai-org/glm-4-9b-chat-hf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import closed_loop as cl  # noqa: E402
from agent import trace_distill as td  # noqa: E402
from agent.model import default_client  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "closed-loop" / "smoke-report.json"

# A small, deterministic uplift suite in the exact shape agent.uplift expects
# ({"id","goal","mode"?,"mustInclude"?}). Agent/tool-use oriented so the domain
# is the executable-truth lane, not open-ended text. Swap for a real hidden
# pack (private/) before a Level-3 run.
DEFAULT_SUITE = [
    {
        "id": "tool_decision_1",
        "goal": "A repo test fails with ImportError. Decide the next diagnostic step and end with a Decision section.",
        "mode": "repo",
        "mustInclude": ["Decision"],
    },
    {
        "id": "tool_decision_2",
        "goal": "Summarize whether the current harness uplift justifies promoting a checkpoint. End with a Decision section.",
        "mode": "repo",
        "mustInclude": ["Decision"],
    },
]


def _live_train_step_factory(runs_root: Path, base_model: str, extra_rlvr_args: list[str]):
    """Build a TrainStep that writes distilled pairs to JSONL and shells out to
    tools/run_rlvr.py to produce a LoRA checkpoint spec. Returns a fresh spec
    like 'lora:<checkpoint-path>' on success, or a no-op outcome on failure.

    Live-only; never imported in --mock."""

    def _step(cycle: int, pairs: list, current_spec: str) -> cl.TrainOutcome:
        if not pairs:
            return cl.TrainOutcome(new_spec=current_spec, ran=False, notes="no distillable pairs this cycle")
        dpo_path = runs_root / f"cycle-{cycle}-dpo.jsonl"
        dpo_path.parent.mkdir(parents=True, exist_ok=True)
        dpo_path.write_text(td.to_jsonl(pairs) + "\n", encoding="utf-8")
        ckpt = ROOT / "training" / "closed-loop" / f"cycle-{cycle}"
        cmd = [
            sys.executable, str(ROOT / "tools" / "run_rlvr.py"),
            "--task", "provenance",  # reward family; the distilled traces add the signal
            "--model", current_spec.removeprefix("lora:") if current_spec.startswith("lora:") else base_model,
            "--output", str(ckpt),
            *extra_rlvr_args,
        ]
        try:
            rc = subprocess.call(cmd, cwd=str(ROOT))
        except FileNotFoundError as exc:
            return cl.TrainOutcome(new_spec=current_spec, ran=False, notes=f"trainer unavailable: {exc}")
        if rc != 0:
            return cl.TrainOutcome(new_spec=current_spec, ran=False, notes=f"trainer exited {rc}")
        return cl.TrainOutcome(
            new_spec=f"lora:{ckpt}", ran=True, artifact=str(ckpt),
            notes=f"DPO/GRPO over {len(pairs)} distilled pairs -> {ckpt}",
        )

    return _step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="offline no-GPU run (CI); train step is a no-op")
    ap.add_argument("--live", action="store_true", help="GPU run: shell out to tools/run_rlvr.py per cycle")
    ap.add_argument("--cycles", type=int, default=2, help="max distill->train->gate cycles (>=2 for non-degeneracy)")
    ap.add_argument("--suite", type=Path, default=None, help="uplift suite JSON (defaults to built-in agent suite)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "agent" / "memory" / "closed-loop")
    ap.add_argument("--provider", default="mock", help="model provider for --mock (default mock)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="base model for --live")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vllm", default="colocate", choices=["colocate", "server", "none"])
    ap.add_argument("--quant", default="4bit", choices=["4bit", "bf16"])
    args = ap.parse_args()

    if args.live and args.mock:
        print("--live and --mock are mutually exclusive")
        return 2
    mode = "live" if args.live else "mock"

    suite = json.loads(args.suite.read_text(encoding="utf-8")) if args.suite else DEFAULT_SUITE

    if mode == "mock":
        # Deterministic offline client via the env-driven mock provider; the loop
        # runs end-to-end with a no-op train step (every cycle => no-candidate).
        def make_client(_spec: str):
            return default_client(args.provider)

        train_step = cl.noop_train_step
        initial_spec = f"mock:{args.provider}"
    else:
        initial_spec = args.base_model

        def make_client(spec: str):
            # A promoted 'lora:<ckpt>' spec is loaded by pointing the provider at
            # the merged checkpoint; for --live the provider preset is set in env.
            return default_client(spec.removeprefix("lora:") if spec.startswith("lora:") else spec)

        train_step = _live_train_step_factory(
            args.runs_root, args.base_model,
            extra_rlvr_args=["--vllm", args.vllm, "--quant", args.quant],
        )

    report = cl.run_closed_loop(
        suite,
        suite_name="agent_tool_use",
        make_client=make_client,
        initial_spec=initial_spec,
        train_step=train_step,
        runs_root=args.runs_root,
        max_cycles=max(2, args.cycles),
        max_retries=args.max_retries,
        bootstrap_seed=args.seed,
    )
    payload = cl.write_report(report, args.out)
    payload["mode"] = mode
    payload["runAt"] = datetime.now().isoformat(timespec="seconds")
    # rewrite with the enriched fields (runAt/mode) for the on-disk artifact
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")
    if not report.non_degenerate:
        print("\nNON-DEGENERACY VIOLATION — loop halted. See haltReason.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
