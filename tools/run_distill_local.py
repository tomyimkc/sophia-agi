#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drive the FULL local distillation pipeline on an NVIDIA DGX Spark (no cloud teacher).

Stages:
  1. generate gate-filtered teacher traces (tools/distill_council_traces.py) with a LOCAL
     teacher served on the Spark (vllm/sglang), so traces are effectively free and the
     teacher != student family firewall holds;
  2. LoRA-SFT the student on the clean traces (tools/train_lora.py, peft backend, bf16);
  3. emit an adapter card stub (from agi-proof/mlops/adapter-card-template.md) filled with
     this run's facts, candidateOnly / canClaimAGI=false;
  (optional, --dpo) 4. build DPO pairs from gate-flagged student misses
     (tools/build_distill_dpo_pairs.py) — the v2 stretch rung.

This is the orchestrator; each stage is an existing tool. ``--dry-run`` prints the plan
(no model load, no GPU); ``--yes`` executes against the host CUDA stack. The distilled
student is an ITERATION-tier artifact — per REPLICATION.md the headline number stays on
x86 RunPod; do not cite a Spark-distilled adapter as the registered result.

    # plan (no GPU):
    python tools/run_distill_local.py --teacher vllm:Qwen/Qwen2.5-14B-Instruct@http://localhost:8000/v1 \
        --student-model Qwen/Qwen2.5-3B-Instruct --dry-run
    # run (needs the Spark + the teacher served):
    python tools/run_distill_local.py --teacher vllm:Qwen/Qwen2.5-14B-Instruct@http://localhost:8000/v1 \
        --student-model Qwen/Qwen2.5-3B-Instruct --yes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TASKS = ROOT / "data" / "council_tasks.json"
DEFAULT_TRACES = ROOT / "training" / "council" / "distill-local.traces.jsonl"
DEFAULT_ADAPTER = ROOT / "training" / "lora" / "checkpoints" / "local-sophia-distilled"


def _stage(cmd: list[str], *, dry_run: bool, log: Path | None = None) -> int:
    """Run one pipeline stage. Reuses tools.runpod_rlvr._stream so the console gets a live
    tee of the stage output (not just the log file) — matching runpod_train --local's UX."""
    from tools.runpod_rlvr import _stream

    print("[distill] " + " ".join(cmd))
    if dry_run:
        return 0
    return _stream(cmd, log if log is not None else Path("/dev/null"))


def _adapter_card(args, *, traces_count: int, adapter_dir: Path) -> dict:
    """Fill the mlops adapter-card template with this run's facts (candidateOnly)."""
    return {
        "schema": "sophia.adapter_card.draft.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "adapterId": adapter_dir.name,
            "baseModel": args.student_model,
            "adapterFormat": "LoRA (bf16, peft)",
            "stage": "distillation (gate-filtered council traces, local teacher)",
        },
        "claimBoundary": (
            "Iteration-tier artifact distilled on an NVIDIA DGX Spark (aarch64/sm_121a). "
            "NOT the registered x86 result — per REPLICATION.md the headline number stays on "
            "x86 RunPod. canClaimAGI=false; promote only via tools/promote_adapter.py."
        ),
        "training": {
            "teacher": args.teacher,
            "teacherStudentFamilyFirewall": "teacher != student family (council-distillation spec)",
            "traceRows": traces_count,
            "epochs": args.epochs,
            "seed": args.seed,
            "antiCircularityFirewall": "only gate-clean traces distilled (no fabricated citation/arithmetic)",
            "adapterPath": str(adapter_dir),
        },
        "nextSteps": [
            "3-condition uplift eval (base-alone / base+council+gate / distilled-alone) on the held-out split",
            "if uplift clears the no-overclaim gate on x86, re-run there and register the x86 adapter",
            "optional v2: DPO from gate-flagged student misses (tools/build_distill_dpo_pairs.py)",
        ],
    }


def build_plan(args) -> list[tuple[str, list[str]]]:
    """Return the (stage_name, argv) plan. Pure — no execution."""
    traces = Path(args.traces_out)
    adapter = Path(args.adapter_out)
    plan: list[tuple[str, list[str]]] = []
    plan.append(("distill", [
        sys.executable, str(ROOT / "tools" / "distill_council_traces.py"),
        "--teacher", args.teacher, "--tasks", args.tasks, "--out", str(traces),
        "--limit", str(args.limit),
    ]))
    train = [sys.executable, str(ROOT / "tools" / "train_lora.py"),
             "--model", args.student_model, "--train", str(traces),
             "--epochs", str(args.epochs), "--seed", str(args.seed),
             "--output", str(adapter)]
    if not args.four_bit:
        train += ["--rslora", "--scaffold", "--guard"]
    plan.append(("train", train))
    if args.dpo:
        dpo_pairs = adapter.parent / (adapter.name + ".dpo_pairs.jsonl")
        plan.append(("dpo-pairs", [
            sys.executable, str(ROOT / "tools" / "build_distill_dpo_pairs.py"),
            "--traces", str(traces), "--out", str(dpo_pairs),
        ]))
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True,
                    help="LOCAL teacher spec, e.g. vllm:Qwen/Qwen2.5-14B-Instruct@http://localhost:8000/v1 "
                         "(must differ from the student family)")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--traces-out", default=str(DEFAULT_TRACES))
    ap.add_argument("--adapter-out", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks (0=all)")
    ap.add_argument("--four-bit", action="store_true", help="QLoRA 4-bit (needs aarch64 bnb); default bf16")
    ap.add_argument("--dpo", action="store_true", help="also build DPO pairs from gate misses (v2)")
    ap.add_argument("--yes", action="store_true", help="execute the plan (default: dry-run only)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, do not execute")
    args = ap.parse_args(argv)

    plan = build_plan(args)
    print(f"[distill] local distillation plan ({len(plan)} stages); teacher={args.teacher} student={args.student_model}")
    for name, cmd in plan:
        print(f"[distill] [{name}] " + " ".join(cmd))

    if args.dry_run or not args.yes:
        print("[distill] dry-run only; pass --yes to execute (needs the teacher served + CUDA)")
        # Still emit the adapter-card DRAFT so the plan is reviewable without a GPU.
        card = _adapter_card(args, traces_count=0, adapter_dir=Path(args.adapter_out))
        print(json.dumps({"plan": [n for n, _ in plan], "adapterCardDraft": card}, indent=2, ensure_ascii=False))
        return 0

    traces = Path(args.traces_out)
    adapter = Path(args.adapter_out)
    log_dir = adapter.parent / "logs"
    for name, cmd in plan:
        code = _stage(cmd, dry_run=False, log=log_dir / f"{name}.log")
        if code != 0:
            print(f"[distill] stage {name} FAILED (exit {code}); see {log_dir / (name + '.log')}")
            return code
    traces_count = 0
    if traces.exists():
        with traces.open(encoding="utf-8") as fh:
            traces_count = sum(1 for _ in fh)
    card = _adapter_card(args, traces_count=traces_count, adapter_dir=adapter)
    card_path = adapter / "ADAPTER-CARD.json"
    adapter.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[distill] DONE. adapter={adapter} traces={traces_count} card={card_path} "
          f"(candidateOnly; canClaimAGI=false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
