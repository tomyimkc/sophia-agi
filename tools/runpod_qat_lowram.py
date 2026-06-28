#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prepare (do NOT auto-run) the paid RunPod QAT + low-RAM certification run.

This is the "one click away" launcher for the Boundary-3 evidence step: a quantization-aware
training run on a pretrained sparse-MoE base, followed by certifying the quantized artifact
against FP16 with ``serving/lowram_eval.py``. It **costs money and is gated**: by default it
only *plans* (prints the exact commands + writes a plan JSON), provisions nothing, and refuses
to emit an executable plan until you supply a base model and a budget. Actually launching is a
separate, explicit step you run after reviewing the plan.

Pipeline it prepares:
  1. **QAT SFT** — ``tools/runpod_train.py`` (which runs ``tools/train_lora.py`` on a pod) with
     ``--qat --qat-scheme`` passed through, so the released adapter co-adapts to its serving
     quantization.
  2. **Low-RAM certification** — on the pod, after training: emit FP16 vs quantized next-token
     distributions over a held-out, decontaminated calibration set and run
     ``serving.lowram_eval.LowRamGate`` (the no-overclaim gate). The logit-extraction glue is
     the GPU-only piece you run on the pod; this planner wires the command and the pass/fail
     contract, not the tensor ops.

Run ``--dry-run`` (default) to see the plan. To launch, review it, then run the emitted
``runpod_train.py`` command yourself with ``--yes`` (and set a real budget cap).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GPU = "A100-80G"
DEFAULT_SCHEME = "nvfp4"
DEFAULT_CALIB = "training/lora/calibration_datasheet.json"


def build_run_plan(*, base_model: str | None, gpu: str, scheme: str, budget_usd: float | None,
                   branch: str, epochs: int, calib: str, target_bits: float) -> dict:
    """Assemble the (training + certification) command plan. Pure data; nothing executes."""
    ready = bool(base_model) and (budget_usd is not None and budget_usd > 0)

    # Step 1: QAT training on a pod (runpod_train.py runs train_lora.py with --qat passthrough).
    train_cmd = [
        "python", "tools/runpod_train.py",
        "--model", base_model or "<SET --base-model>",
        "--gpu", gpu,
        "--branch", branch,
        "--epochs", str(epochs),
        "--train-only",
        # passthrough to train_lora.py on the pod:
        "--extra", f"--qat --qat-scheme {scheme}",
        # cost gate — the planner never adds --yes; you add it after review:
        "--dry-run",
    ]

    # Step 0 (free, local/CI): build the decontaminated calibration datasheet first.
    calib_cmd = [
        "python", "tools/run_calibration.py",
        "--out", calib, "--target-bits", str(target_bits), "--dry-run",
    ]

    # Step 2: on-pod certification after training (GPU-only logit extraction + the gate).
    certify = {
        "description": "After training, on the pod: emit FP16 vs quantized next-token "
                       "distributions over the calibration set and run the no-overclaim gate.",
        "gate": "serving.lowram_eval.LowRamGate",
        "contract": {"max_mean_kl": 0.05, "min_top1_agreement": 0.97,
                     "protected_max_kl": 0.10, "protected_min_agreement": 0.95},
        "inputs": {"full_probs": "fp16 base+adapter logits", "lowram_probs": f"{scheme} served logits",
                   "calibration": calib},
        "gpu_only_glue": "the logit extraction is the on-pod tensor work; the gate itself is CI-tested",
        "pass_means": "quantized artifact retained quality within the bound -> Boundary-3 evidence",
    }

    return {
        "schema": "sophia.runpod_qat_lowram.v1",
        "ready_to_launch": ready,
        "missing": [m for m, ok in (("base_model", bool(base_model)),
                                    ("budget_usd", budget_usd is not None and budget_usd > 0)) if not ok],
        "params": {"base_model": base_model, "gpu": gpu, "scheme": scheme,
                   "budget_usd": budget_usd, "branch": branch, "epochs": epochs,
                   "target_bits": target_bits},
        "steps": [
            {"stage": "calibrate", "where": "local/CI (free)", "command": calib_cmd},
            {"stage": "qat_train", "where": "runpod (PAID)", "command": train_cmd,
             "launch_note": "review, then re-run WITHOUT --dry-run and WITH --yes to spend"},
            {"stage": "certify", "where": "runpod (PAID, after train)", **certify},
        ],
        "cost_note": "This planner provisions nothing. Launching requires you to add --yes to the "
                     "qat_train command and set a real budget cap on RunPod.",
        "honest_scope": (
            "Prepares the Boundary-3 evidence run. A 'low-RAM, capability-retained' claim is only "
            "earned if the certify step passes to the RESULTS.md bar (>=2 judge families, k>=0.40, "
            ">=3 seeds, 95% CIs excluding zero) — one pass is evidence, not the headline."
        ),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base-model", default=None,
                    help="pretrained sparse-MoE (or dense) base to QAT-adapt; REQUIRED to launch")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="cost cap you accept for the paid run; REQUIRED to launch")
    ap.add_argument("--gpu", default=DEFAULT_GPU)
    ap.add_argument("--qat-scheme", dest="scheme", choices=("int8", "nvfp4"), default=DEFAULT_SCHEME)
    ap.add_argument("--branch", default="claude/sophia-v1-lowram-frontier")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--target-bits", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=None, help="write the plan JSON here")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="plan only (default; this tool never launches)")
    args = ap.parse_args(argv)

    plan = build_run_plan(base_model=args.base_model, gpu=args.gpu, scheme=args.scheme,
                          budget_usd=args.budget_usd, branch=args.branch, epochs=args.epochs,
                          calib=args.calib, target_bits=args.target_bits)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"RunPod QAT + low-RAM certification plan (ready_to_launch={plan['ready_to_launch']})")
    if not plan["ready_to_launch"]:
        print(f"  missing to launch: {', '.join(plan['missing'])} "
              f"(set --base-model and --budget-usd)")
    for step in plan["steps"]:
        cmd = step.get("command")
        line = " ".join(cmd) if cmd else step.get("description", "")
        print(f"  [{step['stage']:<9}] ({step['where']}) {line}")
    print(f"  cost: {plan['cost_note']}")
    return 0


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Without base model + budget, the plan is NOT ready to launch (cost gate).
    p0 = build_run_plan(base_model=None, gpu=DEFAULT_GPU, scheme="nvfp4", budget_usd=None,
                        branch="b", epochs=1, calib=DEFAULT_CALIB, target_bits=4.5)
    checks["not_ready_without_inputs"] = p0["ready_to_launch"] is False
    checks["reports_missing"] = set(p0["missing"]) == {"base_model", "budget_usd"}

    # 2. With both supplied, it is ready and the QAT flags are passed through.
    p1 = build_run_plan(base_model="org/MoE-base", gpu="A100-80G", scheme="nvfp4",
                        budget_usd=20.0, branch="b", epochs=1, calib=DEFAULT_CALIB, target_bits=4.5)
    checks["ready_with_inputs"] = p1["ready_to_launch"] is True
    qat_step = next(s for s in p1["steps"] if s["stage"] == "qat_train")
    joined = " ".join(qat_step["command"])
    checks["qat_flags_passthrough"] = "--qat --qat-scheme nvfp4" in joined
    checks["train_is_dry_by_default"] = "--dry-run" in qat_step["command"] and "--yes" not in joined

    # 3. The plan never self-launches: no --yes anywhere; cost note present.
    all_cmds = " ".join(" ".join(s.get("command", [])) for s in p1["steps"])
    checks["never_self_launches"] = "--yes" not in all_cmds
    checks["has_cost_note"] = "provisions nothing" in p1["cost_note"]

    # 4. The certify step wires the gate + contract (Boundary-3 evidence), and carries scope.
    cert = next(s for s in p1["steps"] if s["stage"] == "certify")
    checks["certify_uses_gate"] = cert["gate"] == "serving.lowram_eval.LowRamGate"
    checks["certify_has_contract"] = "max_mean_kl" in cert["contract"]
    checks["scope_present"] = "evidence, not the headline" in p1["honest_scope"]

    # 5. A calibration step precedes training (decontamination before quant).
    checks["calibrate_first"] = p1["steps"][0]["stage"] == "calibrate"

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        ok, detail = offline_invariants()
        print("RunPod QAT+low-RAM plan offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        raise SystemExit(0 if ok else 1)
    raise SystemExit(main())
