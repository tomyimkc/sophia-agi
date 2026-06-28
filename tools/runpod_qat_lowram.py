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

DEFAULT_GPU = "NVIDIA A100-SXM4-80GB"   # a real RunPod --gpu-type string
DEFAULT_SCHEME = "nvfp4"
DEFAULT_CALIB = "training/lora/calibration_datasheet.json"


# Suggested open base models per tier (the "similar params via sparsity" lever).
TIER_BASES = {
    "low":  "allenai/OLMoE-1B-7B-0924-Instruct",   # 7B total / 1B active, fully open, Apache-2.0
    "mid":  "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 47B / 13B
    "high": "mistralai/Mixtral-8x22B-Instruct-v0.1", # 141B / 39B
    "top":  "deepseek-ai/DeepSeek-V3",               # 671B / 37B (adapt + serve, NOT pretrain)
}

# A run target: where training + certification execute.
#   runpod      — paid x86 pods (registered/headline numbers; REPLICATION.md discipline)
#   local-spark — your DGX Spark (GB10 Blackwell, aarch64, 128GB unified, native FP4). FREE.
#                 bf16 LoRA + QAT only (NO bitsandbytes/--4bit, NO unsloth/flash-attn on aarch64);
#                 NVFP4 is Blackwell-native, so the Spark is the ideal low-RAM SERVE/benchmark box.
TARGETS = ("runpod", "local-spark")

# Default GPU count per tier on 80GB cards at 4-bit (matches training/sharding.py invariants:
# Mixtral-8x22B fits 2×80GB, DeepSeek-V3 671B fits 8×80GB). low/mid are single-GPU.
TIER_GPU_COUNT = {"low": 1, "mid": 1, "high": 2, "top": 8}


def build_run_plan(*, base_model: str | None, gpu: str, scheme: str, budget_usd: float | None,
                   branch: str, epochs: int, calib: str, target_bits: float,
                   target: str = "runpod", gpu_count: int = 1) -> dict:
    """Assemble the (calibrate → QAT-train → certify) command plan. Pure data; nothing executes.

    ``target='local-spark'`` plans a FREE local run on the DGX Spark (bf16 + --qat, no
    bitsandbytes/unsloth/flash-attn — all aarch64-blocked); no budget required. ``target='runpod'``
    plans a PAID pod run and requires a budget cap. Either way the planner only *plans*.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    local = target == "local-spark"
    # local-spark needs only a base model (free); runpod also needs a budget cap.
    ready = bool(base_model) and (local or (budget_usd is not None and budget_usd > 0))

    # Step 0 (free, local/CI): build the decontaminated calibration datasheet first.
    calib_cmd = ["python", "tools/run_calibration.py",
                 "--out", calib, "--target-bits", str(target_bits), "--dry-run"]

    if local:
        # On the Spark: bf16 LoRA + QAT directly (aarch64-safe — no --4bit/bitsandbytes,
        # sdpa attention not flash-attn). NVFP4 is the Blackwell-native serving grid.
        train_cmd = [
            "python", "tools/train_lora.py",
            "--model", base_model or "<SET --base-model>",
            "--qat", "--qat-scheme", scheme,
            "--epochs", str(epochs), "--dtype", "bf16", "--attn", "sdpa",
            "--dry-run",   # drop --dry-run to actually train on the Spark
        ]
        train_where = "local DGX Spark (FREE — your hardware, bf16, aarch64-safe)"
        certify_where = "local DGX Spark (FREE, after train — NVFP4 native)"
        cost_note = ("Runs on your DGX Spark — FREE (your hardware). Drop --dry-run on the train "
                     "command to start. Spark numbers are for ITERATION/benchmark, not the "
                     "registered result (REPLICATION.md: headline numbers stay on x86 RunPod).")
    else:
        # On a pod: runpod_train.py runs train_lora.py with --qat passed through. For a
        # multi-GPU MoE base (high/top tiers), add FSDP + expert-parallel sharding.
        extra = f"--qat --qat-scheme {scheme}"
        if gpu_count > 1:
            extra += f" --shard fsdp --expert-parallel --world-size {gpu_count}"
        # Uses runpod_train.py's REAL flags: --gpu-type / --gpu-count and the --extra-train-args
        # passthrough that forwards --qat/--shard to train_lora.py on the pod.
        train_cmd = [
            "python", "tools/runpod_train.py",
            "--model", base_model or "<SET --base-model>",
            "--gpu-type", gpu, "--gpu-count", str(gpu_count),
            "--branch", branch, "--epochs", str(epochs),
            "--extra-train-args", extra,
            "--dry-run",   # the planner never adds --yes; you add it after review
        ]
        train_where = "runpod (PAID)"
        certify_where = "runpod (PAID, after train)"
        cost_note = ("This planner provisions nothing. Launching requires you to add --yes to the "
                     "qat_train command and set a real budget cap on RunPod.")

    certify = {
        "description": "After training: emit FP16 vs quantized next-token distributions over the "
                       "calibration set and run the no-overclaim gate.",
        "gate": "serving.lowram_eval.LowRamGate",
        "contract": {"max_mean_kl": 0.05, "min_top1_agreement": 0.97,
                     "protected_max_kl": 0.10, "protected_min_agreement": 0.95},
        "inputs": {"full_probs": "fp16 base+adapter logits", "lowram_probs": f"{scheme} served logits",
                   "calibration": calib},
        "gpu_only_glue": "logit extraction is the on-device tensor work; the gate itself is CI-tested",
        "pass_means": "quantized artifact retained quality within the bound -> Boundary-3 evidence",
    }

    missing = []
    if not base_model:
        missing.append("base_model")
    if not local and not (budget_usd is not None and budget_usd > 0):
        missing.append("budget_usd")

    return {
        "schema": "sophia.runpod_qat_lowram.v2",
        "target": target,
        "ready_to_launch": ready,
        "missing": missing,
        "params": {"base_model": base_model, "gpu": gpu, "gpu_count": gpu_count, "scheme": scheme,
                   "budget_usd": budget_usd, "branch": branch, "epochs": epochs,
                   "target_bits": target_bits, "target": target},
        "sharded": gpu_count > 1 and not local,
        "steps": [
            {"stage": "calibrate", "where": "local/CI (free)", "command": calib_cmd},
            {"stage": "qat_train", "where": train_where, "command": train_cmd,
             "launch_note": ("drop --dry-run to train on the Spark" if local else
                             "review, then re-run WITHOUT --dry-run and WITH --yes to spend")},
            {"stage": "certify", "where": certify_where, **certify},
        ],
        "cost_note": cost_note,
        "honest_scope": (
            "Prepares the Boundary-3 evidence run. A 'low-RAM, capability-retained' claim is only "
            "earned if the certify step passes to the RESULTS.md bar (>=2 judge families, k>=0.40, "
            ">=3 seeds, 95% CIs excluding zero) — one pass is evidence, not the headline."
        ),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--target", choices=TARGETS, default="runpod",
                    help="where to run: 'local-spark' (FREE, your DGX Spark, bf16) or 'runpod' (PAID x86)")
    ap.add_argument("--tier", choices=tuple(TIER_BASES), default=None,
                    help="fill --base-model with the suggested open base for this tier (low/mid/high/top)")
    ap.add_argument("--base-model", default=None,
                    help="pretrained sparse-MoE (or dense) base to QAT-adapt; REQUIRED to launch")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="cost cap for a PAID runpod run; REQUIRED to launch on runpod (free on Spark)")
    ap.add_argument("--gpu-type", dest="gpu", default=DEFAULT_GPU,
                    help="RunPod --gpu-type for a PAID run (e.g. 'NVIDIA H200', 'NVIDIA A100-SXM4-80GB')")
    ap.add_argument("--gpu-count", dest="gpu_count", type=int, default=0,
                    help="GPUs for a PAID run (0 = tier default; >1 enables FSDP + expert-parallel)")
    ap.add_argument("--qat-scheme", dest="scheme", choices=("int8", "nvfp4"), default=DEFAULT_SCHEME)
    ap.add_argument("--branch", default="claude/sophia-v1-lowram-frontier")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--target-bits", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=None, help="write the plan JSON here")
    args = ap.parse_args(argv)
    # This tool only ever PLANS — it never launches; no dry-run toggle needed.

    base_model = args.base_model or (TIER_BASES[args.tier] if args.tier else None)
    gpu_count = args.gpu_count or (TIER_GPU_COUNT.get(args.tier, 1) if args.tier else 1)
    plan = build_run_plan(base_model=base_model, gpu=args.gpu, scheme=args.scheme,
                          budget_usd=args.budget_usd, branch=args.branch, epochs=args.epochs,
                          calib=args.calib, target_bits=args.target_bits, target=args.target,
                          gpu_count=gpu_count)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"QAT + low-RAM certification plan [target={plan['target']}] "
          f"(ready_to_launch={plan['ready_to_launch']})")
    if base_model:
        print(f"  base model: {base_model}")
    if not plan["ready_to_launch"]:
        hint = "set --base-model (or --tier)" + ("" if args.target == "local-spark"
                                                 else " and --budget-usd")
        print(f"  missing to launch: {', '.join(plan['missing'])} ({hint})")
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
    # The emitted command must use runpod_train.py's REAL flags (--gpu-type/--extra-train-args),
    # not the invented --gpu/--extra that would fail argparse.
    checks["uses_real_runpod_flags"] = ("--gpu-type" in qat_step["command"]
                                        and "--extra-train-args" in qat_step["command"])
    checks["no_invented_flags"] = ("--gpu" not in qat_step["command"]
                                   and "--extra" not in qat_step["command"])

    # Multi-GPU (high/top tier): the plan adds FSDP + expert-parallel sharding.
    pmg = build_run_plan(base_model="org/MoE", gpu="NVIDIA H200", scheme="nvfp4", budget_usd=200.0,
                         branch="b", epochs=1, calib=DEFAULT_CALIB, target_bits=4.5, gpu_count=8)
    mg = " ".join(next(s for s in pmg["steps"] if s["stage"] == "qat_train")["command"])
    checks["sharded_multi_gpu"] = (pmg["sharded"] and "--gpu-count 8" in mg
                                   and "--shard fsdp --expert-parallel --world-size 8" in mg)

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

    # 6. local-spark target: FREE (ready with base model, NO budget needed), bf16 + --qat,
    #    aarch64-safe (no --4bit/bitsandbytes/unsloth/flash-attn), never self-launches.
    sp = build_run_plan(base_model="allenai/OLMoE-1B-7B-0924-Instruct", gpu="-", scheme="nvfp4",
                        budget_usd=None, branch="b", epochs=1, calib=DEFAULT_CALIB,
                        target_bits=4.5, target="local-spark")
    checks["spark_free_no_budget"] = sp["ready_to_launch"] is True and sp["missing"] == []
    spark_train = " ".join(next(s for s in sp["steps"] if s["stage"] == "qat_train")["command"])
    checks["spark_uses_train_lora"] = "tools/train_lora.py" in spark_train
    checks["spark_is_bf16_qat"] = "--qat" in spark_train and "--dtype bf16" in spark_train
    checks["spark_aarch64_safe"] = not any(x in spark_train for x in ("--4bit", "unsloth", "flash_attention_2"))
    checks["spark_dry_by_default"] = "--dry-run" in spark_train and "--yes" not in spark_train
    detail["spark_train_cmd"] = spark_train

    # 7. --tier fills a base model; unknown target rejected.
    checks["tier_base_known"] = TIER_BASES["low"].startswith("allenai/OLMoE")
    try:
        build_run_plan(base_model="x", gpu="-", scheme="nvfp4", budget_usd=1.0, branch="b",
                       epochs=1, calib=DEFAULT_CALIB, target_bits=4.5, target="moon")
        checks["bad_target_rejected"] = False
    except ValueError:
        checks["bad_target_rejected"] = True

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
