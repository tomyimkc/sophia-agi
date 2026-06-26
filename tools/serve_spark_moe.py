#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase 4 of the Spark-MoE workflow: vLLM serve command for the MoE on a DGX Spark (GB10).

Emits the exact ``vllm serve`` invocation for the target sparse-MoE model on the GB10, with a
WORKING quantization. NVFP4 is forcibly excluded -- it is currently broken/underperforming on
GB10 (see docs/06-Roadmap/Spark-MoE-Training-Serve-Workflow.md). Also prints the SOPHIA env
to point the agent stack at the local endpoint and a smoke-test command.

``--dry-run`` prints the plan only (no launch). ``--yes`` launches vLLM (needs the Spark +
GPU + model weights). VERIFY the aarch64+sm_100 vLLM build runs YOUR MoE+quant before trusting
GRPO rollouts on it.

    python tools/serve_spark_moe.py --dry-run
    python tools/serve_spark_moe.py --model Qwen/Qwen3-Next-80B-A3B --quant fp8 --port 8000 --yes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B"
# NVFP4 is broken on GB10 today; only working quants are offered.
WORKING_QUANTS = ["fp8", "int4", "awq", "None"]


def serve_cmd(*, model: str, quant: str, max_model_len: int, port: int, gpu_mem_util: float) -> list[str]:
    cmd = ["vllm", "serve", model, "--port", str(port),
           "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(gpu_mem_util)]
    if quant and quant != "None":
        cmd += ["--quantization", quant]
    return cmd


def sophia_env(port: int) -> dict[str, str]:
    return {"SOPHIA_MODEL_PROVIDER": "vllm",
            "SOPHIA_MODEL_BASE_URL": f"http://localhost:{port}/v1"}


def plan(*, model: str, quant: str, max_model_len: int, port: int, gpu_mem_util: float) -> dict:
    """Pure plan: the serve command + Sophia env + smoke test. Forces NVFP4 -> fp8."""
    forced_from_nvfp4 = (quant or "").lower() == "nvfp4"
    if forced_from_nvfp4:
        quant = "fp8"
    return {
        "model": model,
        "quant": quant,
        "port": port,
        "forcedFromNvfp4": forced_from_nvfp4,
        "serveCmd": serve_cmd(model=model, quant=quant, max_model_len=max_model_len,
                              port=port, gpu_mem_util=gpu_mem_util),
        "sophiaEnv": sophia_env(port),
        "smokeTest": f"python tools/run_local_judge_eval.py --provider vllm --base-url http://localhost:{port}/v1",
        "warning": ("Verify the aarch64+sm_100 vLLM build runs THIS MoE+quant before trusting "
                    "GRPO rollouts. NVFP4 is currently broken on GB10."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default="fp8", help=f'one of {WORKING_QUANTS} (nvfp4 is forced to fp8)')
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true", help="print plan only; do not launch")
    ap.add_argument("--yes", action="store_true", help="launch vLLM (needs the Spark + GPU + weights)")
    args = ap.parse_args(argv)

    p = plan(model=args.model, quant=args.quant, max_model_len=args.max_model_len,
             port=args.port, gpu_mem_util=args.gpu_mem_util)
    if p["forcedFromNvfp4"]:
        print("WARNING: NVFP4 is currently broken/underperforming on GB10; forced quant -> fp8.", file=sys.stderr)
    print(json.dumps(p, ensure_ascii=False, indent=2))

    if args.dry_run or not args.yes:
        return 0
    # Launch (Spark only): replace this process with vllm so it owns the GPU lifecycle.
    cmd = p["serveCmd"]
    print("[serve] launching: " + " ".join(cmd), file=sys.stderr)
    os.execvp(cmd[0], cmd)  # noqa: S606 -- intentional process replace to run vllm
    return 0  # unreachable; kept for clarity


if __name__ == "__main__":
    raise SystemExit(main())
