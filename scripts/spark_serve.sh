#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
# Start the two LOCAL inference tiers of config/inference.local.spark.json on an NVIDIA
# DGX Spark (GB10 Grace Blackwell, aarch64): a vLLM orchestrator (:8000) and an SGLang
# constrained-decoding tool-caller (:30000). The escalation tier stays cloud (Claude) —
# it is not served here; its key is used only by agent.model when a tier routes to it.
#
# This is operational doc-as-code: it documents the exact serve commands the topology
# router (agent.inference_topology) points at. Models/ports are env-overridable so it
# tracks config/inference.local.json without hard-coding.
#
#   bash scripts/spark_serve.sh                 # start both tiers (foreground, logs to ./spark-logs)
#   ORCH_MODEL=Qwen/Qwen2.5-14B-Instruct bash scripts/spark_serve.sh
#
# aarch64/GB10 notes: prefer release vLLM/SGLang containers (build.nvidia.com/spark/{vllm,sglang})
# over head-of-tree. If a model fails to load on sm_121a, fall back to NVFP4 quantization
# (vllm --quantization nvfp4) to fit the 128 GB unified pool. mlx/unsloth prebuilt wheels
# are NOT used here (mlx is Apple-only; this script is the CUDA path).
set -euo pipefail

ORCH_MODEL="${ORCH_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
ORCH_PORT="${ORCH_PORT:-8000}"
TOOL_MODEL="${TOOL_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOOL_PORT="${TOOL_PORT:-30000}"
LOG_DIR="${LOG_DIR:-./spark-logs}"
QUANT="${QUANT:-}"   # e.g. "nvfp4" if a model won't fit on sm_121a

mkdir -p "$LOG_DIR"
QUANT_FLAG=()
if [ -n "$QUANT" ]; then QUANT_FLAG=(--quantization "$QUANT"); fi

echo "[spark_serve] orchestrator: vLLM $ORCH_MODEL on :$ORCH_PORT"
echo "[spark_serve] tool_calls : SGLang $TOOL_MODEL on :$TOOL_PORT (JSON-schema constrained)"
echo "[spark_serve] escalation : cloud (Claude) — served by the provider, not here"
echo "[spark_serve] logs -> $LOG_DIR"

# --- orchestrator tier (vLLM) ---
vllm serve "$ORCH_MODEL" --port "$ORCH_PORT" "${QUANT_FLAG[@]}" \
  > "$LOG_DIR/vllm-orchestrator.log" 2>&1 &
ORCH_PID=$!
echo "[spark_serve] vLLM orchestrator pid=$ORCH_PID (log $LOG_DIR/vllm-orchestrator.log)"

# --- tool_calls tier (SGLang, constrained decoding) ---
# sglang supports JSON-schema constrained decoding for guaranteed well-formed tool calls.
sglang.launch_server --model-path "$TOOL_MODEL" --port "$TOOL_PORT" \
  > "$LOG_DIR/sglang-toolcalls.log" 2>&1 &
TOOL_PID=$!
echo "[spark_serve] SGLang tool_calls pid=$TOOL_PID (log $LOG_DIR/sglang-toolcalls.log)"

cleanup() {
  echo "[spark_serve] stopping tiers (pids $ORCH_PID $TOOL_PID)"
  kill "$ORCH_PID" "$TOOL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[spark_serve] both tiers starting. Verify with:"
echo "  SOPHIA_MODEL_PROVIDER=vllm SOPHIA_MODEL_BASE_URL=http://localhost:$ORCH_PORT/v1 python tools/run_local_judge_eval.py"
echo "  python -m agent.inference_topology   # show resolved tiers from config/inference.local.json"
echo "[spark_serve] waiting on tiers (Ctrl-C to stop)..."
wait
