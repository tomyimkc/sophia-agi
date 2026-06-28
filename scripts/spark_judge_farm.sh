#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
# Start a LOCAL judge farm on an NVIDIA DGX Spark: TWO distinct-vendor models on two
# vLLM ports, so a judged eval can satisfy the no-overclaim >=2-FAMILY gate WITHOUT
# metered cloud judges. provenance_bench.aggregate._distinct_families treats a self-hosted
# vllm/sglang server as an AGGREGATOR (like OpenRouter): the family is the MODEL vendor,
# so Qwen + Llama on two local ports count as TWO distinct families.
#
# Then run a judged eval pointing both judges at the local farm:
#   python tools/run_calibration_judge.py \
#     --judge vllm:Qwen/Qwen2.5-7B-Instruct@http://localhost:8000/v1 \
#     --judge vllm:meta-llama/Llama-3.3-8B-Instruct@http://localhost:8001/v1 ...
#
# (the '@base_url' suffix is the per-spec override added in agent.model.resolve_config —
# without it both judges would hit the vllm preset's default :8000 and the second would
# never be reached.)
#
# aarch64/GB10: prefer release vLLM containers; if a model won't fit on sm_121a, set
# QUANT=nvfp4. Smoke a port before the real eval: tools/run_local_judge_eval.py.
set -euo pipefail

JUDGE_A_MODEL="${JUDGE_A_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_A_PORT="${JUDGE_A_PORT:-8000}"
JUDGE_B_MODEL="${JUDGE_B_MODEL:-meta-llama/Llama-3.3-8B-Instruct}"
JUDGE_B_PORT="${JUDGE_B_PORT:-8001}"
LOG_DIR="${LOG_DIR:-./spark-logs}"
QUANT="${QUANT:-}"

mkdir -p "$LOG_DIR"
QUANT_FLAG=()
if [ -n "$QUANT" ]; then QUANT_FLAG=(--quantization "$QUANT"); fi

echo "[judge_farm] judge A: vLLM $JUDGE_A_MODEL on :$JUDGE_A_PORT"
echo "[judge_farm] judge B: vLLM $JUDGE_B_MODEL on :$JUDGE_B_PORT  (distinct vendor -> 2 families)"

vllm serve "$JUDGE_A_MODEL" --port "$JUDGE_A_PORT" "${QUANT_FLAG[@]}" \
  > "$LOG_DIR/vllm-judgeA.log" 2>&1 &
PID_A=$!
vllm serve "$JUDGE_B_MODEL" --port "$JUDGE_B_PORT" "${QUANT_FLAG[@]}" \
  > "$LOG_DIR/vllm-judgeB.log" 2>&1 &
PID_B=$!
echo "[judge_farm] pids A=$PID_A B=$PID_B"

cleanup() { echo "[judge_farm] stopping (pids $PID_A $PID_B)"; kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[judge_farm] judges starting. Spec for >=2-family judged evals:"
echo "  --judge vllm:${JUDGE_A_MODEL}@http://localhost:${JUDGE_A_PORT}/v1 \\"
echo "  --judge vllm:${JUDGE_B_MODEL}@http://localhost:${JUDGE_B_PORT}/v1"
echo "[judge_farm] waiting (Ctrl-C to stop)..."
wait
