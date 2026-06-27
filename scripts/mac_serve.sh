#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
# Start the two LOCAL Apple-Silicon inference tiers described by
# config/inference.local.json: an MLX orchestrator (:8080) and a llama.cpp
# constrained-decoding tool-calls tier (:8081). Escalation stays cloud.
#
# The router intentionally refuses engine: mlx, so the orchestrator is served
# directly and not resolved through agent.inference_topology. The router should
# still resolve tool_calls + escalation cleanly.
set -euo pipefail

ORCH_MODEL="${ORCH_MODEL:-mlx-community/Qwen2.5-72B-Instruct-4bit}"
ORCH_PORT="${ORCH_PORT:-8080}"
ORCH_HOST="${ORCH_HOST:-127.0.0.1}"
TOOL_REPO="${TOOL_REPO:-zkkdinx/Qwen2.5-1.5B-Instruct-Q4_K_M-GGUF}"
TOOL_FILE="${TOOL_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
TOOL_PORT="${TOOL_PORT:-8081}"
TOOL_HOST="${TOOL_HOST:-127.0.0.1}"
TOOL_CTX="${TOOL_CTX:-8192}"
LOG_DIR="${LOG_DIR:-./mac-logs}"
GRAMMAR_FILE="${GRAMMAR_FILE:-$LOG_DIR/gateway_call.gbnf}"
START_ORCH="${START_ORCH:-1}"
START_TOOL="${START_TOOL:-1}"

mkdir -p "$LOG_DIR"

python - <<'PY' > "$GRAMMAR_FILE"
from agent.structured_output import load_schema, schema_to_gbnf
print(schema_to_gbnf(load_schema("gateway_call")), end="")
PY

echo "[mac_serve] orchestrator: MLX $ORCH_MODEL on $ORCH_HOST:$ORCH_PORT"
echo "[mac_serve] tool_calls : llama.cpp $TOOL_REPO/$TOOL_FILE on $TOOL_HOST:$TOOL_PORT (GBNF)"
echo "[mac_serve] grammar    : $GRAMMAR_FILE"
echo "[mac_serve] escalation : cloud (Claude) -- served by the provider, not here"
echo "[mac_serve] logs -> $LOG_DIR"

pids=()
if [ "$START_ORCH" = "1" ]; then
  mlx_lm.server --model "$ORCH_MODEL" --host "$ORCH_HOST" --port "$ORCH_PORT" \
    > "$LOG_DIR/mlx-orchestrator.log" 2>&1 &
  ORCH_PID="$!"
  pids+=("$ORCH_PID")
  echo "[mac_serve] MLX orchestrator pid=$ORCH_PID (log $LOG_DIR/mlx-orchestrator.log)"
else
  echo "[mac_serve] MLX orchestrator skipped (START_ORCH=$START_ORCH)"
fi

if [ "$START_TOOL" = "1" ]; then
  llama-server \
    --hf-repo "$TOOL_REPO" \
    --hf-file "$TOOL_FILE" \
    --host "$TOOL_HOST" \
    --port "$TOOL_PORT" \
    --ctx-size "$TOOL_CTX" \
    --grammar-file "$GRAMMAR_FILE" \
    > "$LOG_DIR/llamacpp-toolcalls.log" 2>&1 &
  TOOL_PID="$!"
  pids+=("$TOOL_PID")
  echo "[mac_serve] llama.cpp tool_calls pid=$TOOL_PID (log $LOG_DIR/llamacpp-toolcalls.log)"
else
  echo "[mac_serve] llama.cpp tool_calls skipped (START_TOOL=$START_TOOL)"
fi

cleanup() {
  if [ "${#pids[@]}" -gt 0 ]; then
    echo "[mac_serve] stopping tiers (pids ${pids[*]})"
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[mac_serve] verify with:"
echo "  python -m agent.inference_topology"
echo "  python tools/run_local_judge_eval.py --provider llamacpp --base-url http://localhost:$TOOL_PORT/v1 --judge-models $TOOL_REPO"
echo "[mac_serve] waiting on tiers (Ctrl-C to stop)..."
wait
