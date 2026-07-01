#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
#
# Run a sophia train_lora job AND auto-feed its LIVE log to TrainWatch (http://localhost:8420),
# so the run shows up live with step/ETA/loss WITHOUT a forgotten third terminal.
#
# Why this exists: tools/train_lora.py has NO TrainWatch hook — it only prints the step-log
# ("epoch E/N step S/T (..%) loss=.. lr=.."). tools/trainwatch_bridge.py is the REQUIRED follower
# that tails that log and feeds TrainWatch's API. Forgetting to launch it is exactly why a running
# train shows nothing at :8420. This wrapper launches the follower for you, in the background, and
# tees the train's output to the log it follows. (Judging/cert/eval runs are NOT trains and never
# appear in TrainWatch — that activity lives in the bridge STATUS.json; see tools/spark_bridge.py.)
#
# Usage:
#   scripts/train_with_trainwatch.sh <run-name> -- <args passed to tools/train_lora.py ...>
# Example (v5 QAT):
#   scripts/train_with_trainwatch.sh olmoe-qat-v5 -- \
#     --model allenai/OLMoE-1B-7B-0924-Instruct --train training/lora/train.jsonl \
#     --output training/lora/checkpoints/olmoe-qat-spark-v5 --qat --qat-scheme nvfp4 --epochs 3
#
# Make sure `trainwatch serve` is running first (it serves the :8420 dashboard).
# Env overrides: PYTHON (default python3), TW_LOG (default training/lora/<name>.log), TW_IDLE (8).
set -euo pipefail

PY="${PYTHON:-python3}"
if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'   # print the header as help
  exit 0
fi

NAME="$1"; shift
[[ "${1:-}" == "--" ]] && shift   # tolerate the conventional '--' separator

LOG="${TW_LOG:-training/lora/${NAME}.log}"
IDLE="${TW_IDLE:-8}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"   # fresh log so the follower starts at step 0

echo "[train_with_trainwatch] dashboard : http://localhost:8420  (run 'trainwatch serve' if it is not up)"
echo "[train_with_trainwatch] run name  : ${NAME}"
echo "[train_with_trainwatch] live log  : ${LOG}"

# Start the TrainWatch follower in the BACKGROUND. It tolerates the log not existing yet (waits up
# to 60s) and exits ~${IDLE}s after it sees a done marker. If TrainWatch isn't installed the
# follower just dies — the train below is NEVER blocked by it (it runs in the foreground regardless).
"$PY" tools/trainwatch_bridge.py "$LOG" --name "$NAME" --idle-exit "$IDLE" &
TW_PID=$!
cleanup() { kill "$TW_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Run the actual training, teeing its step-log so the follower renders it live at :8420.
set +e
"$PY" tools/train_lora.py "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

# Emit a done marker so the follower marks the run completed (matches trainwatch_bridge._DONE),
# then give it up to ${IDLE}s to flush the final step before the EXIT trap reaps it.
echo "training finished (rc=${RC})" >> "$LOG"
for _ in $(seq 1 "$((IDLE + 4))"); do kill -0 "$TW_PID" 2>/dev/null || break; sleep 1; done

echo "[train_with_trainwatch] train exited rc=${RC}; TrainWatch run '${NAME}' finalized."
exit "$RC"
