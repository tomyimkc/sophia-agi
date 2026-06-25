#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch Qwen2.5-7B QLoRA SFT on RunPod (≥3 seeds). Requires RUNPOD_API_KEY.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BRANCH="${BRANCH:-claude/sophia-7b-train-verify}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
EPOCHS="${EPOCHS:-2}"
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "::error:: RUNPOD_API_KEY unset — record blocker in failure-ledger; dry-run only"
  python3 tools/runpod_train.py --dry-run --branch "$BRANCH" --model "$MODEL" --epochs "$EPOCHS"
  exit 2
fi
for SEED in 0 1 2; do
  echo "===== seed $SEED ====="
  RUNPOD_API_KEY="$RUNPOD_API_KEY" python3 tools/runpod_train.py \
    --yes --branch "$BRANCH" --model "$MODEL" --epochs "$EPOCHS" --seed "$SEED" \
    --name "sophia-7b-sft-seed${SEED}"
done
