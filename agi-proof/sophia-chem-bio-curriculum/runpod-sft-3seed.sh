#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch Qwen2.5-7B QLoRA SFT on the chem-bio curriculum over RunPod (≥3 seeds).
# Requires RUNPOD_API_KEY. POLICY: run via GitHub Actions, NOT a local/agent SSH shell.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BRANCH="${BRANCH:-claude/llm-chemistry-biology-training-4dnakz}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
EPOCHS="${EPOCHS:-2}"
TRAIN_DATA="${TRAIN_DATA:-training/sophia-chem-bio-curriculum/sft_all.jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-training/sophia-chem-bio-curriculum/checkpoints/sophia-cuda-v1}"
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "::error:: RUNPOD_API_KEY unset — record blocker in failure-ledger; dry-run only"
  python3 tools/runpod_train.py --dry-run --branch "$BRANCH" --model "$MODEL" \
    --epochs "$EPOCHS" --train-data "$TRAIN_DATA" \
    --adapter-dir "/workspace/sophia-runpod/sophia-agi/$ADAPTER_DIR" --train-only
  exit 2
fi
for SEED in 0 1 2; do
  echo "===== chem-bio SFT seed $SEED ====="
  RUNPOD_API_KEY="$RUNPOD_API_KEY" python3 tools/runpod_train.py \
    --yes --branch "$BRANCH" --model "$MODEL" --epochs "$EPOCHS" --seed "$SEED" \
    --train-data "$TRAIN_DATA" \
    --adapter-dir "/workspace/sophia-runpod/sophia-agi/${ADAPTER_DIR}-seed${SEED}" \
    --train-only --name "sophia-chem-bio-sft-seed${SEED}"
done
