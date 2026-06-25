#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stage 3: DPO on hard negatives (≥3 seeds) atop per-seed SFT adapters from Stage 2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BRANCH="${BRANCH:-claude/sophia-7b-train-verify}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
EPOCHS="${EPOCHS:-1}"
DPO_PAIRS="${DPO_PAIRS:-training/local_sophia_7b/dpo_hard_negatives.jsonl}"
ARTIFACTS="${ARTIFACTS:-agi-proof/benchmark-results/runpod-train}"
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "::error:: RUNPOD_API_KEY unset — record blocker in failure-ledger; dry-run only"
  python3 tools/runpod_train.py --dry-run --dpo-pairs "$DPO_PAIRS" --branch "$BRANCH" --model "$MODEL" \
    --epochs "$EPOCHS" --sft-adapter-archive "$ARTIFACTS/SEED0.sophia-cuda-v1.tar.gz"
  exit 2
fi
for SEED in 0 1 2; do
  SFT_TAR="$ARTIFACTS/seed${SEED}.sophia-cuda-v1.tar.gz"
  if [[ ! -f "$SFT_TAR" ]]; then
    # fallback: any pod-id prefixed tarball for this seed from Stage-2 promotion manifest
    SFT_TAR="$(ls -1 "$ARTIFACTS"/*seed${SEED}*.sophia-cuda-v1.tar.gz 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$SFT_TAR" || ! -f "$SFT_TAR" ]]; then
    echo "::error:: missing SFT adapter tarball for seed $SEED under $ARTIFACTS"
    exit 1
  fi
  echo "===== DPO seed $SEED (SFT archive: $SFT_TAR) ====="
  RUNPOD_API_KEY="$RUNPOD_API_KEY" python3 tools/runpod_train.py \
    --yes --branch "$BRANCH" --model "$MODEL" --epochs "$EPOCHS" --seed "$SEED" \
    --dpo-pairs "$DPO_PAIRS" --sft-adapter-archive "$SFT_TAR" \
    --name "sophia-7b-dpo-seed${SEED}"
done
