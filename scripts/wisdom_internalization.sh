#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
# End-to-end driver for the wisdom-internalization experiment across the author's
# two-device cluster (config/devices.local.json):
#
#   stage `trace`  -> Mac Studio M3 Ultra (mlx): harvest Sophia's own gated pipeline
#                     into verified, passport-stamped distillation traces.
#   stage `train`  -> DGX Spark (hf/cuda): QLoRA-distill the student on those traces,
#                     emitting checkpoints (the compute axis of the scaling law).
#   stage `ablate` -> Mac Studio (mlx): score the {base,student}x{gate,no-gate} matrix
#                     on the SEALED held-out split + the fabrication-vs-compute curve.
#
# Each stage runs on a different machine, so run them per-device (rsync the artifacts in
# between) — or `all` on one box for a mock end-to-end smoke test.
#
#   bash scripts/wisdom_internalization.sh all                    # mock smoke test (CI/offline)
#   BACKEND=mlx BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh smoke   # ~10s load+gen check
#   BACKEND=mlx BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh trace   # on the Mac
#   BACKEND=hf  BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh train   # on the Spark
#   BACKEND=mlx BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh ablate  # on the Mac
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-all}"
BACKEND="${BACKEND:-mock}"
BASE="${BASE:-mock-base}"
SEED="${SEED:-1337}"
OUT="${OUT:-models/sophia-4b-internalized}"
TRACES="training/council/distill_traces.jsonl"
PY="${PY:-python}"

smoke() {
  echo ">> [smoke] ~10s backend sanity check before the full pass"
  $PY -m tools.model_backends --backend "$BACKEND" --model "$BASE"
}

trace() {
  smoke
  echo ">> [trace] backend=$BACKEND base=$BASE  (run on the Mac for the real teacher farm)"
  $PY tools/gen_distill_traces.py --backend "$BACKEND" --model "$BASE" --seed "$SEED"
  echo ">> traces -> $TRACES   (rsync to the Spark before training)"
}

train() {
  echo ">> [train] backend=peft base=$BASE  (run on the DGX Spark)"
  if [[ "$BACKEND" == "mock" ]]; then
    echo "   (mock) skipping real training; assume checkpoints land in $OUT/checkpoint-*"
    return 0
  fi
  $PY tools/train_lora.py --backend peft --model "$BASE" --4bit \
      --guard --scaffold --distill --distill-file "$TRACES" \
      --anchor-kl "${ANCHOR_KL:-0.05}" \
      --selective-risk-stop --risk-regress-tol "${RISK_TOL:-0.05}" \
      --eval-every 25 --patience 4 --seed "$SEED" --output "$OUT"
  echo ">> adapter/checkpoints -> $OUT   (rsync to the Mac before ablation)"
}

ablate() {
  smoke
  echo ">> [ablate] backend=$BACKEND base=$BASE  (run on the Mac)"
  local ckpts=()
  if [[ "$BACKEND" != "mock" && -d "$OUT" ]]; then
    while IFS= read -r d; do ckpts+=("$d"); done < <(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' | sort)
  fi
  $PY tools/run_wisdom_ablation.py --backend "$BACKEND" --base "$BASE" \
      ${ckpts:+--student-adapter "${ckpts[-1]}" --checkpoints "${ckpts[@]}"}
}

case "$STAGE" in
  smoke)  smoke ;;
  trace)  trace ;;
  train)  train ;;
  ablate) ablate ;;
  all)    trace; train; ablate ;;
  *) echo "usage: $0 {smoke|trace|train|ablate|all}" >&2; exit 2 ;;
esac
