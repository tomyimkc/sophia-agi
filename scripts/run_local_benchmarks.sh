#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
#
# run_local_benchmarks.sh — ONE-COMMAND local benchmark runner for the DGX Spark + Mac Studio cluster.
#
# This script ORCHESTRATES already-wired tools. It does NOT invent results, and it does NOT
# run any GPU/training/judging job by default — `--dry-run` (the default) only prints the plan.
# Pass `--execute` to actually run, and gate the expensive / destructive steps behind their own
# flags (`--run-train`). Read it top-to-bottom before you run it; it is safe to read and to dry-run.
#
# It runs, in order:
#   Benchmark A (headline ≥2-family VALIDATED judging of the M3-SFT source-discipline uplift):
#     1. bring-up reminder (servers you must start by hand on Spark + Mac)
#     2. smoke test ........ tools/run_local_judge_eval.py  (connectivity, CI-safe)
#     3. judge ............. tools/judge_pilot_answers.py    (per-seed answer files, ≥3 seeds, 2 families)
#     4. aggregate + gate .. tools/run_lora_uplift_validation.py  (the no-overclaim VALIDATED gate)
#   Benchmark B (low-RAM NVFP4 certification, Boundary-3):
#     5. OPTIONAL QAT train  tools/train_lora.py --qat --qat-scheme nvfp4   (only with --run-train)
#     6. certify ........... tools/certify_lowram.py --scheme nvfp4
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# PASS BARS (cited exactly from the tools / docs — DO NOT relax these):
#
# Benchmark A — the no-overclaim VALIDATED gate (tools/run_lora_uplift_validation.aggregate,
#   config/inference.local.mac-judge.json gate{}, RESULTS.md). VALIDATED requires ALL of:
#     • notMock subject
#     • ≥2 DISTINCT judge families  (here: qwen via Spark vLLM + mlx via Mac MLX)
#     • judge != subject            (subject lineage allenai/olmoe — clear of qwen & meta-llama)
#     • mean pairwise Cohen's κ ≥ 0.40   (KAPPA_FLOOR)
#     • ≥3 seeds
#     • 95% bootstrap CI on the content-uplift delta EXCLUDES zero
#   NOTE (honesty): the 2026-06-29 session measured κ = 0.24 (< 0.40), so the formal gate is
#   UNMET → result is CANDIDATE, not VALIDATED. This script reports whatever the gate returns;
#   it never forces a VALIDATED verdict.
#
# Benchmark B — low-RAM NVFP4 cert (serving/lowram_eval.LowRamGate / certify_lowram.DEFAULT_CONTRACT):
#     • mean KL ≤ 0.05   AND   top-1 agreement ≥ 0.97        (overall slice)
#     • protected slice: KL ≤ 0.10  AND  agreement ≥ 0.95
#   NOTE (honesty): the best measured run (v3) was mean_kl 0.045 (PASSES ≤0.05) but top1 0.906
#   (< 0.97) → NO-GO on the strict gate. The script reports the gate verdict honestly.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# COST-GUARD (wisdom-gpu-prebaked, .claude/skills/wisdom-gpu-prebaked/SKILL.md):
#   • Owned hardware (DGX Spark + Mac Studio) is FREE — run everything here on it.
#   • RunPod is the ONLY metered/paid path; this script never touches RunPod.
#   • After ANY GPU session: CONFIRM ZERO LEAKED PODS (three documented credit-burn incidents).
#     Cheap validation first (limit=24, runs=1); watch the first ~6 min for restart loops.
# ─────────────────────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Repo root (this script lives in scripts/) ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"

# ── Tunable placeholders (override via env) ───────────────────────────────────────────────────
# Hostnames/ports for the two-box judge farm. Fill these in (link-local IPs, Tailscale, or DNS).
SPARK_HOST="${SPARK_HOST:-SPARK_HOST}"          # DGX Spark — vLLM Qwen judge (family 'qwen')
SPARK_PORT="${SPARK_PORT:-8000}"
MAC_HOST="${MAC_HOST:-MAC_HOST}"                # Mac Studio — MLX Llama judge (family 'mlx')
MAC_PORT="${MAC_PORT:-8080}"

# Judge models (must stay DIFFERENT vendors from each other AND from the subject lineage).
SPARK_JUDGE_MODEL="${SPARK_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAC_JUDGE_MODEL="${MAC_JUDGE_MODEL:-mlx-community/Meta-Llama-3.1-8B-Instruct-4bit}"

# The combined --judges flag (provider:model@base_url, comma-separated). vLLM keys to vendor
# 'qwen'; mlx keys to engine 'mlx' → 2 distinct families. (See Mac-Spark-Judge-Farm.md.)
JUDGES="${JUDGES:-vllm:${SPARK_JUDGE_MODEL}@http://${SPARK_HOST}:${SPARK_PORT}/v1,mlx:${MAC_JUDGE_MODEL}@http://${MAC_HOST}:${MAC_PORT}/v1}"

# Two-box farm config (consumed by run_local_judge_eval.py --config).
JUDGE_CONFIG="${JUDGE_CONFIG:-config/inference.local.mac-judge.json}"

# Benchmark A — per-seed answer files. judge_pilot_answers.py has NO --seed flag: each seed is a
# SEPARATE answers JSON (base_answer + adapter_answer per case). Override SEED_ANSWERS with your
# real per-seed paths (space-separated, ≥3 for the gate). Defaults follow the 2026-06-29 layout.
SEEDS="${SEEDS:-1 2 3}"
ANSWERS_DIR="${ANSWERS_DIR:-agi-proof/benchmark-results/wisdom-market}"
ANSWERS_PREFIX="${ANSWERS_PREFIX:-M3-pilot-answers-seed}"
JUDGE_OUT_DIR="${JUDGE_OUT_DIR:-agi-proof/benchmark-results/wisdom-market/m3-2family-judge}"

# Aggregator (run_lora_uplift_validation.py) consumes a SINGLE judgments JSON (subjectModel,
# judges[], seeds[] with per-item base/adapterContent per family). It is NOT produced verbatim
# by judge_pilot_answers.py — you assemble it from the per-seed judge outputs. Point JUDGMENTS at
# that assembled file. (We surface this clearly rather than pretend the two share a schema.)
JUDGMENTS="${JUDGMENTS:-${JUDGE_OUT_DIR}/judgments.json}"
UPLIFT_OUT="${UPLIFT_OUT:-${JUDGE_OUT_DIR}/uplift-validation.json}"

# Benchmark B — QAT train + NVFP4 cert.
# NB: tools/train_lora.py default --model is Qwen/Qwen2.5-3B-Instruct, so OLMoE MUST be passed
# explicitly; and --qat-scheme defaults to int8, so NVFP4 MUST be passed explicitly.
QAT_BASE="${QAT_BASE:-allenai/OLMoE-1B-7B-0924-Instruct}"
# v5 recipe (default): keep v3's stable lambda=0.001 (v4's 0.01 over-fit and broke mean_kl +
# the protected slice), nudge epochs 2->3 to push top-1, write a v5 adapter + v5 cert artifact.
QAT_ADAPTER="${QAT_ADAPTER:-training/lora/checkpoints/olmoe-qat-spark-v5}"
QAT_DATA="${QAT_DATA:-training/lora/train.jsonl}"
QAT_EPOCHS="${QAT_EPOCHS:-3}"
QAT_LAMBDA="${QAT_LAMBDA:-0.001}"
CERT_CALIB="${CERT_CALIB:-training/lora/train.jsonl}"
# KEEP_SUFFIXES: comma-list of served-linear suffixes to hold in bf16 (mixed precision). Empty =
# full NVFP4 set (v3/v4 behaviour). Try 'down_proj' if top-1 stays < 0.97 at full quantization.
KEEP_SUFFIXES="${KEEP_SUFFIXES:-}"
CERT_OUT="${CERT_OUT:-agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v5.json}"
CERT_NEVAL="${CERT_NEVAL:-256}"

# ── Mode flags ────────────────────────────────────────────────────────────────────────────────
DRY_RUN=1            # default: print the plan, execute nothing
RUN_A=0
RUN_B=0
RUN_TRAIN=0          # gate the long/destructive QAT train behind an explicit flag

usage() {
  cat <<'EOF'
Usage: scripts/run_local_benchmarks.sh [--all|--bench-a|--bench-b] [--execute] [--run-train] [-h]

  --bench-a     Run Benchmark A only (≥2-family VALIDATED judging of M3-SFT uplift).
  --bench-b     Run Benchmark B only (low-RAM NVFP4 certification).
  --all         Run both (default if no bench is selected).
  --execute     Actually run the commands. WITHOUT this flag the script DRY-RUNS (prints only).
  --run-train   In Benchmark B, also run the QAT train step (long, GPU). Default: skip & certify
                an existing adapter at QAT_ADAPTER.
  -h, --help    Show this help.

Key env overrides: SPARK_HOST MAC_HOST SPARK_PORT MAC_PORT JUDGES JUDGE_CONFIG
  SEEDS ANSWERS_DIR ANSWERS_PREFIX JUDGE_OUT_DIR JUDGMENTS UPLIFT_OUT
  QAT_BASE QAT_ADAPTER QAT_DATA QAT_EPOCHS QAT_LAMBDA CERT_CALIB CERT_OUT CERT_NEVAL PYTHON
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-a)   RUN_A=1 ;;
    --bench-b)   RUN_B=1 ;;
    --all)       RUN_A=1; RUN_B=1 ;;
    --execute)   DRY_RUN=0 ;;
    --dry-run)   DRY_RUN=1 ;;   # explicit no-op (dry-run is the default); keeps bridge cmds robust
    --run-train) RUN_TRAIN=1 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

# Default to running both benchmarks if neither was selected.
if [[ "${RUN_A}" -eq 0 && "${RUN_B}" -eq 0 ]]; then RUN_A=1; RUN_B=1; fi

# ── Helpers ───────────────────────────────────────────────────────────────────────────────────
hr()  { printf '%s\n' "────────────────────────────────────────────────────────────────────────────"; }
step() { echo; hr; echo ">> STEP: $*"; hr; }

# run: echo the command, then either execute it (live) or skip it (dry-run).
run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [dry-run] not executed (pass --execute to run)"
  else
    "$@"
  fi
}

cost_guard_reminder() {
  hr
  echo "COST-GUARD (wisdom-gpu-prebaked):"
  echo "  • Owned hardware (Spark + Mac) is FREE — everything here runs on it."
  echo "  • RunPod is the ONLY metered path; this script never touches RunPod."
  echo "  • AFTER any GPU session: confirm ZERO leaked pods. Cheap validation first (limit=24, runs=1)."
  hr
}

echo "Sophia-AGI local benchmark runner"
echo "  mode:        $([[ "${DRY_RUN}" -eq 1 ]] && echo DRY-RUN || echo EXECUTE)"
echo "  benchmark A: $([[ "${RUN_A}" -eq 1 ]] && echo yes || echo no)"
echo "  benchmark B: $([[ "${RUN_B}" -eq 1 ]] && echo yes || echo no) (run-train: $([[ "${RUN_TRAIN}" -eq 1 ]] && echo yes || echo no))"
echo "  repo root:   ${ROOT}"
cost_guard_reminder

# ════════════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK A — ≥2-family VALIDATED judging of the M3-SFT source-discipline uplift
# ════════════════════════════════════════════════════════════════════════════════════════════
if [[ "${RUN_A}" -eq 1 ]]; then

  step "A0 — Bring-up reminder (start these BY HAND before --execute; not automated here)"
  cat <<EOF
  On the DGX Spark (CUDA, vLLM):
    vllm serve ${SPARK_JUDGE_MODEL} --port ${SPARK_PORT}
  On the Mac Studio (Apple Silicon, MLX):
    mlx_lm.server --model ${MAC_JUDGE_MODEL} --port ${MAC_PORT}
  These give 2 distinct judge families: 'qwen' (vLLM→vendor) + 'mlx' (engine). judge != subject
  (subject lineage allenai/olmoe). See docs/11-Platform/Mac-Spark-Judge-Farm.md.
EOF

  step "A1 — Smoke test the two-box farm (connectivity only, CI-safe, no GPU job)"
  # Print-only path: emits the ready --judges flag + each box's serve command from the config.
  run "${PY}" tools/run_local_judge_eval.py --config "${JUDGE_CONFIG}"

  step "A2 — Judge M3-SFT answers across seeds [${SEEDS}] with both families"
  # judge_pilot_answers.py judges ONE answers file per call (one seed). No --seed flag exists:
  # seeds are separate answer JSONs. We loop over seeds and write one judge report per seed.
  for s in ${SEEDS}; do
    ans="${ANSWERS_DIR}/${ANSWERS_PREFIX}${s}.json"
    out="${JUDGE_OUT_DIR}/seed${s}-judge.json"
    echo "  -- seed ${s}: answers=${ans} -> ${out}"
    run "${PY}" tools/judge_pilot_answers.py \
      --answers "${ans}" \
      --judges "${JUDGES}" \
      --out "${out}"
  done

  step "A3 — Aggregate + apply the no-overclaim VALIDATED gate"
  # run_lora_uplift_validation.py consumes ONE assembled judgments JSON (subjectModel + judges[]
  # + per-seed per-item base/adapterContent per family). Assemble \$JUDGMENTS from the A2 outputs
  # first (the upstream labelling step, per the P6 preregistration). Gate bars (ALL required):
  #   notMock; ≥2 families; judge != subject; mean κ ≥ 0.40; ≥3 seeds; 95% CI excludes zero.
  if [[ "${DRY_RUN}" -eq 0 && ! -f "${JUDGMENTS}" ]]; then
    echo "  WARN: ${JUDGMENTS} not found. Assemble it from the per-seed A2 outputs first" >&2
    echo "        (subjectModel='allenai/OLMoE-1B-7B-0924-Instruct', judges[], seeds[].items[])." >&2
  fi
  run "${PY}" tools/run_lora_uplift_validation.py \
    --judgments "${JUDGMENTS}" \
    --out "${UPLIFT_OUT}"

  echo
  echo "Benchmark A done. Verdict is in ${UPLIFT_OUT} ('validated' true/false)."
  echo "If validated==true: promote the row in published-results.json and regenerate RESULTS.md"
  echo "via tools/build_results_page.py. Else: keep it labelled CANDIDATE (current state: κ=0.24)."
fi

# ════════════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK B — low-RAM NVFP4 certification (Boundary-3)
# ════════════════════════════════════════════════════════════════════════════════════════════
if [[ "${RUN_B}" -eq 1 ]]; then

  step "B0 — GPU-free self-test of the cert logic (merge/quant/gate invariants)"
  run "${PY}" tools/certify_lowram.py --selftest

  if [[ "${RUN_TRAIN}" -eq 1 ]]; then
    step "B1 — QAT train (LONG, GPU) — gated behind --run-train"
    cost_guard_reminder
    # train_lora default --model is Qwen/Qwen2.5-3B-Instruct and default --qat-scheme is int8,
    # so OLMoE + nvfp4 + the output dir are ALL passed explicitly.
    run "${PY}" tools/train_lora.py \
      --model "${QAT_BASE}" \
      --train "${QAT_DATA}" \
      --output "${QAT_ADAPTER}" \
      --qat --qat-scheme nvfp4 \
      --qat-lambda "${QAT_LAMBDA}" \
      --epochs "${QAT_EPOCHS}"
  else
    step "B1 — QAT train SKIPPED (no --run-train); will certify existing adapter at ${QAT_ADAPTER}"
  fi

  step "B2 — Certify NVFP4 against BF16 (LowRamGate). Pass bar: mean_kl ≤ 0.05 AND top1 ≥ 0.97"
  # protected slice bar: KL ≤ 0.10 AND agreement ≥ 0.95. Exit 0 = PASS, 2 = FAIL (honest verdict).
  run "${PY}" tools/certify_lowram.py \
    --base-model "${QAT_BASE}" \
    --adapter "${QAT_ADAPTER}" \
    --calib "${CERT_CALIB}" \
    --scheme nvfp4 \
    --keep-suffixes "${KEEP_SUFFIXES}" \
    --n-eval "${CERT_NEVAL}" \
    --out "${CERT_OUT}"

  echo
  echo "Benchmark B done. Report: ${CERT_OUT}. On PASS you may claim ONLY: 'served-quant retains"
  echo "BF16 next-token behavior to a measured bound' — nothing more (Cheap-Compute-Boundary.md)."
fi

echo
cost_guard_reminder
echo "All selected benchmarks complete.$([[ "${DRY_RUN}" -eq 1 ]] && echo '  (DRY-RUN — nothing executed; re-run with --execute.)')"
