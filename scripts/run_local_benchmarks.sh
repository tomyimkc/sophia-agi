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

# Interpreter resolution: GPU steps (train_lora / certify_lowram) need the Spark's torch+numpy
# venv, NOT the bare system python3 (which lacks numpy -> the B0 self-test fails fast). Prefer
# $PYTHON, else the first candidate that can import numpy, else system python3 (caught by preflight).
_pick_py() {
  local c
  for c in "${PYTHON:-}" ".venv/bin/python" "../sophia-agi/.venv/bin/python" \
           "${HOME}/sophia-agi/.venv/bin/python" "${HOME}/.venv/bin/python" python3; do
    [ -n "${c}" ] || continue
    if { command -v "${c}" >/dev/null 2>&1 || [ -x "${c}" ]; } && "${c}" -c "import numpy" >/dev/null 2>&1; then
      echo "${c}"; return 0
    fi
  done
  echo "${PYTHON:-python3}"
}
PY="$(_pick_py)"

# ── Tunable placeholders (override via env) ───────────────────────────────────────────────────
# Hostnames/ports for the two-box judge farm. Fill these in (link-local IPs, Tailscale, or DNS).
# Two-box judge farm. Defaults match the PROVEN 2026-06-29 strong-judge run: Spark Qwen-7B via
# ollama (already serving) + Mac Studio Llama-3.3-70B-4bit via mlx_lm.server reached over the Cat6
# link-local cable with the OpenAI transport. NB the `mlx:` provider is a LOCAL loader (returns
# empty on the Spark — see Mac-Spark-Judge-Farm.md); use `openai:` to a REMOTE mlx server.
SPARK_HOST="${SPARK_HOST:-127.0.0.1}"           # DGX Spark — Qwen judge (family 'qwen', via ollama)
SPARK_PORT="${SPARK_PORT:-11434}"               # ollama serve
MAC_HOST="${MAC_HOST:-169.254.26.171}"          # Mac Studio over the Cat6 link-local cable
MAC_PORT="${MAC_PORT:-8081}"                     # mlx_lm.server (strong 70B judge)

# Judge models (DIFFERENT vendors from each other AND from the subject lineage allenai/olmoe+gemma).
SPARK_JUDGE_MODEL="${SPARK_JUDGE_MODEL:-qwen2.5:7b-instruct}"            # ollama tag
MAC_JUDGE_MODEL="${MAC_JUDGE_MODEL:-mlx-community/Llama-3.3-70B-Instruct-4bit}"

# Auto-start the Mac judge over the Cat6 cable (SSH), so Benchmark A is hands-free. Fixed command,
# gated to EXECUTE mode. Set AUTO_START_JUDGES=0 to require manual bring-up instead.
AUTO_START_JUDGES="${AUTO_START_JUDGES:-1}"
MAC_SSH_USER="${MAC_SSH_USER:-tom}"
MAC_SSH_HOST="${MAC_SSH_HOST:-169.254.26.171}"  # Cat6 link-local IP (Tailscale name also works)
MAC_SSH_KEY="${MAC_SSH_KEY:-${HOME}/.ssh/id_ed25519}"
MAC_MLX_BIN="${MAC_MLX_BIN:-mlx_lm.server}"     # on the Mac login-shell PATH (pyenv 3.10.6)
JUDGE_READY_TRIES="${JUDGE_READY_TRIES:-80}"    # x3s ~= 4 min max wait for the 70B-4bit load

# The combined --judges flag (provider:model@base_url). ollama->family 'qwen'; vllm-transport to
# the remote mlx server->family 'mlx' = 2 distinct families, both != subject. Override JUDGES to tune.
# NB both base_urls MUST end in /v1 — agent.model's per-spec @url OVERRIDES the provider preset's
# base_url, so omitting /v1 makes the ollama judge hit :11434/chat/completions (404) -> every
# verdict None -> all-TIE. (This was the 2026-06-29 bench-a-04 all-tie bug.)
# NB2: the Mac judge uses the `vllm` provider, NOT `openai`. mlx_lm.server is keyless and the
# `openai` preset has no api_key_default — so with OPENAI_API_KEY unset, resolved_key() returns
# None and the OpenAI client throws on EVERY judge call (swallowed -> n=0). `vllm` carries
# api_key_default="EMPTY", which the keyless mlx server accepts. (Root cause of the 2026-06-30 n=0.)
JUDGES="${JUDGES:-ollama:${SPARK_JUDGE_MODEL}@http://${SPARK_HOST}:${SPARK_PORT}/v1,vllm:${MAC_JUDGE_MODEL}@http://${MAC_HOST}:${MAC_PORT}/v1}"

# Two-box farm config (consumed by run_local_judge_eval.py --config).
JUDGE_CONFIG="${JUDGE_CONFIG:-config/inference.local.mac-judge.json}"

# Benchmark A — per-seed answer files. judge_pilot_answers.py has NO --seed flag: each seed is a
# SEPARATE answers JSON (base_answer + adapter_answer per case). Override SEED_ANSWERS with your
# real per-seed paths (space-separated, ≥3 for the gate). Defaults follow the 2026-06-29 layout.
SEEDS="${SEEDS:-1 2 7}"   # committed M3-pilot answer seeds are 1,2,7,8,9,10 (NOT 3) — use 3 of them
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
echo "  python:      ${PY}"
cost_guard_reminder

# Preflight: in EXECUTE mode the chosen interpreter MUST have numpy (and torch for --run-train),
# else fail with an actionable message instead of the cryptic B0 self-test failure (no GPU spent).
if [[ "${DRY_RUN}" -eq 0 ]]; then
  if ! "${PY}" -c "import numpy" >/dev/null 2>&1; then
    echo "FATAL: interpreter '${PY}' has no numpy — GPU steps need the Spark's torch venv." >&2
    echo "  Fix: restart the bridge poller with PYTHON=/abs/path/to/torch-venv/bin/python exported," >&2
    echo "       or place the venv at ./.venv or ../sophia-agi/.venv. Aborting (no GPU spent)." >&2
    exit 3
  fi
  if [[ "${RUN_B}" -eq 1 && "${RUN_TRAIN}" -eq 1 ]] && ! "${PY}" -c "import torch" >/dev/null 2>&1; then
    echo "FATAL: interpreter '${PY}' has no torch; --run-train needs it. Set PYTHON to the torch venv. Aborting." >&2
    exit 3
  fi
fi

# ════════════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK A — ≥2-family VALIDATED judging of the M3-SFT source-discipline uplift
# ════════════════════════════════════════════════════════════════════════════════════════════
if [[ "${RUN_A}" -eq 1 ]]; then

  step "A0 — Bring up the two-box judge farm (Spark ollama Qwen + Mac MLX 70B over Cat6)"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [dry-run] would verify Spark ollama (${SPARK_HOST}:${SPARK_PORT}) has ${SPARK_JUDGE_MODEL}"
    echo "  [dry-run] would ensure Mac judge ${MAC_HOST}:${MAC_PORT}; if down, ssh ${MAC_SSH_USER}@${MAC_SSH_HOST}"
    echo "  [dry-run]   '${MAC_MLX_BIN} --model ${MAC_JUDGE_MODEL} --port ${MAC_PORT}' (nohup, over the Cat6 cable)"
    echo "  [dry-run] judges: ${JUDGES}"
  elif [[ "${AUTO_START_JUDGES}" -eq 1 ]]; then
    # Spark Qwen judge: ollama is a long-running service; just verify it answers.
    if curl -sf "http://${SPARK_HOST}:${SPARK_PORT}/api/tags" >/dev/null 2>&1; then
      echo "  [A0] Spark ollama up on ${SPARK_HOST}:${SPARK_PORT}"
    else
      echo "  [A0] WARN: Spark ollama not answering on ${SPARK_HOST}:${SPARK_PORT} (start 'ollama serve')"
    fi
    # Mac 70B judge: start it over the Cat6 cable via ssh if not already listening.
    if curl -sf "http://${MAC_HOST}:${MAC_PORT}/v1/models" >/dev/null 2>&1; then
      echo "  [A0] Mac MLX judge already listening on ${MAC_HOST}:${MAC_PORT}"
    else
      echo "  [A0] starting Mac MLX judge over Cat6: ssh ${MAC_SSH_USER}@${MAC_SSH_HOST}"
      ssh -i "${MAC_SSH_KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        "${MAC_SSH_USER}@${MAC_SSH_HOST}" \
        "nohup ${MAC_MLX_BIN} --model ${MAC_JUDGE_MODEL} --port ${MAC_PORT} >/tmp/mlx-judge-${MAC_PORT}.log 2>&1 & disown" \
        || echo "  [A0] WARN: ssh start returned non-zero (key/host?); will still poll for readiness"
      echo -n "  [A0] waiting for Mac judge to load (70B-4bit)"
      for _i in $(seq 1 "${JUDGE_READY_TRIES}"); do
        if curl -sf "http://${MAC_HOST}:${MAC_PORT}/v1/models" >/dev/null 2>&1; then echo " ready."; break; fi
        echo -n "."; sleep 3
      done
      curl -sf "http://${MAC_HOST}:${MAC_PORT}/v1/models" >/dev/null 2>&1 \
        || { echo; echo "  [A0] FATAL: Mac judge not ready after ~$((JUDGE_READY_TRIES*3))s — aborting A (need 2 families, judge != subject)" >&2; exit 4; }
    fi
    echo "  [A0] judges: ${JUDGES}"
  else
    echo "  [A0] AUTO_START_JUDGES=0 — assuming judges already up: ${JUDGES}"
  fi

  step "A1 — Smoke test the two-box farm (connectivity only, CI-safe, no GPU job)"
  # Print-only path: emits the ready --judges flag + each box's serve command from the config.
  run "${PY}" tools/run_local_judge_eval.py --config "${JUDGE_CONFIG}"

  step "A2 — Judge M3-SFT answers across seeds [${SEEDS}] with both families"
  # judge_pilot_answers.py judges ONE answers file per call (one seed). No --seed flag exists:
  # seeds are separate answer JSONs. We loop over seeds and write one judge report per seed.
  RAW_SIDECARS=""
  for s in ${SEEDS}; do
    ans="${ANSWERS_DIR}/${ANSWERS_PREFIX}${s}.json"
    out="${JUDGE_OUT_DIR}/seed${s}-judge.json"
    raw="${JUDGE_OUT_DIR}/seed${s}-raw.json"
    echo "  -- seed ${s}: answers=${ans} -> ${out} (+raw ${raw})"
    run "${PY}" tools/judge_pilot_answers.py \
      --answers "${ans}" \
      --judges "${JUDGES}" \
      --forced-choice \
      --seed "${s}" \
      --raw-out "${raw}" \
      --out "${out}"
    RAW_SIDECARS="${RAW_SIDECARS} ${raw}"
  done

  step "A2.5 — Assemble per-seed pairwise verdicts into the A3 judgments.json"
  # Bridges the two protocols: A2 emits pairwise (adapter/base/tie); A3 wants per-family content
  # booleans. assemble_uplift_judgments.py maps them (tie -> at-least-as-good on both) and stamps
  # the subject. delta == net head-to-head preference margin (documented; NOT absolute correctness).
  run "${PY}" tools/assemble_uplift_judgments.py \
    --subject "${QAT_BASE}" \
    --raw ${RAW_SIDECARS} \
    --out "${JUDGMENTS}"

  step "A3 — Aggregate + apply the no-overclaim VALIDATED gate"
  # run_lora_uplift_validation.py consumes the A2.5 judgments JSON (subjectModel + judges[]
  # + per-seed per-item base/adapterContent per family). Gate bars (ALL required):
  #   notMock; ≥2 families; judge != subject; mean κ ≥ 0.40; ≥3 seeds; 95% CI excludes zero.
  if [[ "${DRY_RUN}" -eq 0 && ! -f "${JUDGMENTS}" ]]; then
    echo "  WARN: ${JUDGMENTS} not found — A2.5 assembly did not run or produced nothing." >&2
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
    # --target-modules attn-mlp passes an explicit LIST (q/k/v/o/gate/up/down_proj, suffix-matches
    # the OLMoE experts). The default "all-linear" is a bare string this PEFT build char-splits into
    # {a,l,-,i,n,e,r} -> "Target modules not found" (the 2026-06-29 v5 train crash).
    run "${PY}" tools/train_lora.py \
      --model "${QAT_BASE}" \
      --train "${QAT_DATA}" \
      --output "${QAT_ADAPTER}" \
      --target-modules "${QAT_TARGET_MODULES:-attn-mlp}" \
      --lora-dropout 0 \
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
