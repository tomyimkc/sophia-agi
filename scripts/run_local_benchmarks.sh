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

# The combined --judges flag (provider:model@base_url). ollama->family 'qwen'; openai-transport to
# the remote mlx server->family 'mlx' = 2 distinct families, both != subject. Override JUDGES to tune.
JUDGES="${JUDGES:-ollama:${SPARK_JUDGE_MODEL}@http://${SPARK_HOST}:${SPARK_PORT},openai:${MAC_JUDGE_MODEL}@http://${MAC_HOST}:${MAC_PORT}/v1}"

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
RUN_VIRTUES=0        # cardinal-virtue benchmarks (Sophrosyne + Dikaiosyne real GO-path eval)
RUN_THINKING=0       # thinking-log pipeline benchmark (CPU/offline; opt-in real-model faithfulness)
RUN_FAITH=0          # discriminating CoT-faithfulness battery (CPU integrity; opt-in real-model)
RUN_TRAIN=0          # gate the long/destructive QAT train behind an explicit flag

usage() {
  cat <<'EOF'
Usage: scripts/run_local_benchmarks.sh [--all|--bench-a|--bench-b|--bench-virtues|--bench-thinking] [--execute] [--run-train] [-h]

  --bench-a        Run Benchmark A only (≥2-family VALIDATED judging of M3-SFT uplift).
  --bench-b        Run Benchmark B only (low-RAM NVFP4 certification).
  --bench-virtues  Run the cardinal-virtue real GO-path evals (Sophrosyne + Dikaiosyne):
                   build+decontam batteries, 2-family judge labelling, real-baseline 3-arm
                   scoring. Two judge families + a baseline model required (judge != subject).
  --bench-thinking Run the thinking-log pipeline benchmark (CPU/offline, no GPU): capture
                   coverage (every generate() -> a trace span), A2A delegate/result/synthesis
                   legs, and fail-closed distill yield. With THINKING_MODEL set (+ a key and
                   SOPHIA_CAPTURE_THINKING=1) it also runs a real-model CoT faithfulness pass
                   (a measurement, never a GO/claim). No owned-hardware judge farm needed.
  --bench-faithfulness Run the discriminating CoT-faithfulness battery (CPU integrity + plumbing):
                   intrinsic flip-rate on items whose answer hinges on a reasoning step, plus a
                   cued-vs-uncued split (cue-follow / acknowledge / unfaithful-cue-use). With
                   FAITH_MODEL set (+ key + SOPHIA_CAPTURE_THINKING=1) it runs the real-model
                   measurement over FAITH_SEEDS seeds with bootstrap CIs. A measurement, never a claim.
  --all            Run A and B (default if no bench is selected). Virtues/thinking/faithfulness are opt-in (not in --all).
  --execute        Actually run the commands. WITHOUT this flag the script DRY-RUNS (prints only).
  --run-train      In Benchmark B, also run the QAT train step (long, GPU). Default: skip & certify
                   an existing adapter at QAT_ADAPTER.
  -h, --help       Show this help.

Key env overrides: SPARK_HOST MAC_HOST SPARK_PORT MAC_PORT JUDGES JUDGE_CONFIG
  SEEDS ANSWERS_DIR ANSWERS_PREFIX JUDGE_OUT_DIR JUDGMENTS UPLIFT_OUT
  QAT_BASE QAT_ADAPTER QAT_DATA QAT_EPOCHS QAT_LAMBDA CERT_CALIB CERT_OUT CERT_NEVAL PYTHON
  VIRTUE_JUDGE_A VIRTUE_JUDGE_B VIRTUE_JUDGE_A_NAME VIRTUE_JUDGE_B_NAME VIRTUE_SUBJECT VIRTUE_SEEDS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-a)       RUN_A=1 ;;
    --bench-b)       RUN_B=1 ;;
    --bench-virtues) RUN_VIRTUES=1 ;;
    --bench-thinking) RUN_THINKING=1 ;;
    --bench-faithfulness) RUN_FAITH=1 ;;
    --all)           RUN_A=1; RUN_B=1 ;;
    --execute)       DRY_RUN=0 ;;
    --dry-run)       DRY_RUN=1 ;;   # explicit no-op (dry-run is the default); keeps bridge cmds robust
    --run-train)     RUN_TRAIN=1 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

# Default to running both M3 benchmarks ONLY if no lane at all was selected. A bare
# --bench-virtues must NOT silently also run A+B (it has its own GPU cost profile).
if [[ "${RUN_A}" -eq 0 && "${RUN_B}" -eq 0 && "${RUN_VIRTUES}" -eq 0 && "${RUN_THINKING}" -eq 0 && "${RUN_FAITH}" -eq 0 ]]; then RUN_A=1; RUN_B=1; fi

# Thinking-log / faithfulness model: unset -> CPU/offline only (no real model).
THINKING_MODEL="${THINKING_MODEL:-}"
FAITH_MODEL="${FAITH_MODEL:-}"
FAITH_SEEDS="${FAITH_SEEDS:-3}"

# Cardinal-virtue eval config. Two distinct judge families, BOTH stronger than and DISTINCT
# from the baseline subject (judge != subject). Judge A defaults to a CAPABLE Qwen (32B, NOT
# the 7B SPARK_JUDGE_MODEL, which equals the subject and would both self-judge and deflate κ —
# the M3 weak-judge lesson). Override any of these to retarget. SUBJECT = no-gate/no-auditor baseline.
VIRTUE_JUDGE_A="${VIRTUE_JUDGE_A:-ollama:qwen2.5:32b-instruct@http://${SPARK_HOST}:${SPARK_PORT}/v1}"
VIRTUE_JUDGE_B="${VIRTUE_JUDGE_B:-openai:${MAC_JUDGE_MODEL}@http://${MAC_HOST}:${MAC_PORT}/v1}"
VIRTUE_JUDGE_A_NAME="${VIRTUE_JUDGE_A_NAME:-qwen}"
VIRTUE_JUDGE_B_NAME="${VIRTUE_JUDGE_B_NAME:-llama}"
VIRTUE_SUBJECT="${VIRTUE_SUBJECT:-ollama:qwen2.5:7b-instruct@http://${SPARK_HOST}:${SPARK_PORT}/v1}"
VIRTUE_SEEDS="${VIRTUE_SEEDS:-3}"

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
echo "  virtues:     $([[ "${RUN_VIRTUES}" -eq 1 ]] && echo yes || echo no)"
echo "  thinking:    $([[ "${RUN_THINKING}" -eq 1 ]] && echo "yes (model: ${THINKING_MODEL:-offline})" || echo no)"
echo "  faithfulness:$([[ "${RUN_FAITH}" -eq 1 ]] && echo " yes (model: ${FAITH_MODEL:-offline}, seeds: ${FAITH_SEEDS})" || echo " no")"
echo "  repo root:   ${ROOT}"
echo "  python:      ${PY}"
cost_guard_reminder

# Preflight: in EXECUTE mode the chosen interpreter MUST have numpy (and torch for --run-train),
# else fail with an actionable message instead of the cryptic B0 self-test failure (no GPU spent).
# The thinking lane is pure-stdlib (no numpy/GPU), so it does not require this preflight.
if [[ "${DRY_RUN}" -eq 0 && ( "${RUN_A}" -eq 1 || "${RUN_B}" -eq 1 || "${RUN_VIRTUES}" -eq 1 ) ]]; then
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

# ════════════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK VIRTUES — cardinal-virtue real GO-path evals (Sophrosyne + Dikaiosyne)
# The PR-275 Andreia methodology, generalized. Full runbook: docs/11-Platform/Cardinal-Virtue-Benchmarks.md
# ════════════════════════════════════════════════════════════════════════════════════════════
if [[ "${RUN_VIRTUES}" -eq 1 ]]; then
  cost_guard_reminder
  echo "  virtue judges: A=${VIRTUE_JUDGE_A_NAME} (${VIRTUE_JUDGE_A})  B=${VIRTUE_JUDGE_B_NAME} (${VIRTUE_JUDGE_B})"
  echo "  virtue subject (no-gate baseline): ${VIRTUE_SUBJECT}  seeds=${VIRTUE_SEEDS}"

  step "V0 — Regenerate + freeze the external batteries (deterministic; pre-registration)"
  run "${PY}" tools/build_sophrosyne_external_battery.py
  run "${PY}" tools/build_dikaiosyne_external_battery.py

  step "V1 — Decontam (pillar 6): battery prompts disjoint from all training corpora"
  run "${PY}" tools/assert_sophrosyne_decontam.py
  run "${PY}" tools/assert_dikaiosyne_decontam.py

  step "V2 — Label with 2 independent judge families (consensus ground truth; κ≥0.40 gate)"
  run "${PY}" tools/label_sophrosyne_battery.py \
    --judge-a "${VIRTUE_JUDGE_A}" --judge-a-name "${VIRTUE_JUDGE_A_NAME}" \
    --judge-b "${VIRTUE_JUDGE_B}" --judge-b-name "${VIRTUE_JUDGE_B_NAME}"
  run "${PY}" tools/label_dikaiosyne_battery.py \
    --judge-a "${VIRTUE_JUDGE_A}" --judge-a-name "${VIRTUE_JUDGE_A_NAME}" \
    --judge-b "${VIRTUE_JUDGE_B}" --judge-b-name "${VIRTUE_JUDGE_B_NAME}"

  step "V3 — Score 3 arms vs a REAL no-gate/no-auditor baseline (paired Δ + bootstrap CI)"
  run "${PY}" tools/run_sophrosyne_eval.py --model "${VIRTUE_SUBJECT}" --seeds "${VIRTUE_SEEDS}" --write
  run "${PY}" tools/run_dikaiosyne_eval.py --model "${VIRTUE_SUBJECT}" --seeds "${VIRTUE_SEEDS}" --write

  echo
  echo "Virtue benchmarks done. Receipts:"
  echo "  agi-proof/benchmark-results/sophrosyne/sophrosyne-measure-eval.public-report.json"
  echo "  agi-proof/benchmark-results/dikaiosyne/dikaiosyne-justice-eval.public-report.json"
  echo "GO promotes via published-results.json + build_results_page.py; NO-GO stays candidate"
  echo "(log measured numbers in the temperance/justice ledger). canClaimAGI stays false."
fi

# ════════════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK THINKING — thinking-log pipeline (capture coverage + A2A distill) + faithfulness
# CPU/offline by default; no owned-hardware judge farm, no RunPod. Full design: docs/09-Agent/Thinking-Logs.md
# ════════════════════════════════════════════════════════════════════════════════════════════
if [[ "${RUN_THINKING}" -eq 1 ]]; then
  step "T0 — Thinking-log pipeline + A2A distill (offline capture-coverage + fail-closed yield)"
  if [[ -n "${THINKING_MODEL}" ]]; then
    echo "  real-model faithfulness pass enabled: model=${THINKING_MODEL} (needs key + SOPHIA_CAPTURE_THINKING=1)"
    run "${PY}" tools/run_thinking_bench.py --model "${THINKING_MODEL}"
  else
    echo "  offline only (set THINKING_MODEL=anthropic|deepseek|... + a key for the faithfulness pass)"
    run "${PY}" tools/run_thinking_bench.py --offline
  fi
  echo
  echo "Thinking benchmark done. Receipt: agent/memory/thinking/bench/thinking-bench.json (gitignored)."
  echo "Offline pass = the capture/A2A/distill MECHANISM works; faithfulness Δ is a measurement, not a claim."
fi

# ════════════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK FAITHFULNESS — discriminating CoT-faithfulness battery (intrinsic + cued/uncued)
# CPU/offline integrity by default. Full design: docs/09-Agent/Faithfulness-Battery.md
# ════════════════════════════════════════════════════════════════════════════════════════════
if [[ "${RUN_FAITH}" -eq 1 ]]; then
  step "F0 — Faithfulness battery (integrity + plumbing offline; real-model measurement opt-in)"
  if [[ -n "${FAITH_MODEL}" ]]; then
    echo "  real-model measurement enabled: model=${FAITH_MODEL} seeds=${FAITH_SEEDS} (needs key + SOPHIA_CAPTURE_THINKING=1)"
    run "${PY}" tools/run_faithfulness_battery.py --model "${FAITH_MODEL}" --seeds "${FAITH_SEEDS}"
  else
    echo "  offline only (set FAITH_MODEL=openrouter:deepseek/deepseek-r1|anthropic|... + a key for the real measurement)"
    run "${PY}" tools/run_faithfulness_battery.py
  fi
  echo
  echo "Faithfulness battery done. Receipt: agent/memory/thinking/bench/faithfulness-battery.json (gitignored)."
  echo "Cued split is the disambiguator: unfaithfulCueUseRate = cue-influenced answers whose reasoning hid the cue."
fi

echo
cost_guard_reminder
echo "All selected benchmarks complete.$([[ "${DRY_RUN}" -eq 1 ]] && echo '  (DRY-RUN — nothing executed; re-run with --execute.)')"
