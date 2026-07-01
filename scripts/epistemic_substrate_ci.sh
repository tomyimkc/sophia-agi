# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
#
# epistemic_substrate_ci.sh — enforceable CI lane for the "epistemic-substrate" suite.
#
# Purpose: make the epistemic-substrate gates/tests LOAD-BEARING (not dormant).
# Run from the repo root, under python3.12.
#
#   BLOCKING steps (proven-green today) — the lane FAILS (non-zero exit) if any fail:
#     * the 17 new epistemic-substrate tests (pytest)
#     * the two self-tests that must fire: vov_selftest.py and sleeper_injection_selftest.py
#
#   NON-BLOCKING DIAGNOSTICS — run + print JSON receipt, but NEVER fail the lane.
#   These are pre-registered as not-yet-passing / needing human calibration:
#     * lint_evidence.py
#     * wiki_coupling_gate.py   (currently FAILs the pre-registered floors)
#     * honest_closure_gate.py
#     * fact_recency_gate.py    (requires --records/--today; no self-test mode -> skipped)
#
# OVERALL exit code is 0 iff ALL blocking steps passed. Diagnostic exit codes are
# reported for visibility only.

set -euo pipefail

PY=python3.12

# Resolve repo root as the parent of this script's directory so the lane works
# regardless of the caller's CWD, then cd into it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=================================================================="
echo " epistemic-substrate CI lane"
echo " repo root : ${REPO_ROOT}"
echo " python    : $(${PY} --version 2>&1)"
echo "=================================================================="

# ------------------------------------------------------------------ BLOCKING
BLOCKING_FAIL=0

echo
echo "------------------------------------------------------------------"
echo "[BLOCKING 1/3] 17 epistemic-substrate tests (pytest)"
echo "------------------------------------------------------------------"
EPISTEMIC_TESTS=(
  tests/test_belief_revision_consistency.py
  tests/test_calibration_belief_store.py
  tests/test_evidence_edges.py
  tests/test_evidence_spec.py
  tests/test_fact_recency_gate.py
  tests/test_gap_nodes.py
  tests/test_gate_cost_budget.py
  tests/test_gate_provenance.py
  tests/test_honest_closure_gate.py
  tests/test_label_budget_ledger.py
  tests/test_lint_evidence.py
  tests/test_moral_recall.py
  tests/test_sequence_capability_gate.py
  tests/test_smt_verifier.py
  tests/test_third_party_intake.py
  tests/test_topology_truth_probe.py
  tests/test_verify_verifiers.py
)
if ${PY} -m pytest "${EPISTEMIC_TESTS[@]}" -q; then
  echo "BLOCKING PASS: 17 epistemic-substrate tests"
else
  echo "BLOCKING FAIL: 17 epistemic-substrate tests"
  BLOCKING_FAIL=1
fi

echo
echo "------------------------------------------------------------------"
echo "[BLOCKING 2/3] vov_selftest.py (verifier-of-verifiers must auto-fire)"
echo "------------------------------------------------------------------"
if ${PY} tools/vov_selftest.py; then
  echo "BLOCKING PASS: vov_selftest.py"
else
  echo "BLOCKING FAIL: vov_selftest.py"
  BLOCKING_FAIL=1
fi

echo
echo "------------------------------------------------------------------"
echo "[BLOCKING 3/3] sleeper_injection_selftest.py (super-additivity meta-gate)"
echo "------------------------------------------------------------------"
if ${PY} tools/sleeper_injection_selftest.py; then
  echo "BLOCKING PASS: sleeper_injection_selftest.py"
else
  echo "BLOCKING FAIL: sleeper_injection_selftest.py"
  BLOCKING_FAIL=1
fi

# -------------------------------------------------------------- DIAGNOSTICS
# These NEVER change the lane's exit code. We capture each tool's exit code and
# echo a uniform "DIAGNOSTIC (non-blocking): <tool> exit <code>" line.
echo
echo "=================================================================="
echo " NON-BLOCKING DIAGNOSTICS (pre-registered; do NOT gate the lane)"
echo "=================================================================="

declare -a DIAG_NAMES=()
declare -a DIAG_CODES=()

run_diag () {
  # $1 = human label, rest = command
  local label="$1"; shift
  echo
  echo "------------------------------------------------------------------"
  echo "[DIAGNOSTIC] ${label}"
  echo "------------------------------------------------------------------"
  local code=0
  # Do not let set -e abort on a failing (expected-to-fail) diagnostic.
  "$@" || code=$?
  echo "DIAGNOSTIC (non-blocking): ${label} exit ${code}"
  DIAG_NAMES+=("${label}")
  DIAG_CODES+=("${code}")
}

run_diag "lint_evidence.py"       ${PY} tools/lint_evidence.py
run_diag "wiki_coupling_gate.py"  ${PY} tools/wiki_coupling_gate.py
run_diag "honest_closure_gate.py" ${PY} tools/honest_closure_gate.py

# fact_recency_gate.py needs --records/--today (no runnable no-arg/self-test mode);
# per the lane spec we SKIP it rather than fabricate args.
echo
echo "------------------------------------------------------------------"
echo "[DIAGNOSTIC] fact_recency_gate.py"
echo "------------------------------------------------------------------"
echo "SKIP: fact_recency_gate.py requires --records and --today (no self-test mode); skipping."
DIAG_NAMES+=("fact_recency_gate.py")
DIAG_CODES+=("skipped")

# ------------------------------------------------------------------- SUMMARY
echo
echo "=================================================================="
echo " epistemic-substrate CI lane — SUMMARY"
echo "=================================================================="
echo "Diagnostics (non-blocking):"
for i in "${!DIAG_NAMES[@]}"; do
  printf "  - %-26s exit %s\n" "${DIAG_NAMES[$i]}" "${DIAG_CODES[$i]}"
done
echo
if [ "${BLOCKING_FAIL}" -eq 0 ]; then
  echo "BLOCKING: PASS  (17 tests + vov_selftest + sleeper_injection_selftest all green)"
  echo "OVERALL : PASS"
  exit 0
else
  echo "BLOCKING: FAIL  (see [BLOCKING ...] steps above)"
  echo "OVERALL : FAIL"
  exit 1
fi
