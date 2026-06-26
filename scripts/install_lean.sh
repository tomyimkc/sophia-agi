#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
#
# Install the Lean 4 toolchain (elan + lean + lake) for kernel-checked proof
# verification (agent/lean_verifier.py, selfextend/proof_verifier.py).
#
# Why this exists: the formal-proof verifier is FAIL-CLOSED without Lean — every
# check returns verdict "held" / "lean_unavailable" and never "accepted" (see
# tests/test_lean_verifier.py). That fail-closed path is the tested contract and CI
# is green without Lean. Installing Lean is OPTIONAL: it flips the two real-kernel
# test cases from SKIPPED to PASSED and lets tools/run_formal_proofs_eval.py close
# the loop on the smoke split (held-out proof reward >= threshold). It never
# changes any gate's logic.
#
# Design: idempotent (re-running is a no-op once Lean is present), non-interactive
# (sets elan -y), honours LEAN_VERSION if pinned, fails closed on any error, and
# prints the exact commands it runs. Safe for CI and local use.
#
# Usage:
#   scripts/install_lean.sh                 # install latest stable Lean 4
#   LEAN_VERSION=leanprover/lean4:v4.14.0 scripts/install_lean.sh   # pin a version
#
# Exit codes: 0 = Lean ready; 1 = installation failed; 2 = unsupported platform.

set -euo pipefail

LEAN_VERSION="${LEAN_VERSION:-}"

err()  { echo "install_lean: ERROR: $*" >&2; exit 1; }
log()  { echo "install_lean: $*"; }

# --- platform gate -----------------------------------------------------------
case "$(uname -s)" in
  Linux*)  OS=linux ;;
  Darwin*) OS=darwin ;;
  *)       err "unsupported OS: $(uname -s) (Lean supports linux + darwin)"; exit 2 ;;
esac
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH=x86_64 ;;
  aarch64|arm64) ARCH=aarch64 ;;   # DGX Spark GB10 / Apple Silicon
  *) err "unsupported arch: $ARCH"; exit 2 ;;
esac
log "platform: $OS/$ARCH"

# --- already installed? (idempotent) -----------------------------------------
if command -v lean >/dev/null 2>&1 && command -v lake >/dev/null 2>&1; then
  log "lean + lake already on PATH:"
  log "  lean: $(command -v lean)"
  log "  $(lean --version 2>/dev/null | head -1 || echo '(version unavailable)')"
  log "nothing to do."
  exit 0
fi

# --- install elan (the Lean toolchain manager) -------------------------------
ELAN_DIR="${ELAN_HOME:-$HOME/.elan}"
if [ -x "$ELAN_DIR/bin/elan" ]; then
  log "elan already installed at $ELAN_DIR/bin/elan"
else
  log "installing elan (Lean toolchain manager) to $ELAN_DIR ..."
  # Canonical URL (https://lean-lang.org/install/manual/). The raw GitHub path is
  # https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh — note the
  # repo is `leanprover/elan`, NOT `leanprover/elan-init` (the latter 404s; it was the
  # wrong repo and broke the lean-kernel CI lane on first run).
  ELAN_INIT_URL="${ELAN_INIT_URL:-https://elan.lean-lang.org/elan-init.sh}"
  # elan-init's supported flags are only: -y --default-toolchain <chain> --no-modify-path.
  # The install directory is controlled by the ELAN_HOME env var (NOT a --prefix flag,
  # which elan-init rejects). We export it for the curl|sh invocation below.
  # -y: accept defaults (non-interactive); --no-modify-path: we export PATH ourselves
  # so this script is self-contained and doesn't touch shell rc files.
  ELAN_HOME="$ELAN_DIR" curl -fsSL "$ELAN_INIT_URL" \
    | ELAN_HOME="$ELAN_DIR" sh -s -- -y --default-toolchain none --no-modify-path \
    || err "elan install failed (curl $ELAN_INIT_URL)"
  log "elan installed."
fi

export PATH="$ELAN_DIR/bin:$PATH"

# --- install the Lean 4 default toolchain ------------------------------------
if [ -n "$LEAN_VERSION" ]; then
  log "installing pinned Lean toolchain: $LEAN_VERSION"
  elan default "$LEAN_VERSION" || err "failed to set default toolchain $LEAN_VERSION"
else
  log "installing default Lean 4 toolchain (latest stable) ..."
  # Use leanprover/lean4:stable as the default 4.x channel. elan picks the right
  # binary for the platform automatically.
  elan default leanprover/lean4:stable || err "failed to install leanprover/lean4:stable"
fi

# --- verify ------------------------------------------------------------------
command -v lean >/dev/null 2>&1 || err "lean not on PATH after install"
command -v lake >/dev/null 2>&1 || err "lake not on PATH after install"
log "ready."
log "  lean: $(command -v lean)"
log "  $(lean --version 2>/dev/null | head -1)"
log "  lake: $(command -v lake)"
log ""
log "NOTE: elan was installed to $ELAN_DIR. To make Lean permanent in your shell,"
log "      add to your rc file:  export PATH=\"$ELAN_DIR/bin:\$PATH\""
log "      (this script does not modify your rc files.)"
log ""
log "To verify the fail-closed contract still holds WITH a kernel, run:"
log "  uv run --with pytest python -m pytest tests/test_lean_verifier.py tests/test_selfextend_proof_verifier.py -v"
log "  python tools/run_formal_proofs_eval.py"
