#!/usr/bin/env bash
# SessionStart hook for Sophia-AGI — runs at the start of every Claude Code session.
# Two jobs:
#   1. Auto-unlock the git-crypt-encrypted skills/docs so they are readable this session
#      (the .claude/skills/** IP skills, AGENTS.md, CONTRACT.md, .grok/**, docs/superpowers/**).
#   2. Print a tight orientation block (read-first files, live git state, hard guardrails)
#      so a fresh session bootstraps from the last one instead of starting blind.
#
# It NEVER writes to the repo and NEVER fails the session — every step is best-effort.
# stdout from this hook is injected into the session as context.
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || exit 0

say() { printf '%s\n' "$*"; }

say "=== Sophia-AGI session bootstrap ==="

# --- 1. git-crypt auto-unlock ------------------------------------------------
# Provide the exported symmetric key as base64 in the env var GITCRYPT_KEY_B64
# (Claude Code on the web: add it as an environment secret). Locally, if the repo
# is already unlocked this is a no-op.
unlock_status="locked"
# A git-crypt-encrypted file begins with the magic bytes "\0GITCRYPT\0". If AGENTS.md (an
# encrypted file) reads as plaintext, the repo is already unlocked — independent of the
# git-crypt binary being present.
if [ -f AGENTS.md ] && ! head -c 16 AGENTS.md 2>/dev/null | grep -q 'GITCRYPT'; then
  unlock_status="already-unlocked"
elif [ -n "${GITCRYPT_KEY_B64:-}" ]; then
  if ! command -v git-crypt >/dev/null 2>&1; then
    # Best-effort install (web container is Debian/Ubuntu). Silent if it fails.
    (apt-get install -y git-crypt >/dev/null 2>&1) || true
  fi
  if command -v git-crypt >/dev/null 2>&1; then
    keyfile="$(mktemp)"
    if printf '%s' "$GITCRYPT_KEY_B64" | base64 -d > "$keyfile" 2>/dev/null \
       && git-crypt unlock "$keyfile" >/dev/null 2>&1; then
      unlock_status="unlocked-from-env"
    else
      unlock_status="unlock-FAILED (bad key or git-crypt error)"
    fi
    rm -f "$keyfile"
  else
    unlock_status="locked (git-crypt not installed; could not auto-install)"
  fi
fi
say "git-crypt: ${unlock_status}"
if [ "$unlock_status" = "locked" ]; then
  say "  -> Encrypted skills/AGENTS.md/CONTRACT.md are NOT readable. Set the GITCRYPT_KEY_B64"
  say "     environment secret (see .claude/README.md) so future sessions self-unlock."
fi

# --- 2. live git situational snapshot ---------------------------------------
SNAP=".agents/skills/git-operations/scripts/git_situational_awareness.py"
if [ -f "$SNAP" ]; then
  say ""
  say "--- git situational awareness (run before any git write) ---"
  python "$SNAP" 2>/dev/null | sed -n '1,18p' || say "  (snapshot script error; run it manually)"
else
  say "branch: $(git branch --show-current 2>/dev/null)  head: $(git rev-parse --short HEAD 2>/dev/null)"
fi

# --- 3. orientation: read-first + guardrails + MCP ---------------------------
cat <<'ORIENT'

--- read first (newest handover wins) ---
  * SESSION-HANDOVER-2026-06-28.md   (latest master handover: state + next benchmark)
  * HANDOVER.md                      (consolidation handover)
  * AGENTS.md / CONTRACT.md          (operating contract; encrypted -> needs unlock)
  * agi-proof/failure-ledger.md      (what is NOT proven; 58 open items)

--- skills available this session (auto-trigger by task match) ---
  process : git-discipline, ci-artifact-drift, session-handover   (plaintext, always on)
  repo    : sophia-agi, runpod-mcp, wisdom-gpu-prebaked            (encrypted; need unlock)

--- hard guardrails (do not bypass) ---
  * No overclaiming: `python tools/lint_claims.py` must pass before any commit.
  * The promotion/claim GATES decide validity, never your judgment. religion/history are PROTECTED.
  * RunPod GPU jobs go through GitHub Actions only (never local SSH) + read wisdom-gpu-prebaked first.
  * Before commit/push: run the ci-artifact-drift skill (`make claim-check` + the drift gates).
  * Before merge/branch/rebase: run the git-discipline skill (stale-snapshot waste is the #1 cost here).

--- MCP servers (.mcp.json) ---
  sophia-agi (python sophia_mcp/server.py)  + runpod (@runpod/mcp-server; needs RUNPOD_API_KEY env)
ORIENT

say "=== end bootstrap ==="
exit 0
