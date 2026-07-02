#!/usr/bin/env bash
# PreToolUse(Bash) hook — ADVISORY ONLY, never blocks, exit 0 always.
# When a `git commit` is about to run, execute the claims linter (fast, stdlib-only,
# offline) and inject its verdict as context. This turns the bootstrap's prose rule
# ("lint_claims must pass before any commit") into an ambient check — the model still
# decides; a FAIL here does not block the tool call.
set -u

payload="$(cat 2>/dev/null)"
cmd="$(printf '%s' "$payload" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))
except Exception:
    print("")' 2>/dev/null)"

case "$cmd" in
  *"git commit"*)
    lint_out="$(timeout 30 python3 tools/lint_claims.py 2>&1)"
    lint_rc=$?
    verdict="$(printf '%s' "$lint_out" | tail -3 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null)"
    if [ "$lint_rc" -ne 0 ]; then
      cat <<JSON
{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"claim-guard: tools/lint_claims.py FAILED (rc=$lint_rc) — the no-overclaim gate would go red in CI. Fix the flagged prose BEFORE this commit. Linter tail: $verdict"}}
JSON
    fi
    ;;
esac
exit 0
