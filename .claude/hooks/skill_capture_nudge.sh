#!/usr/bin/env bash
# PostToolUse(Bash) hook — ADVISORY ONLY, never blocks. Exit 0 always.
# When a Bash command clearly FAILED (non-zero exit / traceback / failing tests), inject a
# one-time nudge to consider the `skill-author` skill: capture the lesson as a new/updated,
# auto-triggering skill so a future session does not repeat the issue. This is the
# deterministic half of the issue->skill loop (the model still decides whether to act).
#
# Reads the tool-call JSON from stdin; emits PostToolUse additionalContext only on failure.
set -u

payload="$(cat 2>/dev/null)"

failed="$(printf '%s' "$payload" | python3 -c '
import sys, json, re
try:
    d = json.load(sys.stdin)
except Exception:
    print("0"); raise SystemExit
resp = d.get("tool_response", {})
inp = d.get("tool_input", {})
# Gather any structured failure signal + a text blob to scan.
blob_parts = []
def add(x):
    if isinstance(x, str): blob_parts.append(x)
    elif isinstance(x, (int, float)): blob_parts.append(str(x))
    elif isinstance(x, dict): [add(v) for v in x.values()]
    elif isinstance(x, list): [add(v) for v in x]
add(resp)
blob = "\n".join(blob_parts)
structured_fail = False
if isinstance(resp, dict):
    for k in ("exit_code", "exitCode", "returncode", "code"):
        v = resp.get(k)
        if isinstance(v, (int, float)) and int(v) != 0:
            structured_fail = True
    if resp.get("is_error") in (True, "true"):
        structured_fail = True
    if str(resp.get("interrupted")).lower() == "true":
        structured_fail = False  # user-interrupted, not a captured-able failure
# Text heuristics — conservative, to avoid nagging on benign output.
patterns = [
    r"\bExit code [1-9]\b", r"\bTraceback \(most recent call last\)",
    r"\bFAILED\b", r"\b\d+ failed\b", r"\bE\s+assert", r"command not found",
    r"\bError:\b", r"\bfatal:\b", r"\bnon-zero exit\b", r"AssertionError",
]
text_fail = any(re.search(p, blob) for p in patterns)
print("1" if (structured_fail or text_fail) else "0")
' 2>/dev/null)"

if [ "${failed:-0}" = "1" ]; then
  cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"skill-author nudge: that command failed. If the root cause is reusable knowledge a future session would repeat (a footgun, a non-obvious fix, a repo rule), consider the `skill-author` skill to capture it as a new/updated auto-triggering skill, and log it in agi-proof/failures.jsonl. Skip if it was a transient/typo failure."}}
JSON
fi
exit 0
