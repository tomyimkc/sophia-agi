#!/usr/bin/env bash
# PreToolUse(Bash) hook — ADVISORY ONLY, never blocks.
# When a Bash call is about to push / merge / rebase / force-update in this multi-agent
# repo, inject a one-time reminder to run the git-discipline pre-flight first (stale-snapshot
# and duplicate-PR waste is the most expensive recurring mistake here). Exit 0 always.
#
# Reads the tool-call JSON from stdin; emits PreToolUse additionalContext when relevant.
set -u

payload="$(cat 2>/dev/null)"
cmd="$(printf '%s' "$payload" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))
except Exception:
    print("")' 2>/dev/null)"

case "$cmd" in
  *"git push"*|*"git merge"*|*"git rebase"*|*"gh pr merge"*|*"push -f"*|*"push --force"*)
    cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"git-discipline reminder: this repo has many concurrent advisors. Before this push/merge/rebase, confirm you have run `git fetch origin --prune` and the situational-awareness snapshot (.agents/skills/git-operations/scripts/git_situational_awareness.py). For a merge/unblock, also run the merge-preflight (.agents/skills/multi-agent-merge-preflight/scripts/merge_blockers.py NNN). Stale local state -> wasted work. See the git-discipline skill."}}
JSON
    ;;
esac
exit 0
