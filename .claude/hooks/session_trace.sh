#!/usr/bin/env bash
# Stop hook — ADVISORY ONLY, never blocks, exit 0 always.
# Ambient trajectory capture (ruflo-integration §3.2): append one structured event per
# session stop to agent/memory/session_traces/events.jsonl — the same stream
# sophia_trajectory_record writes and tools/skill_efficacy_report.py reads. This is the
# deterministic half of the learning-signal loop; the trajectory-pack gates decide later
# what (if anything) becomes training data. Nothing here is a claim.
set -u

payload="$(cat 2>/dev/null)"
# payload travels via env: the heredoc already occupies python's stdin
HOOK_PAYLOAD="$payload" python3 - <<'PY' 2>/dev/null
import json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    d = json.loads(os.environ.get("HOOK_PAYLOAD", "") or "{}")
except Exception:
    d = {}

repo = Path.cwd()
out = repo / "agent" / "memory" / "session_traces" / "events.jsonl"

def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""

event = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "kind": "claude_session_stop",
    "sessionId": d.get("session_id", ""),
    "branch": git("branch", "--show-current"),
    "head": git("rev-parse", "--short", "HEAD"),
}
try:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
except Exception:
    pass
PY
exit 0
