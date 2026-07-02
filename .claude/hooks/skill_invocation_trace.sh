#!/usr/bin/env bash
# PostToolUse(Skill) hook — ADVISORY ONLY, never blocks, exit 0 always.
# Log every skill invocation to the session-trace stream so
# tools/skill_efficacy_report.py can correlate skill firings with session outcomes.
# This is the data source the skill flywheel was missing: without it, "did this
# skill prevent waste?" is unanswerable. Capture only the skill name + a short
# args digest — never payload content.
set -u

payload="$(cat 2>/dev/null)"
# payload travels via env: the heredoc already occupies python's stdin
HOOK_PAYLOAD="$payload" python3 - <<'PY' 2>/dev/null
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

try:
    d = json.loads(os.environ.get("HOOK_PAYLOAD", "") or "{}")
except Exception:
    raise SystemExit(0)

tool_input = d.get("tool_input", {}) or {}
skill = tool_input.get("skill", "")
if not skill:
    raise SystemExit(0)
args = str(tool_input.get("args", "") or "")
event = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "kind": "skill_invocation",
    "sessionId": d.get("session_id", ""),
    "skill": skill,
    "argsDigest": hashlib.sha256(args.encode("utf-8")).hexdigest()[:12] if args else "",
}
out = Path.cwd() / "agent" / "memory" / "session_traces" / "events.jsonl"
try:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
except Exception:
    pass
PY
exit 0
