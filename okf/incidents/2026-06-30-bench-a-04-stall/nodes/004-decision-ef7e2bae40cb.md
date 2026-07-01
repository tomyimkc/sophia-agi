---
id: decision:ef7e2bae40cb
node_type: decision
sources: []
links: ["act:a7bf157245d8"]
verifier: null
verdict: null
moral_standard: honest-root-cause: fix the cause, not just restart (no overclaim)
title: root cause = git calls have no timeout; fix = cap them
---

Cap every git call with SOPHIA_BRIDGE_GIT_TIMEOUT (default 120s); on TimeoutExpired return a synthesized returncode=124 so callers retry next tick instead of hanging. The self-reload patch was exonerated (its `_RUNNING is None` gate is correct).
