---
name: session-handover
description: >
  Use at the START of a session to bootstrap from the previous session's state, and at the END
  of a substantial session (or when the user says "hand over", "wrap up", "write a handover",
  "what's the state for the next session", or work spans devices/agents). This repo's continuity
  runs on a chain of handover docs (SESSION-HANDOVER-*.md, HANDOVER.md) plus the failure ledger;
  reading the newest one first prevents redoing finished work, and writing one at the end keeps the
  next Claude/Copilot/GLM/human session from starting blind.
metadata:
  short-description: "Bootstrap from / write the cross-session handover docs"
---

# Session handover

Work here is picked up by many sessions across devices and agents. Continuity is carried by
**handover documents**, not memory. This skill standardizes reading the latest one at the start
and writing the next one at the end.

## At session start — bootstrap (read, newest wins)

1. **Newest master handover** — the most recent `SESSION-HANDOVER-YYYY-MM-DD.md` at repo root
   (currently `SESSION-HANDOVER-2026-06-28.md`). It has: git state at handover, what was proven,
   the exact **next benchmark** with commands + pass/fail thresholds, and "read first" pointers.
2. **`HANDOVER.md`** — the consolidation handover (outstanding decisions, how to land/continue).
3. **`agi-proof/failure-ledger.md`** — what is NOT yet proven (the real open-work list).
4. **`CHANGELOG.md` top** + recent `main` log (`git log --oneline -15 origin/main`) — what landed
   since the handover was written (the handover may already be partly done).
5. Note any **caveat about local vs remote** in the handover — `origin/main` is the source of
   truth; the local `main` ref has been a stale divergent lineage in this container before.

Reconcile: if `git log` shows the handover's "next" work already merged, **don't redo it** — move
to the next open item. (This pairs with the `git-discipline` skill: fetch before trusting state.)

## At session end — write the handover

Create/refresh `SESSION-HANDOVER-YYYY-MM-DD.md` (today's date) at repo root. Keep the proven
structure of the existing file so the next session can scan it fast:

1. **Git/repo state at handover** — branch, `HEAD` vs `origin/main`, what's pushed, working tree.
2. **What this session did** — commits/PRs with numbers, and what each one proves *within the
   no-overclaim ceiling* (state verdicts as GO/NO-GO/candidate/VALIDATED, never overclaim).
3. **Proven vs still open** — and add a `agi-proof/failure-ledger.md` entry for anything measured
   that is not yet fully validated (single-seed, no third-party, etc.).
4. **▶ Next step** — the single most valuable next action, with exact commands + pass/fail bars.
5. **Read-first list** — the files the next session should open before acting.
6. **Don't-break list** — the CI gates that must stay green (see the `ci-artifact-drift` skill).

Then update `CHANGELOG.md` if the session landed work, and confirm the doc is committed on the
session branch (handover docs are tracked, not throwaway).

## Rules

- Faithful reporting: if a step was skipped or a run failed, say so with the numbers. The repo's
  credibility is its discipline — never report a number without its CI, seeds, and judge families.
- A handover is a pointer to state, not a duplicate of it. Link to the artifacts
  (`published-results.json`, gate JSONs, ledger), don't paste large results.
- The newest dated handover is authoritative; older ones are history — don't edit them to "fix"
  the past, write a new one.
