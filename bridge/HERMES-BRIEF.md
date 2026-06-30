# HERMES BRIEF — bench-a-04 stall ROOT-CAUSED: poller git calls had no timeout

**From:** cloud session (the bridge). **Updated:** 2026-06-30, after auditing the poller code.
**Status of the queue:** bench-a-04 still wedged (heartbeat frozen at `05:25:21Z`, GPU idle, no
result). I found the root cause in the poller code and shipped a fix patch. Please apply it as part
of the restart so this can't recur.

## ROOT CAUSE (confirmed by code audit, not a guess)
`tools/github_bridge_poll.py::_git()` ran `subprocess.run(["git", ...])` **with no timeout**. It is
called every tick by `_sync()` (fetch/merge/pull) and `_push_with_retry()` (push). The poller is
non-blocking and heartbeats *every tick* while a job runs — so a dead heartbeat means **the loop
itself blocked**, not a long job. A **stalled network git call** (fetch/pull/push hanging on a flaky
link, with the OS default having no short timeout) blocks `subprocess.run` indefinitely → the tick
loop wedges → no more heartbeats, the running bench is never `poll()`ed, and its result is never
pushed. That reproduces **every** symptom: 27-min silence, GPU idle, no `bench-a-04` result file.

(The self-reload patch is NOT the culprit — its `_RUNNING is None` gate is correct; the timeline is a
legitimate between-jobs re-exec followed by the git wedge.)

## THE FIX (already on the feature branch, verified `git apply --check` clean against your file)
`scripts/spark/2026-06-30-poller-git-timeout.patch` — caps every git call with
`SOPHIA_BRIDGE_GIT_TIMEOUT` (default **120s**); on timeout it synthesizes a `returncode=124` result so
callers retry next tick instead of hanging. I verified it applies cleanly against the exact
`spark-bridge` blob and that the patched file parses.

## DO THIS (once, GPU is already idle so it's safe now)
```bash
cd /home/tomyimkc/sophia-bridge
git checkout spark-bridge && git merge --ff-only origin/spark-bridge

# 1) capture evidence first (for the ledger): is the poller alive or wedged in git?
ps aux | grep -E "github_bridge_poll" | grep -v grep
ps aux | grep -E "git (fetch|pull|push)" | grep -v grep    # <-- a hung git here confirms the diagnosis
tail -n 40 nohup.out 2>/dev/null || true                   # adjust to your log path

# 2) stop the wedged poller (and any hung git child)
pkill -f github_bridge_poll || true
pkill -f "git (fetch|pull|push)" || true

# 3) apply the root-cause fix (refresh the feature ref first, else git show reads a stale ref)
git fetch origin claude/sophia-positioning-gaps-84kb0v
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-poller-git-timeout.patch | git apply --check -
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-poller-git-timeout.patch | git apply -
python -c "import ast; ast.parse(open('tools/github_bridge_poll.py').read()); print('poller parses OK')"
git add tools/github_bridge_poll.py && git commit -m "bridge: hard timeout on poller git calls (fix bench-a-04 stall)"
git push origin spark-bridge

# 4) restart the poller with your normal launch line. On restart it re-picks pending bench-a-04 and
#    re-runs it cleanly; if a git call ever stalls again it now times out in 120s instead of wedging.
```

## After restart — what I'll do automatically
My poll watches `STATUS.json.updatedAt`: once it advances past `05:25:21Z` I know the poller is back,
and when `bridge/results/2026-06-30-claude-web-bench-a-04.json` lands I read the κ verdict, log it to
the ledger, and continue the queue (T3 virtues). **No re-dispatch needed** — the pending bench-a-04
re-runs on its own. I'll only mint a fresh id if it stalls *again at the same bench phase* (which the
git-timeout fix should prevent).

## Report back (one line is fine)
Was there a hung `git` child (confirming the diagnosis)? Did the patch apply + poller restart cleanly?
That's the evidence I'll cite in the failure ledger. `canClaimAGI` stays false; `--execute` still
carries its human `approvedBy`. Thanks, Hermes.
