#!/usr/bin/env python3
# PLANNING/HARNESS ONLY - no capability claim; canClaimAGI stays false.
"""Record the REAL 2026-06-30 bench-a-04 poller-stall incident as an OKF loop trace.

This replays the actual observe -> reason -> decide -> act -> verify -> resolve loop the agent ran
to diagnose and fix the bench-a-04 stall, writing it into the repo OKF structure under
  okf/incidents/2026-06-30-bench-a-04-stall/
via agent/okf_loop.py. Deterministic (fixed clock), so re-running reproduces identical bytes.

Run:  python tools/record_bench_a04_incident.py            # writes into the repo
      python tools/record_bench_a04_incident.py --check    # write to a temp dir + print summary only
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.okf_loop import LoopLog, _fixed_clock  # noqa: E402

INCIDENT_ID = "2026-06-30-bench-a-04-stall"


def record(root: str | Path) -> dict:
    """Write the incident loop and return its summary."""
    log = LoopLog(INCIDENT_ID, root=root, clock=_fixed_clock(base=1_782_795_900.0))

    log.event(
        "bench-a-04 stalled — heartbeat frozen, GPU idle, no result",
        "The cloud-dispatched kappa re-run (seeds {1,2,10}) wedged: STATUS.json heartbeat froze at "
        "05:25:21Z, GPU read 0% idle, and no bridge/results/...bench-a-04.json was ever written — "
        "27+ minutes of silence while the poller still claimed `running`.",
        sources=["bridge/STATUS.json"], verdict="fail",
    )
    log.observe(
        "reconcile TrainWatch vs the bridge",
        "Compared the owner's TrainWatch view (1 serve running, GPU idle) against bridge/STATUS.json. "
        "TrainWatch data matched the mirror; the divergence was the bridge poller's frozen heartbeat "
        "(13 min serve-elapsed) vs the live snapshot (40 min) — a ~27 min gap vs the normal ~38s.",
        actor="Claude (Read/Grep/Bash)", tool="Bash",
    )
    log.reason(
        "a dead heartbeat means the loop itself blocked",
        "The poller is non-blocking and commits a heartbeat EVERY tick while a job runs. So a dead "
        "heartbeat cannot be a long job — the tick loop must be blocked. Most likely a network call "
        "with no timeout. Suspect the git operations in _sync()/_push_with_retry().",
    )
    log.act(
        "audit github_bridge_poll.py for an un-timed blocking call",
        "Read the poller's _git() helper on spark-bridge: it ran subprocess.run(['git', ...]) with "
        "NO timeout, called every tick. A stalled network git fetch/pull/push blocks the loop forever.",
        actor="Claude (Read)", tool="Read",
    )
    log.decide(
        "root cause = git calls have no timeout; fix = cap them",
        "Cap every git call with SOPHIA_BRIDGE_GIT_TIMEOUT (default 120s); on TimeoutExpired return a "
        "synthesized returncode=124 so callers retry next tick instead of hanging. The self-reload "
        "patch was exonerated (its `_RUNNING is None` gate is correct).",
        moral_standard="honest-root-cause: fix the cause, not just restart (no overclaim)",
    )
    log.act(
        "author + verify the git-timeout patch",
        "Wrote scripts/spark/2026-06-30-poller-git-timeout.patch; generated it as an exact diff and "
        "confirmed `git apply --check` clean against the live spark-bridge blob; patched file parses.",
        actor="Claude (Edit/Bash)", tool="Edit",
    )
    log.verify(
        "patch applies clean + parses",
        "git apply --check: CLEAN; ast.parse of the patched poller: OK.",
        verdict="pass", verifier="git apply --check + ast.parse",
    )
    log.act(
        "ship the fix + brief Hermes",
        "Committed the patch (1a9d388b) on the feature branch; updated bridge/HERMES-BRIEF.md "
        "(f50b2ab7) with the root cause + exact apply/restart steps; logged the footgun to the ledger.",
        actor="Claude (Bash/GitHub MCP)", tool="Bash",
    )
    log.observe(
        "Hermes applied the fix + restarted",
        "spark-bridge commit bf926615 'hard timeout on poller git calls' landed; the poller restarted "
        "and heartbeats fresh every ~37s. bench-a-04 itself was SIGKILL'd (exit -9) mid-A2 as a side "
        "effect of stopping the poller to apply the patch — not a bench-logic failure.",
        actor="Hermes (Spark)",
    )
    log.verify(
        "poller healthy after restart",
        "STATUS.json updatedAt advanced past the frozen 05:25:21Z; heartbeat cadence ~37s; "
        "running cleared. The wedge cannot recur (git calls now time out in 120s).",
        verdict="pass", verifier="STATUS.json heartbeat freshness",
    )
    log.act(
        "re-dispatch the killed bench as bench-a-05",
        "Composed + committed bridge/commands/2026-06-30-claude-web-bench-a-05.json (--bench-a "
        "--execute, seeds {1,2,10}); the healthy poller picked it up and it is running under the fix.",
        actor="Claude (GitHub MCP)", tool="Bash",
    )
    log.resolve(
        "incident resolved",
        "Root cause fixed in code + applied to the running poller; bench re-dispatched and progressing. "
        "Ledger row marked RESOLVED. canClaimAGI stays false.",
        verdict="pass",
    )
    return log.summary()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="write to a temp dir and print the summary only (don't touch the repo)")
    args = ap.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory() as td:
            s = record(td)
            print("incident (temp):", s["incident"], "-", s["steps"], "steps")
            print("  kinds:", " -> ".join(s["kinds"]))
            print("  resolved:", s["resolved"], "| first_failed_node:", s["first_failed_node"])
        return 0

    s = record(ROOT)
    print(f"wrote incident '{s['incident']}' ({s['steps']} steps) to okf/incidents/{INCIDENT_ID}/")
    print("  kinds:", " -> ".join(s["kinds"]))
    print("  resolved:", s["resolved"], "| first_failed_node:", s["first_failed_node"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
