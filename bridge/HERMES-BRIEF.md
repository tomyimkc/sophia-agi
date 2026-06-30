# HERMES BRIEF — bench-a-04 looks STALLED, please diagnose + restart the poller

**From:** cloud session (the bridge). **When:** 2026-06-30, after the MegaTrain T5 push.
**Why you're reading this:** I can't SSH to the Spark — this is a Spark-side op. Please run the
checks below and report back via `bridge/results/` or by replying through your channel.

## The symptom (what the bridge data shows)
- `bridge/STATUS.json` still says `running: 2026-06-30-claude-web-bench-a-04`, `pendingCommands:
  ["2026-06-30-claude-web-bench-a-04"]`.
- **But the poller's last heartbeat commit was `05:25:21Z`** (serve-elapsed ~13 min). The TrainWatch
  live view shows the `lora-ollama-serve` process at **serve-elapsed ~40 min** → the poller has been
  **silent for ~27 minutes**. It normally heartbeats every ~38s.
- **GPU is 0% idle** in the live snapshot.
- **No `bridge/results/2026-06-30-claude-web-bench-a-04.json`** was ever written.

→ Three facts that can't all be true if bench-a-04 is genuinely running GPU work: idle GPU +
27-min heartbeat gap + no result. **bench-a-04 (or the poller) is wedged, not in-flight.**

Note: the last few status commits include `e4f8aa79 bridge: poller self-reload between jobs`
(13:24:29) then `9bfe4a79 ... + started 2026-06-30-claude-web-bench-a-04` (13:24:43). So the
self-reload re-exec fired right before bench-a-04 started — worth checking it didn't leave the
process in a bad state (the self-reload is supposed to re-exec ONLY between jobs, never mid-run).

## Diagnostic steps (please run, in order)
```bash
cd /home/tomyimkc/sophia-bridge

# 1) Is the poller process alive? (note its PID + start time)
ps aux | grep -E "github_bridge_poll" | grep -v grep

# 2) Is there a hung benchmark / judge child? (a Mac-judge HTTP call with no timeout is the prime suspect)
ps aux | grep -E "run_local_benchmarks|judge_pilot_answers|assemble_uplift|certify_lowram|python" | grep -v grep

# 3) Tail the poller log (wherever your launch line redirects it — tmux pane or nohup.out)
tail -n 80 nohup.out 2>/dev/null || true   # adjust to your actual log path / tmux capture-pane

# 4) GPU truly idle?
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null || true

# 5) Did bench-a-04 leave a partial run dir / answer sidecars? (tells us WHICH phase it died in:
#    generate-answers vs judge vs assemble)
ls -lt agi-proof/benchmark-results/ 2>/dev/null | head -10
ls -lt /tmp 2>/dev/null | grep -iE "uplift|judg|bench" | head
```

## Remediation (after you've captured the above)
1. **Kill any hung child** from step 2 (the stuck judge/bench process), then **restart the poller**
   with your normal launch line. On restart it will re-pick `bench-a-04` from `pendingCommands` and
   re-run it cleanly — a fresh poller process drops any wedged HTTP socket / stale state.
2. If it **stalls again at the same phase** (re-check heartbeat + GPU after ~3–5 min), then bench-a-04
   itself is the problem (likely a judge endpoint that's down / a Mac-judge URL not responding). In
   that case **leave it** — don't keep re-running. Tell me which phase + the log lines, and I'll
   **re-dispatch a fresh id** (`bench-a-05`) with the judge config fixed rather than trust the wedged
   command.

## What to report back
Please tell me: (a) was the poller process alive or dead? (b) was there a hung child, and which one?
(c) the last ~20 log lines, (d) GPU util, (e) which phase bench-a-04 reached (any partial output).
That tells me whether a plain restart fixed it or whether I need to re-dispatch with a fixed judge
config. `canClaimAGI` stays false; `--execute` still needs the human `approvedBy` (already on the
command). Thanks, Hermes.
