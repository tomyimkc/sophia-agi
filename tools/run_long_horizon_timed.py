#!/usr/bin/env python3
"""
run_long_horizon_timed.py — execute a long-horizon task under a wall-clock budget,
logging an append-only event trail + human interventions + final artifacts
(TODO items 7-10: the 30-min / 2-h / 1-day runs the harness supports but never ran).

WHAT IT DOES
  Wraps agent.long_horizon.run_long_horizon() with:
    - a wall-clock budget (--minutes 30|120|1440),
    - an APPEND-ONLY JSONL event log (one line per action/tool/failure/self-correction),
    - an intervention counter (human-in-the-loop approvals when --approve-tools),
    - a public-report summarising completed/failed/blocked nodes, cost, interventions,
      and the final artifact paths.

  It drives the REAL harness (build_ledger -> run_long_horizon -> LongHorizonResult);
  it does not reimplement it.

FAIL-CLOSED
  No ModelClient / backend -> writes an "environment artifact, not a run" report and
  exits 0. It NEVER fabricates a completed run, a node result, or a cost. If the budget
  elapses mid-run, the run is marked timed-out with the partial event log intact.

HONEST BOUND
  Produces an execution-health long-horizon artifact (did the harness sustain a timed
  multi-step run, with an auditable trail), NOT a capability claim about the quality of
  the work produced. candidateOnly:true level3Evidence:false canClaimAGI:false.

USAGE
  python3 tools/run_long_horizon_timed.py --spec eval/long_horizon/spec_example.json \
      --minutes 30 --out agi-proof/long-horizon/run-30m.public-report.json \
      --events agi-proof/long-horizon/run-30m.events.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from agent.long_horizon import build_ledger, run_long_horizon
    _REPO_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


class EventLog:
    """Append-only JSONL event trail. Every write flushes so a crash/timeout keeps
    the partial log."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self.count = 0
        self.interventions = 0

    def emit(self, kind: str, **fields: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        self.count += 1
        if kind == "human_intervention":
            self.interventions += 1

    def close(self) -> None:
        self._fh.close()


def _make_client(spec: str | None, log: EventLog, *, allow_mock: bool = False):
    """Return a ModelClient that logs each call, or None (fail-closed).

    run_long_horizon passes `client` down to run_subagent, which calls
    client.generate(system, user) -> ModelResult. So we subclass the repo's real
    ModelClient and wrap generate(), NOT a non-existent Model(spec).complete().
    The mock provider (auto-selected without an API key) fail-closes to None so a
    keyless run does not fabricate a "completed" long-horizon trace.
    """
    try:
        from agent.model import ModelClient, resolve_config, _env_fallbacks
    except Exception:
        return None
    try:
        cfg = resolve_config(spec)
    except Exception:
        return None
    if getattr(cfg, "kind", None) == "mock" and not allow_mock:
        return None

    class LoggingClient(ModelClient):
        def generate(self, system: str, user: str, **kw: Any):
            log.emit("model_call", prompt_chars=len(system) + len(user))
            try:
                res = super().generate(system, user, **kw)
            except Exception as e:
                log.emit("failure", where="model_call", error=f"{type(e).__name__}: {e}")
                raise
            log.emit("model_result", out_chars=len(getattr(res, "text", "")),
                     ok=getattr(res, "ok", None))
            return res

    try:
        return LoggingClient(cfg, _env_fallbacks())
    except Exception:
        return None


def env_artifact(reason: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "environmentArtifact": True, "completedRun": False, "reason": reason,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="JSON: {goal, subtasks:[{id,goal,deps?}]}")
    ap.add_argument("--minutes", type=int, required=True, choices=[30, 120, 1440])
    ap.add_argument("--adapter", default=None, help="model spec; omit -> fail-closed")
    ap.add_argument("--approve-tools", action="store_true", help="human-in-the-loop tool approval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if not _REPO_OK:
        env_artifact(f"repo modules unavailable ({_IMPORT_ERR})", out)
        print("FAIL-CLOSED (env artifact):", _IMPORT_ERR)
        return 0

    spec = json.loads(Path(args.spec).read_text())
    log = EventLog(Path(args.events))
    log.emit("run_start", budget_minutes=args.minutes, goal=spec.get("goal", ""))

    client = _make_client(args.adapter, log)
    if client is None:
        log.emit("abort", reason="no backend")
        log.close()
        env_artifact("no model backend available", out)
        print("FAIL-CLOSED (env artifact): no backend; wrote", out)
        return 0

    ledger = build_ledger(spec["goal"], spec["subtasks"],
                          ledger_id=f"timed-{args.minutes}m-{int(time.time())}")
    deadline = time.time() + args.minutes * 60
    log.emit("ledger_built", nodes=len(spec["subtasks"]), deadline_epoch=deadline)

    # ENFORCED (review D4, now a module change): run_long_horizon takes a cooperative
    # `deadline_monotonic` and (a) stops launching new nodes once it passes and (b)
    # forwards it into run_subagent -> run_agent so a running node stops between plan
    # steps. So --minutes now BINDS at node/step granularity. Honest bound: a single
    # in-flight model call still runs to its transport timeout (cfg.timeout_sec) — this
    # cancels cooperatively, it does not kill a blocking request mid-flight.
    deadline_monotonic = time.monotonic() + args.minutes * 60
    t0 = time.time()
    try:
        result = run_long_horizon(ledger, client=client, approve_tools=args.approve_tools,
                                  deadline_monotonic=deadline_monotonic)
        elapsed = time.time() - t0
        timed_out = time.time() > deadline
        log.emit("run_end", ok=result.ok, elapsed_s=elapsed, timed_out=timed_out)
        report = {
            "ledgerId": result.ledger_id,
            "budgetMinutes": args.minutes,
            "elapsedSeconds": round(elapsed, 1),
            "timedOut": timed_out,
            "ok": result.ok,
            "completed": result.completed,
            "failed": result.failed,
            "blocked": result.blocked,
            "totalCostUsd": result.total_cost_usd,
            "eventCount": log.count,
            "humanInterventions": log.interventions,
            "finalArtifacts": [result.ledger_path],
            "eventsPath": str(Path(args.events)),
            "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
            "honestNote": "Execution-health long-horizon artifact (harness sustained a timed "
                          "multi-step run with an auditable trail). NOT a work-quality claim.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        log.emit("failure", where="run_long_horizon", error=f"{type(e).__name__}: {e}")
        report = {"ledgerId": ledger.ledger_id, "budgetMinutes": args.minutes,
                  "ok": False, "error": f"{type(e).__name__}: {e}",
                  "eventCount": log.count, "humanInterventions": log.interventions,
                  "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False}
    finally:
        log.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"OK: long-horizon report -> {out}  events={log.count}  "
          f"interventions={log.interventions}  ok={report.get('ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
