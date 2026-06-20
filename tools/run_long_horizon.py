#!/usr/bin/env python3
"""Long-horizon autonomy harness for the Sophia AGI-candidate proof.

Implements the run logger required by agi-proof/long-horizon-runs/README.md:
initial goal, plan, tool calls, state transitions, failed attempts,
self-corrections, human-intervention count, and the final artifact + verification
command. Logs are append-only JSONL so a run survives interruption and can be
resumed; the public report carries counts and an honest autonomy classification
(full / mostly / partial) with no captured command output.

This harness executes a spec of shell steps, so a *real* 30-minute / 2-hour /
1-day run is produced by pointing it at a longer spec — the same logger and
classification apply. `--self-test` runs a short, read-only built-in plan to
demonstrate the harness end-to-end without external credentials.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EVENT_TYPES = {
    "goal",
    "plan",
    "tool_call",
    "state_transition",
    "failed_attempt",
    "self_correction",
    "human_intervention",
    "artifact",
    "verification",
    "note",
}

TIER_THRESHOLDS = [
    (86400, "long-1day"),
    (7200, "medium-2h"),
    (1800, "short-30min"),
    (0, "below-short-demo"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    """Repo-relative path when inside ROOT, else the absolute path. Uses real
    path containment, not a string-prefix test (which mishandles sibling dirs)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


class LongHorizonRun:
    """Append-only event log for one long-horizon autonomous run."""

    def __init__(self, run_id: str, goal: str, *, log_path: Path, plan: list[str] | None = None):
        self.run_id = run_id
        self.goal = goal
        self.plan = plan or []
        self.log_path = Path(log_path)
        self.events: list[dict[str, Any]] = []
        self.start_monotonic = time.monotonic()
        self.start_iso = _now_iso()

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.events.append(record)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def log(self, event_type: str, message: str, **fields: Any) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        record = {
            "seq": len(self.events),
            "ts": _now_iso(),
            "elapsedSec": round(time.monotonic() - self.start_monotonic, 3),
            "type": event_type,
            "message": message,
            **fields,
        }
        return self._append(record)

    def start(self) -> None:
        self.log("goal", self.goal, runId=self.run_id, startedAt=self.start_iso)
        if self.plan:
            self.log("plan", "initial plan", steps=self.plan)

    def completed_steps(self) -> set[str]:
        return {
            event.get("step")
            for event in self.events
            if event.get("type") == "state_transition" and event.get("status") == "done" and event.get("step")
        }

    @classmethod
    def resume(cls, log_path: Path) -> "LongHorizonRun":
        path = Path(log_path)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        goal_event = next((e for e in events if e.get("type") == "goal"), {})
        plan_event = next((e for e in events if e.get("type") == "plan"), {})
        run = cls(
            goal_event.get("runId", "resumed"),
            goal_event.get("message", "resumed run"),
            log_path=path,
            plan=plan_event.get("steps", []),
        )
        run.events = events
        run.start_iso = goal_event.get("startedAt", _now_iso())
        # Anchor elapsed time to the last recorded elapsed so resumed events continue.
        last_elapsed = events[-1].get("elapsedSec", 0.0) if events else 0.0
        run.start_monotonic = time.monotonic() - float(last_elapsed)
        run.log("note", "run resumed from checkpoint", resumedEventCount=len(events))
        return run

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {etype: 0 for etype in EVENT_TYPES}
        for event in self.events:
            etype = event.get("type", "note")
            counts[etype] = counts.get(etype, 0) + 1
        duration = float(self.events[-1].get("elapsedSec", 0.0)) if self.events else 0.0
        interventions = counts.get("human_intervention", 0)
        tool_calls = counts.get("tool_call", 0)
        failures = counts.get("failed_attempt", 0)
        corrections = counts.get("self_correction", 0)
        autonomy = classify_autonomy(interventions, tool_calls, duration)
        return {
            "runId": self.run_id,
            "goal": self.goal,
            "plan": self.plan,
            "startedAt": self.start_iso,
            "durationSec": round(duration, 2),
            "tier": classify_tier(duration),
            "eventCounts": {etype: counts.get(etype, 0) for etype in sorted(EVENT_TYPES)},
            "toolCalls": tool_calls,
            "failedAttempts": failures,
            "selfCorrections": corrections,
            "humanInterventionCount": interventions,
            "autonomy": autonomy,
            "finalArtifacts": [
                {"message": e["message"], **{k: v for k, v in e.items() if k in ("path", "verificationCommand")}}
                for e in self.events
                if e.get("type") in ("artifact", "verification")
            ],
        }


def classify_tier(duration_sec: float) -> str:
    for threshold, name in TIER_THRESHOLDS:
        if duration_sec >= threshold:
            return name
    return "below-short-demo"


def classify_autonomy(interventions: int, tool_calls: int, duration_sec: float = 0.0) -> dict[str, Any]:
    # A run only earns an autonomy *claim* if it is substantive: at least the
    # 30-minute Short tier OR >=10 tool calls. Shorter runs are labelled as a
    # demo so a trivial 0.1s run with 0 interventions cannot claim "full-autonomy".
    substantive = duration_sec >= 1800 or tool_calls >= 10
    if not substantive:
        level = "no-intervention-demo" if interventions == 0 else "demo-with-intervention"
    elif interventions == 0:
        level = "full-autonomy"
    elif interventions <= math.ceil(tool_calls * 0.1):
        level = "mostly-autonomous"
    else:
        level = "partial-autonomy"
    return {
        "level": level,
        "substantive": substantive,
        "humanInterventionCount": interventions,
        "note": (
            "Runs shorter than the 30-minute Short tier (and under 10 tool calls) are "
            "labelled demos, not autonomy claims. Runs with frequent human steering are "
            "reported as partial autonomy, per agi-proof/long-horizon-runs/README.md."
        ),
    }


def run_step(run: LongHorizonRun, step: dict[str, Any], *, timeout_sec: int) -> bool:
    name = step["name"]
    cmd = step["cmd"]
    argv = cmd if isinstance(cmd, list) else ["bash", "-lc", cmd]
    run.log("state_transition", f"begin step: {name}", step=name, status="begin", purpose=step.get("purpose", ""))
    proc = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec, check=False)
    run.log(
        "tool_call",
        f"ran: {name}",
        step=name,
        argv=argv,
        returncode=proc.returncode,
        stdoutTail=proc.stdout[-500:],
        stderrTail=proc.stderr[-500:],
    )
    ok = proc.returncode == 0 or step.get("allowFailure") is True
    if not ok:
        run.log("failed_attempt", f"step failed: {name}", step=name, returncode=proc.returncode)
        retry = step.get("retryCmd")
        if retry:
            retry_argv = retry if isinstance(retry, list) else ["bash", "-lc", retry]
            retry_proc = subprocess.run(retry_argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec, check=False)
            run.log(
                "self_correction",
                f"retried step: {name}",
                step=name,
                argv=retry_argv,
                returncode=retry_proc.returncode,
                stderrTail=retry_proc.stderr[-500:],
            )
            ok = retry_proc.returncode == 0
    if step.get("verification"):
        run.log(
            "verification",
            f"verification for: {name}",
            step=name,
            verificationCommand=" ".join(argv),
            passed=ok,
        )
    if step.get("artifact"):
        run.log("artifact", f"artifact from: {name}", step=name, path=step["artifact"])
    run.log("state_transition", f"end step: {name}", step=name, status="done", ok=ok)
    return ok


SELF_TEST_SPEC: dict[str, Any] = {
    "runId": "long-horizon-self-test",
    "goal": "Demonstrate the long-horizon harness end-to-end with safe read-only repo steps.",
    "plan": [
        "record environment",
        "inspect repo state",
        "validate proof manifest",
        "intentionally fail then self-correct",
    ],
    "steps": [
        {"name": "record-env", "cmd": ["bash", "-lc", "python3 --version; uname -a"], "purpose": "capture environment"},
        {"name": "repo-head", "cmd": ["git", "rev-parse", "HEAD"], "purpose": "record commit hash"},
        {"name": "repo-status", "cmd": ["git", "status", "--short"], "purpose": "inspect working tree"},
        {
            "name": "validate-manifest",
            "cmd": ["python3", "-m", "json.tool", "agi-proof/evidence-manifest.json"],
            "purpose": "verify proof manifest is valid JSON",
            "verification": True,
        },
        {
            "name": "demonstrate-self-correction",
            "cmd": ["bash", "-lc", "exit 3"],
            "retryCmd": ["bash", "-lc", "echo recovered"],
            "purpose": "show failed_attempt -> self_correction logging",
        },
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-horizon autonomy harness")
    parser.add_argument("--spec", type=Path, help="Run spec JSON ({runId, goal, plan, steps})")
    parser.add_argument("--self-test", action="store_true", help="Run the built-in safe demonstration plan")
    parser.add_argument("--log", type=Path, default=None, help="Append-only JSONL log path")
    parser.add_argument("--report-out", type=Path, default=None, help="Public summary report path")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from an existing JSONL log")
    parser.add_argument("--intervene", default=None, help="Record a human intervention note before continuing")
    parser.add_argument("--overwrite", action="store_true", help="Allow a fresh run to replace an existing same-day log")
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    if args.self_test:
        spec = SELF_TEST_SPEC
    elif args.spec:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    elif args.resume:
        spec = None
    else:
        parser.error("provide --spec, --self-test, or --resume")

    date = datetime.now().date().isoformat()
    default_dir = ROOT / "agi-proof" / "long-horizon-runs"

    if args.resume:
        run = LongHorizonRun.resume(args.resume)
        if args.intervene:
            run.log("human_intervention", args.intervene)
        spec = spec or {"steps": []}
    else:
        run_id = spec.get("runId", f"long-horizon-{date}")
        log_path = args.log or (default_dir / f"{run_id}-{date}.log.jsonl")
        # Fresh (non-resume) runs start a clean log. Refuse to clobber an existing
        # log unless --overwrite (or --self-test, which is a regenerable demo) is
        # set; --resume continues an existing run instead.
        if log_path.exists():
            if not (args.overwrite or args.self_test):
                parser.error(
                    f"log already exists: {log_path}. Use --resume to continue it, "
                    f"--overwrite to replace it, or choose a different --log."
                )
            log_path.unlink()
        run = LongHorizonRun(run_id, spec["goal"], log_path=log_path, plan=spec.get("plan", []))
        run.start()
        if args.intervene:
            run.log("human_intervention", args.intervene)

    completed = run.completed_steps()
    for step in spec.get("steps", []):
        if step["name"] in completed:
            run.log("note", f"skipping already-completed step: {step['name']}", step=step["name"])
            continue
        try:
            run_step(run, step, timeout_sec=args.timeout_sec)
        except subprocess.TimeoutExpired:
            run.log("failed_attempt", f"step timed out: {step['name']}", step=step["name"], timedOut=True)

    summary = run.summary()
    report_path = args.report_out or (default_dir / f"{run.run_id}-{date}.public-report.json")
    report = {
        "visibility": "public-aggregate-no-command-output",
        "logPath": _rel(run.log_path),
        "harness": "tools/run_long_horizon.py",
        "note": (
            "Counts and classification only; full command output stays in the JSONL log. "
            "Tier 'below-short-demo' means this run was shorter than the 30-minute Short tier."
        ),
        **summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
