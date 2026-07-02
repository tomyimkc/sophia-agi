#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill-efficacy report — which skills actually fire, and in what company.

The skills system had no feedback loop: skills are written (skill-author), indexed
(build_skill_index), and trusted to help — but nothing measured whether a skill
firing co-occurred with good or bad session outcomes. The session-trace stream
(``agent/memory/session_traces/events.jsonl``, written by the Stop /
PostToolUse(Skill) hooks and ``sophia_trajectory_record``) now provides the data;
this tool aggregates it.

Run:  python tools/skill_efficacy_report.py [--traces PATH] [--json]

Honest bound (printed in the report): these are CO-OCCURRENCE counts, not causal
effects. A skill that fires in failing sessions may be *responding to* failure
(that is its job), not causing it. Use this to spot dead skills (never fire) and
to shortlist candidates for a real pre-registered comparison — never to promote
or delete a skill by itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "agent" / "memory" / "session_traces" / "events.jsonl"

SCHEMA = "sophia.skill_efficacy.v1"

#: Event kinds that mark a session as having hit trouble. Coarse on purpose.
_FAILURE_KINDS = ("failure", "error", "trap", "regression")


def load_events(path: Path) -> "list[dict]":
    events: list[dict] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def report(events: "list[dict]") -> dict:
    by_session: "dict[str, list[dict]]" = defaultdict(list)
    for ev in events:
        by_session[str(ev.get("sessionId", ""))].append(ev)

    skill_stats: "dict[str, dict]" = defaultdict(lambda: {"invocations": 0,
                                                          "sessions": set(),
                                                          "troubledSessions": set()})
    n_troubled = 0
    for sid, evs in by_session.items():
        troubled = any(any(f in str(ev.get("kind", "")).lower() for f in _FAILURE_KINDS)
                       for ev in evs)
        n_troubled += int(troubled)
        for ev in evs:
            if ev.get("kind") != "skill_invocation":
                continue
            s = skill_stats[str(ev.get("skill", "?"))]
            s["invocations"] += 1
            s["sessions"].add(sid)
            if troubled:
                s["troubledSessions"].add(sid)

    skills = {
        name: {
            "invocations": s["invocations"],
            "sessions": len(s["sessions"]),
            "troubledSessions": len(s["troubledSessions"]),
        }
        for name, s in sorted(skill_stats.items())
    }
    # dead skills: indexed but never observed firing
    indexed = sorted(p.parent.name for p in (ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    never_fired = [name for name in indexed if name not in skills]
    return {
        "schema": SCHEMA,
        "nEvents": len(events),
        "nSessions": len(by_session),
        "nTroubledSessions": n_troubled,
        "skills": skills,
        "indexedButNeverObserved": never_fired,
        "honestBound": ("co-occurrence only, not causal effect; a skill firing in a "
                        "troubled session may be responding to trouble, not causing it. "
                        "Promotion/deletion decisions need a pre-registered comparison."),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, default=TRACES)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = report(load_events(args.traces))
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0
    print(f"events: {rep['nEvents']}  sessions: {rep['nSessions']} "
          f"({rep['nTroubledSessions']} troubled)")
    if not rep["skills"]:
        print("no skill invocations observed yet — the PostToolUse(Skill) hook "
              "accumulates them as sessions run")
    for name, s in rep["skills"].items():
        print(f"  {name}: {s['invocations']} invocation(s) across {s['sessions']} "
              f"session(s), {s['troubledSessions']} troubled")
    if rep["indexedButNeverObserved"]:
        print(f"indexed but never observed: {', '.join(rep['indexedButNeverObserved'])}")
    print(f"note: {rep['honestBound']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
