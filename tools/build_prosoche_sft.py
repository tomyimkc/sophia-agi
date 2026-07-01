#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build the goal-anchored attention SFT pack (thesis §4.1).

Teaches the model to READ its attention anchor and stay on-goal, in THREE balanced
target classes so it learns the *boundary*, not a blanket "ignore everything new":

  1. on_goal           — continue the active goal (the obvious positive).
  2. decline_distractor— a tempting off-goal tangent is present; the gold target
                         names it out-of-scope and returns to the goal.
  3. re_anchor         — a LEGITIMATE goal change / sub-goal is present; the gold
                         target UPDATES the anchor and follows it (omitting this
                         class is how you train a fixated model — thesis §4.2).

Each row carries the AttentionAnchor as a system STABLE-PREFIX (the same prompt
geometry the inference packer pins, so what the model trains to attend to is
byte-identical to what `context_manager` pins). Deterministic + offline.

Closed-loop self-validation (``--check`` / :func:`validate`): every gold target is
re-scored through ``agent.prosoche.focus_reward_axis`` and must land in its intended
band (on_goal/decline -> FOCUS_ONGOAL; re_anchor -> FOCUS_REANCHOR). No fixation
targets exist — fixation is a NEGATIVE, never a gold SFT target. This is NOT a
real-decision claim; it is a dataset-consistency gate. canClaimAGI:false.

    python tools/build_prosoche_sft.py            # write training/prosoche/attention_sft.jsonl
    python tools/build_prosoche_sft.py --check    # exit 1 if committed pack is stale or inconsistent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prosoche import (  # noqa: E402
    FOCUS_ONGOAL,
    FOCUS_REANCHOR,
    AttentionAnchor,
    focus_reward_axis,
)

OUT = ROOT / "training" / "prosoche" / "attention_sft.jsonl"

SYSTEM_PERSONA = (
    "You are Sophia, working under an explicit ATTENTION ANCHOR. Keep your effort on the "
    "anchor's goal. Decline off-goal tangents by naming them out of scope and returning to "
    "the goal. If the goal legitimately changes — or a safety/security signal appears — say so "
    "and RE-ANCHOR to it; never ignore a real change just to stay on the old goal."
)

# (goal, in_scope_entities, on_goal user+target, distractor user+target, shift user+target)
SCENARIOS = [
    {
        "goal": "fix the failing auth login test in services.auth",
        "entities": ["services.auth", "login", "auth test"],
        "on_goal": (
            "What's the next step on the login test?",
            "Next on the failing login test in services.auth: the auth token check rejects valid "
            "sessions, so I'll inspect the token-expiry comparison in the login handler.",
        ),
        "distractor": (
            "While you're in there, can you also redo the Marketing dashboard colours?",
            "That Marketing dashboard work is out of scope for the current goal — I'll note it for "
            "later and stay on the goal: fixing the failing login test in services.auth.",
        ),
        "shift": (
            "Actually stop — drop the login test, the priority now is the broken logout flow.",
            "Understood, the goal has changed — re-anchoring: the new objective is the broken logout "
            "flow, so I'll set the login test aside and start there.",
        ),
    },
    {
        "goal": "optimise the slow checkout database query latency",
        "entities": ["checkout", "database", "query", "latency"],
        "on_goal": (
            "Where should I look to speed up checkout?",
            "Optimising the slow checkout database query: the checkout database query does a full "
            "table scan, so I'll index it to cut the query latency, then re-measure the checkout "
            "database query latency.",
        ),
        "distractor": (
            "By the way, reorganise the office snack inventory while you think.",
            "The office snack inventory is unrelated to this goal — setting that aside and staying on "
            "the goal: optimising the slow checkout database query latency.",
        ),
        "shift": (
            "Change of plan: forget checkout speed, we now need the payment refund bug fixed first.",
            "Got it, the goal has changed — re-anchoring: the new objective is the payment refund bug, "
            "so I'll pause the checkout query work and pick that up.",
        ),
    },
    {
        "goal": "summarise the Q3 revenue report for the board deck",
        "entities": ["Q3", "revenue", "report", "board deck"],
        "on_goal": (
            "How's the Q3 summary coming?",
            "For the board deck Q3 revenue summary: I'll pull the headline revenue figure and the "
            "quarter-over-quarter delta from the Q3 report and write three bullets.",
        ),
        "distractor": (
            "Also, can you plan the team offsite agenda real quick?",
            "The team offsite agenda is out of scope for this goal — I'll flag it separately and keep "
            "to the goal: summarising the Q3 revenue report for the board deck.",
        ),
        "shift": (
            "Hold on — the board moved up, we need the Q4 forecast summarised instead now.",
            "Understood, the goal has changed — re-anchoring: the new objective is the Q4 forecast "
            "summary, so I'll switch from the Q3 report to that.",
        ),
    },
    {
        "goal": "write unit tests for the rate_limiter module",
        "entities": ["rate_limiter", "unit tests", "module"],
        "on_goal": (
            "What test are you adding next?",
            "Next unit test for the rate_limiter module: I'll cover the burst-then-refill path to "
            "assert the rate_limiter blocks once the bucket empties and recovers after the window.",
        ),
        "distractor": (
            "Quick favour — rename all the variables in the logging package too.",
            "Renaming the logging package variables is out of scope for the current goal — noting it "
            "for later and staying on the goal: writing unit tests for the rate_limiter module.",
        ),
        "shift": (
            "Scrap that — a customer is down, the new priority is reproducing their timeout bug.",
            "Understood, the goal has changed — re-anchoring: the new objective is reproducing the "
            "customer timeout bug, so I'll set the rate_limiter tests aside and start the repro.",
        ),
    },
    {
        "goal": "draft the API migration guide for v2 endpoints",
        "entities": ["API", "migration guide", "v2", "endpoints"],
        "on_goal": (
            "What's next in the migration guide?",
            "Next in the v2 API migration guide: I'll document the breaking change to the v2 auth "
            "endpoints and give a before/after request example for each.",
        ),
        "distractor": (
            "Side quest: can you also pick a new logo colour palette?",
            "The logo colour palette is out of scope for this goal — I'll set it aside and keep to the "
            "goal: drafting the API migration guide for the v2 endpoints.",
        ),
        "shift": (
            "Plans changed — leadership wants the v3 deprecation notice written first instead.",
            "Got it, the goal has changed — re-anchoring: the new objective is the v3 deprecation "
            "notice, so I'll pause the v2 migration guide and draft that.",
        ),
    },
]


def _row(anchor: AttentionAnchor, klass: str, user: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PERSONA + "\n\n" + anchor.render()},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "meta": {
            "family": "prosoche_attention",
            "class": klass,
            "anchorId": anchor.id,
            "goalShift": klass == "re_anchor",
            "expectedFocusBand": "reanchor" if klass == "re_anchor" else "ongoal",
        },
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for sc in SCENARIOS:
        anchor = AttentionAnchor(goal=sc["goal"], in_scope_entities=tuple(sc["entities"]))
        rows.append(_row(anchor, "on_goal", *sc["on_goal"]))
        rows.append(_row(anchor, "decline_distractor", *sc["distractor"]))
        rows.append(_row(anchor, "re_anchor", *sc["shift"]))
    return rows


def _score(row: dict) -> float:
    anchor = AttentionAnchor(
        goal=_goal_of(row), in_scope_entities=tuple(_entities_of(row)),
    )
    target = row["messages"][-1]["content"]
    return focus_reward_axis(target, anchor, goal_shift=row["meta"]["goalShift"])


def _goal_of(row: dict) -> str:
    sysmsg = row["messages"][0]["content"]
    for line in sysmsg.splitlines():
        if line.startswith("goal: "):
            return line[len("goal: "):]
    return ""


def _entities_of(row: dict) -> list[str]:
    sysmsg = row["messages"][0]["content"]
    for line in sysmsg.splitlines():
        if line.startswith("in scope: "):
            return [e.strip() for e in line[len("in scope: "):].split(",")]
    return []


def validate(rows: list[dict] | None = None) -> dict:
    """Closed-loop: every gold target must score in its intended focus band."""
    rows = rows if rows is not None else build_rows()
    problems: list[str] = []
    for r in rows:
        s = _score(r)
        klass = r["meta"]["class"]
        if klass in ("on_goal", "decline_distractor"):
            if s != FOCUS_ONGOAL:
                problems.append(f"{r['meta']['anchorId']}/{klass}: expected on-goal {FOCUS_ONGOAL}, got {s}")
        elif klass == "re_anchor":
            if s != FOCUS_REANCHOR:
                problems.append(f"{r['meta']['anchorId']}/{klass}: expected re-anchor {FOCUS_REANCHOR}, got {s}")
    from collections import Counter
    return {"n": len(rows), "byClass": dict(Counter(r["meta"]["class"] for r in rows)),
            "problems": problems, "ok": not problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify committed pack is up to date AND consistent")
    args = ap.parse_args()

    rows = build_rows()
    report = validate(rows)
    serialised = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"

    if args.check:
        if not report["ok"]:
            print("PROSOCHE SFT: INCONSISTENT —", report["problems"], file=sys.stderr)
            return 1
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != serialised:
            print("PROSOCHE SFT: STALE — re-run `python tools/build_prosoche_sft.py`", file=sys.stderr)
            return 1
        print(f"PROSOCHE SFT: OK — {report['n']} rows, classes {report['byClass']}, all gold targets in-band")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(serialised, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  rows={report['n']} classes={report['byClass']} ok={report['ok']}",
          file=sys.stderr)
    if not report["ok"]:
        print("WARNING — inconsistent gold targets:", report["problems"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
