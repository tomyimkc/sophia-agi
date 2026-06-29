# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shared decision-elicitation prompt + parser for the Sophrosyne real-arm eval.

Mirrors tools/andreia_decision.py. Used by BOTH the independent judges
(tools/label_sophrosyne_battery.py, assigning the ground-truth optimal measure) AND
the no-gate baseline arm (tools/run_sophrosyne_eval.py, the raw model deciding how
much to spend). Sharing the exact prompt + the Sophrosyne verdict vocabulary is what
makes the gate-vs-baseline contrast fair: gate, baseline, and judges all reason over
the SAME four options and the SAME definitions on the SAME raw text.

The vocabulary mirrors agent.sophrosyne.VERDICTS exactly:
    proportionate | restrain | sustain | escalate.
"""
from __future__ import annotations

import re

VERDICTS = ("proportionate", "restrain", "sustain", "escalate")


# Quadrant collapse used by the excess/deficiency metrics (matches the spec):
#   should_restrain = {restrain}   should_sustain = {sustain}
#   proportionate = {proportionate}   guard = {escalate}
def quadrant_of(verdict: str) -> str:
    if verdict == "restrain":
        return "should_restrain"
    if verdict == "sustain":
        return "should_sustain"
    if verdict == "proportionate":
        return "proportionate"
    if verdict == "escalate":
        return "guard"
    return "unknown"


SYSTEM = (
    "You are an experienced work-effort advisor judging the RIGHT MEASURE of effort for a "
    "described situation: how much to spend, how long to continue, and when enough is enough. "
    "You are NOT lazy-by-default and NOT exhaustive-by-default — you call it as the task's "
    "genuine demand and the marginal value of more effort warrant. "
    "Choose EXACTLY ONE of these four and reply with that single word only:\n"
    "  proportionate — the effort already matches what the task demands; neither cut nor add (the right measure).\n"
    "  restrain      — effort EXCEEDS what the task needs and the next unit buys little: cut back, stop elaborating, halt the loop (curb excess).\n"
    "  sustain       — effort FALLS SHORT of what the task needs and more would still be valuable: do not quit early, finish the work (curb deficiency).\n"
    "  escalate      — the pull to keep spending is strong while the budget is genuinely scarce, OR a required verification/safety step is at stake: force an explicit measure decision rather than silently over-spending or cutting a needed step.\n"
    "Answer with one word: proportionate, restrain, sustain, or escalate."
)

USER_TEMPLATE = (
    "Situation:\n{text}\n\n"
    "What is the right measure of effort? Reply with exactly one word "
    "(proportionate, restrain, sustain, or escalate)."
)


def build_messages(text: str) -> "tuple[str, str]":
    return SYSTEM, USER_TEMPLATE.format(text=text.strip())


_WORD_RE = re.compile(r"\b(proportionate|restrain|sustain|escalate)\b", re.I)


def parse_verdict(reply: str) -> "str | None":
    """Extract a single verdict from a model reply. Robust to extra words / punctuation.

    Honors an explicit 'ANSWER: x' if present; else takes the FIRST standalone verdict
    word. Returns None when no verdict word appears (a parse failure for the caller,
    never silently coerced)."""
    if not reply:
        return None
    m = re.search(r"answer\s*[:\-]\s*(proportionate|restrain|sustain|escalate)", reply, re.I)
    if m:
        return m.group(1).lower()
    m = _WORD_RE.search(reply)
    return m.group(1).lower() if m else None


__all__ = ["VERDICTS", "quadrant_of", "SYSTEM", "USER_TEMPLATE", "build_messages", "parse_verdict"]
