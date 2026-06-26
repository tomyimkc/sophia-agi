# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sentinel — offline supervision over the agent's own trace logs.

Distilled from AgentArk's Sentinel idea, rebuilt to the Sophia discipline: a
pure, deterministic, offline analyzer that reads the harness run logs
(``agent/memory/agent_runs/*.jsonl``) and the MCP audit log and reports:

  * **failure classification** — histogram of why steps failed
    (``gate_violation``, ``verifier_fail``, ``tool_error``, ``model_error``,
    ``empty_output``, ``max_retries_exhausted``, …), reusing the harness's own
    :data:`agent.harness.FAILURE_CLASSES` vocabulary so the two never drift.
  * **drift detection** — compares the *recent* distribution of gate/critic
    verdicts (and abstention rate) against a committed baseline and flags a
    shift past a threshold. Pure statistics — no model, no network, no ML.
  * **savings ledger** — totals the ``arkdistill`` saved-token events that
    Phase-1 ArkDistill emits into the same logs.

Nothing here executes, mutates, or calls a model: Sentinel only *observes*. It
never raises — a malformed event is skipped, never fatal (fail-open on read).
The active, blocking sibling is :mod:`sophia_mcp.approval` (approval gate) and
the :func:`skills.sentinel_watch` skill.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import ROOT
from agent.harness import FAILURE_CLASSES

RUNS_DIR = ROOT / "agent" / "memory" / "agent_runs"
AUDIT_LOG = ROOT / "agent" / "memory" / "mcp_audit.jsonl"

# Critic/gate verdicts we treat as "the model declined to assert" — the signal
# whose *rate* drift matters most for a fail-closed system (a sudden drop in
# abstention can mean the gate stopped firing; a spike can mean a broken tool).
ABSTAIN_VERDICTS = ("held", "abstain", "abstained", "revise", "retrieve", "clarify", "escalate")
BLOCK_VERDICTS = ("rejected", "block", "blocked")


@dataclass
class SentinelReport:
    """A deterministic supervision summary over a set of run events."""

    runs: int = 0
    steps: int = 0
    failures: Counter = field(default_factory=Counter)   # failure_class -> count
    verdicts: Counter = field(default_factory=Counter)    # gate/critic verdict -> count
    saved_tokens: int = 0
    arkdistill_events: int = 0

    @property
    def failure_rate(self) -> float:
        return round(sum(self.failures.values()) / self.steps, 4) if self.steps else 0.0

    @property
    def abstention_rate(self) -> float:
        total = sum(self.verdicts.values())
        if not total:
            return 0.0
        ab = sum(c for v, c in self.verdicts.items() if v in ABSTAIN_VERDICTS)
        return round(ab / total, 4)

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "steps": self.steps,
            "failures": dict(self.failures),
            "failureRate": self.failure_rate,
            "verdicts": dict(self.verdicts),
            "abstentionRate": self.abstention_rate,
            "savedTokens": self.saved_tokens,
            "arkdistillEvents": self.arkdistill_events,
        }


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #


def classify_failure(event: dict) -> str | None:
    """Map one harness log event to a failure class, or None if it is not a
    failure. Mirrors :data:`agent.harness.FAILURE_CLASSES`.

    Understands the two shapes the harness emits: a ``step_output`` event
    (carries ``passed`` + ``failureClass``) and a ``critic`` event (carries
    ``gatePassed`` / ``verifierPassed``).
    """
    if not isinstance(event, dict):
        return None
    etype = event.get("type")
    if etype == "step_output":
        if event.get("passed"):
            return None
        fc = event.get("failureClass")
        return fc if fc in FAILURE_CLASSES else "unknown"
    if etype == "step_failed":
        fc = event.get("failureClass")
        return fc if fc in FAILURE_CLASSES else "unknown"
    if etype == "critic":
        if event.get("gatePassed") is False:
            return "gate_violation"
        if event.get("verifierPassed") is False:
            return "verifier_fail"
    return None


def _verdict_of(event: dict) -> str | None:
    """Extract a gate/critic verdict label from an event, if any."""
    if not isinstance(event, dict):
        return None
    for key in ("verdict", "gateVerdict"):
        v = event.get(key)
        if isinstance(v, str):
            return v
    if event.get("type") == "critic":
        if event.get("gatePassed") is False or event.get("verifierPassed") is False:
            return "held"
        if event.get("gatePassed") and event.get("verifierPassed"):
            return "accepted"
    return None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def scan_events(events: "list[dict]") -> SentinelReport:
    """Summarize an iterable of harness log events into a :class:`SentinelReport`.

    Deterministic and pure; counts steps by distinct ``step_output`` events.
    """
    rep = SentinelReport()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "step_output":
            rep.steps += 1
        fc = classify_failure(ev)
        if fc:
            rep.failures[fc] += 1
        verdict = _verdict_of(ev)
        if verdict:
            rep.verdicts[verdict] += 1
        if etype == "arkdistill":
            rep.arkdistill_events += 1
            rep.saved_tokens += int(ev.get("savedTokens", 0) or 0)
    return rep


def _read_jsonl(path: Path) -> "list[dict]":
    """Read a JSONL file into dicts; missing file or bad lines are skipped (fail-open)."""
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except FileNotFoundError:
        return out
    except Exception:
        return out
    return out


def scan_runs(runs_dir: Path | None = None) -> SentinelReport:
    """Aggregate every run log under ``runs_dir`` into one report."""
    runs_dir = runs_dir or RUNS_DIR
    rep = SentinelReport()
    try:
        logs = sorted(runs_dir.glob("*.jsonl"))
    except Exception:
        logs = []
    for log in logs:
        events = _read_jsonl(log)
        if not events:
            continue
        sub = scan_events(events)
        rep.runs += 1
        rep.steps += sub.steps
        rep.failures.update(sub.failures)
        rep.verdicts.update(sub.verdicts)
        rep.saved_tokens += sub.saved_tokens
        rep.arkdistill_events += sub.arkdistill_events
    return rep


# --------------------------------------------------------------------------- #
# Drift detection (pure statistics)
# --------------------------------------------------------------------------- #


def verdict_distribution(verdicts: "Counter | dict") -> dict:
    """Normalize a verdict count map to a probability distribution (deterministic)."""
    total = sum(verdicts.values())
    if not total:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(verdicts.items())}


def detect_drift(recent: "Counter | dict", baseline: "Counter | dict",
                 *, threshold: float = 0.15) -> dict:
    """Flag distributional drift between ``recent`` and a committed ``baseline``.

    Two signals, both deterministic and bounded in [0, 1]:
      * ``abstentionDelta`` — absolute change in abstention rate (the fail-closed
        canary: a collapse in abstention is the dangerous direction).
      * ``l1`` — total-variation-style L1 distance over the full verdict
        distribution (catches shifts the abstention rate alone would miss).

    ``drift`` is True iff either signal exceeds ``threshold``. No model, no state.
    """
    rec = verdict_distribution(recent)
    base = verdict_distribution(baseline)

    def _abstain(dist: dict) -> float:
        return round(sum(p for v, p in dist.items() if v in ABSTAIN_VERDICTS), 4)

    abstain_delta = round(abs(_abstain(rec) - _abstain(base)), 4)
    keys = set(rec) | set(base)
    l1 = round(sum(abs(rec.get(k, 0.0) - base.get(k, 0.0)) for k in keys) / 2, 4)

    drift = abstain_delta > threshold or l1 > threshold
    direction = None
    if drift:
        direction = "abstention_drop" if _abstain(rec) < _abstain(base) else "abstention_rise"
    return {
        "drift": drift,
        "abstentionDelta": abstain_delta,
        "l1": l1,
        "direction": direction,
        "threshold": threshold,
        "recent": rec,
        "baseline": base,
    }
