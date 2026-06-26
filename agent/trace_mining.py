# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mine harness decision logs into OutcomePair traces for the world model (Path A).

Path A's `DreamerWorldPredictor` trains on `(state, action, success)` triples.
Sophia's harness already logs these: each `step_output` event in
`agent/memory/agent_runs/*.jsonl` carries `(step, action, passed)`. This module
extracts a clean `(state, action, success)` corpus from those logs, where:

  - ``state``  = a bucket derived from the task goal + step + failure-class-so-far
                 (the adapter `state_key` in planner_learned_sim maps this further).
  - ``action`` = the step's action ("model" | the tool name) or attempt strategy.
  - ``success`` = 1 if the step eventually passed, else 0.

The state-bucketing is deliberately COARSE (goal-slug + step + prior-failure-class)
because the world model's generalization question is about action-outcome
regularities across tasks, not fine-grained token state. A finer adapter is
`planner_learned_sim.default_state_key`; this miner feeds the same protocol.

Discipline: deterministic, offline, provenance-preserving (each pair records its
source task file). No model calls. Output feeds `verified_world_model.make_splits`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agent.verified_world_model import OutcomePair


def _slug(text: str, *, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:max_len]
    return s or "task"


def _bucket(task_goal: str, step_id: str, prior_failure: str | None) -> str:
    """Coarse state bucket: goal-slug : step : prior-failure-class (if any)."""
    parts = [_slug(task_goal), str(step_id or "?")]
    if prior_failure:
        parts.append(prior_failure)
    return ":".join(parts)


def mine_file(path: str | Path) -> list[OutcomePair]:
    """Extract OutcomePairs from one harness run-log JSONL.

    For each step, the OUTCOME is whether that step eventually passed (success=1) or
    exhausted retries (0). The ACTION is the step's action; the STATE is the bucket.
    prior_failure threads the failure-class of the previous step (a cheap context cue)."""
    path = Path(path)
    pairs: list[OutcomePair] = []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    goal = next((e.get("goal", "") for e in events if e.get("type") == "task_start"), "")
    # Group step_output events by step id; a step's success = any attempt passed.
    by_step: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        if ev.get("type") == "step_output":
            by_step.setdefault(str(ev.get("step", "?")), []).append(ev)
    # Track prior failure-class for state bucketing (cheap context cue).
    prior_failure: str | None = None
    for step_id, attempts in by_step.items():
        if not attempts:
            continue
        success = 1 if any(a.get("passed") for a in attempts) else 0
        # action: prefer the step's action from a step_attempt/model_call event, else 'model'
        action = "model"
        for ev in events:
            if ev.get("type") == "step_attempt" and str(ev.get("step")) == step_id:
                action = str(ev.get("action") or "model")
                break
        state = _bucket(goal, step_id, prior_failure)
        pairs.append((state, action, success))
        # update prior_failure from the last failing attempt's failureClass
        if not success:
            prior_failure = str(attempts[-1].get("failureClass") or "") or None
        else:
            prior_failure = None
    return pairs


def mine_dir(runs_dir: str | Path) -> list[OutcomePair]:
    """Mine every *.jsonl run log under ``runs_dir`` (sorted for determinism)."""
    runs_dir = Path(runs_dir)
    pairs: list[OutcomePair] = []
    if not runs_dir.exists():
        return pairs
    for path in sorted(runs_dir.glob("*.jsonl")):
        pairs.extend(mine_file(path))
    return pairs


def corpus_report(pairs: list[OutcomePair]) -> dict[str, Any]:
    """Introspection: the mined corpus shape (state-bucket count, success rate)."""
    if not pairs:
        return {"size": 0, "states": 0, "successRate": 0.0}
    states = {s for s, _, _ in pairs}
    succ = sum(1 for _, _, l in pairs if l)
    return {
        "size": len(pairs),
        "states": len(states),
        "successRate": round(succ / len(pairs), 4),
        "actions": sorted({a for _, a, _ in pairs}),
        "sample": pairs[:3],
    }


__all__ = ["mine_file", "mine_dir", "corpus_report"]
