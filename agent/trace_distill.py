# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Trace distillation — turn harness run logs into preference pairs for the model.

The harness already logs, per step attempt, a ``step_output`` event carrying
``{step, attempt, passed, failureClass, output}`` (see ``agent/harness.py``). When
a step *failed then was fixed* on a later attempt, that single step contains its own
training signal: the passing attempt is the **chosen** completion and an earlier
failing attempt is the **rejected** one, for the *same* prompt. Distilling those
pairs is the data half of the model<->harness co-evolution loop — the harness's
reflect/retry behaviour becomes preference data the next model can internalise, so
future models need fewer retries to reach the same gated answer.

Discipline (Sophia):
  * **fail-closed / gated** — a pair is emitted ONLY when the chosen attempt is
    marked ``passed`` (it cleared the harness critic/gate) and the rejected attempt
    is marked not-``passed``. We never label a still-failing output "chosen", and we
    never pair two passes or two fails.
  * **deterministic / offline** — pure parsing of the append-only JSONL logs; no
    model call, no network. Reproducible from a committed trace.
  * **provenance-preserving** — every pair records the originating task/step and the
    rejected attempt's failure class, so the training signal stays auditable.

This does NOT train anything (no weight updates here) — it produces the dataset; a
downstream DPO/SFT job (out of scope, gated separately) consumes ``to_jsonl``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PreferencePair:
    task_id: str
    step_id: str
    goal: str
    chosen: str  # the passing attempt's output
    rejected: str  # an earlier failing attempt's output on the same step
    rejected_failure_class: str
    chosen_attempt: int
    rejected_attempt: int

    def to_record(self) -> dict:
        """Standard preference-tuning record (prompt / chosen / rejected) plus
        provenance for auditing which harness step produced it."""
        return {
            "prompt": self.goal,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "meta": {
                "taskId": self.task_id,
                "stepId": self.step_id,
                "rejectedFailureClass": self.rejected_failure_class,
                "chosenAttempt": self.chosen_attempt,
                "rejectedAttempt": self.rejected_attempt,
            },
        }


def _goal_of(events: list[dict]) -> str:
    for ev in events:
        if ev.get("type") == "task_start":
            return ev.get("goal", "")
    return ""


def distill_events(events: list[dict]) -> list[PreferencePair]:
    """Extract preference pairs from one run's event list.

    For each step that has at least one passing AND one failing ``step_output``,
    pair the FIRST passing attempt (chosen) with the LAST failing attempt before it
    (rejected — the closest miss, the most informative contrast). A step that never
    passed, or never failed, yields nothing (fail-closed)."""
    goal = _goal_of(events)
    by_step: dict[str, list[dict]] = {}
    for ev in events:
        if ev.get("type") != "step_output":
            continue
        out = ev.get("output")
        if out is None:
            continue
        by_step.setdefault(str(ev.get("step")), []).append(ev)

    pairs: list[PreferencePair] = []
    for step_id, attempts in by_step.items():
        attempts.sort(key=lambda e: e.get("attempt", 0))
        first_pass = next((e for e in attempts if e.get("passed")), None)
        if first_pass is None:
            continue
        prior_fails = [e for e in attempts if not e.get("passed") and e.get("attempt", 0) < first_pass.get("attempt", 0)]
        if not prior_fails:
            continue
        rejected = prior_fails[-1]  # closest miss before the fix
        if not str(first_pass.get("output", "")).strip() or not str(rejected.get("output", "")).strip():
            continue
        pairs.append(
            PreferencePair(
                task_id=_task_id_of(events, step_id),
                step_id=step_id,
                goal=goal,
                chosen=first_pass["output"],
                rejected=rejected["output"],
                rejected_failure_class=str(rejected.get("failureClass") or "unknown"),
                chosen_attempt=int(first_pass.get("attempt", 0)),
                rejected_attempt=int(rejected.get("attempt", 0)),
            )
        )
    return pairs


def _task_id_of(events: list[dict], default: str) -> str:
    for ev in events:
        if ev.get("type") == "task_start" and ev.get("taskId"):
            return ev["taskId"]
    return default


def distill_file(path: str | Path) -> list[PreferencePair]:
    """Distill one ``*.jsonl`` run log. Skips malformed lines, never raises on a
    corrupt trace (a partial log still yields whatever pairs it can)."""
    path = Path(path)
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return distill_events(events)


def distill_dir(runs_dir: str | Path) -> list[PreferencePair]:
    """Distill every ``*.jsonl`` run log under ``runs_dir`` (sorted for determinism)."""
    runs_dir = Path(runs_dir)
    pairs: list[PreferencePair] = []
    if not runs_dir.exists():
        return pairs
    for path in sorted(runs_dir.glob("*.jsonl")):
        pairs.extend(distill_file(path))
    return pairs


def to_jsonl(pairs: Iterable[PreferencePair]) -> str:
    """Serialise pairs as JSONL preference records (one per line)."""
    return "\n".join(json.dumps(p.to_record(), ensure_ascii=False) for p in pairs)
