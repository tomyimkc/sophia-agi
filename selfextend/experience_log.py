# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Experience log — append-only record of completed/corrected runs, the fuel for
:mod:`selfextend.evolve`.

Distilled from AgentArk's GEPA/Evolve "experience_runs" store, rebuilt to the
Sophia discipline: an append-only JSONL (schema ``sophia.experience.v1``),
provenance-tagged, offline, never mutated in place. Evolve reads this log to
propose prompt/policy/profile candidates; nothing here trains or executes.

An :class:`Experience` is one outcome: a target artifact (``prompt:advisor``,
``policy:route``, ``profile:browser_html``, …), the input, the produced output,
a verifier/gate ``outcome`` (``pass|fail|abstain``) and a bounded ``reward`` in
[-1, 1]. The reward MUST come from a verifier or gate — never a model's
self-score — so Evolve optimises against the same hard signal the gate uses.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent.config import ROOT

EXPERIENCE_LOG = ROOT / "agent" / "memory" / "experience_log.jsonl"
SCHEMA = "sophia.experience.v1"

# An outcome must come from one of these hard sources (mirrors the failure_memory
# discipline: no model self-judgement is admissible as a reward).
ALLOWED_OUTCOMES = ("pass", "fail", "abstain")


@dataclass
class Experience:
    """One verifier/gate-scored run outcome for an evolvable ``target``."""

    target: str            # "prompt:advisor" | "policy:route" | "profile:browser_html" | ...
    input: str
    output: str
    outcome: str           # one of ALLOWED_OUTCOMES
    reward: float = 0.0    # bounded [-1, 1]; from a verifier/gate, never self-score
    provenance: str = ""   # e.g. "gate.check_response", "verifier.math_sound", run id
    ts: str = ""
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.outcome not in ALLOWED_OUTCOMES:
            raise ValueError(f"outcome must be one of {ALLOWED_OUTCOMES}, got {self.outcome!r}")
        # clamp reward into bounds (fail-closed: a runaway reward can never escape)
        self.reward = max(-1.0, min(1.0, float(self.reward)))
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat()


def record(exp: Experience, *, path: Path | None = None) -> dict:
    """Append one experience. Returns a small ack dict. Never mutates prior lines."""
    path = path or EXPERIENCE_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(exp), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return {"ok": True, "target": exp.target, "outcome": exp.outcome, "reward": exp.reward}


def load(target: str | None = None, *, path: Path | None = None) -> "list[Experience]":
    """Load experiences (optionally filtered to one ``target``). Fail-open on read."""
    path = path or EXPERIENCE_LOG
    out: list[Experience] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return out
    except Exception:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            exp = Experience(
                target=row["target"], input=row.get("input", ""), output=row.get("output", ""),
                outcome=row["outcome"], reward=float(row.get("reward", 0.0)),
                provenance=row.get("provenance", ""), ts=row.get("ts", ""),
            )
        except Exception:
            continue  # a malformed line is skipped, never fatal
        if target is None or exp.target == target:
            out.append(exp)
    return out


def labelled_examples(target: str, *, path: Path | None = None) -> "list[tuple[str, bool]]":
    """Project a target's experiences into ``(text, label)`` pairs for verifier
    synthesis: label = True iff the run passed. ``abstain`` is excluded (it is not
    a positive or a negative — the system correctly declined)."""
    pairs: list[tuple[str, bool]] = []
    for exp in load(target, path=path):
        if exp.outcome == "abstain":
            continue
        pairs.append((exp.output or exp.input, exp.outcome == "pass"))
    return pairs
