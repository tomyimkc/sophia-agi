# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A2A distillation — turn logged inter-agent messages into reusable skills + training rows.

The complement of :mod:`agent.trace_distill` (which mines single-agent harness logs for
DPO pairs). When the swarm runs in agent-to-agent mode, every inter-agent prompt+response is
logged as an ``a2a_message`` span by :mod:`agent.thinking_trace`. Those exchanges are the data
half of A2A self-improvement: a *successful, gated* delegation chain is a worked example of
"this kind of task routes to this kind of agent and gets this kind of answer", which can be
(a) lifted into a **skill candidate** for the skill layer, or (b) emitted as an **SFT row** so
the next model internalises the A2A chain and future swarms need fewer hops.

Discipline (mirrors agent/trace_distill.py exactly):
  * **fail-closed / gated** — a message becomes training data ONLY when it succeeded
    (``ok``) AND was not gated out (``gate`` is absent or ``"accept"``) AND carries a
    non-empty response. Abstains/blocks/empties are never turned into "good" examples.
  * **deterministic / offline** — pure parsing of the append-only JSONL trace; no model call,
    no network. Reproducible from a committed trace.
  * **provenance-preserving** — every row records its originating trace id, sender, receiver,
    and leg (``a2aKind``) so the training signal stays auditable.
  * **needs verbatim capture** — rows can only be built from spans logged under
    ``SOPHIA_CAPTURE_THINKING`` (which stores the actual prompt/response text). Hash-only
    spans are skipped, and that count is reported rather than silently dropped.

This does NOT train anything and does NOT register skills — it produces candidate datasets a
gated downstream job (out of scope) consumes. A2A peer output is untrusted external data, so a
human/gate must vet skill candidates before promotion; nothing here auto-promotes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Legs we accept as training signal. "delegate" alone has no response (it's the outbound
# task), so it is used for skill-shape mining but not for SFT rows.
_ANSWER_LEGS = frozenset({"result", "synthesis", "peer"})


@dataclass
class A2ATrainingRow:
    """One supervised example distilled from a successful A2A exchange."""

    prompt: str
    completion: str
    trace_id: str
    sender: str
    receiver: str
    a2a_kind: str

    def to_record(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "meta": {
                "traceId": self.trace_id,
                "sender": self.sender,
                "receiver": self.receiver,
                "a2aKind": self.a2a_kind,
                "source": "a2a",
            },
        }


@dataclass
class SkillCandidate:
    """A proposed skill mined from recurring successful delegations. NOT a registered skill —
    a candidate for human/gate review before it enters the skill layer."""

    intent: str
    receiver_role: str
    support: int  # how many successful exchanges back this candidate
    example_prompt: str
    example_completion: str

    def to_record(self) -> dict:
        return {
            "name": f"a2a:{self.receiver_role}:{self.intent}",
            "whenToUse": f"Delegate a '{self.intent}' task to a '{self.receiver_role}' agent.",
            "workflow": [
                f"Route the task to a '{self.receiver_role}' agent.",
                "Gate the returned answer (untrusted peer output) before acting on it.",
            ],
            "support": self.support,
            "example": {"prompt": self.example_prompt, "completion": self.example_completion},
            "status": "candidate",  # never auto-promoted; review/gate before use
        }


def _is_good(ev: dict) -> bool:
    """Fail-closed acceptance: a usable exchange succeeded and cleared any gate."""
    if ev.get("kind") != "a2a_message":
        return False
    if not ev.get("ok"):
        return False
    gate = ev.get("gate")
    if gate not in (None, "accept"):
        return False
    return bool(str(ev.get("response", "")).strip() and str(ev.get("prompt", "")).strip())


def _good_delegate(ev: dict) -> bool:
    """A usable routing example: a delegate leg that ran with a non-empty task prompt.

    Delegate legs carry no response (the response is the later ``result`` leg), so they
    are judged on dispatch success, not on a gated answer."""
    return (
        ev.get("kind") == "a2a_message"
        and ev.get("a2aKind") == "delegate"
        and bool(ev.get("ok", True))
        and bool(str(ev.get("prompt", "")).strip())
    )


# Common function words that carry no routing signal — dropped so the intent slug keys on
# the content words (e.g. "review the gacha odds" -> "review-gacha", not "review-the").
_STOPWORDS = frozenset("the a an of to for our my your this that with and or on in at is are be".split())


def _intent_of(prompt: str) -> str:
    """A coarse, deterministic intent slug from the prompt's leading content words. Cheap and
    offline by design — it groups similar delegations without a model in the loop. Two content
    words keep the grouping coarse enough that near-identical tasks share a candidate."""
    words = [w for w in re.findall(r"[a-z]+", prompt.lower()) if w not in _STOPWORDS]
    return "-".join(words[:2]) if words else "task"


def _receiver_role(receiver: str) -> str:
    """Normalise a receiver id (e.g. 'parent.sub1-review-the-odds') to a role slug."""
    tail = receiver.split("-", 1)[1] if "-" in receiver.split(".")[-1] else receiver.split(".")[-1]
    words = re.findall(r"[a-z]+", tail.lower())
    return "-".join(words[:2]) if words else (receiver or "agent")


def a2a_training_rows(events: list[dict]) -> list[A2ATrainingRow]:
    """Build SFT rows from the good answer-bearing exchanges in one trace's events."""
    rows: list[A2ATrainingRow] = []
    for ev in events:
        if ev.get("a2aKind") not in _ANSWER_LEGS or not _is_good(ev):
            continue
        rows.append(
            A2ATrainingRow(
                prompt=ev["prompt"],
                completion=ev["response"],
                trace_id=str(ev.get("traceId", "")),
                sender=str(ev.get("sender", "")),
                receiver=str(ev.get("receiver", "")),
                a2a_kind=str(ev.get("a2aKind", "")),
            )
        )
    return rows


def skill_candidates(events: list[dict], *, min_support: int = 2) -> list[SkillCandidate]:
    """Mine recurring (intent, receiver-role) delegation shapes into skill candidates.

    Only successful, gated exchanges count toward support; a shape must recur at least
    ``min_support`` times to surface (a one-off delegation is not yet a skill)."""
    # Index successful results by sender so a candidate can show a real worked answer.
    results_by_sender: dict[str, dict] = {}
    for ev in events:
        if ev.get("a2aKind") == "result" and _is_good(ev):
            results_by_sender.setdefault(str(ev.get("sender", "")), ev)

    by_key: dict[tuple[str, str], list[dict]] = {}
    for ev in events:
        if not _good_delegate(ev):
            continue
        key = (_intent_of(ev["prompt"]), _receiver_role(str(ev.get("receiver", ""))))
        by_key.setdefault(key, []).append(ev)

    out: list[SkillCandidate] = []
    for (intent, role), evs in sorted(by_key.items()):
        if len(evs) < min_support:
            continue
        sample = evs[0]
        # The matching answer (if any) is the result leg whose sender is this delegate's receiver.
        answer = results_by_sender.get(str(sample.get("receiver", "")), {}).get("response", "")
        out.append(
            SkillCandidate(
                intent=intent,
                receiver_role=role,
                support=len(evs),
                example_prompt=sample["prompt"],
                example_completion=answer,
            )
        )
    return out


def _load(path: str | Path) -> list[dict]:
    """Read one trace JSONL, keeping only a2a_message spans; never raise on a corrupt line."""
    path = Path(path)
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") == "a2a_message":
            events.append(ev)
    return events


@dataclass
class DistillReport:
    """What a distillation pass produced — including what it had to skip, never silently."""

    rows: list[A2ATrainingRow] = field(default_factory=list)
    candidates: list[SkillCandidate] = field(default_factory=list)
    total_messages: int = 0
    hash_only_skipped: int = 0  # spans without verbatim text (SOPHIA_CAPTURE_THINKING was off)


def distill_events(events: list[dict], *, min_support: int = 2) -> DistillReport:
    a2a = [e for e in events if e.get("kind") == "a2a_message"]
    hash_only = sum(1 for e in a2a if "prompt" not in e)
    return DistillReport(
        rows=a2a_training_rows(events),
        candidates=skill_candidates(events, min_support=min_support),
        total_messages=len(a2a),
        hash_only_skipped=hash_only,
    )


def distill_dir(thinking_dir: str | Path, *, min_support: int = 2) -> DistillReport:
    """Distill every ``*.jsonl`` trace under ``thinking_dir`` (sorted for determinism)."""
    thinking_dir = Path(thinking_dir)
    events: list[dict] = []
    if thinking_dir.exists():
        for path in sorted(thinking_dir.glob("*.jsonl")):
            events.extend(_load(path))
    return distill_events(events, min_support=min_support)


def rows_to_jsonl(rows: Iterable[A2ATrainingRow]) -> str:
    """Serialise SFT rows as JSONL (one prompt/completion record per line)."""
    return "\n".join(json.dumps(r.to_record(), ensure_ascii=False) for r in rows)
