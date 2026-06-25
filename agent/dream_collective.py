# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""REM dream collective — offline generative batch + wake consolidation gate.

REM phase: candidates are written to a non-canonical dream ledger only.
Wake phase: each candidate passes ``conscience_check``; only accepted items may
call ``memory_consolidation.consolidate_result``. Dreams never upsert wiki directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent.config import MEMORY_DIR
from agent.conscience import conscience_check
from agent.memory_consolidation import consolidate_result

DREAMS_DIR = MEMORY_DIR / "agent_dreams"
LEDGER_PATH = DREAMS_DIR / "dream_ledger.jsonl"

SCHEMA = "sophia.dream_collective.v1"


@dataclass
class DreamCandidate:
    """One REM-phase generative candidate (non-canonical until wake gate)."""

    dream_id: str
    goal: str
    text: str
    source: str = "rem_batch"
    createdAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "dreamId": self.dream_id,
            "goal": self.goal,
            "text": self.text,
            "source": self.source,
            "createdAt": self.createdAt,
            "canonical": False,
        }


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:48] or "dream"


def _holdout_questions() -> set[str]:
    from agent.memory_consolidation import _benchmark_questions

    return _benchmark_questions()


def _generality_prompts() -> set[str]:
    path = Path(__file__).resolve().parents[1] / "data" / "generality_tasks.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(t.get("prompt", "")).strip().lower() for t in data.get("tasks", []) if t.get("prompt")}


def contamination_blocked(text: str, goal: str) -> list[str]:
    """Fail-closed: dream must not echo eval/holdout prompts."""
    reasons: list[str] = []
    blob = f"{goal}\n{text}".strip().lower()
    for q in _holdout_questions():
        if q and q in blob:
            reasons.append(f"benchmark question leak: {q[:60]}")
    for p in _generality_prompts():
        if p and p in blob:
            reasons.append(f"generality holdout leak: {p[:60]}")
    return reasons


def append_dream_ledger(candidate: DreamCandidate, *, ledger_path: Path = LEDGER_PATH) -> dict:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    leak = contamination_blocked(candidate.text, candidate.goal)
    entry = {**candidate.to_dict(), "contaminationBlocked": bool(leak), "leakReasons": leak}
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def rem_phase(candidates: list[DreamCandidate], *, ledger_path: Path = LEDGER_PATH) -> dict:
    """Write dream candidates to the non-canonical ledger (never wiki)."""
    written = blocked = 0
    entries: list[dict] = []
    for cand in candidates:
        entry = append_dream_ledger(cand, ledger_path=ledger_path)
        entries.append(entry)
        if entry.get("contaminationBlocked"):
            blocked += 1
        else:
            written += 1
    return {
        "schema": SCHEMA,
        "phase": "rem",
        "candidateOnly": True,
        "written": written,
        "contaminationBlocked": blocked,
        "entries": entries,
    }


_CONSCIENCE_WAKE_CONTEXT = {
    "canClaimAGI": False,
    "trustUpstreamVerdict": True,
    "factVerdict": "non_factual",
}


def wake_phase(
    *,
    ledger_path: Path = LEDGER_PATH,
    limit: int | None = None,
    tier: str = "memory",
) -> dict:
    """Run conscience gate + consolidation for ledger entries that are not blocked."""
    if not ledger_path.exists():
        return {
            "schema": SCHEMA,
            "phase": "wake",
            "candidateOnly": True,
            "consolidated": 0,
            "rejected": 0,
            "skippedBlocked": 0,
            "entries": [],
        }

    consolidated = rejected = skipped = 0
    results: list[dict] = []
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[-limit:]

    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("contaminationBlocked"):
            skipped += 1
            results.append({"dreamId": entry.get("dreamId"), "verdict": "skipped_contamination"})
            continue

        text = str(entry.get("text", ""))
        goal = str(entry.get("goal", entry.get("dreamId", "dream")))
        decision = conscience_check(
            text,
            mode="memory",
            context=_CONSCIENCE_WAKE_CONTEXT,
        )
        verdict = decision.verdict

        if verdict in ("allow", "revise"):
            # revise still may consolidate if text is usable; use original for demo
            cons = consolidate_result(goal, text, task_id=entry.get("dreamId"), mode="dream", tier=tier)
            if cons.get("ok"):
                consolidated += 1
                results.append({"dreamId": entry.get("dreamId"), "verdict": verdict, "consolidated": True})
            else:
                rejected += 1
                results.append(
                    {
                        "dreamId": entry.get("dreamId"),
                        "verdict": verdict,
                        "consolidated": False,
                        "reasons": cons.get("reasons", []),
                    }
                )
        else:
            rejected += 1
            results.append({"dreamId": entry.get("dreamId"), "verdict": verdict, "consolidated": False})

    return {
        "schema": SCHEMA,
        "phase": "wake",
        "candidateOnly": True,
        "consolidated": consolidated,
        "rejected": rejected,
        "skippedBlocked": skipped,
        "entries": results,
    }


def run_dream_cycle(
    candidates: list[DreamCandidate],
    *,
    ledger_path: Path = LEDGER_PATH,
    tier: str = "memory",
) -> dict:
    """REM then wake in one offline cycle."""
    rem = rem_phase(candidates, ledger_path=ledger_path)
    wake = wake_phase(ledger_path=ledger_path, tier=tier)
    return {
        "schema": SCHEMA,
        "candidateOnly": True,
        "level3Evidence": False,
        "rem": rem,
        "wake": wake,
        "ok": wake["consolidated"] >= 0,
    }
