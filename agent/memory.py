# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decision memory log (local, gitignored)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agent.config import MEMORY_DIR


def log_decision(
    *,
    mode: str,
    question: str,
    answer: str,
    sources: list[str],
    gate: dict,
    tools_run: list[str] | None = None,
) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / "decisions.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "question": question,
        "answer": answer[:4000],
        "sources": sources,
        "gate": gate,
        "tools_run": tools_run or [],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # T-4: also fold this decision into the provenance-gated semantic-memory path
    # (opt-in via SOPHIA_CONSOLIDATE_DECISIONS; fail-soft — never breaks logging).
    if os.environ.get("SOPHIA_CONSOLIDATE_DECISIONS", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from agent.memory_consolidation import consolidate_result
            consolidate_result(question, answer, task_id=f"dec_{path.stem}", mode=mode)
        except Exception:
            pass
    return path


def recent_decisions(*, limit: int = 5) -> list[dict]:
    path = MEMORY_DIR / "decisions.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return entries[-limit:]