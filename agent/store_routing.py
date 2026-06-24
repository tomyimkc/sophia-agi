# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Two-store routing for promotion gates (the v4 learning-under-shift diagnosis).

PR #82's architecture splits knowledge into two stores:
  - HABIT store      ("neocortex" / LoRA weights): HOW the model answers — qualification,
                     source discipline, attribution habits. Shaped by training.
  - KNOWLEDGE store  ("hippocampus" / OKF graph + retrieval): WHAT new facts it knows.
                     Learned by appending pages and retrieving them at inference.

v4 seed 0 failed `learning-under-shift` with pre 0/10 AND post 0/10 while old-task
retention held. That signature is STRUCTURAL: learning-under-shift teaches by appending
declarative records to memory and post-tests whether they're used — but it was gated
against a frozen LoRA adapter (`--backend adapter`), which cannot absorb newly-appended
facts and whose answer path did not retrieve them. So post == pre by construction.

The fix is NOT to weaken the gate. It is to route each signal to the store that can
actually move it, and to treat a knowledge-store signal measured on the habit store as an
INVALID measurement (quarantine — neither promote nor reject), forcing re-measurement on
the graph/retrieval path. This module is that router. It is store-classification + verdict
logic only; it changes no thresholds and trains nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STORE_HABIT = "habit"
STORE_KNOWLEDGE = "knowledge"

# Named promotion/eval signals → the store that can move them.
_KNOWLEDGE_SIGNALS = {
    "learning-under-shift", "learning_under_shift", "learningundershift",
    "distribution-shift", "distribution_shift", "distributionshift",
    "continual-qa", "continual_qa", "continualqa", "cpqa",
    "retention-of-new-facts", "new-fact-recall", "fact-acquisition",
}
_HABIT_SIGNALS = {
    "eval-ladder", "eval_ladder", "ladder", "seib", "seib-qualification",
    "source-discipline", "qualification", "abstention-calibration", "attribution",
    "philosophy", "psychology", "history", "religion", "math", "coding", "personality",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")


def classify_signal(name: str) -> str:
    """Route a named gate/eval signal to STORE_HABIT or STORE_KNOWLEDGE.

    Knowledge-store signals are the ones that require LEARNING NEW DECLARATIVE FACTS
    (shift, continual-QA, fact recall). Everything else — domain ladders, SEIB
    qualification, source-discipline habits — is a habit-store signal.
    """
    n = _norm(name)
    if n in _KNOWLEDGE_SIGNALS:
        return STORE_KNOWLEDGE
    if n in _HABIT_SIGNALS:
        return STORE_HABIT
    if any(tok in n for tok in ("shift", "continual", "cpqa", "acquisition", "recall")):
        return STORE_KNOWLEDGE
    return STORE_HABIT  # default: most eval signals shape habits


@dataclass
class StoreAwareVerdict:
    verdict: str                       # promote | quarantine | reject
    habit_verdict: str
    reasons: list[str] = field(default_factory=list)
    knowledge_store: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "habitVerdict": self.habit_verdict,
            "reasons": self.reasons,
            "knowledgeStore": self.knowledge_store,
            "note": self.note,
        }


def store_aware_adapter_verdict(
    *,
    habit_pass: bool,
    habit_reasons: list[str] | None = None,
    knowledge_goals: list[str] | None = None,
    knowledge_validated: bool | None = None,
    mismatch_signals: list[str] | None = None,
) -> StoreAwareVerdict:
    """Verdict for a LoRA adapter, which is a HABIT-store artifact.

    Rules (none of which weaken the gate):
      1. A habit-store failure REJECTS — the adapter did not do its job.
      2. A knowledge-store signal measured on the habit store (``mismatch_signals``,
         e.g. learning-under-shift run with ``--backend adapter``) is an INVALID
         measurement → QUARANTINE. It is recorded, never used to promote OR reject the
         adapter, and must be re-measured on the graph/retrieval path.
      3. Knowledge goals that were claimed but not validated on the knowledge store →
         QUARANTINE (unproven), not promote.
      4. Otherwise → PROMOTE the habit adapter; knowledge goals are gated separately.
    """
    habit_reasons = list(habit_reasons or [])
    knowledge_goals = list(knowledge_goals or [])
    mismatch_signals = list(mismatch_signals or [])
    reasons: list[str] = []

    ks = {
        "goals": knowledge_goals,
        "validatedOnKnowledgeStore": knowledge_validated,
        "mismatchSignals": mismatch_signals,
        "route": "graph/retrieval (OKF + CPQA) — NOT the LoRA adapter",
    }

    habit_verdict = "pass" if habit_pass else "fail"

    if not habit_pass:
        reasons.extend(f"habit-store failure: {r}" for r in (habit_reasons or ["habit metrics did not pass"]))
        return StoreAwareVerdict("reject", habit_verdict, reasons, ks,
                                 note="Rejected on the habit store — the adapter's own job.")

    if mismatch_signals:
        reasons.append(
            "INVALID measurement: knowledge-store signal(s) "
            f"{mismatch_signals} were measured on the habit store (frozen adapter, no "
            "retrieval) — neither promote nor reject; re-measure on the graph/CPQA path."
        )
        return StoreAwareVerdict("quarantine", habit_verdict, reasons, ks,
                                 note="Habit store passed; a knowledge signal was mis-measured. "
                                      "Quarantine until re-run on the knowledge store.")

    if knowledge_goals and not knowledge_validated:
        reasons.append(
            f"knowledge goals {knowledge_goals} not yet validated on the knowledge store "
            "(graph/CPQA); habit adapter is sound but knowledge learning is unproven."
        )
        return StoreAwareVerdict("quarantine", habit_verdict, reasons, ks,
                                 note="Promote the habit adapter only after knowledge goals "
                                      "are validated on the graph/retrieval path.")

    reasons.append("habit store passed; no mis-measured or unvalidated knowledge signals.")
    return StoreAwareVerdict("promote", habit_verdict, reasons, ks,
                             note="Habit-store adapter promotable; knowledge goals gated separately.")
