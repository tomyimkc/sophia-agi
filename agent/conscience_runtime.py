# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Conscience-gated agent runtime (trust-layer ticket T-2).

The Conscience Kernel (``agent.conscience``) and its enforcement API
(``agent.conscience_enforcement``) were built but **not consulted in the live
``run_agent`` loop** (``grep conscience agent/harness.py`` was empty). T-2 closes
that gap: :func:`agent.harness.run_agent` now applies :func:`maybe_gate_result`
to the final emit. This module holds the gate logic, shared by the in-loop hook
and the ergonomic wrapper.

This is **gating/orchestration, not intelligence** — no AGI claim, and no public
number ships from it until one is measured under the no-overclaim gate.

Sophia discipline:
  * **opt-in** — off by default; enable with ``conscience_gate=True`` on
    :func:`run_agent` / :func:`run_agent_with_conscience`, or globally via the
    ``SOPHIA_CONSCIENCE_GATE`` env var. A no-op otherwise (same pattern as the
    graded-confidence router).
  * **fail-closed / downgrade-only** — on ``block``/``abstain``/``escalate`` the
    final answer is replaced with an abstention. The blocked text is appended to
    the run trace for audit only and is NEVER returned as ``final_text``.
  * **candidateOnly** — gating infrastructure; the flag is carried on emitted
    trace events.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from agent.conscience_enforcement import EnforcementDecision, enforce_conscience
from agent.harness import AgentResult, AgentTask, run_agent

GATE_ENV = "SOPHIA_CONSCIENCE_GATE"
_FINALIZE_ACTION = "finalize_answer"
_ABSTAIN_TEXT = (
    "Conscience gate held this answer: insufficient verified basis to finalize "
    "(see run trace for the verdict). Abstaining rather than emitting."
)


def gate_enabled(flag: bool | None) -> bool:
    """True if the gate is on: explicit flag wins, else the env var."""
    if flag is not None:
        return bool(flag)
    return os.environ.get(GATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _finalize_gate(text: str) -> EnforcementDecision:
    """Conscience verdict on the final answer (high-impact finalize action)."""
    return enforce_conscience(action=_FINALIZE_ACTION, text=text, high_impact=True)


def _log_hold(trace_path: str, *, decision: EnforcementDecision, withheld: str) -> None:
    """Append a conscience_hold event to the JSONL run trace. Never raises."""
    try:
        event = {
            "type": "conscience_hold",
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": decision.action,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "withheldPreview": withheld[:200],
            "candidateOnly": True,
        }
        with open(trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        # Logging must never break the fail-closed downgrade.
        pass


def apply_to_result(result: AgentResult) -> AgentResult:
    """Apply the conscience gate to a finished result's final emit.

    Downgrade-only: returns ``result`` unchanged if the final text is empty or the
    conscience allows it. Otherwise returns a NEW result with the abstention text,
    ``ok=False``, a ``conscience:<verdict>`` failure, and the withheld text logged.
    """
    if not result.final_text.strip():
        return result
    decision = _finalize_gate(result.final_text)
    if decision.allowed:
        return result
    _log_hold(result.trace_path, decision=decision, withheld=result.final_text)
    return AgentResult(
        task_id=result.task_id,
        ok=False,
        final_text=_ABSTAIN_TEXT,
        steps=result.steps,
        failures=(result.failures or []) + [f"conscience:{decision.verdict}"],
        cost_usd=result.cost_usd,
        latency_sec=result.latency_sec,
        trace_path=result.trace_path,
    )


def maybe_gate_result(result: AgentResult, flag: bool | None) -> AgentResult:
    """The in-loop hook: gate only when enabled (flag or env). No-op otherwise."""
    return apply_to_result(result) if gate_enabled(flag) else result


def run_agent_with_conscience(
    task: AgentTask,
    *,
    client: Any = None,
    verifier: Any = None,
    conscience_gate: bool | None = None,
    **run_kwargs: Any,
) -> AgentResult:
    """Ergonomic wrapper: run the harness loop with the conscience gate armed.

    The gate now lives inside :func:`agent.harness.run_agent` (its
    ``conscience_gate`` param); this wrapper just forwards the flag and is kept for
    callers that want the intent-explicit name. Defaults to opt-in (off unless the
    flag or ``SOPHIA_CONSCIENCE_GATE`` says otherwise).
    """
    return run_agent(
        task, client=client, verifier=verifier, conscience_gate=conscience_gate, **run_kwargs
    )


__all__ = [
    "GATE_ENV",
    "gate_enabled",
    "apply_to_result",
    "maybe_gate_result",
    "run_agent_with_conscience",
]
