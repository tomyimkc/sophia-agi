# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Agent harness for Sophia AGI: plan -> execute -> critic -> reflect/retry.

Built on the unified model adapter (agent/model.py). Reuses existing pieces as
components rather than rebuilding them:
  - critic     : agent.gate.check_response (epistemic gate) by default; any
                 verifier callable can be plugged in (unit tests, score_pack, etc.).
  - executor   : agent.tools.run_tool (approval-gated repo tools) + registered
                 python callables.
  - context    : agent.retrieval (optional), skill workflows.

Provides: model-driven planner, tool calling, a reflection/retry loop, append-only
decision logs, task-state persistence with checkpoint/resume, and failure
classification. Offline-testable via the model adapter's mock provider.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.config import ROOT
from agent.gate import check_response
from agent.model import ModelClient, ModelResult, default_client
from agent.prompts import MODE_PROMPTS
from agent.tools import TOOL_CATALOG, parse_tool_requests, run_tools
from agent.untrusted import wrap_untrusted

RUNS_DIR = ROOT / "agent" / "memory" / "agent_runs"

FAILURE_CLASSES = (
    "empty_output",
    "model_error",
    "tool_error",
    "gate_violation",
    "verifier_fail",
    "exception",
    "max_retries_exhausted",
    "unknown",
)

# A verifier takes (text, task, step) and returns {"passed", "reasons", "detail"}.
Verifier = Callable[[str, "AgentTask", dict], dict]


@dataclass
class AgentTask:
    goal: str
    mode: str = "advisor"  # advisor | repo | life
    task_id: str = ""
    context: str = ""
    skill: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            slug = re.sub(r"[^a-z0-9]+", "-", self.goal.lower())[:40].strip("-") or "task"
            self.task_id = slug


@dataclass
class StepResult:
    step_id: str
    description: str
    action: str
    output: str = ""
    ok: bool = False
    attempts: int = 0
    failure_class: str | None = None
    reasons: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_sec: float = 0.0


@dataclass
class AgentResult:
    task_id: str
    ok: bool
    final_text: str
    steps: list[StepResult]
    failures: list[str]
    cost_usd: float
    latency_sec: float
    trace_path: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Append-only decision log + checkpoint state for one agent run."""

    def __init__(self, task_id: str, *, runs_dir: Path = RUNS_DIR):
        self.task_id = task_id
        self.log_path = runs_dir / f"{task_id}.jsonl"
        self.state_path = runs_dir / f"{task_id}.state.json"
        self.events: list[dict] = []
        self.state: dict[str, Any] = {"taskId": task_id, "completedSteps": [], "final": None}

    def fresh(self) -> "RunStore":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._save_state()
        return self

    def resume(self) -> "RunStore":
        if self.log_path.exists():
            self.events = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        return self

    def log(self, event_type: str, **fields: Any) -> None:
        record = {"ts": _now(), "type": event_type, **fields}
        self.events.append(record)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def mark_done(self, step_id: str, output: str) -> None:
        if step_id not in self.state["completedSteps"]:
            self.state["completedSteps"].append(step_id)
        self.state.setdefault("stepOutputs", {})[step_id] = output
        self._save_state()

    def set_final(self, text: str) -> None:
        self.state["final"] = text
        self._save_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def classify_failure(*, result: ModelResult | None, gate: dict | None, verifier: dict | None, tool_results: list[dict] | None) -> str:
    if result is not None and not result.ok:
        return "model_error"
    if result is not None and result.ok and not result.text.strip():
        return "empty_output"
    if tool_results and any(not t.get("ok") for t in tool_results):
        return "tool_error"
    if gate is not None and not gate.get("passed", True):
        return "gate_violation"
    if verifier is not None and not verifier.get("passed", True):
        return "verifier_fail"
    return "unknown"  # cause not identified — do NOT credit it to the verifier (taints ablations)


def gate_verifier(text: str, task: "AgentTask", step: dict) -> dict:
    gate = check_response(text, mode=task.mode if task.mode in {"advisor", "repo", "life"} else "advisor", question=task.goal)
    return {
        "passed": bool(gate.get("passed")),
        "reasons": list(gate.get("warnings", [])) + list(gate.get("violations", [])),
        "detail": {"gate": gate},
    }


def skill_keyword_verifier(must_include: list[str]) -> Verifier:
    """Verifier that the output mentions each required token (skill output check)."""

    def _verify(text: str, task: "AgentTask", step: dict) -> dict:
        lowered = text.lower()
        missing = [kw for kw in must_include if kw.lower() not in lowered]
        return {"passed": not missing, "reasons": [f"missing: {kw}" for kw in missing], "detail": {"missing": missing}}

    return _verify


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def _memory_recall(goal: str, *, max_pages: int = 3, search_fn=None, graph=None) -> str:
    """Recall relevant OKF wiki pages at plan time (prior knowledge + provenance
    warnings), annotated with the belief graph's EFFECTIVE confidence so a page
    that launders weak provenance into a confident claim is shown as capped, not
    at face value. Never raises — recall failure must not break planning.

    ``search_fn`` / ``graph`` are injectable for testing; by default they wire
    wiki_store search and a graph built over the whole corpus.
    """
    try:
        if search_fn is None:
            from agent import wiki_store

            search_fn = wiki_store.search
        pages = search_fn(goal, top_k=max_pages)
        if not pages:
            return ""
        if graph is None:
            import okf
            from agent import wiki_store

            graph = okf.build_graph(wiki_store.load_all_pages())
        lines = [_recall_line(page, graph) for page in pages]
        return "## Memory (prior knowledge — build on it, respect the warnings)\n" + "\n".join(lines)
    except Exception:
        return ""


def _recall_line(page, graph) -> str:
    """One memory bullet: page id + effective (graph) confidence + provenance warnings."""
    import okf

    dna = page.meta.get("doNotAttributeTo") or []
    dnm = page.meta.get("doNotMergeWith") or []
    conf = page.meta.get("authorConfidence")
    belief = okf.belief(graph, page.id) if graph is not None else {}
    note = ""
    if conf:
        note += f" (confidence={conf}"
        if belief.get("found"):
            note += f", effectiveRank={belief['effectiveConfidenceRank']}/{belief['confidenceRank']}"
        note += ")"
    if belief.get("confidenceLaundered"):
        note += " ⚠ confidence capped by weak provenance."
    if dna:
        note += f" ⚠ do-not-attribute: {', '.join(dna)}."
    if dnm:
        note += f" ⚠ do-not-merge: {', '.join(dnm)}."
    return f"- [[{page.id}]]{note}"


def plan(task: AgentTask, client: ModelClient, *, max_steps: int = 4) -> list[dict]:
    """Ask the model for a compact JSON plan; fall back to a single answer step."""
    skill_block = ""
    if task.skill:
        skill_block = (
            f"\nUse this skill workflow:\n- when: {task.skill.get('whenToUse', '')}\n"
            + "\n".join(f"- step: {s}" for s in task.skill.get("workflow", []))
        )
    memory_block = _memory_recall(task.goal)
    system = "You are a planner. Output ONLY a JSON array of steps."
    parts = [f"Goal: {task.goal}\nMode: {task.mode}\n{task.context}{skill_block}"]
    if memory_block:
        parts.append(memory_block)
    parts.append(
        f"Available repo tools: {', '.join(TOOL_CATALOG)}.\n"
        f"Return up to {max_steps} steps as JSON: "
        '[{"id":"s1","description":"...","action":"model|tool","tool":"<tool name or empty>"}].'
    )
    user = "\n\n".join(parts)
    result = client.generate(system, user)
    steps = _parse_plan(result.text, max_steps=max_steps)
    if not steps:
        steps = [{"id": "s1", "description": task.goal, "action": "model", "tool": ""}]
    return steps


def _parse_plan(text: str, *, max_steps: int) -> list[dict]:
    candidate = text
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    steps: list[dict] = []
    for index, item in enumerate(data[:max_steps], 1):
        if not isinstance(item, dict):
            continue
        steps.append(
            {
                "id": str(item.get("id") or f"s{index}"),
                "description": str(item.get("description") or item.get("desc") or "step"),
                "action": "tool" if str(item.get("action")) == "tool" and item.get("tool") in TOOL_CATALOG else "model",
                "tool": item.get("tool") if item.get("tool") in TOOL_CATALOG else "",
            }
        )
    return steps


# --------------------------------------------------------------------------- #
# Executor + reflection/retry loop
# --------------------------------------------------------------------------- #


def _build_step_prompt(task: AgentTask, step: dict, prior: str, reflection: str | None) -> tuple[str, str]:
    system = MODE_PROMPTS.get(task.mode, MODE_PROMPTS["advisor"])
    parts = [f"## Goal\n{task.goal}", f"## Current step\n{step['description']}"]
    if task.context:
        # external context is untrusted -> fence it against prompt injection
        parts.append("## Context\n" + wrap_untrusted(task.context, "task-context"))
    if task.skill:
        parts.append("## Skill verification\n" + "\n".join(f"- {v}" for v in task.skill.get("verification", [])))
    if prior:
        parts.append(f"## Prior step outputs\n{prior}")
    if reflection:
        parts.append(f"## Reflection on the previous failed attempt — fix these\n{reflection}")
    parts.append("End with a Decision section and a short 中文摘要.")
    return system, "\n\n".join(parts)


def _execute_step(
    task: AgentTask,
    step: dict,
    *,
    client: ModelClient,
    store: RunStore,
    verifier: Verifier,
    prior: str,
    max_retries: int,
    approve_tools: bool,
) -> StepResult:
    result = StepResult(step_id=step["id"], description=step["description"], action=step["action"])
    reflection: str | None = None
    for attempt in range(1, max_retries + 2):
        result.attempts = attempt
        store.log("step_attempt", step=step["id"], attempt=attempt, action=step["action"])
        system, user = _build_step_prompt(task, step, prior, reflection)
        model_result = client.generate(system, user)
        store.log("model_call", step=step["id"], **model_result.to_log())
        result.cost_usd += model_result.cost_usd
        result.latency_sec += model_result.latency_sec
        text = model_result.text

        tool_results: list[dict] = []
        if step["action"] == "tool" or "{\"tools\"" in text or "```json" in text:
            requested = parse_tool_requests(text)
            if requested:
                tool_results = run_tools(requested, approved=approve_tools)
                store.log("tool_call", step=step["id"], requested=requested, results=tool_results)
        result.tool_results = tool_results

        gate_check = gate_verifier(text, task, step)
        verify = verifier(text, task, step) if verifier is not gate_verifier else gate_check
        store.log("critic", step=step["id"], gatePassed=gate_check["passed"], verifierPassed=verify["passed"], reasons=verify.get("reasons", []))

        passed = model_result.ok and bool(text.strip()) and verify.get("passed", False) and all(t.get("ok") for t in tool_results)
        result.output = text
        fclass = None if passed else classify_failure(result=model_result, gate=gate_check, verifier=verify, tool_results=tool_results)
        # full output is logged (gitignored run dir) so collect_traces can build SFT/DPO data
        store.log("step_output", step=step["id"], attempt=attempt, passed=passed, failureClass=fclass, output=text)
        if passed:
            result.ok = True
            result.reasons = []
            result.failure_class = None
            store.log("step_done", step=step["id"], attempts=attempt)
            return result

        result.failure_class = fclass
        result.reasons = verify.get("reasons", []) or ([model_result.error] if model_result.error else [])
        if attempt <= max_retries:
            reflection = _reflect(client, task, text, result.reasons)
            store.log("reflect", step=step["id"], failureClass=result.failure_class, reflection=reflection)
        else:
            result.failure_class = "max_retries_exhausted"
            store.log("step_failed", step=step["id"], failureClass=result.failure_class, reasons=result.reasons)
    return result


def _reflect(client: ModelClient, task: AgentTask, last_output: str, reasons: list[str]) -> str:
    system = "You are a critic. In 2-4 bullet points, say exactly how to fix the answer. No rewrite."
    user = f"Goal: {task.goal}\nReasons it failed: {reasons}\nLast answer:\n{last_output[:1500]}"
    result = client.generate(system, user)
    return result.text.strip() if result.ok else f"Address: {', '.join(reasons)}"


def run_agent(
    task: AgentTask,
    *,
    client: ModelClient | None = None,
    verifier: Verifier | None = None,
    max_retries: int = 2,
    max_steps: int = 4,
    approve_tools: bool = False,
    resume: bool = False,
    consolidate: bool = False,
) -> AgentResult:
    """Run the full plan -> execute -> critic -> reflect/retry loop.

    When ``consolidate=True``, a verified run folds its conclusion into a gated OKF
    memory page so future runs can recall it (continual learning, off by default).
    """
    client = client or default_client()
    verifier = verifier or gate_verifier
    store = RunStore(task.task_id)
    store.resume() if resume else store.fresh()
    store.log("task_start", goal=task.goal, mode=task.mode, skill=(task.skill or {}).get("name"), resumed=resume)

    steps = plan(task, client, max_steps=max_steps)
    store.log("plan", steps=[s["id"] for s in steps], detail=steps)

    completed = set(store.state.get("completedSteps", [])) if resume else set()
    step_results: list[StepResult] = []
    prior_outputs: list[str] = []
    total_cost = 0.0
    total_latency = 0.0
    for step in steps:
        if step["id"] in completed:
            store.log("step_skip", step=step["id"], reason="already completed (resume)")
            prior_outputs.append(store.state.get("stepOutputs", {}).get(step["id"], ""))
            continue
        outcome = _execute_step(
            task, step, client=client, store=store, verifier=verifier,
            prior="\n\n".join(prior_outputs)[-4000:], max_retries=max_retries, approve_tools=approve_tools,
        )
        step_results.append(outcome)
        total_cost += outcome.cost_usd
        total_latency += outcome.latency_sec
        if outcome.ok:
            store.mark_done(step["id"], outcome.output)
            prior_outputs.append(outcome.output)
        else:
            break  # stop at first unrecoverable step

    final_text = prior_outputs[-1] if prior_outputs else ""
    failures = [f"{s.step_id}:{s.failure_class}" for s in step_results if not s.ok]
    # ok = no failures and every executed step passed (a fully-resumed task with
    # all steps already complete has no new step_results and is also ok).
    ok = not failures and all(s.ok for s in step_results)
    store.set_final(final_text)
    if consolidate and ok and final_text.strip():
        try:
            from agent.memory_consolidation import consolidate_result

            cons = consolidate_result(task.goal, final_text, task_id=task.task_id, mode=task.mode)
            store.log("consolidate", ok=cons.get("ok"), id=cons.get("id"), reasons=cons.get("reasons"))
        except Exception as exc:  # consolidation must never fail the run
            store.log("consolidate", ok=False, error=repr(exc))
    store.log("task_end", ok=ok, failures=failures, costUsd=round(total_cost, 6), latencySec=round(total_latency, 3))
    return AgentResult(
        task_id=task.task_id,
        ok=ok,
        final_text=final_text,
        steps=step_results,
        failures=failures,
        cost_usd=total_cost,
        latency_sec=total_latency,
        trace_path=str(store.log_path),
    )
