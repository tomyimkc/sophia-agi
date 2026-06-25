# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-horizon execution engine — a durable task tree over the delegation layer,
with a recovery memory that feeds recurring-failure hints back into retries.

``agent/horizon.py`` *measures* the effective-horizon capability curve (METR-style,
oracle-judged). This module is the *engine* that survives long, dependent task
chains — the property that makes long horizons hard is that one slip fails the whole
task, so the engine must (a) persist progress so a crash/resume does not redo work,
and (b) learn from a failure within the run so it does not slip the same way twice.

Design (Sophia discipline — deterministic, offline-testable, fail-closed):

  * **Durable ledger** — the task tree (:class:`TaskLedger` of :class:`SubtaskNode`)
    is persisted to JSON after every node transition, so a resumed run skips
    already-``done`` nodes and re-attempts only what is ``pending``/``failed``. This
    extends the harness's per-task checkpoint to a *multi-task* tree.
  * **Dependency ordering** — a node runs only once its ``deps`` are all ``done``;
    a node whose dependency failed is left ``blocked`` (fail-closed: we never run a
    step on top of an unmet prerequisite).
  * **Recovery memory** — on failure the engine records a hint keyed by the node's
    *failure signature*; before re-attempting a similar node it recalls the hint and
    injects it into the child's context. Dependency-free (no numpy) so it stays
    offline-deterministic; the richer provenance-specific store in
    ``agent/failure_memory.py`` is complementary, not required here.
  * **Execution via delegation** — each node runs through
    ``agent.subagent.run_subagent``, inheriting isolation, least-privilege tool
    scope, and per-node budgets.

Honest bound: recovery is *within-run* hint injection, not weight learning; success
of the whole tree is still meant to be judged by an external oracle (see
``horizon.py``), never by the engine itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import ROOT
from agent.model import ModelClient, default_client
from agent.subagent import SubagentSpec, run_subagent

LEDGERS_DIR = ROOT / "agent" / "memory" / "task_ledgers"
RECOVERY_PATH = ROOT / "agent" / "memory" / "recovery_memory.jsonl"

# Node lifecycle.
PENDING, RUNNING, DONE, FAILED, BLOCKED = "pending", "running", "done", "failed", "blocked"


def _signature(goal: str) -> str:
    """A coarse, deterministic signature of a subtask goal for recovery lookup:
    lowercased significant tokens, sorted and joined. Coarse on purpose — it should
    match *similar* tasks (e.g. retried phrasings), not only byte-identical ones."""
    tokens = sorted(set(re.findall(r"[a-z0-9]+", goal.lower())) - _STOPWORDS)
    return " ".join(tokens[:12])


_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on", "step", "then"}


# --------------------------------------------------------------------------- #
# Recovery memory (dependency-free, append-only, deterministic)
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryMemory:
    """Append-only store of (signature, failure_class) -> hint. Latest write wins
    on recall. Never raises on a malformed line — a corrupt entry is skipped."""

    path: Path = RECOVERY_PATH

    def record(self, *, signature: str, failure_class: str, hint: str) -> None:
        if not (signature and hint):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"signature": signature, "failureClass": failure_class, "hint": hint.strip()[:600]}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def recall(self, *, signature: str, failure_class: str | None = None) -> str | None:
        if not signature or not self.path.exists():
            return None
        best: str | None = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("signature") != signature:
                continue
            if failure_class is not None and row.get("failureClass") not in (None, failure_class):
                continue
            best = row.get("hint") or best  # latest matching hint wins
        return best


# --------------------------------------------------------------------------- #
# Task tree
# --------------------------------------------------------------------------- #


@dataclass
class SubtaskNode:
    id: str
    goal: str
    deps: list[str] = field(default_factory=list)
    mode: str = "advisor"
    allowed_tools: "set[str] | None" = None
    max_steps: int = 3
    max_retries: int = 1
    cost_budget_usd: "float | None" = None
    status: str = PENDING
    attempts: int = 0
    result_text: str = ""
    failure: str | None = None
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["allowed_tools"] = sorted(self.allowed_tools) if self.allowed_tools is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubtaskNode":
        d = dict(d)
        if d.get("allowed_tools") is not None:
            d["allowed_tools"] = set(d["allowed_tools"])
        return cls(**d)


@dataclass
class TaskLedger:
    goal: str
    ledger_id: str
    nodes: list[SubtaskNode] = field(default_factory=list)
    ledgers_dir: Path = LEDGERS_DIR

    @property
    def path(self) -> Path:
        return self.ledgers_dir / f"{self.ledger_id}.json"

    def by_id(self, node_id: str) -> SubtaskNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"goal": self.goal, "ledgerId": self.ledger_id, "nodes": [n.to_dict() for n in self.nodes]}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, ledger_id: str, *, ledgers_dir: Path = LEDGERS_DIR) -> "TaskLedger | None":
        path = ledgers_dir / f"{ledger_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            goal=data["goal"],
            ledger_id=data["ledgerId"],
            nodes=[SubtaskNode.from_dict(n) for n in data.get("nodes", [])],
            ledgers_dir=ledgers_dir,
        )

    def _deps_satisfied(self, node: SubtaskNode) -> bool:
        return all((self.by_id(dep) or SubtaskNode(id=dep, goal="")).status == DONE for dep in node.deps)

    def _dep_failed(self, node: SubtaskNode) -> bool:
        return any((self.by_id(dep) or SubtaskNode(id=dep, goal="")).status in (FAILED, BLOCKED) for dep in node.deps)

    def next_runnable(self) -> SubtaskNode | None:
        """First pending node whose deps are all done. Marks dependency-blocked
        nodes as BLOCKED (fail-closed) so they are never executed."""
        for node in self.nodes:
            if node.status != PENDING:
                continue
            if self._dep_failed(node):
                node.status = BLOCKED
                node.failure = "dependency_failed"
                continue
            if self._deps_satisfied(node):
                return node
        return None


@dataclass
class LongHorizonResult:
    ledger_id: str
    ok: bool
    completed: list[str]
    failed: list[str]
    blocked: list[str]
    total_cost_usd: float
    ledger_path: str

    def to_dict(self) -> dict:
        return {
            "ledgerId": self.ledger_id,
            "ok": self.ok,
            "completed": self.completed,
            "failed": self.failed,
            "blocked": self.blocked,
            "totalCostUsd": round(self.total_cost_usd, 6),
        }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def build_ledger(goal: str, subtasks: list[dict], *, ledger_id: str, ledgers_dir: Path = LEDGERS_DIR) -> TaskLedger:
    """Build a fresh ledger from a list of subtask dicts (``{id, goal, deps?, ...}``)."""
    nodes = []
    for i, st in enumerate(subtasks, 1):
        node_id = str(st.get("id") or f"n{i}")
        nodes.append(
            SubtaskNode(
                id=node_id,
                goal=str(st["goal"]),
                deps=[str(d) for d in st.get("deps", [])],
                mode=st.get("mode", "advisor"),
                allowed_tools=set(st["allowed_tools"]) if st.get("allowed_tools") is not None else None,
                max_steps=int(st.get("max_steps", 3)),
                max_retries=int(st.get("max_retries", 1)),
                cost_budget_usd=st.get("cost_budget_usd"),
            )
        )
    return TaskLedger(goal=goal, ledger_id=ledger_id, nodes=nodes, ledgers_dir=ledgers_dir)


def run_long_horizon(
    ledger: TaskLedger,
    *,
    client: ModelClient | None = None,
    recovery: RecoveryMemory | None = None,
    approve_tools: bool = False,
    max_nodes: int = 256,
) -> LongHorizonResult:
    """Execute a task tree to completion, persisting after every node transition.

    Each node runs through the delegation layer (isolation + tool scope + budget).
    On failure, a recovery hint is recorded; before each attempt the engine recalls
    a hint for the node's signature and injects it into the child's context. The
    ledger is durable: re-running with a resumed ledger skips ``done`` nodes.
    """
    client = client or default_client()
    recovery = recovery if recovery is not None else RecoveryMemory()
    total_cost = 0.0
    steps = 0
    while steps < max_nodes:
        node = ledger.next_runnable()
        if node is None:
            break
        steps += 1
        node.status = RUNNING
        node.attempts += 1
        ledger.save()

        sig = _signature(node.goal)
        hint = recovery.recall(signature=sig)
        context = node_context(ledger, node, hint)
        spec = SubagentSpec(
            goal=node.goal,
            mode=node.mode,
            allowed_tools=node.allowed_tools,
            max_steps=node.max_steps,
            max_retries=node.max_retries,
            cost_budget_usd=node.cost_budget_usd,
            context=context,
            label=node.id,
        )
        child = run_subagent(spec, parent_id=ledger.ledger_id, index=steps, client=client, approve_tools=approve_tools)
        node.cost_usd += child.cost_usd
        total_cost += child.cost_usd

        if child.ok:
            node.status = DONE
            node.result_text = child.final_text
            node.failure = None
        else:
            node.status = FAILED
            node.failure = ";".join(child.failures) or "unknown"
            # Record a recovery hint so a later similar node avoids this failure mode.
            recovery.record(
                signature=sig,
                failure_class=node.failure.split(":")[0].split(";")[0],
                hint=f"A prior attempt at '{node.goal}' failed ({node.failure}). Address that explicitly.",
            )
        ledger.save()

    completed = [n.id for n in ledger.nodes if n.status == DONE]
    failed = [n.id for n in ledger.nodes if n.status == FAILED]
    blocked = [n.id for n in ledger.nodes if n.status == BLOCKED]
    return LongHorizonResult(
        ledger_id=ledger.ledger_id,
        ok=not failed and not blocked and len(completed) == len(ledger.nodes),
        completed=completed,
        failed=failed,
        blocked=blocked,
        total_cost_usd=total_cost,
        ledger_path=str(ledger.path),
    )


def node_context(ledger: TaskLedger, node: SubtaskNode, hint: str | None) -> str:
    """Assemble the context handed to a node: the dependency outputs it builds on,
    plus any recalled recovery hint. Kept small; the child's own context manager
    will budget it further."""
    parts = [f"Overall goal: {ledger.goal}"]
    for dep in node.deps:
        dn = ledger.by_id(dep)
        if dn and dn.result_text:
            parts.append(f"Result of prerequisite {dep}:\n{dn.result_text}")
    if hint:
        parts.append(f"Recovery hint from a prior similar failure — heed it:\n{hint}")
    return "\n\n".join(parts)
