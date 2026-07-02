# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""GOAP planner — goal-oriented action planning with preconditions/effects, bounded
A* search, and replan-from-current-state.

The long-horizon engine (:mod:`agent.long_horizon`) executes a *static* dependency
tree: when a node fails, the engine retries with hints, but the tree's shape never
changes. This module adds the missing planning altitude: actions declare
*preconditions* and *effects* over a small typed state vocabulary, a bounded A*
search finds the cheapest action sequence from the current state to the goal, and —
the point — after a failure the caller replans from the *actual* current state
instead of retrying a dead branch. ``plan_to_subtasks`` lowers a plan to the exact
subtask dicts :func:`agent.long_horizon.build_ledger` accepts, so the two layers
compose without either knowing the other's internals.

Sophia discipline:

  * **Deterministic + offline + stdlib-only.** Ties in the A* frontier break on the
    lexicographic action path, so the same inputs always yield the same plan.
  * **Fail-closed.** An unreachable goal returns ``None`` (never a partial plan);
    an action whose preconditions are unmet is unreachable in search rather than
    failed at runtime. Resource claims (e.g. ``resource:spark-gpu:free``) belong in
    preconditions, so a plan that would violate a live claim is never generated.
  * **Bounded.** ``max_expansions`` caps search work; hitting the cap returns
    ``None`` (honest "don't know") rather than looping.
  * **Replanning is auditable data.** :func:`replan` returns the new plan plus a
    structured event (abandoned prefix, reason, state) the caller can append to its
    ledger — and the (abandoned, successful) pair is exactly the preference shape
    ``tools/build_trajectory_pack.py`` mines into DPO negatives/positives.

Honest bound: the heuristic (unsatisfied goal atoms) is admissible only while every
action has cost ≥ 1 and each atom is added by some single action — with cheaper or
compound actions A* remains correct but may expand more nodes. This module plans; it
never executes, verifies, or claims success — execution stays behind the delegation
layer and its verifiers.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

SCHEMA = "sophia.goap_plan.v1"

State = frozenset  # of str atoms, e.g. "artifact:report.md:exists", "gate:claims:passed"


@dataclass(frozen=True)
class Action:
    """One planable step: applicable when ``preconditions ⊆ state``; applying it
    yields ``(state - delete) | add``. ``cost`` ≥ 1 keeps the heuristic admissible."""

    name: str
    preconditions: frozenset = frozenset()
    add: frozenset = frozenset()
    delete: frozenset = frozenset()
    cost: float = 1.0
    # Optional lowering metadata for plan_to_subtasks (tool scope stays least-privilege).
    goal_text: str = ""
    allowed_tools: "frozenset[str] | None" = None
    max_steps: int = 3

    def applicable(self, state: State) -> bool:
        return self.preconditions <= state

    def apply(self, state: State) -> State:
        return frozenset((state - self.delete) | self.add)


@dataclass
class Plan:
    actions: "list[Action]"
    start: State
    goal: frozenset
    expansions: int

    @property
    def cost(self) -> float:
        return sum(a.cost for a in self.actions)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "actions": [a.name for a in self.actions],
            "cost": self.cost,
            "start": sorted(self.start),
            "goal": sorted(self.goal),
            "expansions": self.expansions,
        }


def _heuristic(state: State, goal: frozenset) -> int:
    return len(goal - state)


def plan(start: "frozenset | set", goal: "frozenset | set", actions: "list[Action]",
         *, max_expansions: int = 10_000) -> "Plan | None":
    """Bounded A* from ``start`` to ``goal ⊆ state``. Returns ``None`` when the goal
    is unreachable or the expansion budget is exhausted (fail-closed, never partial)."""
    start = frozenset(start)
    goal = frozenset(goal)
    if goal <= start:
        return Plan(actions=[], start=start, goal=goal, expansions=0)

    # Frontier entries: (f, path_names, g, state, path). path_names is the
    # deterministic tie-break — no wall clock, no insertion counter.
    frontier: "list[tuple[float, tuple[str, ...], float, State, list[Action]]]" = [
        (float(_heuristic(start, goal)), (), 0.0, start, [])
    ]
    best_g: "dict[State, float]" = {start: 0.0}
    expansions = 0
    while frontier and expansions < max_expansions:
        _f, _names, g, state, path = heapq.heappop(frontier)
        if g > best_g.get(state, float("inf")):
            continue  # stale entry
        expansions += 1
        if goal <= state:
            return Plan(actions=path, start=start, goal=goal, expansions=expansions)
        for action in actions:
            if not action.applicable(state):
                continue
            nxt = action.apply(state)
            ng = g + action.cost
            if ng < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng
                npath = path + [action]
                heapq.heappush(frontier, (
                    ng + _heuristic(nxt, goal),
                    tuple(a.name for a in npath),
                    ng, nxt, npath,
                ))
    return None


def replan(current: "frozenset | set", goal: "frozenset | set", actions: "list[Action]",
           *, failed_action: "str | None" = None, abandoned: "list[str] | None" = None,
           max_expansions: int = 10_000) -> dict:
    """Replan from the *actual* current state after a failure. Returns a structured,
    ledger-appendable event: the new plan (or a fail-closed hold) plus what was
    abandoned and why — the auditable branch point the trajectory miner consumes."""
    new_plan = plan(current, goal, actions, max_expansions=max_expansions)
    event = {
        "schema": "sophia.goap_replan.v1",
        "failedAction": failed_action,
        "abandonedPlan": list(abandoned or []),
        "currentState": sorted(frozenset(current)),
        "replanned": new_plan is not None,
    }
    if new_plan is None:
        event["verdict"] = "held"
        event["heldReason"] = "goal unreachable from current state (fail-closed; no partial plan)"
    else:
        event["plan"] = new_plan.to_dict()
    return event


def plan_to_subtasks(p: Plan) -> "list[dict]":
    """Lower a plan to ``agent.long_horizon.build_ledger`` subtask dicts. The chain
    is sequential (each node depends on the previous) because A* already ordered the
    actions; parallelizable structure belongs in the action model, not here."""
    subtasks: list[dict] = []
    prev: "str | None" = None
    for i, action in enumerate(p.actions, 1):
        node_id = f"g{i}-{action.name}"
        subtasks.append({
            "id": node_id,
            "goal": action.goal_text or action.name,
            "deps": [prev] if prev else [],
            "allowed_tools": (sorted(action.allowed_tools)
                              if action.allowed_tools is not None else None),
            "max_steps": action.max_steps,
        })
        prev = node_id
    return subtasks


# --------------------------------------------------------------------------- #
# Offline invariants (CI-gated; deterministic, stdlib-only)
# --------------------------------------------------------------------------- #

def _demo_actions() -> "list[Action]":
    return [
        Action("gather_sources", preconditions=frozenset(),
               add=frozenset({"sources:collected"}), goal_text="Gather and cite sources"),
        Action("draft", preconditions=frozenset({"sources:collected"}),
               add=frozenset({"draft:exists"}), goal_text="Draft the report"),
        Action("verify_claims", preconditions=frozenset({"draft:exists"}),
               add=frozenset({"gate:claims:passed"}), goal_text="Run the claims gate"),
        Action("publish", preconditions=frozenset({"draft:exists", "gate:claims:passed",
                                                   "resource:ci:free"}),
               add=frozenset({"report:published"}), goal_text="Publish the report"),
        # A tempting shortcut that skips the gate — it still needs the CI resource
        # (every publish does), deletes the gate atom, and costs more, so it can
        # never appear in a cheapest plan whose goal includes the gate atom.
        Action("publish_unverified", preconditions=frozenset({"draft:exists",
                                                              "resource:ci:free"}),
               add=frozenset({"report:published"}), delete=frozenset({"gate:claims:passed"}),
               cost=5.0, goal_text="Publish without verification"),
    ]


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    actions = _demo_actions()
    goal = frozenset({"report:published", "gate:claims:passed"})

    # 1) Plans through the gate, not around it (the shortcut deletes the gate atom).
    p = plan(frozenset({"resource:ci:free"}), goal, actions)
    checks["plans_through_gate"] = p is not None and [a.name for a in p.actions] == [
        "gather_sources", "draft", "verify_claims", "publish"]

    # 2) Resource precondition is binding: without the claim, the goal is unreachable.
    checks["missing_resource_unreachable"] = plan(frozenset(), goal, actions) is None

    # 3) Already-satisfied goal → empty plan (idempotent).
    p0 = plan(goal | frozenset({"resource:ci:free"}), goal, actions)
    checks["satisfied_goal_empty_plan"] = p0 is not None and p0.actions == []

    # 4) Replan from mid-state skips completed work.
    mid = frozenset({"sources:collected", "draft:exists", "resource:ci:free"})
    ev = replan(mid, goal, actions, failed_action="verify_claims",
                abandoned=["verify_claims", "publish"])
    checks["replan_skips_done_work"] = ev["replanned"] and \
        ev["plan"]["actions"] == ["verify_claims", "publish"]

    # 5) Unreachable replan is held, never partial.
    ev2 = replan(frozenset({"draft:exists"}), goal, actions)
    checks["unreachable_replan_held"] = (not ev2["replanned"]) and ev2["verdict"] == "held"

    # 6) Expansion budget is honored (fail-closed on exhaustion).
    checks["budget_fail_closed"] = plan(frozenset({"resource:ci:free"}), goal, actions,
                                        max_expansions=1) is None

    # 7) Determinism: identical inputs → identical plan dict.
    q = plan(frozenset({"resource:ci:free"}), goal, actions)
    checks["deterministic"] = p is not None and q is not None and p.to_dict() == q.to_dict()

    # 8) Lowering preserves ordering + least privilege shape for build_ledger.
    subs = plan_to_subtasks(p)  # type: ignore[arg-type]
    checks["lowering_well_formed"] = (
        len(subs) == 4 and subs[0]["deps"] == [] and
        all(subs[i]["deps"] == [subs[i - 1]["id"]] for i in range(1, 4)) and
        all(set(s) >= {"id", "goal", "deps", "allowed_tools", "max_steps"} for s in subs)
    )

    ok = all(checks.values())
    return ok, {"checks": checks}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    for name, passed in detail["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"{'PASS' if ok else 'FAIL'} goap_planner offline_invariants")
    raise SystemExit(0 if ok else 1)
