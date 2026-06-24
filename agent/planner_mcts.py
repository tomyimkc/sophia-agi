# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""MCTS-style deliberate planning over verification tool calls.

The planner searches over *verification strategies*, not final claims. A plan wins
only if it reaches an accepted/rejected state with enough independent evidence;
unsupported majority/judge shortcuts receive low reward. This is an offline,
deterministic harness for the long-horizon planning component.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from agent.fact_check_gate import AtomicClaim, classify_claim, risk_for


@dataclass(frozen=True)
class PlannerState:
    claim: str
    claim_type: str
    risk: str
    entailing_sources: int = 0
    contradicting_sources: int = 0
    deterministic_done: bool = False
    judged: bool = False
    used_actions: tuple[str, ...] = ()
    cost: float = 0.0
    terminal: str | None = None  # accepted|rejected|held

    def key(self) -> tuple[Any, ...]:
        return (self.claim_type, self.risk, self.entailing_sources, self.contradicting_sources, self.deterministic_done, self.judged, self.used_actions, self.terminal)


@dataclass(frozen=True)
class Action:
    name: str
    cost: float
    source_gain: int = 0
    contradiction_gain: int = 0
    deterministic: str | None = None
    judge: bool = False


DEFAULT_ACTIONS = (
    Action("deterministic_type_check", cost=0.2, deterministic="maybe"),
    Action("wikidata_or_authority", cost=0.5, source_gain=1),
    Action("crossref_openalex", cost=0.6, source_gain=1),
    Action("macro_stat_source", cost=0.7, source_gain=1),
    Action("independent_web_source", cost=0.8, source_gain=1),
    Action("adversarial_contradiction_search", cost=0.9, contradiction_gain=1),
    Action("competent_judge_with_evidence", cost=1.0, judge=True),
    Action("abstain", cost=0.1, deterministic="hold"),
)


@dataclass
class Node:
    state: PlannerState
    parent: "Node | None" = None
    action: Action | None = None
    visits: int = 0
    value: float = 0.0
    children: dict[str, "Node"] = field(default_factory=dict)

    def uct(self, child: "Node", c: float = 1.4) -> float:
        if child.visits == 0:
            return float("inf")
        return child.value / child.visits + c * math.sqrt(math.log(max(1, self.visits)) / child.visits)


class VerificationSimulator:
    """Small deterministic transition model for planning.

    ``profiles`` can override action outcomes per claim substring for tests/live
    adapters. Outcomes: ``entails``, ``contradicts``, ``none``, ``accept``,
    ``reject``, ``hold``.
    """

    def __init__(self, profiles: dict[str, dict[str, str]] | None = None):
        self.profiles = profiles or {}

    def required_sources(self, state: PlannerState) -> int:
        return 3 if state.risk == "high" else 2

    def actions(self, state: PlannerState) -> list[Action]:
        if state.terminal:
            return []
        acts = list(DEFAULT_ACTIONS)
        used = set(state.used_actions)
        # Deterministic probes and each source-family call are single-use; repeated
        # calls to the same family would be source laundering, not independence.
        acts = [a for a in acts if a.name not in used]
        if state.deterministic_done:
            acts = [a for a in acts if a.name != "deterministic_type_check"]
        if state.claim_type in {"math", "doi", "url", "date_temporal", "code_python", "subjective"}:
            acts = [a for a in acts if a.name in {"deterministic_type_check", "abstain"}]
        if not state.claim_type.startswith("econ"):
            acts = [a for a in acts if a.name != "macro_stat_source"]
        return acts

    def outcome(self, state: PlannerState, action: Action) -> str:
        for needle, mapping in self.profiles.items():
            if needle.lower() in state.claim.lower():
                return mapping.get(action.name, "none")
        if action.name == "deterministic_type_check":
            if state.claim_type in {"math", "subjective", "code_python"}:
                return "accept"
            return "none"
        if action.name == "adversarial_contradiction_search":
            return "none"
        if action.name == "abstain":
            return "hold"
        return "entails"

    def step(self, state: PlannerState, action: Action) -> PlannerState:
        out = self.outcome(state, action)
        ent = state.entailing_sources
        con = state.contradicting_sources
        det = state.deterministic_done or action.name == "deterministic_type_check"
        judged = state.judged or action.judge
        terminal = state.terminal
        if out == "accept":
            terminal = "accepted"
        elif out == "reject" or out == "contradicts":
            con += 1
            terminal = "rejected"
        elif out == "hold":
            terminal = "held"
        elif out == "entails":
            ent += action.source_gain or 1
        if terminal is None and con > 0:
            terminal = "rejected"
        if terminal is None and ent >= self.required_sources(state):
            terminal = "accepted"
        if terminal is None and judged and ent >= max(1, self.required_sources(state) - 1):
            # Judge cannot create support from nothing, but can resolve with evidence.
            terminal = "accepted"
        return PlannerState(
            claim=state.claim,
            claim_type=state.claim_type,
            risk=state.risk,
            entailing_sources=ent,
            contradicting_sources=con,
            deterministic_done=det,
            judged=judged,
            used_actions=state.used_actions + (action.name,),
            cost=round(state.cost + action.cost, 4),
            terminal=terminal,
        )

    def reward(self, state: PlannerState) -> float:
        if state.terminal == "accepted":
            return 10.0 - state.cost
        if state.terminal == "rejected":
            return 8.0 - state.cost
        if state.terminal == "held":
            return 2.5 - state.cost
        # non-terminal rollouts prefer more evidence, lower cost.
        return state.entailing_sources * 1.2 - state.contradicting_sources * 2.0 - state.cost


def initial_state(claim: str) -> PlannerState:
    ctype = classify_claim(claim)
    return PlannerState(claim=claim, claim_type=ctype, risk=risk_for(claim))


def _select(node: Node) -> Node:
    while node.children and not node.state.terminal:
        node = max(node.children.values(), key=lambda ch: node.uct(ch))
    return node


def _expand(node: Node, sim: VerificationSimulator) -> Node:
    if node.state.terminal:
        return node
    tried = set(node.children)
    for action in sim.actions(node.state):
        if action.name not in tried:
            child = Node(state=sim.step(node.state, action), parent=node, action=action)
            node.children[action.name] = child
            return child
    return node


def _rollout(state: PlannerState, sim: VerificationSimulator, rng: random.Random, depth: int) -> float:
    current = state
    for _ in range(depth):
        if current.terminal:
            break
        acts = sim.actions(current)
        if not acts:
            break
        # Bias rollouts toward source-gathering before judges/abstention.
        weights = [3.0 if a.source_gain else 1.0 for a in acts]
        action = rng.choices(acts, weights=weights, k=1)[0]
        current = sim.step(current, action)
    return sim.reward(current)


def run_mcts(claim: str, *, simulator: VerificationSimulator | None = None, iterations: int = 160, rollout_depth: int = 5, seed: int = 0) -> dict[str, Any]:
    sim = simulator or VerificationSimulator()
    root = Node(initial_state(claim))
    rng = random.Random(seed)
    for _ in range(iterations):
        leaf = _select(root)
        child = _expand(leaf, sim)
        reward = _rollout(child.state, sim, rng, rollout_depth)
        cur: Node | None = child
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent
    plan = []
    node = root
    while node.children and not node.state.terminal and len(plan) < rollout_depth:
        node = max(node.children.values(), key=lambda ch: (ch.visits, ch.value / max(ch.visits, 1)))
        plan.append(node.action.name if node.action else "?")
    return {
        "schema": "sophia.verification_mcts_plan.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claim": claim,
        "initialType": root.state.claim_type,
        "risk": root.state.risk,
        "plan": plan,
        "predictedTerminal": node.state.terminal or "nonterminal",
        "predictedEvidence": {"entailingSources": node.state.entailing_sources, "contradictingSources": node.state.contradicting_sources, "cost": node.state.cost},
        "rootVisits": root.visits,
        "decisionRule": "execute chosen tool plan, then pass every resulting claim through fact_check_gate; unsupported votes cannot publish",
    }


__all__ = ["PlannerState", "Action", "VerificationSimulator", "initial_state", "run_mcts"]
