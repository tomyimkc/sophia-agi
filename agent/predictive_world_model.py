# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A tiny learned predictive world model for Sophia's agent loop.

This is the missing *model of consequences* layer: learn P(next_state, reward | state,
action) from traces, expose uncertainty/OOD, and plan only when predictions are
supported. It is intentionally discrete and auditable, not a neural world model.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Transition:
    state: str
    action: str
    next_state: str
    reward: float = 0.0


class PredictiveWorldModel:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.rewards: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    def observe(self, state: str, action: str, next_state: str, reward: float = 0.0) -> None:
        key = (str(state), str(action))
        ns = str(next_state)
        self.counts[key][ns] += 1
        self.rewards[(key[0], key[1], ns)].append(float(reward))

    def fit(self, traces: list[dict[str, Any] | Transition]) -> "PredictiveWorldModel":
        for t in traces:
            if isinstance(t, Transition):
                self.observe(t.state, t.action, t.next_state, t.reward)
            else:
                self.observe(str(t["state"]), str(t["action"]), str(t["next_state"]), float(t.get("reward", 0.0)))
        return self

    def predict_distribution(self, state: str, action: str) -> dict[str, float]:
        c = self.counts.get((str(state), str(action)), Counter())
        total = sum(c.values())
        if total == 0:
            return {}
        return {k: round(v / total, 4) for k, v in sorted(c.items())}

    def expected_reward(self, state: str, action: str) -> float:
        dist = self.predict_distribution(state, action)
        if not dist:
            return 0.0
        total = 0.0
        for ns, p in dist.items():
            rs = self.rewards.get((str(state), str(action), ns), [])
            total += p * (sum(rs) / len(rs) if rs else 0.0)
        return round(total, 4)

    def uncertainty(self, state: str, action: str) -> float:
        dist = self.predict_distribution(state, action)
        if not dist:
            return 1.0
        h = -sum(p * math.log(p, 2) for p in dist.values() if p > 0)
        return round(h / max(1.0, math.log(max(2, len(dist)), 2)), 4)

    def is_ood(self, state: str, action: str, *, min_count: int = 2) -> bool:
        return sum(self.counts.get((str(state), str(action)), Counter()).values()) < min_count

    def choose_action(self, state: str, actions: list[str], *, uncertainty_floor: float = 0.75) -> dict[str, Any]:
        scored = []
        for a in actions:
            scored.append({
                "action": a,
                "expectedReward": self.expected_reward(state, a),
                "uncertainty": self.uncertainty(state, a),
                "ood": self.is_ood(state, a),
            })
        safe = [s for s in scored if not s["ood"] and s["uncertainty"] <= uncertainty_floor]
        if not safe:
            return {"verdict": "hold", "reason": "world-model OOD/uncertain; require exploration or verifier", "actions": scored}
        best = max(safe, key=lambda s: s["expectedReward"])
        return {"verdict": "act", "chosen": best, "actions": scored}

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for (s, a), c in self.counts.items():
            rows.append({"state": s, "action": a, "distribution": self.predict_distribution(s, a), "expectedReward": self.expected_reward(s, a)})
        return {"schema": "sophia.predictive_world_model.v1", "rows": rows}


def demo_world_model_report() -> dict[str, Any]:
    traces = [
        {"state": "claim_held", "action": "wikidata", "next_state": "one_source", "reward": 1.0},
        {"state": "claim_held", "action": "wikidata", "next_state": "one_source", "reward": 1.0},
        {"state": "claim_held", "action": "guess", "next_state": "rejected", "reward": -5.0},
        {"state": "one_source", "action": "crossref", "next_state": "accepted", "reward": 8.0},
        {"state": "one_source", "action": "crossref", "next_state": "accepted", "reward": 8.0},
        {"state": "one_source", "action": "judge_without_evidence", "next_state": "held", "reward": -1.0},
    ]
    wm = PredictiveWorldModel().fit(traces)
    decision = wm.choose_action("one_source", ["crossref", "judge_without_evidence", "unknown_tool"])
    return {
        "schema": "sophia.world_model_demo.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "model": wm.to_dict(),
        "decision": decision,
        "invariants": {
            "chooses_supported_high_reward_action": decision.get("chosen", {}).get("action") == "crossref",
            "holds_on_unseen_state_action": wm.choose_action("new_state", ["unknown_tool"])["verdict"] == "hold",
        },
    }


def write_world_model_report(out: str | Path) -> dict[str, Any]:
    report = demo_world_model_report()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = ["Transition", "PredictiveWorldModel", "demo_world_model_report", "write_world_model_report"]
