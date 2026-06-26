# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Continual-Governed-RL (Phase 1 wiring): generation → reward → fail-closed admit.

Phase 1 of
[docs/11-Platform/Continual-Governed-RL.md](../docs/11-Platform/Continual-Governed-RL.md):
a runnable end-to-end loop that composes the real seams —

    agent.model (generation)  →  provenance_bench.rl_reward (verifier-as-reward)
      →  governed_rl.FailClosedReplayBuffer (gate + OKF grounding + staleness)
      →  trainer step (version bump; the GPU GRPO update stays gated)

— so the whole pipeline is exercised, not just its parts. The generator defaults
to ``agent.model.complete`` (the unified adapter): **mock backend offline**, and a
**live provider (DeepSeek/any) is gated behind env** exactly like the existing
RLVR run. For deterministic CI and to exercise the governors, a scripted policy
generator is provided whose admitted-rate *rises* as its skill improves — a
synthetic stand-in for the optimizer, which is the only mocked part.

No GPU. The trainer "update" is a version bump + a skill proxy; it trains no
weights (that is the gated live GRPO step). What is real and end-to-end here: a
model produces completions, the deterministic verifier scores them, and only
verified+grounded+fresh trajectories are admitted toward an update.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from provenance_bench.async_rl import Trajectory, grpo_advantages
from provenance_bench.governed_rl import FailClosedReplayBuffer, make_provenance_gate


@dataclass
class CaseSpec:
    prompt: str
    work: str
    label: str = "false"
    gold_author: str = ""
    claimed_author: Optional[str] = None


# generate_fn(case, version, rng) -> completion text
GenerateFn = Callable[[CaseSpec, int, random.Random], str]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def model_generate_fn(spec: Optional[str] = None, *, system: str = "") -> GenerateFn:
    """Real generator via the unified model adapter (mock offline, live gated).

    Calls ``agent.model.complete`` — which resolves to the ``mock`` backend with no
    provider configured (offline, deterministic) and to a live provider when
    ``SOPHIA_MODEL_PROVIDER`` / keys are set. Proves the loop composes over the
    actual inference seam.
    """
    from agent.model import complete

    sys_prompt = system or ("Answer with source discipline; do not assert an "
                            "undocumented or forbidden attribution.")

    def gen(case: CaseSpec, version: int, rng: random.Random) -> str:
        return complete(sys_prompt, case.prompt)

    return gen


def scripted_policy_generate_fn(skill_ref: "list[float]") -> GenerateFn:
    """Deterministic improving policy: emits a correct refusal with prob = skill,
    else a fabricated attribution. ``skill_ref`` is a 1-elem list the trainer
    mutates, so admitted-rate rises as the policy improves.
    """
    def gen(case: CaseSpec, version: int, rng: random.Random) -> str:
        skill = skill_ref[0]
        if rng.random() < skill:
            who = case.gold_author or "the documented author"
            return (f"No, {case.claimed_author} did not write the {case.work}; "
                    f"it was written by {who}.")
        return f"{case.claimed_author} wrote the {case.work}."   # fabrication → gate floors it

    return gen


def build_reward_fn(records: "dict | None") -> Callable[[str, CaseSpec], float]:
    """Wrap the real RLVR reward (verifier-as-reward) as reward_fn(completion, case)."""
    from provenance_bench import rl_reward

    grpo = rl_reward.make_grpo_reward(records=records)

    def reward_fn(completion: str, case: CaseSpec) -> float:
        return grpo([case.prompt], [completion], label=case.label,
                    gold_author=case.gold_author, claimed_author=case.claimed_author)[0]

    return reward_fn


# ---------------------------------------------------------------------------
# The continual loop
# ---------------------------------------------------------------------------

@dataclass
class LoopReport:
    rounds: int
    generated: int
    admitted: int
    train_steps: int
    final_version: int
    final_skill: float
    admit_stats: dict
    early_admit_rate: float
    late_admit_rate: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class ContinualGovernedLoop:
    """End-to-end: rollout → real reward → fail-closed admit → trainer step."""

    def __init__(
        self,
        cases: list[CaseSpec],
        records: "dict | None",
        *,
        generate_fn: Optional[GenerateFn] = None,
        gate=None,
        grounded=None,
        group_size: int = 6,
        batch_size: int = 12,
        max_staleness: int = 2,
        lr: float = 0.06,
        seed: int = 0,
    ) -> None:
        if not cases:
            raise ValueError("need at least one case")
        self.cases = cases
        self.records = records
        self._skill = [0.15]                       # mutable policy proxy
        self.generate_fn = generate_fn or scripted_policy_generate_fn(self._skill)
        self.reward_fn = build_reward_fn(records)
        self.gate = gate if gate is not None else make_provenance_gate(records)
        self.buffer = FailClosedReplayBuffer(
            capacity=256, max_staleness=max_staleness, gate=self.gate, grounded=grounded)
        self.group_size = group_size
        self.batch_size = batch_size
        self.lr = lr
        self.version = 0
        self._rng = random.Random(seed)
        self._gid = 0
        self._admit_history: list[int] = []        # 1 admitted / 0 rejected, per trajectory

    def step(self) -> None:
        case = self.cases[self._gid % len(self.cases)]
        self._gid += 1
        completions = [self.generate_fn(case, self.version, self._rng)
                       for _ in range(self.group_size)]
        rewards = [self.reward_fn(c, case) for c in completions]
        advs = grpo_advantages(rewards)
        for c, r, a in zip(completions, rewards, advs):
            traj = Trajectory(self._gid, case.prompt, c, r, a, self.version)
            decision = self.buffer.offer(traj, work=case.work)
            self._admit_history.append(1 if decision.admitted else 0)
        # trainer step: consume a fresh-enough batch, bump version, improve skill
        if len(self.buffer) >= self.batch_size:
            batch = self.buffer.sample(self.batch_size, self.version)
            if batch:
                good = sum(1 for t in batch if t.advantage > 0) / len(batch)
                self._skill[0] = min(1.0, self._skill[0] + self.lr * good)
                self.version += 1

    def run(self, rounds: int) -> LoopReport:
        for _ in range(rounds):
            self.step()
        s = self.buffer.admit_stats
        hist = self._admit_history
        half = max(1, len(hist) // 2)
        early = sum(hist[:half]) / half
        late = sum(hist[half:]) / max(1, len(hist) - half)
        return LoopReport(
            rounds=rounds, generated=s["offered"], admitted=s["admitted"],
            train_steps=self.version, final_version=self.version,
            final_skill=round(self._skill[0], 4), admit_stats=dict(s),
            early_admit_rate=round(early, 4), late_admit_rate=round(late, 4),
        )


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

_RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter",
                        "doNotAttributeTo": ["Alice"]}}
_CASES = [CaseSpec(prompt="Did Alice write the Project Phoenix Charter?",
                   work="Project Phoenix Charter", label="false",
                   gold_author="the founding committee", claimed_author="Alice")]


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. End-to-end composes and accounting closes.
    loop = ContinualGovernedLoop(_CASES, _RECORDS, group_size=6, batch_size=12, seed=1)
    rep = loop.run(rounds=40)
    s = rep.admit_stats
    checks["accounting_closes"] = (
        s["admitted"] + s["low_reward"] + s["ungated"] + s["ungrounded"] == s["offered"]
    )
    checks["loop_produces_train_steps"] = rep.train_steps > 0
    detail["report"] = rep.as_dict()

    # 2. Governors active: a never-improving fabricating policy is fully rejected.
    fab_skill = [0.0]
    bad = ContinualGovernedLoop(
        _CASES, _RECORDS, generate_fn=scripted_policy_generate_fn(fab_skill), seed=2)
    bad_rep = bad.run(rounds=20)
    checks["fabrication_policy_admits_nothing"] = bad_rep.admitted == 0
    checks["fabrication_no_training"] = bad_rep.train_steps == 0

    # 3. Governed improvement: admitted-rate rises as the policy improves.
    checks["admit_rate_rises"] = rep.late_admit_rate > rep.early_admit_rate

    # 4. The real model adapter seam produces a completion offline (mock backend).
    gen = model_generate_fn()
    out = gen(_CASES[0], 0, random.Random(0))
    checks["model_seam_returns_text"] = isinstance(out, str) and len(out) > 0

    # 5. Determinism: same seed → identical report.
    a = ContinualGovernedLoop(_CASES, _RECORDS, seed=7).run(30).as_dict()
    b = ContinualGovernedLoop(_CASES, _RECORDS, seed=7).run(30).as_dict()
    checks["deterministic"] = a == b

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Continual-Governed-RL (Phase 1) offline invariants:",
          "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    r = detail.get("report", {})
    print(f"  admitted {r.get('admitted')}/{r.get('generated')} | "
          f"train_steps {r.get('train_steps')} | "
          f"admit-rate {r.get('early_admit_rate')} → {r.get('late_admit_rate')} | "
          f"skill → {r.get('final_skill')}")
    raise SystemExit(0 if ok else 1)
