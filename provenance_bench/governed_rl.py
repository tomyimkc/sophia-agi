# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-0 of Continual-Governed-RL: a fail-closed replay buffer.

This is the concrete first step of
[docs/11-Platform/Continual-Governed-RL.md](../docs/11-Platform/Continual-Governed-RL.md):
turn the async-RL replay buffer (``provenance_bench.async_rl.ReplayBuffer``,
which already bounds *off-policy staleness*) into a buffer that also enforces the
**trust bound** — a self-generated trajectory is admitted into a policy update
only if it clears every governor:

    admit(traj) iff
        reward earned  (deterministic, bounded; not a learnable RM → unhackable)
      and gate passes  (agent.verifiers.provenance_faithful — the SAME fail-closed
                        seam the verifier-as-reward uses; a fabricated attribution
                        is rejected regardless of how high its reward is)
      and grounded     (okf.counterfactual.is_grounded against the trusted corpus;
                        an off-corpus or orphaned belief is too far off-trust)
      and staleness ok (delegated to the wrapped ReplayBuffer at sample time)

Nothing here trains a model — that is the gated live path. This is the
**admission policy**, proven by deterministic offline invariants on any machine
(no torch/GPU). The governors are the repo's real seams, not mocks: the gate is
``agent.verifiers.provenance_faithful``; grounding is the real OKF belief graph.

Offline invariants (``offline_invariants()``, CI-gated):
  1. No ungated promotion — a high-reward *fabrication* is dropped by the gate.
  2. Trust bound is real — an off-corpus / orphaned belief is dropped (ungrounded),
     a grounded one is admitted.
  3. Unhackable reward — a below-floor reward is dropped; reward is deterministic.
  4. Staleness still bounded — admitted trajectories past max_staleness are never
     sampled (carried from async_rl, re-checked end-to-end here).
  5. Accounting closes — admitted + each rejection reason == offered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from provenance_bench.async_rl import ReplayBuffer, Trajectory

# admit predicate: (trajectory, work) -> bool
Predicate = Callable[[Trajectory, Optional[str]], bool]

# Reward floor: a trajectory must beat this to be admitted. The verifier reward is
# bounded in [-1, 1] with -1 the "asserted forbidden / unverifiable" floor, so the
# default rejects exactly the floor (a hair above -1 to admit legitimate -1+ε).
DEFAULT_REWARD_FLOOR = -0.999


@dataclass
class AdmitDecision:
    admitted: bool
    reason: str   # "admitted" | "low_reward" | "ungated" | "ungrounded"


def make_provenance_gate(records: "dict | None") -> Predicate:
    """A fail-closed admit predicate from the real provenance gate.

    Wraps ``agent.verifiers.provenance_faithful`` — the same verifier the RLVR
    reward is built on — so admission and reward agree on what "fabricated" means.
    Admits iff the completion does not assert a forbidden attribution.
    """
    from agent.verifiers import provenance_faithful

    gate = provenance_faithful(records)

    def predicate(traj: Trajectory, work: Optional[str] = None) -> bool:
        return bool(gate(traj.completion, None, {})["passed"])

    return predicate


def make_okf_grounding(graph) -> Predicate:
    """A trust-bound predicate over the real OKF belief graph.

    Admits iff the trajectory's subject (``work``, else the prompt) resolves to a
    present node that ``is_grounded`` — i.e. it has a provenance ground in the
    trusted corpus. An off-corpus subject (resolves to nothing) or an orphaned
    derived claim (its ``derivesFrom`` support is gone) is **ungrounded** → dropped.
    This is the fail-closed reading of "too far off-trust to train on".
    """
    from okf.counterfactual import is_grounded
    from okf.graph import resolve

    def predicate(traj: Trajectory, work: Optional[str] = None) -> bool:
        target = work or traj.prompt
        nid = resolve(graph, target)
        if nid is None:
            return False          # off-corpus → off-trust
        return is_grounded(graph, nid)

    return predicate


class FailClosedReplayBuffer:
    """A staleness-bounded replay buffer that admits only verified, grounded data.

    Wraps ``async_rl.ReplayBuffer`` (staleness + capacity bounds) and adds the
    version-independent governors at ``offer`` time, so unverified/ungrounded
    trajectories never enter the buffer at all. ``gate`` and ``grounded`` are
    optional governors; when supplied they are enforced **fail-closed** (a
    predicate that raises is treated as a rejection, never a silent pass).
    """

    def __init__(
        self,
        capacity: int,
        max_staleness: int,
        *,
        gate: Optional[Predicate] = None,
        grounded: Optional[Predicate] = None,
        reward_floor: float = DEFAULT_REWARD_FLOOR,
    ) -> None:
        self._buf = ReplayBuffer(capacity=capacity, max_staleness=max_staleness)
        self.gate = gate
        self.grounded = grounded
        self.reward_floor = reward_floor
        self.admit_stats: dict[str, int] = {
            "offered": 0, "admitted": 0,
            "low_reward": 0, "ungated": 0, "ungrounded": 0,
        }

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def buffer_stats(self):
        """Staleness/overflow stats from the wrapped async_rl buffer."""
        return self._buf.stats

    def _safe(self, pred: Predicate, traj: Trajectory, work: Optional[str]) -> bool:
        """Evaluate a governor fail-closed: any exception == rejection."""
        try:
            return bool(pred(traj, work))
        except Exception:
            return False

    def admit(self, traj: Trajectory, work: Optional[str] = None) -> AdmitDecision:
        """Version-independent admission decision (does not enqueue)."""
        if traj.reward <= self.reward_floor:
            return AdmitDecision(False, "low_reward")
        if self.gate is not None and not self._safe(self.gate, traj, work):
            return AdmitDecision(False, "ungated")
        if self.grounded is not None and not self._safe(self.grounded, traj, work):
            return AdmitDecision(False, "ungrounded")
        return AdmitDecision(True, "admitted")

    def offer(self, traj: Trajectory, work: Optional[str] = None) -> AdmitDecision:
        """Admit-then-enqueue. Rejected trajectories are counted, never buffered."""
        self.admit_stats["offered"] += 1
        decision = self.admit(traj, work)
        self.admit_stats[decision.reason] += 1
        if decision.admitted:
            self._buf.push(traj)
        return decision

    def offer_group(self, trajs: list[Trajectory], work: Optional[str] = None) -> list[AdmitDecision]:
        return [self.offer(t, work) for t in trajs]

    def sample(self, batch_size: int, current_version: int) -> list[Trajectory]:
        """Staleness-bounded sample (delegates to the wrapped buffer)."""
        return self._buf.sample(batch_size, current_version)


# ---------------------------------------------------------------------------
# Offline invariants — deterministic, real seams, no GPU.
# ---------------------------------------------------------------------------

# Synthetic provenance records (mirrors tests/test_rlvr.py so the real gate fires
# deterministically without depending on the seeded corpus).
_RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter",
                        "doNotAttributeTo": ["Alice"]}}
_WORK = "Project Phoenix Charter"
_GOOD = "No, Alice did not write the Project Phoenix Charter; the founding committee did."
_FABRICATION = "Alice wrote the Project Phoenix Charter."


def _traj(completion: str, reward: float, version: int = 0) -> Trajectory:
    return Trajectory(group_id=0, prompt=f"Who wrote {_WORK}?", completion=completion,
                      reward=reward, advantage=0.0, policy_version=version)


def _build_corpus_graph(tmpdir):
    """A tiny real OKF graph: one grounded primary page + one orphaned derived page."""
    from pathlib import Path

    from okf import frontmatter, graph as okf_graph, page as okf_page

    specs = [
        # primary, self-grounded (no derivesFrom) — the work in the corpus
        ("project-phoenix-charter.md",
         {"id": "project-phoenix-charter", "pageType": "text",
          "aliases": ["Project Phoenix Charter"]},
         "Authored by the founding committee."),
        # derived claim whose support is missing → orphaned → not grounded
        ("orphan-claim.md",
         {"id": "orphan-claim", "pageType": "text",
          "aliases": ["Orphan Claim"], "derivesFrom": ["nonexistent-source"]},
         "A claim with a vanished source."),
    ]
    for rel, meta, body in specs:
        p = Path(tmpdir) / rel
        p.write_text(frontmatter.serialize(meta, body), encoding="utf-8")
    return okf_graph.build(okf_page.load_pages(tmpdir))


def offline_invariants() -> "tuple[bool, dict]":
    import tempfile

    checks: dict[str, bool] = {}
    gate = make_provenance_gate(_RECORDS)

    # 1. No ungated promotion: a FABRICATION with an artificially HIGH reward is
    #    still dropped by the gate (defense-in-depth — gate is independent of reward).
    b1 = FailClosedReplayBuffer(capacity=64, max_staleness=4, gate=gate)
    d_fab = b1.offer(_traj(_FABRICATION, reward=1.0))     # high reward, fabricated
    d_good = b1.offer(_traj(_GOOD, reward=0.7))
    checks["fabrication_rejected_despite_high_reward"] = (
        not d_fab.admitted and d_fab.reason == "ungated"
    )
    checks["verified_admitted"] = d_good.admitted and len(b1) == 1

    # 2. Trust bound: grounded subject admitted; off-corpus & orphaned rejected.
    with tempfile.TemporaryDirectory() as tmp:
        graph = _build_corpus_graph(tmp)
        grounding = make_okf_grounding(graph)
        b2 = FailClosedReplayBuffer(capacity=64, max_staleness=4, grounded=grounding)
        in_corpus = b2.offer(_traj(_GOOD, reward=0.7), work=_WORK)
        off_corpus = b2.offer(_traj(_GOOD, reward=0.7), work="Nonexistent Treatise")
        orphaned = b2.offer(_traj(_GOOD, reward=0.7), work="Orphan Claim")
        checks["grounded_admitted"] = in_corpus.admitted
        checks["off_corpus_rejected"] = (
            not off_corpus.admitted and off_corpus.reason == "ungrounded"
        )
        checks["orphaned_rejected"] = (
            not orphaned.admitted and orphaned.reason == "ungrounded"
        )

    # 3. Unhackable reward: a below-floor reward is dropped before any other check.
    b3 = FailClosedReplayBuffer(capacity=64, max_staleness=4, gate=gate)
    d_low = b3.offer(_traj(_GOOD, reward=-1.0))
    checks["low_reward_rejected"] = not d_low.admitted and d_low.reason == "low_reward"

    # 4. Staleness still bounded end-to-end: admit fresh-enough trajectories at
    #    several versions, then sample — over-stale ones are never returned.
    b4 = FailClosedReplayBuffer(capacity=64, max_staleness=2, gate=gate)
    for v in range(6):
        b4.offer(_traj(_GOOD, reward=0.7, version=v))     # all admitted
    sampled = b4.sample(10, current_version=5)            # staleness = 5 - v
    checks["staleness_bounded"] = all(5 - t.policy_version <= 2 for t in sampled)
    checks["all_admitted_were_verified"] = b4.admit_stats["admitted"] == 6

    # 5. Accounting closes: admitted + rejections == offered.
    s = b1.admit_stats
    checks["accounting_closes"] = (
        s["admitted"] + s["low_reward"] + s["ungated"] + s["ungrounded"] == s["offered"]
    )

    ok = all(checks.values())
    return ok, {"checks": checks, "admit_stats": b1.admit_stats}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Governed-RL (fail-closed buffer) offline invariants:",
          "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)
