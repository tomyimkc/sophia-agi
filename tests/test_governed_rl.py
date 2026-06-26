# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-0 invariants for the fail-closed replay buffer (real gate + OKF, no GPU).

Verifies the trust-bound admission policy of Continual-Governed-RL against the
repo's real seams: the provenance gate (agent.verifiers.provenance_faithful, the
same one the RLVR reward uses) and the real OKF belief graph (okf.counterfactual
.is_grounded). No model is trained — this proves the *admission* logic.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import governed_rl, rl_reward  # noqa: E402
from provenance_bench.async_rl import Trajectory  # noqa: E402
from provenance_bench.dataset import Case  # noqa: E402
from provenance_bench.governed_rl import (  # noqa: E402
    FailClosedReplayBuffer,
    make_okf_grounding,
    make_provenance_gate,
)

RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter",
                       "doNotAttributeTo": ["Alice"]}}
WORK = "Project Phoenix Charter"
GOOD = "No, Alice did not write the Project Phoenix Charter; the founding committee did."
FABRICATION = "Alice wrote the Project Phoenix Charter."


def _traj(completion, reward, version=0):
    return Trajectory(0, f"Who wrote {WORK}?", completion, reward, 0.0, version)


def test_offline_invariants() -> None:
    ok, detail = governed_rl.offline_invariants()
    assert ok, detail["checks"]


def test_gate_rejects_fabrication_independent_of_reward() -> None:
    gate = make_provenance_gate(RECORDS)
    buf = FailClosedReplayBuffer(64, 4, gate=gate)
    # A fabrication with a *high* reward is still rejected by the gate.
    d = buf.offer(_traj(FABRICATION, reward=1.0))
    assert not d.admitted and d.reason == "ungated"
    assert len(buf) == 0


def test_verified_trajectory_admitted() -> None:
    gate = make_provenance_gate(RECORDS)
    buf = FailClosedReplayBuffer(64, 4, gate=gate)
    d = buf.offer(_traj(GOOD, reward=0.7))
    assert d.admitted and len(buf) == 1


def test_low_reward_rejected() -> None:
    gate = make_provenance_gate(RECORDS)
    buf = FailClosedReplayBuffer(64, 4, gate=gate)
    d = buf.offer(_traj(GOOD, reward=-1.0))
    assert not d.admitted and d.reason == "low_reward"


def test_grounding_governor_with_real_okf_graph() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        graph = governed_rl._build_corpus_graph(tmp)
        grounding = make_okf_grounding(graph)
        buf = FailClosedReplayBuffer(64, 4, grounded=grounding)
        assert buf.offer(_traj(GOOD, 0.7), work=WORK).admitted
        assert buf.offer(_traj(GOOD, 0.7), work="Nonexistent Treatise").reason == "ungrounded"
        assert buf.offer(_traj(GOOD, 0.7), work="Orphan Claim").reason == "ungrounded"


def test_staleness_bound_holds_after_admission() -> None:
    gate = make_provenance_gate(RECORDS)
    buf = FailClosedReplayBuffer(64, max_staleness=2, gate=gate)
    for v in range(6):
        buf.offer(_traj(GOOD, 0.7, version=v))
    sampled = buf.sample(10, current_version=5)
    assert all(5 - t.policy_version <= 2 for t in sampled)


def test_fail_closed_on_raising_predicate() -> None:
    def boom(traj, work=None):
        raise RuntimeError("predicate blew up")

    buf = FailClosedReplayBuffer(64, 4, gate=boom)
    d = buf.offer(_traj(GOOD, 0.7))
    assert not d.admitted and d.reason == "ungated"   # exception == rejection


def test_admission_agrees_with_real_reward_seam() -> None:
    """The gate's admit verdict matches the sign of the real RLVR reward."""
    gate_v = rl_reward.make_grpo_reward(records=RECORDS)
    fab_reward = gate_v(["p"], [FABRICATION], label="false", gold_author="the founding committee",
                        claimed_author="Alice")[0]
    good_reward = gate_v(["p"], [GOOD], label="false", gold_author="the founding committee",
                         claimed_author="Alice")[0]
    # Real reward floors the fabrication and rewards the correction...
    assert fab_reward == rl_reward.REWARD_MIN
    assert good_reward > 0
    # ...and the buffer, fed those real rewards, admits exactly the good one.
    gate = make_provenance_gate(RECORDS)
    buf = FailClosedReplayBuffer(64, 4, gate=gate)
    assert not buf.offer(_traj(FABRICATION, fab_reward)).admitted
    assert buf.offer(_traj(GOOD, good_reward)).admitted


def test_accounting_closes() -> None:
    gate = make_provenance_gate(RECORDS)
    buf = FailClosedReplayBuffer(64, 4, gate=gate)
    buf.offer(_traj(FABRICATION, 1.0))
    buf.offer(_traj(GOOD, 0.7))
    buf.offer(_traj(GOOD, -1.0))
    s = buf.admit_stats
    assert s["admitted"] + s["low_reward"] + s["ungated"] + s["ungrounded"] == s["offered"]
