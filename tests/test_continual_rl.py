# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-1 invariants for the continual-governed-RL loop (real seams, no GPU).

Exercises the full pipeline — agent.model generation (mock backend) → real
verifier-as-reward → fail-closed admission → trainer step — deterministically.
No weights are trained; the live GPU GRPO update stays gated.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import continual_rl  # noqa: E402
from provenance_bench.continual_rl import (  # noqa: E402
    ContinualGovernedLoop,
    model_generate_fn,
    scripted_policy_generate_fn,
)

CASES = continual_rl._CASES
RECORDS = continual_rl._RECORDS


def test_offline_invariants() -> None:
    ok, detail = continual_rl.offline_invariants()
    assert ok, detail["checks"]


def test_loop_accounting_closes() -> None:
    loop = ContinualGovernedLoop(CASES, RECORDS, seed=1)
    rep = loop.run(30)
    s = rep.admit_stats
    assert s["admitted"] + s["low_reward"] + s["ungated"] + s["ungrounded"] == s["offered"]


def test_fabricating_policy_admits_nothing() -> None:
    bad = ContinualGovernedLoop(
        CASES, RECORDS, generate_fn=scripted_policy_generate_fn([0.0]), seed=2)
    rep = bad.run(20)
    assert rep.admitted == 0
    assert rep.train_steps == 0


def test_governed_improvement_raises_admit_rate() -> None:
    rep = ContinualGovernedLoop(CASES, RECORDS, seed=1).run(40)
    assert rep.late_admit_rate > rep.early_admit_rate


def test_real_model_adapter_seam_offline() -> None:
    gen = model_generate_fn()
    out = gen(CASES[0], 0, random.Random(0))
    assert isinstance(out, str) and len(out) > 0      # mock backend, offline


def test_deterministic() -> None:
    a = ContinualGovernedLoop(CASES, RECORDS, seed=7).run(30).as_dict()
    b = ContinualGovernedLoop(CASES, RECORDS, seed=7).run(30).as_dict()
    assert a == b


def test_cli_check_runs() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import run_continual_rl  # noqa: E402

    # --check path returns 0 when invariants pass
    sys.argv = ["run_continual_rl.py", "--check"]
    assert run_continual_rl.main() == 0
