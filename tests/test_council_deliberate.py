#!/usr/bin/env python3
"""Tests for council deliberation (map-reduce + per-seat gate) and the uplift harness.

Uses STUB clients (no model calls) so behaviour is deterministic offline. These
verify the orchestration + gating mechanics, not real model quality.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import deliberate  # noqa: E402

FINANCE_Q = "Model our 18-month runway and flag AML for Stripe payouts."


def _client(text: str):
    return SimpleNamespace(generate=lambda system, user: SimpleNamespace(ok=True, text=text))


def _broken_client():
    def boom(system, user):
        raise RuntimeError("model down")
    return SimpleNamespace(generate=boom)


def test_deliberate_runs_seats_and_synthesizes() -> None:
    d = deliberate(FINANCE_Q, client=_client("Clean finance analysis. 中文摘要：見上。"), max_seats=3)
    assert d.councilId == "financial"
    assert 1 <= len(d.seats) <= 3            # capped at max_seats
    assert d.guardians                        # guardian seats present
    assert d.synthesis.strip()
    assert all(s.gatePassed for s in d.seats)  # clean answers pass the gate


def test_seat_gating_quarantines_fabricated_citation() -> None:
    bad = "Per Wong v Lee [2099] HKCFI 9999 the answer is yes. 中文摘要：見上。"
    d = deliberate(FINANCE_Q, client=_client(bad), max_seats=3, gate=True)
    assert d.gatedOutSeatIds                          # at least one seat quarantined
    assert all(not s.gatePassed for s in d.seats)     # every seat had the fabricated cite
    # with no clean seats, synthesis must abstain rather than repeat the fabrication
    assert "insufficient" in d.synthesis.lower()


def test_no_gate_keeps_seats() -> None:
    bad = "Per Wong v Lee [2099] HKCFI 9999. 中文摘要：見上。"
    d = deliberate(FINANCE_Q, client=_client(bad), max_seats=3, gate=False)
    assert d.gatedOutSeatIds == [] and all(s.gatePassed for s in d.seats)


def test_no_council_falls_back_to_single_pass() -> None:
    d = deliberate("What is the meaning of a quiet life?", client=_client("A reflective answer."))
    assert d.councilId is None and d.seats == [] and d.synthesis.strip()


def test_broken_client_abstains() -> None:
    d = deliberate(FINANCE_Q, client=_broken_client(), max_seats=3, gate=True)
    # every seat empty -> no clean seats -> abstaining synthesis (never a crash)
    assert "insufficient" in d.synthesis.lower()


def _load_uplift():
    spec = importlib.util.spec_from_file_location("rcu", ROOT / "tools" / "run_council_uplift.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_uplift_harness_three_conditions() -> None:
    m = _load_uplift()
    res = m.run_uplift(m.DEMO_TASKS, _client("Clean answer. 中文摘要：見上。"))
    assert set(res["conditions"]) == {"alone", "council", "council+gate"}
    for r in res["conditions"].values():
        assert 0.0 <= r["cleanRate"] <= 1.0


def test_uplift_positive_when_model_hallucinates() -> None:
    m = _load_uplift()
    # a model that always fabricates a citation: alone is dirty, council+gate abstains -> clean
    hallucinating = _client("Per Chan v SC [2099] HKCFI 9999, yes. 中文摘要：見上。")
    res = m.run_uplift(m.DEMO_TASKS, hallucinating)
    assert res["conditions"]["alone"]["cleanRate"] == 0.0
    assert res["conditions"]["council+gate"]["cleanRate"] == 1.0
    assert res["deltaCleanRate"] == 1.0  # the uplift the thesis predicts


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_council_deliberate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
