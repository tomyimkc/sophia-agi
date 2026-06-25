#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W4 Option A — prove the ``use_claim_router`` ablation flag flows to the live gate.

The per-claim routing seam (``agent/claim_router.py``) is dormant on the live path:
``run_case`` calls ``agent.gate.check_response`` without ``route_claims``. W4 turns it
into an OFF-by-default, ablatable lever. These tests pin the wiring WITHOUT any network
or model backend:

- the model-generation step (``call_model``) and all IO are stubbed deterministically;
- ``check_response`` is patched to capture the ``route_claims`` keyword it receives;
- with the flag OFF (default ``sophia-full``) the gate is called with
  ``route_claims=False`` and the router seam is never entered;
- with the flag ON (``sophia-claim-router``) the gate is called with
  ``route_claims=True`` and the real ``route_and_check`` seam IS reached.

No live delta is produced or claimed — only that the seam is now measurable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_hidden_eval_sophia as runner  # noqa: E402

CONFIG = runner.RunConfig(backend="fake", timeout_sec=1)

CODING_CASE = {
    "id": "coding_router_001",
    "domain": "coding",
    "prompt": "Fix a Python bug in the retry logic.",
    "materials": [],
    "scoring": {"maxPoints": 1, "rubric": ["x"], "mustInclude": ["Decision"]},
}


def _fake_answer() -> dict:
    return {
        "backend": "fake",
        "returncode": 0,
        "elapsedSec": 0.01,
        "answer": "Source discipline applies. Decision: yes. 中文摘要: 好.",
        "stderrTail": "",
    }


def _stub_pipeline() -> dict:
    """Replace IO/network/corpus calls with deterministic stubs (mirror test_ablation_runner)."""
    saved = {
        "call_model": runner.call_model,
        "retrieve": runner.retrieve,
        "gather_evidence": runner.gather_evidence,
        "run_operational_tools": runner.run_operational_tools,
        "append_learning_memory": runner.append_learning_memory,
    }

    class FakeChunk:
        path = "data/attributions.json"
        title = "stub"
        excerpt = "stub excerpt"
        score = 0.9

    runner.call_model = lambda system, user, *, backend, timeout_sec, grok_cwd=None: {
        **_fake_answer(),
        "_system": system,
        "_user": user,
    }
    runner.retrieve = lambda query, *, top_k=8: [FakeChunk()]
    runner.gather_evidence = lambda query, **kw: {"localSources": [{"url": "docs/x.md"}], "web": {"online": False, "sources": []}}
    runner.run_operational_tools = lambda case: ({"commands": [{"cmd": "git status", "returncode": 0}]} if case.get("requiresToolLog") else {})
    runner.append_learning_memory = lambda case: {"appended": True, "oldHashChanged": False, "memoryFile": "stub", "entryRecordId": "stub"}
    return saved


def _restore(saved: dict) -> None:
    for name, value in saved.items():
        setattr(runner, name, value)


def _capture_route_claims(ablation: runner.Ablation) -> list[bool]:
    """Run one case under ``ablation`` and return the ``route_claims`` value(s) the
    gate was invoked with."""
    seen: list[bool] = []

    def fake_check_response(text, **kwargs):
        seen.append(kwargs.get("route_claims"))
        return dict(runner.NEUTRAL_GATE)

    saved = _stub_pipeline()
    try:
        with mock.patch.object(runner, "check_response", side_effect=fake_check_response):
            runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=ablation)
    finally:
        _restore(saved)
    return seen


def test_flag_defaults_off_on_sophia_full() -> None:
    assert runner.SOPHIA_FULL.use_claim_router is False
    assert runner.Ablation().use_claim_router is False


def test_router_mode_registered_but_not_canonical() -> None:
    assert "sophia-claim-router" in runner.ABLATION_MODES
    assert runner.ABLATION_MODES["sophia-claim-router"].use_claim_router is True


def test_flag_off_calls_gate_with_route_claims_false() -> None:
    seen = _capture_route_claims(runner.SOPHIA_FULL)
    assert seen, "gate should be invoked at least once on sophia-full"
    assert all(v is False for v in seen), f"expected route_claims=False, saw {seen}"


def test_flag_on_calls_gate_with_route_claims_true() -> None:
    seen = _capture_route_claims(runner.ABLATION_MODES["sophia-claim-router"])
    assert seen, "gate should be invoked at least once with the router mode"
    assert all(v is True for v in seen), f"expected route_claims=True, saw {seen}"


def test_router_seam_reached_only_when_flag_on() -> None:
    """At the gate boundary, ``route_and_check`` must be entered iff route_claims=True."""
    import agent.claim_router as claim_router
    import agent.gate as gate

    sentinel = {"perClaim": [], "violations": []}

    with mock.patch.object(claim_router, "route_and_check", return_value=sentinel) as router_off:
        gate.check_response("Some answer with a 中文 summary.", mode="advisor", question="q", route_claims=False)
    assert router_off.call_count == 0

    with mock.patch.object(claim_router, "route_and_check", return_value=sentinel) as router_on:
        gate.check_response("Some answer with a 中文 summary.", mode="advisor", question="q", route_claims=True)
    assert router_on.call_count == 1


def main() -> int:
    test_flag_defaults_off_on_sophia_full()
    test_router_mode_registered_but_not_canonical()
    test_flag_off_calls_gate_with_route_claims_false()
    test_flag_on_calls_gate_with_route_claims_true()
    test_router_seam_reached_only_when_flag_on()
    print("test_claim_router_ablation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
