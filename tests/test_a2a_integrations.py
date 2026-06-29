# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the A2A real-machinery integrations (agent/a2a_integrations.py).

Deterministic, no network/key/real model — the swarm path uses the ``mock`` client.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import a2a  # noqa: E402
from agent import a2a_integrations as ai  # noqa: E402
from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402

_TRAP = "Confucius wrote the Dao De Jing."          # forbidden lineage merge
_OK = "The Analects is associated with the Confucian tradition."


def _client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


# --- ① source-discipline gate ---------------------------------------------- #
def test_sophia_gate_blocks_attribution_trap() -> None:
    v = ai.sophia_gate(_TRAP)
    assert v.accept is False
    assert v.label == "block"
    assert "source-discipline" in v.reason


def test_sophia_gate_allows_benign_and_inherits_markers() -> None:
    assert ai.sophia_gate(_OK).accept is True
    # the conservative markers still apply on top of the verifier
    assert ai.sophia_gate("this is definitely AGI").label == "block"
    assert ai.sophia_gate("").accept is False


def test_sophia_gate_is_fail_closed_offline(monkeypatch=None) -> None:
    # If the verifier can't load, we must keep the conservative verdict, never upgrade to trust.
    import agent.a2a_integrations as mod
    orig = mod._provenance_verdict
    mod._provenance_verdict = lambda _t: None
    try:
        assert mod.sophia_gate("a plain benign sentence").accept is True   # marker gate still ran
        assert mod.sophia_gate("this is definitely AGI").accept is False
    finally:
        mod._provenance_verdict = orig


# --- ③ contract handshake on the card -------------------------------------- #
def test_full_card_has_skills_and_validates() -> None:
    card = ai.sophia_agent_card_full("http://spark.local:8080")
    ok, problems = card.validate()
    assert ok, problems
    d = card.to_dict()
    assert {s["id"] for s in d["skills"]} >= {"swarm.delegate", "provenance.validate", "epistemic.gate_check"}


def test_contract_version_advertised_when_present() -> None:
    meta = ai.sophia_contract_meta()
    card = ai.sophia_agent_card_full("http://x.local")
    if meta.get("version"):
        assert card.to_dict()["x-sophia"]["contractVersion"] == meta["version"]
    else:
        # offline-safe: absent contract -> key simply omitted, card still valid
        assert "contractVersion" not in card.to_dict()["x-sophia"]


# --- ② real skill routing --------------------------------------------------- #
def test_skill_directive_parsing() -> None:
    assert ai._parse_skill_directive("@skill provenance.validate\nhello") == ("provenance.validate", "hello")
    assert ai._parse_skill_directive("just a task") == ("swarm.delegate", "just a task")
    assert ai._parse_skill_directive("@skill epistemic.gate_check: x")[0] == "epistemic.gate_check"


def test_invoke_provenance_validate_skill_blocks_trap() -> None:
    server = ai.make_sophia_a2a_server("http://x.local", client=_client())
    client = a2a.A2AClient(a2a.StubA2ATransport(server), gate=ai.sophia_gate)
    task = ai.invoke_skill(client, "provenance.validate", _TRAP)
    assert task.state == a2a.COMPLETED
    assert "block" in task.answer().lower()


def test_invoke_provenance_validate_skill_allows_benign() -> None:
    server = ai.make_sophia_a2a_server("http://x.local", client=_client())
    client = a2a.A2AClient(a2a.StubA2ATransport(server), gate=ai.sophia_gate)
    task = ai.invoke_skill(client, "provenance.validate", _OK)
    assert task.state == a2a.COMPLETED
    assert "allow" in task.answer().lower()


def test_default_routes_to_swarm() -> None:
    with tempfile.TemporaryDirectory() as td:
        h.RUNS_DIR = Path(td)
        server = ai.make_sophia_a2a_server("http://x.local", client=_client())
        client = a2a.A2AClient(a2a.StubA2ATransport(server), gate=ai.sophia_gate)
        task = client.send_task("summarise the discipline")   # no @skill -> swarm
        assert task.state == a2a.COMPLETED
        assert task.est_cost_steps >= 1


def test_unknown_skill_falls_back_to_swarm() -> None:
    with tempfile.TemporaryDirectory() as td:
        h.RUNS_DIR = Path(td)
        server = ai.make_sophia_a2a_server("http://x.local", client=_client())
        client = a2a.A2AClient(a2a.StubA2ATransport(server), gate=ai.sophia_gate)
        task = ai.invoke_skill(client, "no.such.skill", "a real task")
        assert task.state == a2a.COMPLETED   # fail-closed to the swarm default, not an error


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} integration tests passed")
