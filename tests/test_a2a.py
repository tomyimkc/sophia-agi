# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline, deterministic tests for the A2A interoperability layer (agent/a2a.py).

No network, key, or real model — the whole client<->server round trip runs under the
``mock`` model client, matching the rest of the suite (tests/test_swarm_router.py).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import a2a  # noqa: E402
from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402

_SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema" / "agent-card-1.0.0.json").read_text())


def _client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


def _server(**kw) -> a2a.A2AServer:
    return a2a.A2AServer(a2a.sophia_agent_card("http://test.local:8080"), client=_client(), **kw)


# --- Agent Card ------------------------------------------------------------- #
def test_agent_card_valid_and_least_privilege() -> None:
    card = a2a.sophia_agent_card("http://spark.local:8080")
    ok, problems = card.validate()
    assert ok, problems
    d = card.to_dict()
    # only advertised skills are reachable
    assert d["skills"], "must advertise at least one skill"
    assert "swarm.delegate" in card.skill_ids()
    # Sophia discipline flag is present
    assert d["x-sophia"]["provenanceDiscipline"] is True
    assert d["x-sophia"]["reduce"] == "fail_closed_synthesis"


def test_agent_card_matches_schema_shape() -> None:
    d = a2a.sophia_agent_card("https://x.local").to_dict()
    for req in _SCHEMA["required"]:
        assert req in d, f"missing required card field {req}"
    assert d["schema"] == "sophia.agent_card.v1"
    assert d["url"].startswith("https://")


def test_agent_card_rejects_bad_url_and_dupe_skills() -> None:
    bad = a2a.AgentCard(name="x", description="d", url="ftp://nope",
                        skills=(a2a.AgentSkill("a", "A", "d"), a2a.AgentSkill("a", "A2", "d")))
    ok, problems = bad.validate()
    assert not ok
    assert any("url must be http" in p for p in problems)
    assert any("duplicate skill" in p for p in problems)


def test_server_rejects_invalid_card() -> None:
    try:
        a2a.A2AServer(a2a.AgentCard(name="", description="d", url="http://x", skills=()),
                      client=_client())
        assert False, "should have rejected an invalid card"
    except ValueError:
        pass


# --- Discovery + task lifecycle -------------------------------------------- #
def test_discovery_returns_card() -> None:
    client = a2a.A2AClient(a2a.StubA2ATransport(_server()))
    card = client.get_card()
    assert card["name"] == "Sophia"
    assert "swarm.delegate" in {s["id"] for s in card["skills"]}


def test_send_task_completes_with_artifacts() -> None:
    with tempfile.TemporaryDirectory() as td:
        h.RUNS_DIR = Path(td)
        client = a2a.A2AClient(a2a.StubA2ATransport(_server()))
        task = client.send_task("summarise the provenance discipline")
        assert task.state == a2a.COMPLETED
        assert task.id
        assert any(art.get("kind") == "text" for art in task.artifacts)
        assert task.gate is not None
        assert task.est_cost_steps >= 1


def test_empty_task_fails_closed_no_swarm() -> None:
    server = _server()
    client = a2a.A2AClient(a2a.StubA2ATransport(server))
    task = client.send_task("   ")
    assert task.state == a2a.FAILED
    assert "empty task" in task.error


def test_tasks_get_and_cancel() -> None:
    with tempfile.TemporaryDirectory() as td:
        h.RUNS_DIR = Path(td)
        server = _server()
        client = a2a.A2AClient(a2a.StubA2ATransport(server))
        task = client.send_task("a real task here")
        # completed tasks are pollable
        again = client.get_task(task.id)
        assert again.id == task.id
        # cancel on a terminal task is a no-op (stays completed)
        canceled = client.cancel_task(task.id)
        assert canceled.state == a2a.COMPLETED


def test_unknown_method_and_unknown_task_error() -> None:
    server = _server()
    bad = server.handle({"jsonrpc": "2.0", "id": 1, "method": "no/such"})
    assert bad["error"]["code"] == -32601
    miss = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": "nope"}})
    assert miss["error"]["code"] == -32602


# --- Fail-closed trust gate ------------------------------------------------- #
def test_default_gate_is_fail_closed() -> None:
    assert a2a.default_gate("").accept is False
    assert a2a.default_gate("a clear, verified answer with substance").accept is True
    assert a2a.default_gate("I cannot verify this claim").label == "abstain"
    assert a2a.default_gate("this is definitely AGI").label == "block"


def test_peer_output_passes_through_our_gate() -> None:
    # A peer whose runner emits an over-claim must NOT be trusted by the caller.
    def overclaim_runner(_task: str):
        return "This is definitely AGI and 100% accurate.", 0.0, 1, {"mock": True}

    server = a2a.A2AServer(a2a.sophia_agent_card("http://x.local"), runner=overclaim_runner)
    client = a2a.A2AClient(a2a.StubA2ATransport(server))
    verdict = client.delegate_to_peer("anything")
    assert verdict.accept is False
    assert verdict.label == "block"


def test_trusted_peer_output_is_accepted() -> None:
    def good_runner(_task: str):
        return "A substantive, verified, well-grounded answer.", 0.0, 2, {"mock": True}

    client = a2a.A2AClient(a2a.StubA2ATransport(
        a2a.A2AServer(a2a.sophia_agent_card("http://x.local"), runner=good_runner)))
    verdict = client.delegate_to_peer("anything")
    assert verdict.accept is True


# --- Auth ------------------------------------------------------------------- #
def test_require_auth_blocks_without_key_allows_with_key() -> None:
    server = _server(require_auth=True, api_keys=frozenset({"secret"}))
    # card is public even with auth on
    assert "result" in server.handle({"id": 1, "method": "agent/getCard"})
    # a task without the key is rejected
    no_key = a2a.A2AClient(a2a.StubA2ATransport(server))
    bad = no_key.transport.call("message/send", {"message": "hi"})
    assert bad["error"]["code"] == -32001
    # with the key it works
    with tempfile.TemporaryDirectory() as td:
        h.RUNS_DIR = Path(td)
        ok = a2a.A2AClient(a2a.StubA2ATransport(server, api_key="secret"))
        task = ok.send_task("a real task")
        assert task.state == a2a.COMPLETED


# --- MCP-as-agent bridge ---------------------------------------------------- #
def test_mcp_as_agent_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        h.RUNS_DIR = Path(td)
        client = a2a.A2AClient(a2a.StubA2ATransport(_server()))
        bridge = a2a.A2AMcpTransport(client)
        out = bridge.call("sophia.delegate", {"task": "summarise the discipline"})
        assert "text" in out and "sources" in out
        assert out["sources"] == ["a2a:remote-sophia"]
        assert isinstance(out["accept"], bool)


def test_mcp_bridge_rejects_unknown_tool() -> None:
    bridge = a2a.A2AMcpTransport(a2a.A2AClient(a2a.StubA2ATransport(_server())))
    try:
        bridge.call("sophia.unknown", {})
        assert False, "should reject unknown tool"
    except RuntimeError:
        pass


def test_mcp_tool_specs_are_gated() -> None:
    specs = a2a.agent_mcp_tool_specs()
    assert specs and specs[0]["id"] == "sophia.delegate"
    # peer output is verified before use (not UNCLASSIFIED, has a verifier)
    assert specs[0]["verifier_ref"] != "none"
    assert specs[0]["blp_level"] == "CONFIDENTIAL"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
