# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drop the governance gate into a LangGraph pipeline.

The node factories are plain ``state -> state`` callables (the LangGraph node
contract), so they work in any LangGraph ``StateGraph`` *and* run without LangGraph
installed. ``build_contract_graph`` returns a compiled LangGraph when the library is
present; ``run_contract_flow`` is a dependency-free executor of the same
record → verify → route logic for CI and for orchestrators other than LangGraph.

State keys (a plain dict):
  in:  idempotency_key, content, sources[], parents[], blp_level, role?, clearance?
  out: claim, verdict, publishable (bool), route ("publish" | "review" | "reject")

    from sophia_contract import SophiaContract
    from sophia_contract.langgraph_nodes import run_contract_flow
    final = run_contract_flow(SophiaContract(), {
        "idempotency_key": "k1", "content": "...", "sources": ["s1"]})
    final["route"]  # -> "publish" | "review" | "reject"
"""

from __future__ import annotations

from sophia_contract.service import SophiaContract


def make_record_node(contract: "SophiaContract"):
    """Node: record_claim from state, store the Claim (or error) on the state."""

    def record(state: dict) -> dict:
        req = {
            "idempotency_key": state["idempotency_key"],
            "content": state["content"],
            "sources": state.get("sources", []),
            "parents": state.get("parents", []),
            "blp_level": state.get("blp_level", "UNCLASSIFIED"),
        }
        if state.get("role"):
            req["role"] = state["role"]
        result = contract.record_claim(req)
        if "error" in result:
            return {**state, "error": result["error"], "route": "reject"}
        return {**state, "claim": result}

    return record


def make_verify_node(contract: "SophiaContract"):
    """Node: verify the recorded claim, set verdict / publishable / route."""

    def verify(state: dict) -> dict:
        if state.get("error") or not state.get("claim"):
            return {**state, "route": "reject"}
        req = {"claim_id": state["claim"]["claim_id"]}
        if state.get("role"):
            req["role"] = state["role"]
        verdict = contract.verify_claim(req, clearance=state.get("clearance")
                                        or state.get("blp_level", "UNCLASSIFIED"))
        if "error" in verdict:
            return {**state, "error": verdict["error"], "route": "reject"}
        accepted = verdict["verdict"] == "accepted"
        return {**state, "verdict": verdict, "publishable": accepted,
                "route": route_after_verify(verdict)}

    return verify


def route_after_verify(verdict: dict) -> str:
    """Conditional edge: only 'accepted' may publish; 'held' goes to human review;
    everything else is rejected. Fail-closed."""
    v = verdict.get("verdict")
    if v == "accepted":
        return "publish"
    if v == "held":
        return "review"
    return "reject"


def run_contract_flow(contract: "SophiaContract", state: dict) -> dict:
    """Dependency-free executor: record -> verify -> route. Same semantics a LangGraph
    of these nodes would have; usable in CI and non-LangGraph orchestrators."""
    state = make_record_node(contract)(dict(state))
    state = make_verify_node(contract)(state)
    return state


def build_contract_graph(contract: "SophiaContract"):
    """Compile a real LangGraph StateGraph (record -> verify -> {publish|review|reject})
    when langgraph is installed. Raises ImportError otherwise — use run_contract_flow
    for the dependency-free path."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised only with langgraph present
        raise ImportError("langgraph is not installed; use run_contract_flow() instead") from exc

    graph = StateGraph(dict)
    graph.add_node("record", make_record_node(contract))
    graph.add_node("verify", make_verify_node(contract))
    graph.add_node("publish", lambda s: {**s, "published": s.get("publishable", False)})
    graph.add_node("review", lambda s: {**s, "awaiting_human": True})
    graph.add_node("reject", lambda s: {**s, "rejected": True})
    graph.add_edge(START, "record")
    graph.add_edge("record", "verify")
    graph.add_conditional_edges("verify", lambda s: s.get("route", "reject"),
                                {"publish": "publish", "review": "review", "reject": "reject"})
    for terminal in ("publish", "review", "reject"):
        graph.add_edge(terminal, END)
    return graph.compile()
