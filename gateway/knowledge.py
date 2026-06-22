"""Knowledge MCP (P3): expose the OKF belief graph as gated gateway tools.

Any agent can query the shared, provenance-tracked, self-healing knowledge base through
the same fail-closed pipeline. Outputs are grounding-verified, so an ungrounded belief
answer is held (never surfaced) like any other tool output.
"""

from __future__ import annotations

from gateway.registry import ToolEntry


def register_knowledge_tools(gateway, *, blp_level: str = "UNCLASSIFIED") -> "list[ToolEntry]":
    """Register kb.belief / kb.counterfactual as gated gateway tools backed by okf."""
    from sophia_mcp import tools_impl

    def _belief(args):
        out = tools_impl.belief(args["entity"])
        # surface a grounding source so the grounding verifier can accept a found belief
        sources = ["okf://belief-graph"] if out.get("found") else []
        return {"answer": out, "sources": sources}

    def _counterfactual(args):
        out = tools_impl.counterfactual(args["source"], args.get("query"))
        return {"answer": out, "sources": ["okf://belief-graph"]}

    entries = [
        ToolEntry(id="kb.belief", handler=_belief, kind="native", verifier_ref="grounding",
                  blp_level=blp_level, side_effects="read",
                  description="OKF belief-graph lookup (effective confidence, contradictions)"),
        ToolEntry(id="kb.counterfactual", handler=_counterfactual, kind="native",
                  verifier_ref="grounding", blp_level=blp_level, side_effects="read",
                  description="OKF counterfactual: what if a source were removed"),
    ]
    return [gateway.register(e) for e in entries]
