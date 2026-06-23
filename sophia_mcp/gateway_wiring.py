"""Wire the fail-closed gateway into the live MCP server's side-effecting tools.

Until now ``gateway/interceptor.py`` was tested as a library but had **zero
references in the served server** — the authz / firewall / kill-switch / BLP /
output-verify pipeline never ran on a real MCP call. This module closes that gap
for the tools where governance actually matters: the writes and external egress
(``wiki_upsert``, ``export_corpus``, ``web_evidence_search``, ``openclaw_infer``).
Read-only lookups stay direct, to limit blast radius and avoid false positives.

Opt-in: routing is active only when ``SOPHIA_MCP_GATEWAY=1`` (see
``boundary.gateway_enabled``), so the server's default behavior is unchanged.
When active, every governed call passes the gateway's **pre-dispatch governance**,
in order: served kill-switch → caller identity (env, never args) → per-tool
authz → injection scan on args → contract kill-switch → BLP no-read-up → dispatch.
Any of those that fails-closed **skips dispatch entirely** (no side effect). On
dispatch the tool's OWN internal gate decides its result; the gateway tags external
/ unverified output as ``untrusted`` (Biba low-integrity) so it cannot be laundered
into a high-integrity sink, and writes an audit span.

Output is NOT re-verified through the contract's claim store here: these tools
already self-gate (wiki_upsert's source-discipline gate, export's approval gate),
and tool outputs are not idempotent claims. The gateway shares the server's
contract singleton, so kill-switch and audit are one unified store.
"""

from __future__ import annotations

from gateway import Gateway, ToolEntry, firewall
from sophia_contract import blp, errors
from sophia_mcp import boundary
from sophia_mcp import tools_impl as impl

# The side-effecting / external tools that route through the gateway when enabled.
GOVERNED_TOOLS = (
    "sophia_wiki_upsert",
    "sophia_export_corpus",
    "sophia_web_evidence_search",
    "sophia_openclaw_infer",
)

_GATEWAY: "Gateway | None" = None


def _entries() -> "list[ToolEntry]":
    # verifier_ref="none" keeps the tools' own internal gating authoritative; the
    # gateway adds the served-surface pre-dispatch governance + taint label
    # (external/none output is tagged "untrusted" by gateway.firewall.taint_label).
    return [
        ToolEntry(
            id="sophia_wiki_upsert", handler=lambda a: impl.wiki_upsert(**a),
            side_effects="write", verifier_ref="none", blp_level="UNCLASSIFIED",
            description="Create/update an agent-owned wiki page (gated, audited write).",
        ),
        ToolEntry(
            id="sophia_export_corpus", handler=lambda a: impl.export_corpus(**a),
            side_effects="write", verifier_ref="none", blp_level="UNCLASSIFIED",
            description="Export training/examples/*.json to training/corpus.jsonl.",
        ),
        ToolEntry(
            id="sophia_web_evidence_search", handler=lambda a: impl.web_evidence_search(**a),
            side_effects="external", verifier_ref="none", blp_level="UNCLASSIFIED",
            description="Local RAG plus optional Brave/Tavily/SerpAPI web evidence.",
        ),
        ToolEntry(
            id="sophia_openclaw_infer", handler=lambda a: impl.openclaw_infer(**a),
            side_effects="external", verifier_ref="none", blp_level="UNCLASSIFIED",
            description="Read-only text inference via the local OpenClaw gateway CLI.",
        ),
    ]


def gateway() -> "Gateway":
    """Lazily build the singleton gateway, sharing the server's contract store."""
    global _GATEWAY
    if _GATEWAY is None:
        gw = Gateway(contract=impl._contract())
        for entry in _entries():
            gw.register(entry)
        _GATEWAY = gw
    return _GATEWAY


def reset() -> None:
    """Drop the cached gateway (tests that toggle env/contract call this)."""
    global _GATEWAY
    _GATEWAY = None


def _fail(err: dict, *, held: "str | None" = None) -> dict:
    gov = {"verdict": "unavailable" if held == "kill_switch" else "rejected", "held_reason": held}
    return {**err, "result": None, "_governance": gov}


def _held(reason: str, *, reasons=None, suggested_fix=None) -> dict:
    return {
        "result": None,
        "held_reason": reason,
        "reasons": reasons or [],
        "suggested_fix": suggested_fix,
        "_governance": {"verdict": "held", "held_reason": reason},
    }


def governed(tool_id: str, args: dict) -> dict:
    """Run the gateway's pre-dispatch governance, then dispatch, then tag + audit.

    A fail-closed check returns ``result: None`` with the error/held reason and
    **does not dispatch** (no side effect). On a pass the tool's own result dict is
    returned shape-preserved, annotated with a ``_governance`` block (verdict +
    Biba integrity label). This is the served-surface enforcement that was missing.
    """
    args = dict(args or {})
    entry = gateway().registry.get(tool_id)
    if entry is None:
        return _fail(errors.error("BAD_REQUEST", f"unknown governed tool {tool_id!r}"))

    # 1) Served kill switch (operator surface), before anything else.
    if boundary.kill_switch_engaged():
        return _fail(errors.error("UNAVAILABLE", "served kill switch engaged", retryable=True),
                     held="kill_switch")

    # 2) Caller identity from env (never from args), then per-tool authz.
    role, clearance = boundary.caller_identity()
    if entry.allowed_roles is not None and (role is None or role not in entry.allowed_roles):
        return _fail(errors.error("UNAUTHENTICATED", f"role {role!r} not permitted for {tool_id}"))

    # 3) Firewall: scan untrusted call args for injection/exfiltration markers.
    hits = firewall.scan_args(args)
    if hits:
        return _fail(errors.error("BAD_REQUEST", f"injection blocked by firewall: {hits}"))

    # 4) Contract kill switch + 5) BLP no-read-up (clearance must dominate the tool).
    contract = impl._contract()
    if contract.health().get("checks", {}).get("kill_switch_engaged"):
        return _fail(errors.error("UNAVAILABLE", "contract kill switch engaged", retryable=True),
                     held="kill_switch")
    read_up = blp.read_up_violation(clearance, entry.blp_level)
    if read_up:
        return _held("blp_violation", reasons=[read_up],
                     suggested_fix="request access at the tool's clearance level")

    # 6) Dispatch — the tool's own internal gate decides its result.
    try:
        result = entry.handler(args)
    except Exception as exc:  # transport / tool fault
        return _fail(errors.error("UNAVAILABLE", f"tool {tool_id} failed: {exc!r}", retryable=True))

    integrity = firewall.taint_label(entry.verifier_ref, entry.side_effects)
    try:  # 7) audit span (best-effort; never fail the call on a tracing hiccup)
        contract.tracer.span(
            "mcp.governed", input={"tool_id": tool_id, "role": role, "clearance": clearance},
            output={"integrity": integrity}, level="DEFAULT", metadata={"side_effects": entry.side_effects})
    except Exception:
        pass

    out = dict(result) if isinstance(result, dict) else {"result": result}
    out["_governance"] = {"verdict": "accepted", "integrity": integrity,
                          "role": role, "clearance": clearance}
    return out
