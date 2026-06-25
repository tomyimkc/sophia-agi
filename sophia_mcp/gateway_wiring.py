# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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

On the served surface the output is **re-verified** through Sophia's epistemic gate
when ``SOPHIA_MCP_OUTPUT_VERIFY=1`` (opt-in, default off): the tool's returned text is
re-checked for fabricated attributions / bad citations, and a violating payload is
**withheld** (fail-closed) rather than laundered back to the caller. This is
defense-in-depth — the tools already self-gate (wiki_upsert's source-discipline gate,
export's approval gate), but the inference/search tools return external or
model-generated text whose *content* the pre-dispatch checks never inspect. Honest
bound: it is a read-back integrity check on the served payload — it cannot roll back a
side effect that already completed (write tools self-gate *before* mutating); its primary
targets are the external/inference reads. The gateway shares the server's contract
singleton, so kill-switch and audit are one unified store.
"""

from __future__ import annotations

from gateway import Gateway, ToolEntry, firewall
from sophia_contract import blp, errors
from sophia_mcp import boundary
from sophia_mcp import tools_impl as impl

# Result keys whose values are free text a model/web source produced — the laundering risk.
_OUTPUT_TEXT_KEYS = ("text", "answer", "response", "summary", "content")

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


def _output_text(result: dict) -> str:
    """Concatenate the free-text fields of a tool result that a model/web source produced.

    Pulls the known text keys plus any ``results``/``evidence`` snippet text, so the
    re-verifier inspects the *content* served back (where a fabricated attribution would
    live), not the structural metadata.
    """
    if not isinstance(result, dict):
        return str(result or "")
    parts: list[str] = []
    for key in _OUTPUT_TEXT_KEYS:
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    for coll_key in ("results", "evidence", "sources"):
        coll = result.get(coll_key)
        if isinstance(coll, list):
            for item in coll:
                if isinstance(item, dict):
                    for k in ("text", "snippet", "excerpt", "title"):
                        v = item.get(k)
                        if isinstance(v, str) and v.strip():
                            parts.append(v)
                elif isinstance(item, str):
                    parts.append(item)
    return "\n".join(parts).strip()


def verify_output(tool_id: str, args: dict, out: dict) -> dict:
    """Re-verify a governed tool's served output, fail-closed, when output-verify is on.

    Runs the served free text back through ``agent.gate.check_response`` and keys on its
    substantive **violations** (fabricated attribution / bad citation / numeric) — NOT the
    style warnings, which would wrongly fail arbitrary tool output. On a violation the
    payload body is **withheld** and the call is marked held; otherwise the result is
    annotated ``outputVerified: True``. A no-text or already-failed result is passed through
    untouched (nothing to launder). Best-effort: a gate fault never crashes the call.
    """
    if not boundary.output_verify_enabled():
        return out
    if not isinstance(out, dict) or out.get("result") is None and out.get("error"):
        return out
    text = _output_text(out)
    gov = out.get("_governance") or {}
    if not text:
        gov["outputVerified"] = None  # nothing inspectable
        out["_governance"] = gov
        return out
    try:
        from agent.gate import check_response

        question = (args.get("question") or args.get("query") or args.get("prompt") or "")
        gate = check_response(text, mode="advisor", question=question or None,
                              strict_attribution=True)
        violations = list(gate.get("violations") or [])
    except Exception:
        # Fail-closed posture, but do not crash: flag unverifiable rather than withhold.
        gov["outputVerified"] = "error"
        out["_governance"] = gov
        return out

    if violations:
        return {
            "result": None,
            "held_reason": "output_failed_reverification",
            "reasons": violations,
            "suggested_fix": "the served output asserted an unverified attribution/citation; "
                             "re-answer with source-grounded support or abstain",
            "_governance": {**gov, "verdict": "held",
                            "held_reason": "output_failed_reverification",
                            "outputVerified": False},
        }
    gov["outputVerified"] = True
    out["_governance"] = gov
    return out


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

    # 8) Served-output re-verification (opt-in): re-check the returned text through the
    #    epistemic gate so fabricated attributions cannot be laundered back. Fail-closed.
    return verify_output(tool_id, args, out)
