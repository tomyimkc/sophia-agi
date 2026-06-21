"""Planner — turn a trusted request into a validated PLAN for the interpreter (M2.3).

The privileged planner (a deterministic template planner, or a real P-LLM via the
model adapter) reads ONLY the trusted user request + the tool schemas — never any
untrusted data — and emits a plan. `parse_plan` validates the plan into the
constrained step set, allowing only known ops and manifest tools (a `retrieve` must
be a READ tool), and fails CLOSED (`PlanError`) on anything malformed.

Scope — what parse_plan does and does NOT guarantee (honest):
  - It DOES stop an unknown/dangerous tool, an unknown op, a write disguised as a
    read, and malformed shapes. It cannot be made to admit a non-manifest tool.
  - It does NOT, by itself, stop a *well-formed* Call to a legitimate WRITE/EGRESS
    tool whose args are trusted Consts — that is allowed by design (trusted input
    into a sink is fine under the lethal-trifecta rule). Safety against a *malicious
    planner* therefore rests on the **request/planner being the trust root**
    (see interpreter.py). If requests can be attacker-influenced, run the
    interpreter with ``approve_sinks=True`` so every WRITE/EGRESS call needs
    explicit human approval regardless of taint.
  - Untrusted *data* still cannot escalate: it can never become a plan step
    (control-flow integrity) nor reach a sink untainted (the firewall).

Plan JSON shape (a list of steps), e.g.:
    [{"op": "const",    "var": "q",   "value": "who wrote it"},
     {"op": "retrieve", "var": "doc", "tool": "sophia_wiki_read", "query": "q"},
     {"op": "extract",  "var": "sum", "src": "doc", "instruction": "summarize"},
     {"op": "concat",   "var": "ans", "parts": ["Answer: ", "sum"]},
     {"op": "call",     "var": "w",   "tool": "sophia_wiki_upsert", "args": ["ans"]}]
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.dataflow.capabilities import Effect
from agent.dataflow.interpreter import Call, Concat, Const, Extract, Retrieve
from agent.dataflow.manifest import TOOL_CAPS, cap_for

_OPS = {"const", "retrieve", "extract", "concat", "call"}


class PlanError(ValueError):
    """A planner emitted an invalid/unsafe plan; the validator fails closed."""


def parse_plan(spec: Any, *, allowed_tools: "set | None" = None) -> list:
    """Validate a plan spec (JSON string or list of dict steps) into typed Steps.

    Only known ops and known tools are accepted (default: the manifest); a
    ``retrieve`` must name a READ tool; ``var`` must be a non-empty string. Anything
    malformed raises :class:`PlanError` — the planner cannot smuggle an unknown
    tool, an unknown op, or a write disguised as a read.
    """
    allowed = set(allowed_tools) if allowed_tools is not None else set(TOOL_CAPS)
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise PlanError(f"plan is not valid JSON: {exc}") from exc
    if not isinstance(spec, list):
        raise PlanError("plan must be a list of steps")

    steps: list = []
    for i, raw in enumerate(spec):
        if not isinstance(raw, dict):
            raise PlanError(f"step {i}: not an object")
        op = str(raw.get("op", "")).lower()
        if op not in _OPS:
            raise PlanError(f"step {i}: unknown op {raw.get('op')!r}")
        var = raw.get("var")
        if not isinstance(var, str) or not var:
            raise PlanError(f"step {i}: 'var' must be a non-empty string")

        if op == "const":
            steps.append(Const(var, raw.get("value")))
        elif op == "retrieve":
            tool = raw.get("tool")
            if not isinstance(tool, str) or tool not in allowed:
                raise PlanError(f"step {i}: tool {tool!r} not in allowed set")
            if cap_for(tool).effect != Effect.READ:
                raise PlanError(f"step {i}: retrieve must name a READ tool, not {tool!r}")
            steps.append(Retrieve(var, tool, raw.get("query", "")))
        elif op == "extract":
            steps.append(Extract(var, str(raw.get("src", "")), str(raw.get("instruction", ""))))
        elif op == "concat":
            parts = raw.get("parts")
            if not isinstance(parts, list):
                raise PlanError(f"step {i}: concat 'parts' must be a list")
            steps.append(Concat(var, [str(p) for p in parts]))
        elif op == "call":
            tool = raw.get("tool")
            if not isinstance(tool, str) or tool not in allowed:
                raise PlanError(f"step {i}: tool {tool!r} not in allowed set")
            args = raw.get("args", [])
            if not isinstance(args, list):
                raise PlanError(f"step {i}: call 'args' must be a list")
            steps.append(Call(var, tool, [str(a) for a in args]))
    return steps


# --------------------------------------------------------------------------- #
# Planners: request -> plan. The planner sees only the (trusted) request.
# --------------------------------------------------------------------------- #

Planner = Callable[[str], list]   # (request) -> [Step]


def template_planner(*, allowed_tools: "set | None" = None) -> Planner:
    """A deterministic, offline planner for a few task shapes (CI-friendly).

    It NEVER consults untrusted data — it keys only on the trusted request — and
    routes its output through :func:`parse_plan`, so it obeys the same trust
    boundary as a model planner.
    """

    def plan(request: str) -> list:
        r = (request or "").lower()
        if "save" in r or "write" in r or "upsert" in r:
            spec = [
                {"op": "const", "var": "q", "value": request},
                {"op": "retrieve", "var": "doc", "tool": "sophia_wiki_read", "query": "q"},
                {"op": "extract", "var": "sum", "src": "doc", "instruction": "summarize"},
                {"op": "call", "var": "w", "tool": "sophia_wiki_upsert", "args": ["sum"]},
            ]
        else:  # default: retrieve + summarise (no sink)
            spec = [
                {"op": "const", "var": "q", "value": request},
                {"op": "retrieve", "var": "doc", "tool": "sophia_wiki_read", "query": "q"},
                {"op": "extract", "var": "sum", "src": "doc", "instruction": "summarize"},
            ]
        return parse_plan(spec, allowed_tools=allowed_tools)

    return plan


def model_planner(spec: "str | None" = None, *, allowed_tools: "set | None" = None) -> Planner:
    """A real privileged-planner LLM via the model adapter. It is shown ONLY the
    request + the allowed tools; its JSON output is validated by :func:`parse_plan`
    (fail-closed). Offline-testable with the mock provider (set SOPHIA_MOCK_RESPONSE
    to a plan JSON)."""
    from agent.model import default_client

    client = default_client(spec)
    tools = sorted(allowed_tools) if allowed_tools is not None else sorted(TOOL_CAPS)
    system = (
        "You are a PRIVILEGED PLANNER. Output ONLY a JSON array of steps for a "
        "data-flow interpreter. Ops: const, retrieve, extract, concat, call. Use "
        "variables (never inline fetched data). Allowed tools: " + ", ".join(tools) + ". "
        "Do not invent tools. Retrieve must use a read tool."
    )

    def plan(request: str) -> list:
        out = getattr(client.generate(system, request), "text", "") or ""
        return parse_plan(out, allowed_tools=allowed_tools)

    return plan
