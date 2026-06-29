# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A2A (Agent2Agent) interoperability layer for the Sophia harness.

There are two complementary layers in the agent stack — this module adds the second:

    MCP  (gateway/federation.py)  — agent -> TOOL    (Sophia already speaks this)
    A2A  (this module)            — agent -> AGENT   (the inter-agent coordination layer)

It makes Sophia's in-process swarm (``agent.swarm_router`` -> ``agent.subagent.delegate``)
reachable as a networked **A2A peer**: it publishes an **Agent Card** (discovery), accepts
tasks over a JSON-RPC-shaped envelope with the standard A2A **task lifecycle**
(submitted -> working -> completed/failed/canceled), and runs each task through the existing
fan-out + fail-closed reduce. Symmetrically it can act as a **client** to other A2A agents
(yours, or third-party LLM agents) via a pluggable transport.

Mirrors A2A v1.x as governed by the Linux Foundation (JSON-RPC 2.0 over HTTP, Agent Card
discovery, task states). This is a *spike* of the contract, not a full SDK: the wire shapes
are A2A-compatible so a real ``a2a-sdk`` peer can talk to it, but the runtime is Sophia's.

Sophia discipline (same as every ``agent/*`` module):
  * **deterministic + offline-testable** — :class:`StubA2ATransport` needs no network, key,
    or real model; the whole client<->server round trip runs under the ``mock`` model client.
  * **fail-closed** — a peer's output is UNTRUSTED external data. It passes through a gate
    hook (:func:`default_gate`, swap in the real epistemic gate) before this agent acts on
    it; an unverified/garbage response yields ABSTAIN, never silent trust.
  * **least privilege** — the Agent Card advertises only scoped skills; an inbound task is
    lowered to a least-privilege ``SwarmPlan``/``SubagentSpec`` and never widens tool scope.
  * **honest cost** — every task result carries the swarm's ``est_cost_steps`` and the
    actual delegation cost; nothing is hidden.

Quick start (offline)::

    from agent.a2a import A2AServer, A2AClient, StubA2ATransport, sophia_agent_card
    from agent import model as m
    server = A2AServer(sophia_agent_card("http://spark.local:8080"),
                       client=m.ModelClient(m.resolve_config("mock")))
    client = A2AClient(StubA2ATransport(server))
    card = client.get_card()                       # discovery
    task = client.send_task("validate this attribution: ...")   # delegate
    verdict = client.delegate_to_peer("...")       # delegate + run answer through OUR gate
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

A2A_PROTOCOL_VERSION = "1.0"
AGENT_CARD_SCHEMA = "sophia.agent_card.v1"
AGENT_CARD_SCHEMA_VERSION = "1.0.0"

# A2A standard task lifecycle states (the subset Sophia uses).
SUBMITTED = "submitted"
WORKING = "working"
INPUT_REQUIRED = "input-required"
COMPLETED = "completed"
FAILED = "failed"
CANCELED = "canceled"
TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELED})


# --------------------------------------------------------------------------- #
# Agent Card (discovery)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentSkill:
    """One advertised capability on the Agent Card. Maps to a Sophia entrypoint."""

    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = ("text/plain",)
    output_modes: tuple[str, ...] = ("text/plain", "application/json")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "inputModes": list(self.input_modes),
            "outputModes": list(self.output_modes),
        }


@dataclass(frozen=True)
class AgentCard:
    """Sophia's self-description, served at ``/.well-known/agent-card.json`` (A2A discovery).

    Only *scoped* skills are listed (least privilege) — a peer cannot ask for a capability
    that is not advertised. ``provenance_discipline`` is a Sophia-specific extension flag so
    peers know responses are verifier-gated and may ABSTAIN rather than fabricate.
    """

    name: str
    description: str
    url: str
    version: str = "0.1.0"
    protocol_version: str = A2A_PROTOCOL_VERSION
    skills: tuple[AgentSkill, ...] = ()
    streaming: bool = False
    push_notifications: bool = False
    security_schemes: tuple[str, ...] = ("apiKey",)  # apiKey | oauth2 | mtls | none
    provider: str = "Sophia-AGI"
    provenance_discipline: bool = True

    def skill_ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.skills)

    def to_dict(self) -> dict:
        return {
            "schema": AGENT_CARD_SCHEMA,
            "schemaVersion": AGENT_CARD_SCHEMA_VERSION,
            "protocolVersion": self.protocol_version,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "provider": {"organization": self.provider},
            "capabilities": {
                "streaming": self.streaming,
                "pushNotifications": self.push_notifications,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "securitySchemes": list(self.security_schemes),
            "skills": [s.to_dict() for s in self.skills],
            # Sophia extension: peers can rely on the fail-closed / abstaining contract.
            "x-sophia": {"provenanceDiscipline": self.provenance_discipline,
                         "reduce": "fail_closed_synthesis"},
        }

    def validate(self) -> tuple[bool, list[str]]:
        """Deterministic structural check (no network). Returns (ok, problems)."""
        problems: list[str] = []
        if not self.name:
            problems.append("name is empty")
        if not (self.url.startswith("http://") or self.url.startswith("https://")):
            problems.append(f"url must be http(s): {self.url!r}")
        if not self.skills:
            problems.append("no skills advertised (a peer could not discover any capability)")
        if "none" in self.security_schemes and len(self.security_schemes) > 1:
            problems.append("security scheme 'none' cannot be combined with others")
        seen: set[str] = set()
        for s in self.skills:
            if s.id in seen:
                problems.append(f"duplicate skill id {s.id!r}")
            seen.add(s.id)
        return (not problems, problems)


# The skills Sophia advertises — each is verifier-gated and may abstain.
SOPHIA_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        id="swarm.delegate",
        name="Delegate a task to the Sophia swarm",
        description=("Route a task through the Agentic-MoE swarm-router (solo-vs-fan-out, "
                     "least-privilege teams) and return a fail-closed synthesised answer."),
        tags=("delegation", "swarm", "reasoning"),
    ),
    AgentSkill(
        id="provenance.validate",
        name="Validate an attribution / provenance claim",
        description=("Check a citation/attribution against the corpus discipline; abstains or "
                     "blocks on doNotAttributeTo / uncertain authorship rather than fabricating."),
        tags=("provenance", "attribution", "verification"),
    ),
    AgentSkill(
        id="epistemic.gate_check",
        name="Run a draft answer through the epistemic gate",
        description=("Accept / abstain / block a candidate answer for attribution traps and "
                     "over-claims (the no-overclaim contract)."),
        tags=("gate", "calibration", "safety"),
    ),
)


def sophia_agent_card(url: str, *, name: str = "Sophia", version: str = "0.1.0",
                      skills: "tuple[AgentSkill, ...] | None" = None,
                      streaming: bool = False, push_notifications: bool = False,
                      security_schemes: tuple[str, ...] = ("apiKey",)) -> AgentCard:
    """Build Sophia's Agent Card from its real, scoped capabilities."""
    return AgentCard(
        name=name,
        description=("Provenance-aware, verifier-gated reasoning agent. Abstains instead of "
                     "fabricating; every answer is fail-closed synthesised. NOT AGI."),
        url=url,
        version=version,
        skills=skills if skills is not None else SOPHIA_SKILLS,
        streaming=streaming,
        push_notifications=push_notifications,
        security_schemes=security_schemes,
    )


# --------------------------------------------------------------------------- #
# Fail-closed gate hook (peer output is untrusted external data)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateVerdict:
    """The decision about whether to TRUST a peer's (or our own draft) answer."""

    accept: bool
    label: str          # "accept" | "abstain" | "block"
    reason: str
    text: str = ""      # the answer, echoed for convenience

    def to_dict(self) -> dict:
        return {"accept": self.accept, "label": self.label, "reason": self.reason}


# Markers a conservative gate treats as "do not present as a settled answer".
_ABSTAIN_MARKERS = ("insufficient verified basis", "cannot verify", "i don't know",
                    "no sources", "abstain", "unable to")
_OVERCLAIM_MARKERS = ("definitely agi", "100% accurate", "0 hallucination",
                      "guaranteed correct", "proven agi")


def default_gate(text: str) -> GateVerdict:
    """Deterministic, conservative trust gate for offline use and as a safe default.

    This is intentionally simple and FAIL-CLOSED — empty/short output abstains, over-claim
    markers block. In production, swap this for the real epistemic gate (e.g. the
    ``sophia_gate_check`` MCP tool or ``agent.grounded_*``) via the ``gate=`` argument on
    :class:`A2AServer` / :class:`A2AClient`. The *contract* (GateVerdict) does not change.
    """
    t = (text or "").strip()
    low = t.lower()
    if len(t) < 3:
        return GateVerdict(False, "abstain", "empty or trivial response", t)
    if any(m in low for m in _OVERCLAIM_MARKERS):
        return GateVerdict(False, "block", "over-claim marker present", t)
    if any(m in low for m in _ABSTAIN_MARKERS):
        return GateVerdict(False, "abstain", "peer abstained / no verified basis", t)
    return GateVerdict(True, "accept", "passes conservative trust gate", t)


# --------------------------------------------------------------------------- #
# Task object
# --------------------------------------------------------------------------- #
@dataclass
class A2ATask:
    """An A2A task with its lifecycle state and result artifacts."""

    id: str
    context_id: str
    state: str
    input_text: str
    artifacts: list[dict] = field(default_factory=list)
    error: str = ""
    cost_usd: float = 0.0
    est_cost_steps: int = 0
    gate: "GateVerdict | None" = None

    def answer(self) -> str:
        """The first text artifact, or empty string."""
        for a in self.artifacts:
            if a.get("kind") == "text":
                return a.get("text", "")
        return ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.state},
            "artifacts": self.artifacts,
            "error": self.error or None,
            "metadata": {
                "costUsd": round(self.cost_usd, 6),
                "estCostSteps": self.est_cost_steps,
                "gate": self.gate.to_dict() if self.gate else None,
            },
        }


# A runner maps an inbound task string -> (answer_text, cost_usd, est_cost_steps, raw).
Runner = Callable[[str], tuple]


def _sophia_swarm_runner(client: Any) -> Runner:
    """Default runner: route through the swarm and return its fail-closed synthesis."""

    def _run(task_text: str) -> tuple:
        from agent import swarm_router as sr  # local import: keep a2a import cheap/offline
        plan, result = sr.run_swarm(task_text, client=client, parent_id="a2a")
        return result.synthesis, result.total_cost_usd, plan.est_cost_steps, result.to_dict()

    return _run


# --------------------------------------------------------------------------- #
# Server side — Sophia AS an A2A peer
# --------------------------------------------------------------------------- #
class A2AServer:
    """Serve Sophia's swarm behind the A2A contract.

    ``handle(request)`` dispatches a JSON-RPC-shaped request and returns a JSON-RPC-shaped
    response, so it can sit behind any HTTP handler (or be called directly in tests). Tasks
    are kept in an in-memory store and are pollable (``tasks/get``) — the interface is
    async-shaped even though v1 executes inline. For genuinely long-running work, swap the
    inline ``_execute`` for an enqueue onto a bus/worker (see docs/09-Agent/A2A-Sophia-Interop.md).
    """

    def __init__(self, card: AgentCard, *, runner: "Runner | None" = None,
                 client: Any | None = None, gate: Callable[[str], GateVerdict] = default_gate,
                 require_auth: bool = False, api_keys: "frozenset[str] | None" = None):
        ok, problems = card.validate()
        if not ok:
            raise ValueError(f"invalid Agent Card: {problems}")
        self.card = card
        self.gate = gate
        self.require_auth = require_auth
        self.api_keys = api_keys or frozenset()
        self._runner = runner or _sophia_swarm_runner(client)
        self._tasks: dict[str, A2ATask] = {}
        self._seq = 0

    # -- public JSON-RPC dispatch -------------------------------------------
    def handle(self, request: dict, *, auth: "str | None" = None) -> dict:
        rid = request.get("id", 1)
        method = request.get("method", "")
        params = request.get("params") or {}
        try:
            if self.require_auth and method != "agent/getCard":
                if auth not in self.api_keys:
                    return self._err(rid, -32001, "unauthorized (bad or missing API key)")
            if method in ("agent/getCard", "agent/getAuthenticatedExtendedCard"):
                return self._ok(rid, self.card.to_dict())
            if method == "message/send":
                return self._ok(rid, self._message_send(params).to_dict())
            if method == "tasks/get":
                t = self._tasks.get(params.get("id", ""))
                if t is None:
                    return self._err(rid, -32602, "unknown task id")
                return self._ok(rid, t.to_dict())
            if method == "tasks/cancel":
                return self._ok(rid, self._cancel(params.get("id", "")).to_dict())
            return self._err(rid, -32601, f"method not found: {method!r}")
        except Exception as exc:  # never leak a stack trace to a peer; fail-closed
            return self._err(rid, -32603, f"internal error: {type(exc).__name__}")

    # -- task lifecycle ------------------------------------------------------
    def _message_send(self, params: dict) -> A2ATask:
        text = _extract_text(params.get("message", params))
        self._seq += 1
        tid = f"task-{self._seq}"
        ctx = params.get("contextId") or f"ctx-{self._seq}"
        task = A2ATask(id=tid, context_id=ctx, state=SUBMITTED, input_text=text)
        self._tasks[tid] = task
        if not text.strip():
            # fail-closed: an empty task does not spawn a swarm.
            task.state = FAILED
            task.error = "empty task: nothing to do"
            return task
        return self._execute(task)

    def _execute(self, task: A2ATask) -> A2ATask:
        task.state = WORKING
        answer, cost, est_steps, raw = self._runner(task.input_text)
        task.cost_usd = float(cost or 0.0)
        task.est_cost_steps = int(est_steps or 0)
        # Gate OUR OWN answer before we hand it to a peer — abstain honestly if it fails.
        verdict = self.gate(answer)
        task.gate = verdict
        # Inbound A2A task -> our answer, recorded as a peer message (gate verdict attached).
        from agent.thinking_trace import maybe_record_a2a

        maybe_record_a2a(
            sender=f"peer:{task.context_id}", receiver=self.card.name, prompt=task.input_text,
            response=answer, ok=verdict.accept, gate=verdict.label, cost_usd=task.cost_usd, kind="peer",
        )
        task.artifacts = [
            {"kind": "text", "text": answer if verdict.accept else
             f"[abstain:{verdict.label}] {verdict.reason}"},
            {"kind": "data", "name": "sophia.delegation", "data": raw},
        ]
        task.state = COMPLETED  # A2A has no "abstain" state; the gate verdict carries it.
        return task

    def _cancel(self, tid: str) -> A2ATask:
        t = self._tasks.get(tid)
        if t is None:
            raise KeyError(tid)
        if t.state not in TERMINAL_STATES:
            t.state = CANCELED
        return t

    # -- JSON-RPC envelopes --------------------------------------------------
    @staticmethod
    def _ok(rid, result) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def _err(rid, code, message) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _extract_text(message: Any) -> str:
    """Pull plain text out of an A2A message (parts[].text) or a bare string/dict."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if "text" in message and isinstance(message["text"], str):
            return message["text"]
        parts = message.get("parts")
        if isinstance(parts, list):
            return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    return ""


# --------------------------------------------------------------------------- #
# Client side — Sophia (or anything) CALLING an A2A peer
# --------------------------------------------------------------------------- #
class StubA2ATransport:
    """Offline transport: routes JSON-RPC requests straight into a local :class:`A2AServer`.

    Mirrors ``gateway.federation.StubTransport`` so the whole client<->server path is
    testable with no network, key, or real model.
    """

    def __init__(self, server: A2AServer, *, api_key: "str | None" = None):
        self.server = server
        self.api_key = api_key
        self.calls: list[dict] = []

    def call(self, method: str, params: dict) -> dict:
        req = {"jsonrpc": "2.0", "id": len(self.calls) + 1, "method": method, "params": params}
        self.calls.append(req)
        return self.server.handle(req, auth=self.api_key)


class HttpA2ATransport:
    """A2A JSON-RPC over HTTP(S). Used only live (mirrors HttpMcpTransport)."""

    def __init__(self, base_url: str, *, api_key: "str | None" = None, timeout_sec: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def call(self, method: str, params: dict) -> dict:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(self.base_url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"A2A call failed: {exc!r}") from exc


class A2AClient:
    """Talk to an A2A peer through a transport, then TRUST its output only via our gate."""

    def __init__(self, transport, *, gate: Callable[[str], GateVerdict] = default_gate):
        self.transport = transport
        self.gate = gate

    def _rpc(self, method: str, params: "dict | None" = None) -> dict:
        resp = self.transport.call(method, params or {})
        if "error" in resp and resp["error"]:
            raise RuntimeError(f"peer error {resp['error'].get('code')}: {resp['error'].get('message')}")
        return resp.get("result", {})

    def get_card(self) -> dict:
        """Discovery: fetch the peer's Agent Card."""
        return self._rpc("agent/getCard")

    def send_task(self, text: str, *, context_id: "str | None" = None) -> A2ATask:
        """Delegate a task and return the (terminal or pollable) task object."""
        params: dict = {"message": {"role": "user", "parts": [{"kind": "text", "text": text}]}}
        if context_id:
            params["contextId"] = context_id
        return _task_from_dict(self._rpc("message/send", params))

    def get_task(self, task_id: str) -> A2ATask:
        return _task_from_dict(self._rpc("tasks/get", {"id": task_id}))

    def cancel_task(self, task_id: str) -> A2ATask:
        return _task_from_dict(self._rpc("tasks/cancel", {"id": task_id}))

    def delegate_to_peer(self, text: str) -> GateVerdict:
        """High-level: delegate, then run the peer's answer through OUR gate (fail-closed).

        This is the trust boundary — a remote agent's claims are untrusted external data, so
        we never act on them until our own gate accepts. Returns the gate verdict (with the
        peer's text attached), or an abstain verdict if the task did not complete cleanly.
        """
        task = self.send_task(text)
        if task.state != COMPLETED:
            return GateVerdict(False, "abstain", f"peer task did not complete (state={task.state})",
                               task.error)
        # If the peer's OWN gate already rejected, propagate that verdict — never upgrade a
        # peer's abstain/block to trust. Otherwise re-gate locally (untrusted external data).
        from agent.thinking_trace import maybe_record_a2a

        if task.gate is not None and not task.gate.accept:
            propagated = GateVerdict(False, task.gate.label, f"peer gate: {task.gate.reason}", task.answer())
            maybe_record_a2a(sender="self", receiver="peer", prompt=text, response=task.answer(),
                             ok=False, gate=propagated.label, cost_usd=task.cost_usd, kind="peer")
            return propagated
        local = self.gate(task.answer())
        maybe_record_a2a(sender="self", receiver="peer", prompt=text, response=task.answer(),
                         ok=local.accept, gate=local.label, cost_usd=task.cost_usd, kind="peer")
        return local


def _task_from_dict(d: dict) -> A2ATask:
    status = (d.get("status") or {})
    meta = (d.get("metadata") or {})
    gate_d = meta.get("gate")
    return A2ATask(
        id=d.get("id", ""),
        context_id=d.get("contextId", ""),
        state=status.get("state", FAILED),
        input_text="",
        artifacts=d.get("artifacts", []) or [],
        error=d.get("error") or "",
        cost_usd=float(meta.get("costUsd", 0.0) or 0.0),
        est_cost_steps=int(meta.get("estCostSteps", 0) or 0),
        gate=(GateVerdict(gate_d["accept"], gate_d["label"], gate_d["reason"]) if gate_d else None),
    )


# --------------------------------------------------------------------------- #
# MCP-as-agent bridge (option 2: reach a remote agent through the MCP path)
# --------------------------------------------------------------------------- #
# Sophia already federates MCP servers (gateway.federation). This bridge lets a remote
# Sophia A2A peer be registered as an ordinary *gated MCP tool* (e.g. ``sophia.delegate``),
# so the existing tool path can spawn cross-agent work with zero new transport — the lowest
# friction way to get agents delegating to each other today.

def agent_mcp_tool_specs() -> list[dict]:
    """Tool metadata for ``gateway.federation.register_mcp_server`` — a remote agent as a tool."""
    return [{
        "id": "sophia.delegate",
        "description": "Delegate a task to a remote Sophia A2A agent and return its gated answer.",
        "blp_level": "CONFIDENTIAL",
        "verifier_ref": "grounding",   # peer output is verified before use
        "risk_tier": "medium",
        "side_effects": "read",
    }]


class A2AMcpTransport:
    """Adapt an A2A peer to the MCP ``transport.call(tool_id, args)`` interface.

    Register a remote agent as a tool::

        from gateway.federation import register_mcp_server
        register_mcp_server(gw, "spark-sophia", A2AMcpTransport(A2AClient(http_transport)),
                            tools=agent_mcp_tool_specs())

    A ``tools/call`` for ``sophia.delegate`` is routed to the peer's ``message/send`` and the
    gated answer is returned in MCP's ``{text, sources}`` shape.
    """

    def __init__(self, client: A2AClient):
        self.client = client
        self.calls: list = []

    def call(self, tool_id: str, args: dict):
        self.calls.append((tool_id, args))
        if tool_id != "sophia.delegate":
            raise RuntimeError(f"A2AMcpTransport: unknown tool {tool_id!r}")
        text = args.get("task") or args.get("text") or args.get("goal") or ""
        verdict = self.client.delegate_to_peer(text)
        return {"text": verdict.text if verdict.accept else f"[{verdict.label}] {verdict.reason}",
                "sources": ["a2a:remote-sophia"], "accept": verdict.accept}
