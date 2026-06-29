# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Wire Sophia's REAL governance machinery into the A2A interop layer.

`agent/a2a.py` is the transport/contract scaffolding with a placeholder ``default_gate`` and
named-but-unwired Agent Card skills. This module connects the scaffolding to the things that
already exist in the repo, so an A2A peer (yours or a third-party LLM agent) actually gets
Sophia's discipline, not a stub:

  ① Real trust gate   — :func:`sophia_gate` runs the source-discipline verifier
                        (``agent.verifiers.provenance_faithful`` = the "don't merge lineages"
                        rule) on top of the conservative over-claim/abstain markers. Pure,
                        deterministic, offline — so the whole path stays CI-testable.
  ② Real card skills  — :func:`sophia_skill_handlers` binds ``swarm.delegate`` /
                        ``provenance.validate`` / ``epistemic.gate_check`` to real impls;
                        :func:`sophia_router_runner` routes an inbound task to the requested
                        skill (``@skill <id>`` directive). A peer can now *use* the skills it
                        discovers, not just see their names.
  ③ Contract handshake — :func:`sophia_contract_meta` reads the versioned ``sophia_contract``
                        (the seam aihk-os / OpenClaw consume) and advertises its version on the
                        Agent Card, so A2A peers can negotiate capability versions.

Every external dependency is imported lazily and guarded: if a module is missing the function
FAILS CLOSED to the safe scaffolding behavior (e.g. ``sophia_gate`` falls back to
``a2a.default_gate``) rather than raising — the same offline-or-degrade discipline as
``gateway.federation``.

Build a fully-wired peer in one call::

    from agent import a2a_integrations as ai, model as m
    server = ai.make_sophia_a2a_server("http://spark.local:8080",
                                       client=m.ModelClient(m.resolve_config("mock")))
    client = a2a.A2AClient(a2a.StubA2ATransport(server), gate=ai.sophia_gate)
    ai.invoke_skill(client, "provenance.validate", "Confucius wrote the Dao De Jing")  # -> block
"""

from __future__ import annotations

from typing import Any, Callable

from agent.a2a import (
    SOPHIA_SKILLS,
    A2AServer,
    A2ATask,
    AgentCard,
    GateVerdict,
    default_gate,
)

SKILL_DIRECTIVE = "@skill"


# --------------------------------------------------------------------------- #
# ① Real source-discipline trust gate
# --------------------------------------------------------------------------- #
def _provenance_verdict(text: str) -> "dict | None":
    """Run the real provenance verifier; None if it can't be loaded (offline-safe)."""
    try:
        from agent.verifiers import provenance_faithful
        return provenance_faithful()(text, None, {})
    except Exception:
        return None


def sophia_gate(text: str) -> GateVerdict:
    """The production trust gate: conservative markers + the source-discipline verifier.

    Fail-closed and composable: the cheap marker gate runs first (catches over-claims and
    abstentions); if it accepts, the real ``provenance_faithful`` "don't merge lineages" check
    runs. A forbidden attribution (e.g. *"Confucius wrote the Dao De Jing"*) is BLOCKED. If the
    verifier cannot be loaded (bare env), we fall back to the marker verdict — never silently
    upgrade to trust.
    """
    base = default_gate(text)
    if not base.accept:
        return base  # already block/abstain on markers (over-claim, empty, "cannot verify")
    res = _provenance_verdict(text)
    if res is None:
        return base  # offline fallback — keep the conservative verdict, do not invent trust
    if not res.get("passed", True):
        reasons = "; ".join(res.get("reasons", [])) or "source-discipline violation"
        return GateVerdict(False, "block", f"source-discipline: {reasons}", text)
    return GateVerdict(True, "accept", "passes source-discipline + trust gate", text)


# --------------------------------------------------------------------------- #
# ② Real Agent Card skills + a skill-routing runner
# --------------------------------------------------------------------------- #
# A skill handler maps task text -> (answer_text, cost_usd, est_cost_steps, raw_dict), the
# same tuple the A2AServer runner contract expects.

def _h_swarm(client: Any) -> Callable[[str], tuple]:
    def _run(text: str) -> tuple:
        from agent import swarm_router as sr
        plan, result = sr.run_swarm(text, client=client, parent_id="a2a")
        return result.synthesis, result.total_cost_usd, plan.est_cost_steps, result.to_dict()
    return _run


def _h_provenance_validate(text: str) -> tuple:
    res = _provenance_verdict(text)
    if res is None:
        return "[abstain] provenance verifier unavailable", 0.0, 1, {"available": False}
    ok = res.get("passed", False)
    answer = ("allow: no attribution/provenance violation" if ok
              else "block: " + "; ".join(res.get("reasons", [])) or "block")
    return answer, 0.0, 1, {"passed": ok, "reasons": res.get("reasons", []), "skill": "provenance.validate"}


def _h_gate_check(text: str) -> tuple:
    try:
        from sophia_mcp.tools_impl import check_claim
        res = check_claim(text)
    except Exception:
        # fall back to the provenance verifier, then to the marker gate
        res = _provenance_verdict(text)
        if res is None:
            v = default_gate(text)
            return f"[{v.label}] {v.reason}", 0.0, 1, v.to_dict()
    ok = bool(res.get("passed", False))
    answer = ("accept: claim passes the source-discipline gate" if ok
              else "block: " + "; ".join(res.get("reasons", []) or res.get("violations", [])) or "block")
    return answer, 0.0, 1, {"passed": ok, "skill": "epistemic.gate_check", **res}


def sophia_skill_handlers(client: Any) -> dict[str, Callable[[str], tuple]]:
    """Bind the advertised Agent Card skills to their real implementations."""
    return {
        "swarm.delegate": _h_swarm(client),
        "provenance.validate": _h_provenance_validate,
        "epistemic.gate_check": _h_gate_check,
    }


def _parse_skill_directive(text: str) -> tuple[str, str]:
    """A leading ``@skill <id>`` directive selects a skill; otherwise default to swarm.delegate.

    The id is terminated by the first newline or colon, so both forms work:
    ``@skill provenance.validate\\n<body>`` and ``@skill epistemic.gate_check: <body>``.
    (Skill ids use dots, never colons, so colon-splitting is safe.)
    """
    stripped = text.lstrip()
    if not stripped.startswith(SKILL_DIRECTIVE):
        return "swarm.delegate", text
    after = stripped[len(SKILL_DIRECTIVE):].lstrip()
    idx = len(after)
    for sep in ("\n", ":"):
        pos = after.find(sep)
        if pos != -1:
            idx = min(idx, pos)
    skill_id = after[:idx].strip()
    body = after[idx + 1:].lstrip() if idx < len(after) else ""
    return (skill_id or "swarm.delegate"), body


def sophia_router_runner(client: Any, handlers: "dict | None" = None) -> Callable[[str], tuple]:
    """A2AServer runner that routes a task to the requested skill (fail-closed default)."""
    handlers = handlers or sophia_skill_handlers(client)

    def _run(text: str) -> tuple:
        skill_id, body = _parse_skill_directive(text)
        handler = handlers.get(skill_id) or handlers["swarm.delegate"]
        return handler(body)

    return _run


def invoke_skill(a2a_client, skill_id: str, text: str) -> A2ATask:
    """Client helper: send a task addressed to a specific advertised skill."""
    return a2a_client.send_task(f"{SKILL_DIRECTIVE} {skill_id}\n{text}")


# --------------------------------------------------------------------------- #
# ③ Governance contract handshake
# --------------------------------------------------------------------------- #
def sophia_contract_meta() -> dict:
    """Read the versioned governance contract (offline-safe). {version, capabilities}."""
    try:
        from sophia_contract import SophiaContract
        d = SophiaContract().describe()
        return {"version": d.get("version", ""), "capabilities": d.get("capabilities", [])}
    except Exception:
        return {"version": "", "capabilities": []}


# --------------------------------------------------------------------------- #
# Fully-wired card + server
# --------------------------------------------------------------------------- #
def sophia_agent_card_full(url: str, *, name: str = "Sophia", version: str = "0.1.0",
                           security_schemes: tuple[str, ...] = ("apiKey",)) -> AgentCard:
    """Sophia's Agent Card with real skills AND the governance contract version advertised."""
    return AgentCard(
        name=name,
        description=("Provenance-aware, verifier-gated reasoning agent. Source-discipline gate "
                     "(don't-merge-lineages) governs every answer; abstains instead of fabricating."),
        url=url,
        version=version,
        skills=SOPHIA_SKILLS,
        security_schemes=security_schemes,
        contract_version=sophia_contract_meta().get("version", ""),
    )


def make_sophia_a2a_server(url: str, *, client: Any, gate: Callable[[str], GateVerdict] = sophia_gate,
                           require_auth: bool = False, api_keys: "frozenset[str] | None" = None,
                           **card_kw) -> A2AServer:
    """One call: an A2AServer wired with the real card, skill router, and source-discipline gate."""
    card = sophia_agent_card_full(url, **card_kw)
    return A2AServer(card, runner=sophia_router_runner(client), gate=gate,
                     require_auth=require_auth, api_keys=api_keys)
