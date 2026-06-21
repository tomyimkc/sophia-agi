"""Guarded completion loop — the spine of the discipline layer.

``guarded_complete`` wraps a small/local model with Sophia's machine-checked
provenance gate so that an answer which asserts a forbidden attribution
("don't merge lineages") never escapes unchecked:

    retrieve + format_context  ->  generate  ->  judge (provenance_faithful)
        pass        -> return the answer ("clean")
        violation   -> on_fail decides:
            repair      -> ONE bounded re-generation that must fix it;
                           still bad -> cited abstention ("abstained")
            abstain     -> immediate cited abstention
            hedge       -> keep the answer but prepend a visible disclaimer
            passthrough -> return the unguarded answer (explicitly opt-out)

The cited abstention is templated so it itself passes the gate — it never
restates the forbidden author→title claim.

``check_claim`` is the mode-free verifier surface: given any text, return
``{passed, reasons, violations}`` under the same gate, with no question, mode,
or style coupling (unlike the moded epistemic ``gate_check``).

Everything here is offline-testable: ``generate``, ``retrieve_fn`` and
``format_context_fn`` are injectable; the defaults wire the real model client
and corpus retrieval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from agent.retrieval import format_context, retrieve
from agent.verifiers import provenance_faithful

# Generation strategies when the gate fails. "repair" is the default spine.
ON_FAIL_MODES = ("repair", "abstain", "hedge", "passthrough")

DEFAULT_SYSTEM = (
    "You answer strictly from the provided sources and practice source discipline: "
    "never assert an authorship/origin the sources do not establish, and never merge "
    "distinct lineages. If unsure, say what is and is not supported."
)

GenerateFn = Callable[[str, str], "object"]  # (system, user) -> ModelResult-like


@dataclass
class GuardedResult:
    """Outcome of a guarded completion."""

    text: str
    ok: bool                       # safe to surface (clean / repaired / abstained / hedged)
    passed: bool                   # did the FINAL text pass the provenance gate
    action: str                    # clean | repaired | abstained | hedged | passthrough | model_error
    attempts: int = 0              # number of model generations performed
    violations: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    context_used: bool = False

    def to_log(self) -> dict:
        return {
            "action": self.action,
            "ok": self.ok,
            "passed": self.passed,
            "attempts": self.attempts,
            "violations": self.violations,
            "reasons": self.reasons,
        }


def check_claim(text: str, *, records: "dict | None" = None) -> dict:
    """Mode-free provenance check: ``{passed, reasons, violations}`` for any text.

    No question, mode, or style scoring — just Sophia's "don't merge lineages"
    rule. Suitable as a CLI/MCP surface and as the judge inside the loop.
    """
    verify = provenance_faithful(records)
    result = verify(text or "", None, {})
    detail = result.get("detail") or {}
    return {
        "passed": bool(result.get("passed")),
        "reasons": list(result.get("reasons") or []),
        "violations": list(detail.get("violations") or []),
    }


def _cited_abstention(query: str, context: str, violations: list) -> str:
    """A safe answer that does NOT restate the forbidden attribution.

    Phrased so the provenance gate passes it: the only mention of the disputed
    author/title is inside an explicit "will not attribute" clause (which the
    gate treats as a correction, not an assertion).
    """
    note = "; ".join(violations) if violations else "the specific authorship asked about"
    return (
        "I will not attribute authorship the available sources do not establish, so I can't "
        f"confirm the requested attribution ({note}). "
        "Based on the retrieved sources I can share what is supported, but I won't merge "
        "distinct lineages. If you can point me to a primary source, I'll re-check."
    )


def _hedged(text: str, violations: list) -> str:
    """Keep the model's answer but prepend a visible unverified-attribution banner."""
    note = "; ".join(violations) if violations else "an attribution the sources do not establish"
    banner = (
        f"⚠ Unverified attribution — the answer below may assert {note}, "
        "which the available sources do not confirm. Treat it as uncertain.\n\n"
    )
    return banner + text


def _repair_prompt(query: str, context: str, bad_answer: str, violations: list) -> str:
    note = "; ".join(violations) if violations else "a forbidden attribution"
    return (
        f"Your previous answer violated source discipline ({note}).\n\n"
        f"Question:\n{query}\n\n"
        f"Sources:\n{context}\n\n"
        f"Previous answer (DO NOT repeat its forbidden attribution):\n{bad_answer}\n\n"
        "Rewrite the answer so it does NOT assert any authorship/origin the sources do not "
        "establish. State plainly what the sources do and do not support. Do not merge lineages."
    )


def _generic_repair_prompt(query: str, context: str, bad_answer: str, reasons: list, hint: str) -> str:
    """Policy-agnostic repair prompt: tell the model which checks failed and the
    policy's targeted hint for fixing them."""
    note = "; ".join(reasons) if reasons else "the policy's checks"
    return (
        f"Your previous answer failed these checks: {note}.\n\n"
        f"Question:\n{query}\n\n"
        f"Sources:\n{context}\n\n"
        f"Previous answer (do NOT repeat the failure):\n{bad_answer}\n\n"
        f"Rewrite the answer so it passes: {hint}."
    )


def guarded_complete(
    query: str,
    *,
    system: str = DEFAULT_SYSTEM,
    on_fail: "str | None" = None,
    generate: "GenerateFn | None" = None,
    records: "dict | None" = None,
    policy: "str | None" = None,
    verifier: "Callable | None" = None,
    sources: "list | None" = None,
    top_k: int = 8,
    retrieve_fn: Callable[..., list] = retrieve,
    format_context_fn: Callable[[list], str] = format_context,
) -> GuardedResult:
    """Generate an answer to ``query`` and enforce a machine-checked gate on it.

    The gate is selectable at runtime (default: provenance, unchanged):
      - ``verifier`` — any ``(text, task, step) -> {passed,...}`` gate (e.g. a
        synthesised gate from :mod:`agent.verifier_synthesis`); highest priority;
      - ``policy``   — a named policy (``provenance|citation|arithmetic|code``),
        else ``$SOPHIA_POLICY``; ``sources`` feeds the citation policy;
      - neither      — the provenance gate (original behaviour, zero-config).

    ``on_fail`` defaults to ``$SOPHIA_ON_FAIL`` then ``"repair"``. ``generate`` is
    a ``(system, user) -> ModelResult`` callable; when omitted, the default model
    client is used.
    """
    mode = (on_fail or os.environ.get("SOPHIA_ON_FAIL") or "repair").strip().lower()
    if mode not in ON_FAIL_MODES:
        raise ValueError(f"invalid on_fail mode {mode!r}; valid: {', '.join(ON_FAIL_MODES)}")

    # --- resolve the gate (preserve the provenance default path exactly) ----- #
    from agent import policies as _policies

    env_policy = os.environ.get("SOPHIA_POLICY")
    if verifier is not None:
        pol = _policies.from_verifier(verifier)
        verify_fn = verifier
    elif policy or env_policy:
        pol = _policies.get_policy(policy or env_policy, records=records, sources=sources)
        verify_fn = pol.verifier
    else:
        pol = None                                  # provenance default
        verify_fn = provenance_faithful(records)

    if generate is None:
        from agent.model import default_client

        client = default_client()
        generate = lambda s, u: client.generate(s, u)  # noqa: E731

    chunks = retrieve_fn(query, top_k=top_k)
    context = format_context_fn(chunks)
    context_used = bool(chunks)
    user = f"Sources:\n{context}\n\nQuestion:\n{query}\n\nAnswer from the sources, with source discipline."

    attempts = 0

    def _judge(text: str) -> dict:
        r = verify_fn(text or "", None, {})
        detail = r.get("detail") or {}
        reasons = list(r.get("reasons") or [])
        return {
            "passed": bool(r.get("passed")),
            "reasons": reasons,
            "violations": list(detail.get("violations") or reasons),
        }

    # --- first generation -------------------------------------------------- #
    first = generate(system, user)
    attempts += 1
    if not getattr(first, "ok", True):
        return GuardedResult(
            text="", ok=False, passed=False, action="model_error", attempts=attempts,
            reasons=[getattr(first, "error", None) or "model call failed"], context_used=context_used,
        )

    text = getattr(first, "text", "") or ""
    verdict = _judge(text)
    if verdict["passed"]:
        return GuardedResult(
            text=text, ok=True, passed=True, action="clean", attempts=attempts,
            context_used=context_used,
        )

    violations = verdict["violations"]
    reasons = verdict["reasons"]

    # --- gate failed: branch on mode -------------------------------------- #
    if mode == "passthrough":
        return GuardedResult(
            text=text, ok=False, passed=False, action="passthrough", attempts=attempts,
            violations=violations, reasons=reasons, context_used=context_used,
        )

    if mode == "hedge":
        return GuardedResult(
            text=_hedged(text, violations), ok=True, passed=False, action="hedged", attempts=attempts,
            violations=violations, reasons=reasons, context_used=context_used,
        )

    if mode == "repair":
        repair_user = (
            _repair_prompt(query, context, text, violations) if pol is None
            else _generic_repair_prompt(query, context, text, reasons, pol.repair_hint)
        )
        repair = generate(system, repair_user)
        attempts += 1
        if getattr(repair, "ok", True):
            repaired = getattr(repair, "text", "") or ""
            rv = _judge(repaired)
            if rv["passed"]:
                return GuardedResult(
                    text=repaired, ok=True, passed=True, action="repaired", attempts=attempts,
                    context_used=context_used,
                )
        # repair did not clear the gate -> fall through to cited abstention.

    # mode == "abstain", or repair exhausted.
    abstention = _cited_abstention(query, context, violations) if pol is None else pol.abstention
    av = _judge(abstention)
    return GuardedResult(
        text=abstention, ok=True, passed=av["passed"], action="abstained", attempts=attempts,
        violations=violations, reasons=reasons, context_used=context_used,
    )
