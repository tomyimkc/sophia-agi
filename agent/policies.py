"""Selectable verifier policies for the guarded loop.

The guarded completion spine (retrieve → generate → gate → repair/abstain) was
wired to one gate: the provenance verifier. A *policy* generalises that: a named
composition of machine-checked verifiers, plus the repair hint and the
abstention text that fit it, chosen per call or via ``$SOPHIA_POLICY`` — so the
same spine can enforce provenance, citation-faithfulness, arithmetic soundness,
executable code, or a *synthesised* gate, at runtime, without forking the loop.

A policy carries three things:
  - ``verifier`` — the ``(text, task, step) -> {passed, reasons, detail}`` gate;
  - ``repair_hint`` — what to tell the model when it fails (so repair is targeted);
  - ``abstention`` — a safe answer that itself passes the gate (no confabulation).

The ``provenance`` policy reproduces the loop's original behaviour exactly, so it
stays the zero-config default; the others reuse the verifiers already in
``agent/verifiers.py``. A synthesised gate (``agent/verifier_synthesis.py``) drops
in via :func:`from_verifier`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.verifiers import (
    Verifier,
    all_of,
    arithmetic_sound,
    citation_faithful,
    claim_supported,
    code_tests_pass,
    no_secret_leak,
    provenance_faithful,
)

# A generic safe answer used by non-provenance policies. It asserts nothing
# checkable, so it passes citation / arithmetic / synthesised gates; code-exec
# policies are noted below (their abstention legitimately cannot pass a "must run
# code" gate, which the loop records honestly via passed=False).
_GENERIC_ABSTENTION = (
    "I can't verify this to the standard this task requires, so I won't assert it. "
    "Here is what I can support from the available information, with the unverifiable "
    "parts flagged; point me to a checkable source and I'll re-run the check."
)


@dataclass
class Policy:
    """A named gate + the repair/abstention behaviour that fits it.

    Whether the abstention text actually clears the gate is not declared here (a
    static flag would drift); the guarded loop re-judges the abstention and
    surfaces the truth via ``action`` (``abstained`` vs ``abstained_unverified``).
    """

    name: str
    verifier: Verifier
    repair_hint: str
    abstention: str


#: name -> builder. Each builder takes the same kwargs and ignores what it doesn't need.
def _build_provenance(*, records=None, sources=None, secrets=None, extra=None) -> Policy:
    # The guarded loop builds provenance's gate-passing *cited* abstention
    # dynamically (it needs the query/violations), keyed on name == "provenance",
    # so this static abstention is only a fallback for direct use of the policy.
    return Policy(
        name="provenance",
        verifier=provenance_faithful(records),
        repair_hint=(
            "do NOT assert any authorship/origin the sources do not establish, and do not "
            "merge distinct lineages"
        ),
        abstention=_GENERIC_ABSTENTION,
    )


def _build_citation(*, records=None, sources=None, secrets=None, extra=None) -> Policy:
    srcs = list(sources or [])
    return Policy(
        name="citation",
        verifier=citation_faithful(srcs),
        repair_hint=(
            "every cited sentence must be supported by the source it cites; remove or fix "
            "citations the source does not back, and do not cite out-of-range markers"
        ),
        abstention=_GENERIC_ABSTENTION,
    )


def _build_arithmetic(*, records=None, sources=None, secrets=None, extra=None) -> Policy:
    return Policy(
        name="arithmetic",
        verifier=arithmetic_sound(),
        repair_hint="every stated equality (a OP b = c) must be arithmetically correct; recompute and fix any that are wrong",
        abstention=_GENERIC_ABSTENTION,
    )


def _build_nli(*, records=None, sources=None, secrets=None, extra=None) -> Policy:
    # Semantic faithfulness (entailment of cited claims). Needs an NLI model
    # (opt-in); without one, claim_supported fails closed — see agent/verifiers.py.
    return Policy(
        name="nli",
        verifier=claim_supported(list(sources or [])),
        repair_hint="every cited sentence must be ENTAILED by the source it cites; fix the claim or the citation",
        abstention=_GENERIC_ABSTENTION,
    )


def _build_confidentiality(*, records=None, sources=None, secrets=None, extra=None) -> Policy:
    return Policy(
        name="confidentiality",
        verifier=no_secret_leak(secrets or []),
        repair_hint="remove any classified/secret value from the answer; never echo a value the sources marked secret",
        abstention=_GENERIC_ABSTENTION,
    )


def _build_code(*, records=None, sources=None, secrets=None, extra=None) -> Policy:
    # Note: a no-code abstention cannot pass a "run the code" gate; the loop
    # reports that honestly as action="abstained_unverified".
    return Policy(
        name="code",
        verifier=code_tests_pass(),
        repair_hint="the code block must run to completion and exit 0; fix the error and include the corrected, self-checking code",
        abstention=_GENERIC_ABSTENTION,
    )


POLICY_BUILDERS: dict[str, Callable[..., Policy]] = {
    "provenance": _build_provenance,
    "citation": _build_citation,
    "arithmetic": _build_arithmetic,
    "code": _build_code,
    "confidentiality": _build_confidentiality,
    "nli": _build_nli,
}


def get_policy(name: str, *, records=None, sources=None, secrets=None, extra=None) -> Policy:
    """Build a named policy. ``records`` (provenance), ``sources`` (citation), and
    ``secrets`` (confidentiality) are forwarded to the verifiers that use them."""
    key = (name or "provenance").strip().lower()
    if key not in POLICY_BUILDERS:
        raise KeyError(f"unknown policy {name!r}; known: {', '.join(sorted(POLICY_BUILDERS))} (or pass a verifier)")
    return POLICY_BUILDERS[key](records=records, sources=sources, secrets=secrets, extra=extra)


def from_verifier(verifier: Verifier, *, name: str = "custom",
                  repair_hint: str = "satisfy the stated checks") -> Policy:
    """Wrap any verifier (e.g. a synthesised gate from agent.verifier_synthesis)
    as a runtime policy for the guarded loop."""
    return Policy(name=name, verifier=verifier, repair_hint=repair_hint,
                  abstention=_GENERIC_ABSTENTION)


def compose_policies(*policies: Policy, name: str = "composite") -> Policy:
    """AND several policies into one gate (all must pass). The repair hint and
    abstention concatenate/fall back sensibly."""
    if not policies:
        raise ValueError("compose_policies needs at least one policy")
    verifier = all_of(*(p.verifier for p in policies))
    hint = "; ".join(p.repair_hint for p in policies)
    return Policy(name=name, verifier=verifier, repair_hint=hint, abstention=_GENERIC_ABSTENTION)
