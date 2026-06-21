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
    code_tests_pass,
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
    """A named gate + the repair/abstention behaviour that fits it."""

    name: str
    verifier: Verifier
    repair_hint: str
    abstention: str
    # Whether the abstention text is expected to pass this gate. For code-exec
    # policies it cannot (no code block), and the loop surfaces that truthfully.
    abstention_passes: bool = True


def _provenance_abstention() -> str:
    # Kept as a constant marker; the loop uses its own cited-abstention builder
    # for the provenance policy to preserve the original, gate-passing wording.
    return "__PROVENANCE_CITED_ABSTENTION__"


#: name -> builder. Each builder takes the same kwargs and ignores what it doesn't need.
def _build_provenance(*, records=None, sources=None, extra=None) -> Policy:
    return Policy(
        name="provenance",
        verifier=provenance_faithful(records),
        repair_hint=(
            "do NOT assert any authorship/origin the sources do not establish, and do not "
            "merge distinct lineages"
        ),
        abstention=_provenance_abstention(),
    )


def _build_citation(*, records=None, sources=None, extra=None) -> Policy:
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


def _build_arithmetic(*, records=None, sources=None, extra=None) -> Policy:
    return Policy(
        name="arithmetic",
        verifier=arithmetic_sound(),
        repair_hint="every stated equality (a OP b = c) must be arithmetically correct; recompute and fix any that are wrong",
        abstention=_GENERIC_ABSTENTION,
    )


def _build_code(*, records=None, sources=None, extra=None) -> Policy:
    return Policy(
        name="code",
        verifier=code_tests_pass(),
        repair_hint="the code block must run to completion and exit 0; fix the error and include the corrected, self-checking code",
        abstention=_GENERIC_ABSTENTION,
        abstention_passes=False,   # a no-code abstention cannot pass a "run the code" gate
    )


POLICY_BUILDERS: dict[str, Callable[..., Policy]] = {
    "provenance": _build_provenance,
    "citation": _build_citation,
    "arithmetic": _build_arithmetic,
    "code": _build_code,
}


def get_policy(name: str, *, records=None, sources=None, extra=None) -> Policy:
    """Build a named policy. ``records`` (provenance) and ``sources`` (citation)
    are forwarded to the verifiers that use them."""
    key = (name or "provenance").strip().lower()
    if key not in POLICY_BUILDERS:
        raise KeyError(f"unknown policy {name!r}; known: {', '.join(sorted(POLICY_BUILDERS))} (or pass a verifier)")
    return POLICY_BUILDERS[key](records=records, sources=sources, extra=extra)


def from_verifier(verifier: Verifier, *, name: str = "custom",
                  repair_hint: str = "satisfy the stated checks", abstention_passes: bool = True) -> Policy:
    """Wrap any verifier (e.g. a synthesised gate from agent.verifier_synthesis)
    as a runtime policy for the guarded loop."""
    return Policy(name=name, verifier=verifier, repair_hint=repair_hint,
                  abstention=_GENERIC_ABSTENTION, abstention_passes=abstention_passes)


def compose_policies(*policies: Policy, name: str = "composite") -> Policy:
    """AND several policies into one gate (all must pass). The repair hint and
    abstention concatenate/fall back sensibly."""
    if not policies:
        raise ValueError("compose_policies needs at least one policy")
    verifier = all_of(*(p.verifier for p in policies))
    hint = "; ".join(p.repair_hint for p in policies)
    return Policy(
        name=name, verifier=verifier, repair_hint=hint,
        abstention=_GENERIC_ABSTENTION,
        abstention_passes=all(p.abstention_passes for p in policies),
    )
