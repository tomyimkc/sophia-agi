"""Bounded, logged declassification — the controlled relief for label creep (#3).

Combining sources drives confidentiality up (max-over-chain), so without a downgrade
path everything trends to SECRET and the lattice becomes unusable. Declassification
is the *only* sanctioned way to lower a label — and it is deliberately narrow:

  - **bounded:** a rule may only LOWER confidentiality (never raise it), and only
    from the level it declares;
  - **predicate-gated:** a deterministic check must pass (e.g. a redaction verifier
    confirms the secret content was removed);
  - **approval-gated:** an approver must explicitly return True (fail closed on a
    missing, raising, or non-True approver);
  - **audited:** every outcome — granted OR denied — is written to a tamper-evident
    :class:`~agent.security.audit.AuditLog`.

Integrity is never silently raised here either; "endorsement" (raising integrity)
would be a separate, equally-audited operation — not implemented in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.security.audit import AuditLog
from agent.security.labels import Conf, Label


class DeclassError(ValueError):
    """A declassification rule is malformed (e.g. it would raise, not lower, conf)."""


@dataclass(frozen=True)
class DeclassRule:
    name: str
    from_conf: Conf
    to_conf: Conf
    predicate: Callable          # (value) -> bool : deterministic precondition (e.g. redaction passed)
    justification: str = ""

    def __post_init__(self):
        if int(self.to_conf) >= int(self.from_conf):
            raise DeclassError(f"{self.name}: declassification must LOWER confidentiality "
                               f"({self.from_conf.name} -> {self.to_conf.name})")


def declassify(label: Label, value, rule: DeclassRule, *, approver, audit: AuditLog,
               context: str = "") -> "Label | None":
    """Attempt to downgrade ``label`` per ``rule``. Returns the new (lower) label on
    success, or None if refused. Every outcome is audited; refusal is fail-closed."""
    base = {"rule": rule.name, "from": label.conf.name, "to": rule.to_conf.name,
            "justification": rule.justification, "context": context}

    # The rule must apply to this label's level (don't let a weak rule downgrade
    # data more secret than it was written for).
    if label.conf != rule.from_conf:
        audit.append("declassify_denied", {**base, "why": f"rule targets {rule.from_conf.name}, data is {label.conf.name}"})
        return None
    # Predicate (e.g. redaction) must hold.
    try:
        ok = bool(rule.predicate(value))
    except Exception as exc:  # noqa: BLE001 — fail closed on a raising predicate
        audit.append("declassify_denied", {**base, "why": f"predicate raised: {exc}"})
        return None
    if not ok:
        audit.append("declassify_denied", {**base, "why": "predicate failed"})
        return None
    # Human approval, fail-closed (missing / raising / non-True all refuse).
    granted = False
    if approver is not None:
        try:
            granted = approver(rule, label, context) is True
        except Exception:  # noqa: BLE001
            granted = False
    if not granted:
        audit.append("declassify_denied", {**base, "why": "not approved"})
        return None

    new_label = Label(rule.to_conf, label.integ, label.compartments)
    audit.append("declassify", base)
    return new_label
