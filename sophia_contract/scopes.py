"""Capability scopes per role — least privilege for the 9 aihk-os role pipelines.

A role may only perform the operations in its scope, up to a maximum classification.
Scopes are data the founder configures; the service enforces them when a request
carries a ``role``. With no registry configured (or no role on a request) the call is
unrestricted — so scopes are opt-in and never silently block an unconfigured caller,
but once configured they fail closed (an unknown role is denied).

Denials are ``UNAUTHENTICATED`` (authorization), kept distinct from ``BLP_VIOLATION``
(the lattice flow rules in blp.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sophia_contract import blp
from sophia_contract.errors import ContractError

# Operations a scope can grant.
OPS = ("record_claim", "verify_claim", "enqueue_task", "admin")


@dataclass(frozen=True)
class Scope:
    ops: frozenset = field(default_factory=frozenset)
    max_blp: str = "UNCLASSIFIED"
    dry_run_only: bool = False  # role may validate but never persist/publish

    def __post_init__(self):
        object.__setattr__(self, "ops", frozenset(self.ops))
        if not blp.is_level(self.max_blp):
            raise ValueError(f"bad max_blp {self.max_blp!r}")


class ScopeRegistry:
    """role -> Scope. Empty registry == unrestricted (scopes opt-in)."""

    def __init__(self, roles: "dict[str, Scope] | None" = None):
        self.roles = dict(roles or {})

    def __bool__(self) -> bool:
        return bool(self.roles)

    def check(self, role: "str | None", op: str, *, blp_level: str = "UNCLASSIFIED",
              dry_run: bool = False) -> None:
        """Raise UNAUTHENTICATED if ``role`` may not perform ``op`` at ``blp_level``.
        No registry or no role -> allowed (unrestricted)."""
        if not self.roles or role is None:
            return
        scope = self.roles.get(role)
        if scope is None:
            raise ContractError("UNAUTHENTICATED", f"unknown role {role!r}")
        if op not in scope.ops:
            raise ContractError("UNAUTHENTICATED", f"role {role!r} is not scoped for {op}")
        if blp.rank(blp_level) > blp.rank(scope.max_blp):
            raise ContractError("UNAUTHENTICATED",
                                f"role {role!r} is capped at {scope.max_blp}, not {blp_level}")
        if scope.dry_run_only and not dry_run:
            raise ContractError("UNAUTHENTICATED",
                                f"role {role!r} is dry-run only and may not persist/publish")


# A sensible default the founder can override. Names are illustrative; aihk-os passes
# its own registry. Least-privilege: most roles read/verify; few may write high.
DEFAULT_ROLES = ScopeRegistry({
    "reader": Scope(ops={"verify_claim"}, max_blp="UNCLASSIFIED"),
    "researcher": Scope(ops={"record_claim", "verify_claim"}, max_blp="CONFIDENTIAL"),
    "auditor": Scope(ops={"verify_claim"}, max_blp="SECRET", dry_run_only=True),
    "operator": Scope(ops={"record_claim", "verify_claim", "enqueue_task"}, max_blp="SECRET"),
    "founder": Scope(ops=set(OPS), max_blp="TOP_SECRET"),
})
