"""Tool/skill registry — the catalog every gateway call is checked against.

Each entry declares its risk tier, classification, allowed roles, side-effects, and how
its OUTPUT should be verified (``verifier_ref``). P0 registers native/mock tools (an
in-process handler); downstream MCP federation (stdio/HTTP) is P1+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sophia_contract import blp

RISK_TIERS = ("low", "medium", "high")
SIDE_EFFECTS = ("none", "read", "write", "external")
# how to verify a tool's output (see gateway.verify_router)
VERIFIERS = ("none", "deterministic", "grounding", "env:arithmetic", "env:regex")


@dataclass
class ToolEntry:
    id: str
    handler: "Callable[[dict], object]"          # native/mock: args -> output
    kind: str = "native"                          # native | mock | mcp
    risk_tier: str = "low"
    blp_level: str = "UNCLASSIFIED"
    allowed_roles: "frozenset | None" = None      # None => any role
    side_effects: str = "read"
    dry_run_supported: bool = True
    verifier_ref: str = "none"
    description: str = ""

    def __post_init__(self):
        if not blp.is_level(self.blp_level):
            raise ValueError(f"bad blp_level {self.blp_level!r}")
        if self.risk_tier not in RISK_TIERS:
            raise ValueError(f"bad risk_tier {self.risk_tier!r}")
        if self.side_effects not in SIDE_EFFECTS:
            raise ValueError(f"bad side_effects {self.side_effects!r}")
        if self.allowed_roles is not None:
            self.allowed_roles = frozenset(self.allowed_roles)

    def public(self, reliability: float) -> dict:
        return {
            "tool_id": self.id, "kind": self.kind, "risk_tier": self.risk_tier,
            "blp_level": self.blp_level, "side_effects": self.side_effects,
            "allowed_roles": sorted(self.allowed_roles) if self.allowed_roles else None,
            "dry_run_supported": self.dry_run_supported, "verifier_ref": self.verifier_ref,
            "reliability": reliability, "description": self.description,
        }


class Registry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, entry: "ToolEntry") -> "ToolEntry":
        self._tools[entry.id] = entry
        return entry

    def get(self, tool_id: str) -> "ToolEntry | None":
        return self._tools.get(tool_id)

    def list(self, *, role: "str | None" = None) -> "list[ToolEntry]":
        out = []
        for e in self._tools.values():
            if role is not None and e.allowed_roles is not None and role not in e.allowed_roles:
                continue
            out.append(e)
        return out
