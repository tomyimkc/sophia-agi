"""Taint labels for data-flow tracking.

A value's taint is a set of labels describing *where it came from*. Untrusted
sources (retrieved documents, tool output, anything an attacker can influence)
carry ``"untrusted"``. Taint propagates through derivation (``combine``), so a
value computed from any untrusted input is itself untrusted. The firewall reads
these labels to decide whether a value may flow into a sink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TRUSTED: frozenset = frozenset()
UNTRUSTED: frozenset = frozenset({"untrusted"})


@dataclass(frozen=True)
class Labeled:
    """A value tagged with a taint set and a short provenance note."""

    value: Any
    taint: frozenset = field(default=TRUSTED)
    origin: str = "?"

    def is_tainted(self) -> bool:
        return bool(self.taint)


def taint_of(x: Any) -> frozenset:
    """The taint of any value (untracked plain values are TRUSTED by default)."""
    return x.taint if isinstance(x, Labeled) else TRUSTED


def unwrap(x: Any) -> Any:
    return x.value if isinstance(x, Labeled) else x


def combine(*xs: Any) -> frozenset:
    """Union of the taints of all inputs — the propagation rule for derived data."""
    out: set = set()
    for x in xs:
        out |= taint_of(x)
    return frozenset(out)


def untrusted(value: Any, origin: str = "retrieved") -> Labeled:
    """Tag a value as attacker-influenceable (retrieved content, tool output, …)."""
    return Labeled(value, UNTRUSTED, origin)


def trusted(value: Any, origin: str = "user") -> Labeled:
    return Labeled(value, TRUSTED, origin)
