"""Classification lattice — Bell-LaPadula confidentiality + Biba integrity (#3).

The original review found the confidentiality story incomplete: max-over-chain
classification causes **label creep** (everything drifts to SECRET), and there was
**no integrity axis** (Biba) and no declassification. This adds the lattice; the
audited downgrade lives in :mod:`agent.security.declassify`.

A ``Label`` carries two ordered axes plus need-to-know compartments:
  - **confidentiality** (PUBLIC < INTERNAL < CONFIDENTIAL < SECRET) — Bell-LaPadula:
    *no write down* (high-conf data must not flow to a lower-cleared sink).
  - **integrity** (UNTRUSTED < COMMUNITY < CURATED < AUTHORITATIVE) — Biba:
    *no write up* (low-integrity data must not flow into a higher-integrity sink).
  - **compartments** (need-to-know): the data's tags must be covered by the sink.

Propagation when combining sources: confidentiality = **max** (most secret wins —
this is what causes creep, which declassification then relieves), integrity =
**min** (least-trusted wins), compartments = **union**. Deterministic, no model.

The dataflow taint axis (``agent/dataflow/taint.py``: untrusted/trusted) is the
2-level projection of this integrity axis; this generalises it and adds the
confidentiality axis BLP needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Conf(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3


class Integ(IntEnum):
    UNTRUSTED = 0
    COMMUNITY = 1
    CURATED = 2
    AUTHORITATIVE = 3


@dataclass(frozen=True)
class Label:
    conf: Conf = Conf.PUBLIC
    integ: Integ = Integ.AUTHORITATIVE
    compartments: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        # accept plain ints / strings for ergonomics, normalise to enums + frozenset
        object.__setattr__(self, "conf", Conf(int(self.conf)) if not isinstance(self.conf, Conf) else self.conf)
        object.__setattr__(self, "integ", Integ(int(self.integ)) if not isinstance(self.integ, Integ) else self.integ)
        object.__setattr__(self, "compartments", frozenset(self.compartments))


@dataclass(frozen=True)
class FlowDecision:
    allowed: bool
    reasons: tuple = ()


def combine(*labels: Label) -> Label:
    """Propagation rule for derived data: conf = max (creep), integ = min, need-to-
    know = union. With no inputs, returns the most-permissive bottom label."""
    if not labels:
        return Label(Conf.PUBLIC, Integ.AUTHORITATIVE, frozenset())
    conf = max(label.conf for label in labels)
    integ = min(label.integ for label in labels)
    comps: set = set()
    for label in labels:
        comps |= label.compartments
    return Label(conf, integ, frozenset(comps))


def can_flow(data: Label, sink: Label) -> FlowDecision:
    """May ``data`` flow into a destination cleared/typed as ``sink``?

    Bell-LaPadula (no write down): ``sink.conf >= data.conf``.
    Biba (no write up): ``data.integ >= sink.integ``.
    Need-to-know: ``data.compartments`` ⊆ ``sink.compartments``.
    """
    reasons = []
    if sink.conf < data.conf:
        reasons.append(f"write-down blocked: {data.conf.name} data -> {sink.conf.name} sink (Bell-LaPadula)")
    if data.integ < sink.integ:
        reasons.append(f"write-up blocked: {data.integ.name} data -> {sink.integ.name} sink (Biba)")
    missing = data.compartments - sink.compartments
    if missing:
        reasons.append(f"need-to-know: sink missing compartments {sorted(missing)}")
    return FlowDecision(allowed=not reasons, reasons=tuple(reasons))
