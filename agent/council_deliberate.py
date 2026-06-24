# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council deliberation — map-reduce a hard query across constrained seats.

The upgrade that makes the council a real uplift for SMALL models: instead of one
shallow pass over a big multi-seat prompt, decompose into a few NARROW seat passes
(a weak model answers one focused role well), GATE each seat output (deterministic
checks catch weak-model errors cheaply), then REDUCE — synthesise under the
guardian seats + decision contract.

    map:    route -> N substantive seats -> one focused pass each
    gate:   each seat output is gate-checked; flagged seats are quarantined
    reduce: synthesise the clean seat outputs under the guardian checklist

``client`` is any object with ``generate(system, user) -> result`` where result has
``.ok`` and ``.text`` (agent.model.ModelClient, or a stub in tests). Nothing here
requires a specific provider, so it runs offline with the mock client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agent.gate import check_response
from agent.sector_council import detect_council, load_council, route_council


@dataclass
class SeatResult:
    seatId: str
    displayName: str
    answer: str
    ok: bool
    gatePassed: bool
    violations: list[str] = field(default_factory=list)
    model: str = ""


@dataclass
class Deliberation:
    query: str
    councilId: "str | None"
    seats: list[SeatResult]
    guardians: list[str]
    synthesis: str
    gatedOutSeatIds: list[str]
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _flatten(route: dict, *, core: bool) -> list[dict]:
    out: list[dict] = []
    for group in route.get("selected", {}).values():
        if bool(group.get("core")) is core:
            out += [s for s in group.get("seats", []) if isinstance(s, dict) and s.get("seatId")]
    return out


def _seat_system(seat: dict, council_name: str) -> str:
    emphasis = "; ".join(str(x) for x in seat.get("decisionEmphasis", [])[:4])
    frame = seat.get("sourceFrame") or seat.get("representativeFrame") or "a constrained expert seat"
    boundary = seat.get("speakerBoundary", "")
    return (
        f"You are the **{seat.get('displayName', seat.get('seatId'))}** on the {council_name}. "
        f"Source frame: {frame}. Focus ONLY on your seat's concern: {emphasis}. "
        "Answer in 2-4 sentences from this single perspective. Cite a source path/authority if you "
        "rely on one. If you lack a basis, say 'insufficient basis' rather than guess. "
        f"{boundary}"
    ).strip()


def _seat_gate(answer: str, query: str) -> dict:
    """Seat-level gate: fail only on substantive VIOLATIONS (fabricated citation,
    false arithmetic, forbidden attribution) — not on style warnings like a missing
    中文 summary, which don't apply to a single short seat answer."""
    g = check_response(answer, mode="advisor", question=query)
    return {"passed": not g["violations"], "violations": g["violations"]}


def deliberate(query: str, *, client, council_id: "str | None" = None, max_seats: int = 4,
               gate: bool = True, materials: "list[str] | None" = None,
               seat_clients: "list | None" = None) -> Deliberation:
    """Map-reduce a query across council seats.

    ``client`` runs every seat (homogeneous: one model wearing N hats — its
    "members" share weights, so their errors are correlated). ``seat_clients``, a
    list of clients cycled across the substantive seats, makes the panel
    HETEROGENEOUS: each seat is a *different* model, so the members are genuinely
    independent voters. The synthesis chair always uses ``client``.
    """
    cid = council_id or detect_council(query)
    if not cid:
        # No sector matched — single direct pass (still gated), so the API always works.
        ans = _gen(client, "You are a careful, source-disciplined advisor. Be concise; cite sources; "
                           "say what you cannot verify.", query)
        gres = _seat_gate(ans, query) if gate else {"passed": True, "violations": []}
        return Deliberation(query=query, councilId=None, seats=[], guardians=[],
                            synthesis=ans, gatedOutSeatIds=[],
                            note="no council matched; single direct pass" + ("" if gres["passed"] else " (gate flagged)"))

    route = route_council(load_council(cid), query, materials)
    council_name = route.get("displayName", cid)
    substantive = _flatten(route, core=False)[:max_seats]
    guardians = _flatten(route, core=True)

    pool = [c for c in (seat_clients or []) if c is not None] or [client]

    seats: list[SeatResult] = []
    gated_out: list[str] = []
    for i, seat in enumerate(substantive):
        seat_client = pool[i % len(pool)]
        ans = _gen(seat_client, _seat_system(seat, council_name), query)
        ok = bool(ans.strip()) and "insufficient basis" not in ans.lower()
        gres = _seat_gate(ans, query) if gate else {"passed": True, "violations": []}
        seats.append(SeatResult(seatId=seat["seatId"], displayName=seat.get("displayName", seat["seatId"]),
                                answer=ans, ok=ok, gatePassed=gres["passed"], violations=gres["violations"],
                                model=getattr(seat_client, "spec", "") or getattr(seat_client, "model", "")))
        if gate and not gres["passed"]:
            gated_out.append(seat["seatId"])

    clean = [s for s in seats if s.ok and s.gatePassed]
    synthesis = _synthesize(client, query, council_name, clean, guardians, route)
    return Deliberation(
        query=query, councilId=cid, seats=seats,
        guardians=[g.get("displayName", g["seatId"]) for g in guardians],
        synthesis=synthesis, gatedOutSeatIds=gated_out,
        note=f"{len(clean)}/{len(seats)} seats passed the gate; synthesised under {len(guardians)} guardian seats.",
    )


def _synthesize(client, query: str, council_name: str, clean_seats: list[SeatResult],
                guardians: list[dict], route: dict) -> str:
    if not clean_seats:
        return ("Insufficient verified basis to answer: every seat was gated out (e.g. unverifiable "
                "citation or unsound figure). Escalate to a human. Not advice.")
    seat_block = "\n".join(f"- {s.displayName}: {s.answer}" for s in clean_seats)
    checklist = "; ".join(
        e for g in guardians for e in [str(x) for x in g.get("decisionEmphasis", [])[:2]]
    ) or "source/quote traceability; high-risk escalation"
    contract = "; ".join(str(x) for x in route.get("decisionContract", [])[:4])
    boundary = " ".join(str(x) for x in route.get("humanBoundary", [])[:1])
    system = (
        f"You are the synthesis chair of the {council_name}. Combine the gate-passed seat findings "
        f"into one decision. Apply the guardian checklist ({checklist}). Follow the decision contract: "
        f"{contract}. Be concise. Label clearly as not professional advice. {boundary}"
    )
    user = f"Question: {query}\n\nGate-passed seat findings:\n{seat_block}\n\nProduce the council's decision."
    return _gen(client, system, user)


def _gen(client, system: str, user: str) -> str:
    try:
        res = client.generate(system, user)
    except Exception:  # noqa: BLE001 - a broken client yields no content, not a crash
        return ""
    return (getattr(res, "text", "") or "").strip() if getattr(res, "ok", False) else ""
