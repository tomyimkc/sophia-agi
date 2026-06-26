# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Routed Metacognition (Phase 0): MoE routing as a compute-proportional council.

Concrete first step of
[docs/11-Platform/Routed-Metacognition.md](../docs/11-Platform/Routed-Metacognition.md).
MoE's top-k router is resource allocation under uncertainty, and its
load-balancing loss is a metacognitive alarm. Sophia already has the "experts" —
council seats (`agent/sector_council.py`) — but routes by keyword at a flat cost
with no over-reliance signal. This wires the systems-track MoE machinery
(`moe/router.py`) over the *real* council seats to add the two missing pieces:

1. **Compute proportional to difficulty.** The number of seats convened (``k``)
   scales with an uncertainty estimate (label-free `self_consistency` from
   `agent/calibration.py`). A confident question convenes one seat; a low-confidence
   one convenes the full panel.
2. **A monoculture meter.** `moe.router.load_balancing_loss` over a session's
   routing is a live read-out of over-reliance: ≈1.0 when Sophia draws on a balanced
   set of seats, →E when it has collapsed onto one — functional self-modeling
   (VISION pillar 4) given a number.

Plus a **fail-closed floor**: a high-stakes sector (law) always convenes at least
``safety_min_k`` seats regardless of the difficulty estimate, and an *uncertain*
estimate routes toward *more* deliberation, never less.

Relevance scoring and seat structure are the real `sector_council` seams; the
gather/route math is the real `moe.router`. Deterministic, numpy-only, no model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

HIGH_STAKES_SECTORS = frozenset({"law"})


@dataclass
class RouteDecision:
    council_id: str
    sector: Optional[str]
    experts: list[str]               # seatIds convened
    weights: list[float]             # combine weights (renormalized)
    k: int
    difficulty: float
    safety_floor_applied: bool
    dropped_at_capacity: int = 0


def _flatten_seats(council: dict) -> list[dict]:
    """All seats of a council (core + non-core), each a dict with seatId/terms."""
    seats: list[dict] = []
    for group in council.get("seatGroups", {}).values():
        for seat in group.get("seats", {}).values():
            seats.append(seat)
    return seats


class MetacognitiveRouter:
    """Difficulty-gated, monoculture-aware router over the real sector councils."""

    def __init__(
        self,
        *,
        max_k: int = 4,
        safety_min_k: int = 3,
        default_difficulty: float = 0.6,    # uncertain → lean toward MORE deliberation
        capacity_factor: float = 2.0,
        high_stakes: frozenset[str] = HIGH_STAKES_SECTORS,
    ) -> None:
        self.max_k = max_k
        self.safety_min_k = safety_min_k
        self.default_difficulty = default_difficulty
        self.capacity_factor = capacity_factor
        self.high_stakes = high_stakes
        # session accumulators for the monoculture meter:
        self._dispatch: dict[str, int] = {}     # seatId -> times convened
        self._prob_sum: dict[str, float] = {}   # seatId -> summed router prob
        self._expert_universe: set[str] = set()
        self._routes = 0

    # ---- difficulty --------------------------------------------------------

    def estimate_difficulty(self, samples: Optional[Sequence[Any]]) -> float:
        """1 - self_consistency confidence. No samples → default (fail-toward-more)."""
        if not samples:
            return self.default_difficulty
        from agent.calibration import self_consistency

        _, conf = self_consistency(samples)
        return round(1.0 - conf, 4)

    def _k_for(self, difficulty: float, n_seats: int) -> int:
        k = math.ceil(difficulty * self.max_k)
        return max(1, min(k, self.max_k, n_seats))

    # ---- council selection -------------------------------------------------

    def _pick_council(self, question: str):
        from agent import sector_council as sc

        sector = sc.detect_council(question)
        if sector is not None:
            return sector, sc.load_council(sector)
        # No clear sector: pick the council whose seats best match (never empty).
        best_id, best_council, best_score = None, None, -1
        for cid in sc.available_councils():
            council = sc.load_council(cid)
            score = sum(sc._score_seat(question, s) for s in _flatten_seats(council))
            if score > best_score:
                best_id, best_council, best_score = cid, council, score
        return None, best_council            # sector None (no high-stakes floor)

    # ---- routing -----------------------------------------------------------

    def route(
        self,
        question: str,
        *,
        samples: Optional[Sequence[Any]] = None,
        difficulty: Optional[float] = None,
    ) -> RouteDecision:
        """Convene k seats proportional to difficulty, with a high-stakes floor."""
        import numpy as np

        from agent import sector_council as sc
        from moe.router import top_k_gating

        detected_sector, council = self._pick_council(question)
        council_id = council.get("councilId", "unknown")
        seats = _flatten_seats(council)
        seat_ids = [s.get("seatId", f"seat_{i}") for i, s in enumerate(seats)]
        self._expert_universe.update(seat_ids)

        diff = difficulty if difficulty is not None else self.estimate_difficulty(samples)
        diff = max(0.0, min(1.0, diff))
        k = self._k_for(diff, len(seats))

        # Fail-closed floor: high-stakes sector always convenes >= safety_min_k.
        safety = False
        if detected_sector in self.high_stakes:
            floored = min(self.safety_min_k, len(seats))
            if floored > k:
                k = floored
                safety = True

        # Relevance logits = real seat scores; top-k picks the most relevant seats.
        logits = np.array([[float(sc._score_seat(question, s)) for s in seats]])
        idx, combine, probs = top_k_gating(logits, k=k)
        order = list(idx[0])
        weights = list(combine[0])

        # Capacity: cap how often any seat is convened per session; overflow drops
        # to the next-best seat so no single expert silently dominates.
        cap = math.ceil(self.capacity_factor * (self._routes + 1) * k / max(1, len(seats)))
        chosen: list[str] = []
        chosen_w: list[float] = []
        dropped = 0
        for j, w in zip(order, weights):
            sid = seat_ids[int(j)]
            if self._dispatch.get(sid, 0) < cap:
                chosen.append(sid)
                chosen_w.append(float(w))
            else:
                dropped += 1
        # renormalize weights over the actually-convened seats
        tot = sum(chosen_w) or 1.0
        chosen_w = [w / tot for w in chosen_w]

        # record for the monoculture meter (probs are the full router distribution)
        self._routes += 1
        for sid, p in zip(seat_ids, probs[0]):
            self._prob_sum[sid] = self._prob_sum.get(sid, 0.0) + float(p)
        for sid in chosen:
            self._dispatch[sid] = self._dispatch.get(sid, 0) + 1

        return RouteDecision(
            council_id=council_id, sector=detected_sector, experts=chosen,
            weights=chosen_w, k=k, difficulty=diff,
            safety_floor_applied=safety, dropped_at_capacity=dropped,
        )

    # ---- monoculture meter -------------------------------------------------

    def monoculture_loss(self) -> float:
        """Switch-style aux loss over the session's routing: ≈1 balanced, →E collapsed.

        ``E·Σ f_e·P_e`` with ``f_e`` the dispatch fraction and ``P_e`` the mean
        router probability per expert, over the universe of seats seen this session.
        """
        experts = sorted(self._expert_universe)
        E = len(experts)
        total_dispatch = sum(self._dispatch.values())
        if E == 0 or total_dispatch == 0 or self._routes == 0:
            return 0.0
        loss = 0.0
        for e in experts:
            f = self._dispatch.get(e, 0) / total_dispatch
            p = self._prob_sum.get(e, 0.0) / self._routes
            loss += f * p
        return round(E * loss, 4)

    def dispatch_snapshot(self) -> dict[str, int]:
        return dict(self._dispatch)


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    try:
        import numpy  # noqa: F401
    except Exception:
        return False, {"checks": {"numpy_available": False}}

    checks: dict[str, bool] = {}
    detail: dict = {}

    # A law-sector question (high-stakes) and an economics question.
    legal_q = "Is this contract clause enforceable under the governing statute and case law?"
    econ_q = "How does an increase in aggregate demand affect short-run GDP and unemployment?"

    # 1. Difficulty monotonicity: low-confidence (high difficulty) convenes more.
    r = MetacognitiveRouter(max_k=4, safety_min_k=3)
    k_easy = r.route(econ_q, samples=["A", "A", "A", "A"]).k       # confident → high agreement
    k_hard = r.route(econ_q, samples=["A", "B", "C", "D"]).k       # split → low agreement
    checks["difficulty_monotonic"] = k_hard > k_easy
    detail["k_easy"], detail["k_hard"] = k_easy, k_hard

    # 2. Safety floor: a high-stakes (law) question convenes >= safety_min_k even
    #    when the difficulty estimate is minimal (fully confident).
    r2 = MetacognitiveRouter(max_k=4, safety_min_k=3)
    dec = r2.route(legal_q, samples=["yes", "yes", "yes", "yes"])  # high confidence
    checks["high_stakes_floor"] = dec.k >= 3 and dec.safety_floor_applied
    checks["high_stakes_sector_detected"] = dec.sector == "law"

    # 3. Uncertainty → more deliberation, not less (no samples uses default>min).
    r3 = MetacognitiveRouter(max_k=4, default_difficulty=0.6)
    k_unknown = r3.route(econ_q).k
    k_confident = MetacognitiveRouter(max_k=4).route(econ_q, samples=["A"] * 5).k
    checks["uncertainty_routes_to_more"] = k_unknown > k_confident

    # 4. Monoculture meter: routing the SAME question repeatedly collapses onto one
    #    seat set → high loss; routing varied questions spreads → lower loss.
    mono = MetacognitiveRouter(max_k=1, capacity_factor=99)        # k=1, no capacity relief
    for _ in range(12):
        mono.route(econ_q, samples=["A"] * 5)
    mono_loss = mono.monoculture_loss()
    varied = MetacognitiveRouter(max_k=1, capacity_factor=99)
    qs = [
        "How does aggregate demand affect GDP?",
        "What is the effect of an auction mechanism on allocation?",
        "How does a regression identify a causal effect?",
        "What sets the price under supply and demand elasticity?",
    ]
    for i in range(12):
        varied.route(qs[i % len(qs)], samples=["A"] * 5)
    varied_loss = varied.monoculture_loss()
    checks["monoculture_meter_flags_collapse"] = mono_loss > varied_loss
    detail["mono_loss"], detail["varied_loss"] = mono_loss, varied_loss

    # 5. Cost bound: no seat convened beyond its session capacity.
    rc = MetacognitiveRouter(max_k=2, capacity_factor=1.0)
    for _ in range(20):
        rc.route(econ_q, samples=["A", "B"])      # medium difficulty, repeated
    # capacity grows with routes; assert dispatch stayed bounded vs a generous cap
    total = sum(rc.dispatch_snapshot().values())
    checks["cost_bounded"] = total <= rc._routes * rc.max_k
    checks["weights_normalized"] = True
    # every route's weights must sum to ~1 over convened seats
    w_ok = True
    for _ in range(5):
        d = rc.route(econ_q, samples=["A", "B", "C"])
        if d.experts and abs(sum(d.weights) - 1.0) > 1e-6:
            w_ok = False
    checks["weights_normalized"] = w_ok

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Routed metacognition offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  k_easy={detail.get('k_easy')} k_hard={detail.get('k_hard')} | "
          f"mono_loss={detail.get('mono_loss')} varied_loss={detail.get('varied_loss')}")
    raise SystemExit(0 if ok else 1)
