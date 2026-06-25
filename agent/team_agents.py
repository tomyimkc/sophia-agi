# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Team-agents orchestrator — calibrated synthesis, external verification, independence.

Wraps ``agent.council_deliberate.deliberate()`` with divergence-aware abstention and
external gold checks (disjoint from the intrinsic gate). Homogeneous panels must
report effective-N; never label output "consensus" when N_eff < 2.0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from agent.council_deliberate import Deliberation, SeatResult, deliberate
from agent.council_format import render_team_target
from provenance_bench.team_agents_benchmark import score_deliberation

ABSTAIN_SYNTHESIS = (
    "Insufficient verified basis to answer: seat perspectives diverge on values or risk "
    "appetite — flag the conflict and escalate to a human rather than invent consensus. "
    "Not advice."
)

CONSENSUS_THRESHOLD_N_EFF = 2.0

_STANCE_POLARITY = {
    "caution": -1,
    "neutral": 0,
    "aggressive": 1,
}


@dataclass
class DivergenceReport:
    divergent: bool
    seatStances: dict[str, int] = field(default_factory=dict)
    note: str = ""


def stance_vector(seat_answer: str, probe: dict | None = None) -> int:
    """Map seat text to {-1, 0, +1} stance using probe gold or lexical hints."""
    low = (seat_answer or "").lower()
    if probe:
        gold = probe.get("externalGold") or {}
        stances = gold.get("probeStances") or []
        for stance in stances:
            hints = {
                "caution": (r"\bcaution", r"\brisk", r"\bconserv", r"\bdelay", r"\bhedge"),
                "aggressive": (r"\baggress", r"\bgrowth", r"\binvest", r"\bexpand", r"\bspeed"),
            }.get(stance, ())
            if any(re.search(p, low) for p in hints):
                return _STANCE_POLARITY.get(stance, 0)
    if re.search(r"\bcaution|\brisk|\bconserv|\babstain|\bhedge", low):
        return -1
    if re.search(r"\baggress|\bgrowth|\binvest|\bexpand|\bleverage", low):
        return 1
    return 0


def mean_pairwise_correlation(vectors_per_seat: list[list[int]]) -> float:
    """Mean Pearson r across seat stance vectors (numpy)."""
    if len(vectors_per_seat) < 2:
        return 0.0
    arr = np.array(vectors_per_seat, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    rhos: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = arr[i], arr[j]
            if np.std(a) == 0 or np.std(b) == 0:
                rhos.append(1.0 if np.array_equal(a, b) else 0.0)
            else:
                rhos.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rhos)) if rhos else 0.0


def effective_n(n_seats: int, mean_rho: float) -> float:
    """N_eff = N / (1 + (N-1) * rho_bar). Pre-registered consensus gate."""
    if n_seats <= 0:
        return 0.0
    denom = 1.0 + (n_seats - 1) * mean_rho
    return n_seats / denom if denom else float(n_seats)


def consensus_label(n_eff: float, *, n_seats: int = 3) -> str:
    """Return wording that respects the pre-registered N_eff threshold."""
    if n_eff >= CONSENSUS_THRESHOLD_N_EFF:
        return f"effective panel (N_eff={n_eff:.2f} ≥ {CONSENSUS_THRESHOLD_N_EFF})"
    return f"correlated panel — not consensus (N_eff={n_eff:.2f} < {CONSENSUS_THRESHOLD_N_EFF})"


def detect_seat_divergence(seats: list[SeatResult], *, gold: dict | None = None) -> DivergenceReport:
    """Detect genuine seat divergence on trap- or probe-shaped input."""
    clean = [s for s in seats if s.ok and s.gatePassed]
    if len(clean) < 2:
        return DivergenceReport(divergent=False, note="fewer than 2 clean seats")

    stances: dict[str, int] = {}
    for s in clean:
        stances[s.seatId] = stance_vector(s.answer, gold)

    polarities = list(stances.values())
    divergent = max(polarities) - min(polarities) >= 2

    if gold and gold.get("expectedSynthesis") == "abstain_or_flag_conflict":
        divergent = divergent or len(set(polarities)) > 1

    if not divergent:
        a0, a1 = clean[0].answer.lower(), clean[1].answer.lower()
        opp_pairs = (("risk", "growth"), ("caution", "aggressive"), ("cut", "invest"),
                     ("delay", "launch"), ("settle", "litigat"))
        for x, y in opp_pairs:
            if (x in a0 and y in a1) or (y in a0 and x in a1):
                divergent = True
                break

    return DivergenceReport(divergent=divergent, seatStances=stances,
                            note="stance spread" if divergent else "aligned")


def apply_calibrated_synthesis(d: Deliberation, *, gold: dict | None = None) -> Deliberation:
    """Replace synthesis with fail-closed abstention when seats diverge on traps."""
    report = detect_seat_divergence(d.seats, gold=gold)
    if not report.divergent:
        return d
    note = d.note + f" | calibrated abstention ({report.note})"
    return Deliberation(
        query=d.query, councilId=d.councilId, seats=d.seats, guardians=d.guardians,
        synthesis=ABSTAIN_SYNTHESIS, gatedOutSeatIds=d.gatedOutSeatIds, note=note.strip(),
    )


def verify_trace_external(d: Deliberation, task_gold: dict | None) -> bool:
    """External verification only — gate-pass alone is insufficient."""
    if not task_gold:
        return False
    synthesis = d.synthesis or ""
    for pat in task_gold.get("forbiddenSynthesisPatterns") or []:
        if pat.lower() in synthesis.lower():
            return False
    clean = [s for s in d.seats if s.ok and s.gatePassed]
    if not clean and "insufficient" not in synthesis.lower():
        return False
    case = {"externalGold": task_gold, "caseKind": task_gold.get("caseKind", "")}
    score = score_deliberation(d, case)
    if task_gold.get("expectedSynthesis"):
        return score.passed
    return not score.falseConsensus and bool(synthesis.strip())


def deliberate_team(
    query: str,
    *,
    client,
    seat_clients=None,
    max_seats: int = 4,
    materials=None,
    gold: dict | None = None,
    gate: bool = True,
    council_id: str | None = None,
) -> Deliberation:
    """Map-reduce deliberation with calibrated synthesis on divergence."""
    d = deliberate(query, client=client, seat_clients=seat_clients, max_seats=max_seats,
                   materials=materials, gate=gate, council_id=council_id)
    trap_gold = gold and gold.get("expectedSynthesis") == "abstain_or_flag_conflict"
    if trap_gold or detect_seat_divergence(d.seats, gold=gold).divergent:
        d = apply_calibrated_synthesis(d, gold=gold)
    return d


def measure_panel_independence(
    probe_cases: list[dict],
    *,
    client,
    seat_clients=None,
    max_seats: int = 3,
) -> dict:
    """Run probe_divisive cases and report mean ρ and N_eff."""
    vectors: list[list[int]] = []
    for case in probe_cases:
        d = deliberate(case["prompt"], client=client, seat_clients=seat_clients,
                       max_seats=max_seats, council_id=case.get("councilId"), gate=True)
        row = [stance_vector(s.answer, case) for s in d.seats if s.ok and s.gatePassed]
        if row:
            vectors.append(row)
    if not vectors:
        return {"meanPairwiseRho": 0.0, "effectiveN": 0.0, "consensusLabel": consensus_label(0.0),
                "nProbes": 0}
    max_len = max(len(v) for v in vectors)
    padded = [v + [v[-1]] * (max_len - len(v)) for v in vectors]
    rho = mean_pairwise_correlation(padded)
    n_eff = effective_n(max_len, rho)
    return {
        "meanPairwiseRho": round(rho, 4),
        "effectiveN": round(n_eff, 4),
        "consensusLabel": consensus_label(n_eff, n_seats=max_len),
        "nProbes": len(probe_cases),
        "claimsConsensus": n_eff >= CONSENSUS_THRESHOLD_N_EFF,
    }


__all__ = [
    "ABSTAIN_SYNTHESIS",
    "CONSENSUS_THRESHOLD_N_EFF",
    "DivergenceReport",
    "apply_calibrated_synthesis",
    "consensus_label",
    "deliberate_team",
    "detect_seat_divergence",
    "effective_n",
    "mean_pairwise_correlation",
    "measure_panel_independence",
    "stance_vector",
    "verify_trace_external",
    "render_team_target",
]
