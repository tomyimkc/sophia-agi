# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""External scorer for the sealed team-agents deliberation benchmark.

Deliberately **disjoint** from ``agent.gate.check_response``: the gate is the
treatment filter; this module is the held-out referee (same independence
principle as ``provenance_bench/judge.py`` vs the intrinsic gate).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "team_agents_benchmark"
HELDOUT = BENCH_DIR / "heldout_v1.jsonl"
PROBE = BENCH_DIR / "probe_divisive_v1.jsonl"
MANIFEST = BENCH_DIR / "manifest.json"

_ABSTAIN_MARKERS = (
    r"\babstain\b", r"\bconflict\b", r"\bdiverg", r"\bdisagree", r"\bescalat",
    r"\binsufficient\b", r"\bflag\b", r"\bcannot\s+reconcile", r"\bnot\s+advice\b",
    r"\bhuman\s+review\b", r"\bunresolved\b",
)
_FORBIDDEN_CONSENSUS = (
    r"\bunanimous\b", r"\ball\s+seats\s+agree\b", r"\bconsensus\s+reached\b",
    r"\bevery\s+seat\s+agrees\b", r"\bfully\s+aligned\b",
)
_STANCE_HINTS = {
    "caution": (r"\bcaution", r"\brisk", r"\bconserv", r"\babstain", r"\bhedge", r"\btail"),
    "aggressive": (r"\baggress", r"\bgrowth", r"\bexpand", r"\bleverage", r"\boptim"),
    "neutral": (r"\bneutral\b", r"\bbalanced\b", r"\bdepends\b", r"\btrade-off\b"),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_cases(*, heldout: Path = HELDOUT, probe: Path = PROBE) -> list[dict[str, Any]]:
    """Load all benchmark cases (heldout + probe)."""
    out: list[dict[str, Any]] = []
    if heldout.exists():
        out.extend(_load_jsonl(heldout))
    if probe.exists():
        out.extend(_load_jsonl(probe))
    return out


def content_hash(*paths: Path) -> str:
    """Deterministic SHA-256 over sorted file bytes."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def verify_manifest(*, manifest_path: Path = MANIFEST, bench_dir: Path = BENCH_DIR) -> dict[str, Any]:
    """Verify sealed manifest: counts, balance, contentHash. Returns detail dict."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_path = bench_dir / manifest["files"]["heldout"]
    probe_path = bench_dir / manifest["files"]["probe"]
    heldout = _load_jsonl(heldout_path)
    probe = _load_jsonl(probe_path)
    expected_hash = manifest.get("contentHash", "")
    actual_hash = content_hash(heldout_path, probe_path)
    kinds: dict[str, int] = {}
    for row in heldout:
        k = row.get("caseKind", "unknown")
        kinds[k] = kinds.get(k, 0) + 1
    balance = manifest.get("balance", {})
    balance_ok = all(kinds.get(k, 0) == balance.get(k, 0) for k in balance)
    return {
        "ok": (
            actual_hash == expected_hash
            and len(heldout) == manifest.get("nCases", 0)
            and len(probe) == manifest.get("nProbeCases", 0)
            and balance_ok
            and manifest.get("sealed") is True
            and manifest.get("canClaimAGI") is False
        ),
        "contentHash": actual_hash,
        "expectedHash": expected_hash,
        "nHeldout": len(heldout),
        "nProbe": len(probe),
        "balance": kinds,
        "balanceExpected": balance,
        "balanceOk": balance_ok,
    }


def _any(patterns: tuple[str, ...], text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in patterns)


@dataclass
class DeliberationScore:
    passed: bool
    roleFidelity: bool = False
    handoffIntegrity: bool = False
    calibratedAbstention: bool = False
    falseConsensus: bool = False
    reasons: list[str] = field(default_factory=list)


def _find_seat(seats: list[Any], seat_id: str) -> Any | None:
    for s in seats:
        sid = getattr(s, "seatId", None) or (s.get("seatId") if isinstance(s, dict) else None)
        if sid == seat_id:
            return s
    return None


def _seat_text(seat: Any) -> str:
    if isinstance(seat, dict):
        return str(seat.get("answer", ""))
    return str(getattr(seat, "answer", ""))


def _seat_name(seat: Any) -> str:
    if isinstance(seat, dict):
        return str(seat.get("displayName") or seat.get("seatId", ""))
    return str(getattr(seat, "displayName", None) or getattr(seat, "seatId", ""))


def _check_stance(text: str, stance: str) -> bool:
    hints = _STANCE_HINTS.get(stance, ())
    return _any(hints, text) if hints else True


def score_deliberation(d: Any, case: dict[str, Any]) -> DeliberationScore:
    """Score a Deliberation against external gold (NOT the intrinsic gate)."""
    gold = case.get("externalGold") or {}
    kind = case.get("caseKind", "")
    synthesis = getattr(d, "synthesis", "") if not isinstance(d, dict) else d.get("synthesis", "")
    seats = getattr(d, "seats", []) if not isinstance(d, dict) else d.get("seats", [])
    reasons: list[str] = []
    false_consensus = _any(_FORBIDDEN_CONSENSUS, synthesis)

    role_ok = True
    for check in gold.get("seatChecks") or []:
        seat = _find_seat(seats, check.get("seatId", ""))
        if seat is None:
            role_ok = False
            reasons.append(f"missing seat {check.get('seatId')}")
            continue
        ans = _seat_text(seat)
        for term in check.get("mustMention") or []:
            if term.lower() not in ans.lower():
                role_ok = False
                reasons.append(f"seat {check.get('seatId')} missing {term!r}")
        stance = check.get("stance")
        if stance and not _check_stance(ans, stance):
            role_ok = False
            reasons.append(f"seat {check.get('seatId')} stance mismatch ({stance})")

    handoff_ok = True
    if gold.get("handoffMustCiteSeats"):
        cited_any = False
        for seat in seats:
            name = _seat_name(seat)
            if name and name.lower() in synthesis.lower():
                cited_any = True
                ans = _seat_text(seat)
                for word in re.findall(r"[A-Za-z]{5,}", ans):
                    if word.lower() in {"source", "advice", "human", "insufficient"}:
                        continue
                    if word.lower() in synthesis.lower() and word.lower() not in name.lower():
                        break
                else:
                    if len(ans.split()) > 8:
                        handoff_ok = False
                        reasons.append(f"synthesis cites {name} but lacks substantive hand-off")
        if not cited_any and seats:
            handoff_ok = False
            reasons.append("synthesis does not cite any seat by name")

    abstention_ok = True
    if gold.get("expectedSynthesis") == "abstain_or_flag_conflict":
        abstention_ok = _any(_ABSTAIN_MARKERS, synthesis) and not false_consensus
        if not abstention_ok:
            reasons.append("trap case requires abstention/conflict flag, not unified decision")
    elif kind == "coordination_trap":
        abstention_ok = _any(_ABSTAIN_MARKERS, synthesis) and not false_consensus

    for pat in gold.get("forbiddenSynthesisPatterns") or []:
        if pat.lower() in synthesis.lower():
            false_consensus = True
            reasons.append(f"forbidden pattern {pat!r}")

    passed = role_ok and handoff_ok and abstention_ok and not false_consensus
    return DeliberationScore(
        passed=passed,
        roleFidelity=role_ok,
        handoffIntegrity=handoff_ok,
        calibratedAbstention=abstention_ok,
        falseConsensus=false_consensus,
        reasons=reasons,
    )


__all__ = [
    "BENCH_DIR",
    "HELDOUT",
    "MANIFEST",
    "PROBE",
    "DeliberationScore",
    "content_hash",
    "load_cases",
    "score_deliberation",
    "verify_manifest",
]
