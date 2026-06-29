# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""External scorer for the sealed team-agents **long-task** benchmark.

Deliberately **disjoint** from ``agent.gate.check_response`` and extends the
short-horizon team scorer with sub-step coverage and task-completion rubrics
for multi-step / long-horizon deliberation cases.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from provenance_bench.team_agents_benchmark import (
    DeliberationScore,
    _find_seat,
    _seat_text,
    score_deliberation,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "team_agents_longtask"
HELDOUT = BENCH_DIR / "heldout_v1.jsonl"
MANIFEST = BENCH_DIR / "manifest.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_cases(*, heldout: Path = HELDOUT) -> list[dict[str, Any]]:
    """Load all long-task benchmark cases."""
    if heldout.exists():
        return _load_jsonl(heldout)
    return []


def content_hash(*paths: Path) -> str:
    """Deterministic SHA-256 over sorted file bytes."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def verify_manifest(*, manifest_path: Path = MANIFEST, bench_dir: Path = BENCH_DIR) -> dict[str, Any]:
    """Verify sealed manifest: counts, balance, contentHash."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_path = bench_dir / manifest["files"]["heldout"]
    heldout = _load_jsonl(heldout_path)
    expected_hash = manifest.get("contentHash", "")
    actual_hash = content_hash(heldout_path)
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
            and balance_ok
            and manifest.get("sealed") is True
            and manifest.get("canClaimAGI") is False
        ),
        "contentHash": actual_hash,
        "expectedHash": expected_hash,
        "nHeldout": len(heldout),
        "balance": kinds,
        "balanceExpected": balance,
        "balanceOk": balance_ok,
    }


def _synthesis_text(d: Any) -> str:
    if isinstance(d, dict):
        return str(d.get("synthesis", ""))
    return str(getattr(d, "synthesis", ""))


def _seats_list(d: Any) -> list[Any]:
    if isinstance(d, dict):
        return d.get("seats", []) or []
    return getattr(d, "seats", []) or []


def score_sub_step_coverage(d: Any, case: dict[str, Any]) -> tuple[float, bool]:
    """Fraction of subSteps covered in seat answers or synthesis."""
    sub_steps = case.get("subSteps") or []
    if not sub_steps:
        return 1.0, True
    synthesis = _synthesis_text(d)
    seats = _seats_list(d)
    covered = 0
    for step in sub_steps:
        terms = step.get("mustMention") or []
        seat_id = step.get("seatId")
        texts: list[str] = [synthesis]
        if seat_id:
            seat = _find_seat(seats, seat_id)
            if seat is not None:
                texts.append(_seat_text(seat))
        hit = all(
            any(term.lower() in t.lower() for t in texts)
            for term in terms
        )
        if hit:
            covered += 1
    rate = covered / len(sub_steps)
    gold = case.get("externalGold") or {}
    min_rate = float(gold.get("subStepCoverageMin", 0.67))
    return round(rate, 4), rate >= min_rate


@dataclass
class LongTaskScore(DeliberationScore):
    subStepCoverage: float = 0.0
    subStepCoverageOk: bool = False
    taskCompletion: bool = False


def score_longtask(d: Any, case: dict[str, Any]) -> LongTaskScore:
    """Score a deliberation on long-task external gold (NOT the intrinsic gate)."""
    base = score_deliberation(d, case)
    coverage_rate, coverage_ok = score_sub_step_coverage(d, case)
    kind = case.get("caseKind", "")
    task_ok = base.roleFidelity and coverage_ok
    if kind == "long_coordination_trap":
        task_ok = base.calibratedAbstention and not base.falseConsensus
    elif kind in ("multi_domain_chain", "chained_subquestions"):
        task_ok = task_ok and base.handoffIntegrity
    passed = base.passed and coverage_ok and task_ok
    reasons = list(base.reasons)
    if not coverage_ok:
        reasons.append(f"subStepCoverage {coverage_rate} below minimum")
    if not task_ok and kind != "long_coordination_trap":
        reasons.append("taskCompletion rubric failed")
    return LongTaskScore(
        passed=passed,
        roleFidelity=base.roleFidelity,
        handoffIntegrity=base.handoffIntegrity,
        calibratedAbstention=base.calibratedAbstention,
        falseConsensus=base.falseConsensus,
        reasons=reasons,
        subStepCoverage=coverage_rate,
        subStepCoverageOk=coverage_ok,
        taskCompletion=task_ok,
    )


__all__ = [
    "BENCH_DIR",
    "HELDOUT",
    "LongTaskScore",
    "MANIFEST",
    "content_hash",
    "load_cases",
    "score_longtask",
    "score_sub_step_coverage",
    "verify_manifest",
]
