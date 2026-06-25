# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decidable numeric invariants for self-gate promotion proofs.

Each function returns a ``formal_verifier`` result dict. Missing inputs degrade to
``verdict="held"`` (fail-closed). Only ``verdict="accepted"`` clears an invariant.
"""

from __future__ import annotations

from typing import Any

from agent.formal_verifier import check_lattice_consistency, require_z3, z3_available


def _held(reason: str, *, backend: str = "n/a") -> dict[str, Any]:
    return {
        "verdict": "held",
        "backend": backend,
        "status": "missing_inputs",
        "reasons": [reason],
        "model": None,
    }


def protected_floor_content(
    metrics: list[dict[str, Any]],
    *,
    tolerance: float,
    protected_suites: tuple[str, ...] = ("religion", "history"),
) -> dict[str, Any]:
    """Protected suite s: CONTENT_after >= CONTENT_before - tolerance."""
    if not metrics:
        return _held("no protected CONTENT metrics supplied")
    protected_set = {s.lower() for s in protected_suites}
    assignments: dict[str, int] = {}
    constraints: list[tuple[str, str, str]] = []
    found = False
    for m in metrics:
        suite = str(m.get("suite", "")).lower()
        if suite not in protected_set:
            continue
        before = m.get("contentBefore")
        after = m.get("contentAfter")
        if before is None or after is None:
            return _held(f"missing CONTENT channel for protected suite {suite!r}")
        found = True
        var = f"content_after_{suite}"
        assignments[var] = round(float(after) * 1000)
        floor_rank = round((float(before) - tolerance) * 1000)
        constraints.append((var, ">=", str(floor_rank)))
    if not found:
        return _held("no protected CONTENT metrics supplied")
    return check_lattice_consistency(assignments, constraints)


def no_total_regression(
    *,
    before_total: float | None,
    after_total: float | None,
    tolerance: float,
) -> dict[str, Any]:
    """COMBINED total_after >= total_before - tolerance."""
    if before_total is None or after_total is None:
        return _held("missing COMBINED total scores")
    assignments = {"total_after": round(float(after_total) * 1000)}
    floor_rank = round((float(before_total) - tolerance) * 1000)
    return check_lattice_consistency(assignments, [("total_after", ">=", str(floor_rank))])


def contamination_zero(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Decontamination overlap count from build manifest must be zero."""
    if manifest is None:
        return _held("manifest not supplied")
    contamination = manifest.get("contamination")
    if not isinstance(contamination, dict):
        return _held("manifest lacks contamination block")
    overlap = None
    for key in ("eval", "vsEval"):
        block = contamination.get(key)
        if isinstance(block, dict) and block.get("overlapCount") is not None:
            overlap = block.get("overlapCount")
            break
    if overlap is None and contamination.get("clean") is True:
        overlap = 0
    if overlap is None:
        return _held("manifest lacks contamination overlap count")
    if int(overlap) == 0:
        backend = "z3" if z3_available() else "fallback"
        return {
            "verdict": "accepted",
            "backend": backend,
            "status": "consistent",
            "reasons": ["decontamination eval overlap count is 0"],
            "model": {"overlapCount": 0},
        }
    return {
        "verdict": "rejected",
        "backend": "fallback",
        "status": "contradiction",
        "reasons": [f"decontamination eval overlap count is {overlap}, expected 0"],
        "model": {"overlapCount": int(overlap)},
    }


def _trace_has_provenance_citation(trace: dict[str, Any]) -> bool:
    meta = trace.get("metadata") or {}
    for key in ("sourceCitation", "provenance", "citation", "source"):
        val = meta.get(key)
        if val:
            return True
    messages = trace.get("messages") or []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = str(msg.get("content", ""))
            if "http://" in content or "https://" in content or "doi:" in content.lower():
                return True
    return False


def provenance_complete(traces: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Promoted traces lacking provenance citation must be zero."""
    if traces is None:
        return _held("training traces not supplied")
    lacking = [t for t in traces if not _trace_has_provenance_citation(t)]
    if lacking:
        return {
            "verdict": "rejected",
            "backend": "fallback",
            "status": "contradiction",
            "reasons": [f"{len(lacking)} trace(s) lack provenance citation"],
            "model": {"lackingCount": len(lacking)},
        }
    backend = "z3" if z3_available() else "fallback"
    return {
        "verdict": "accepted",
        "backend": backend,
        "status": "consistent",
        "reasons": ["all supplied traces carry provenance citation"],
        "model": {"lackingCount": 0},
    }


def solver_attestation() -> dict[str, Any]:
    """Optional z3 attestation over the empty constraint set (fail-closed without z3)."""
    return require_z3(check_lattice_consistency, {}, [])


def run_invariant_suite(
    *,
    protected_content_metrics: list[dict[str, Any]],
    before_total: float | None,
    after_total: float | None,
    manifest: dict[str, Any] | None,
    traces: list[dict[str, Any]] | None,
    tolerance: float,
    protected_suites: tuple[str, ...] = ("religion", "history"),
) -> dict[str, dict[str, Any]]:
    return {
        "protected_floor_content": protected_floor_content(
            protected_content_metrics, tolerance=tolerance, protected_suites=protected_suites
        ),
        "no_total_regression": no_total_regression(
            before_total=before_total, after_total=after_total, tolerance=tolerance
        ),
        "contamination_zero": contamination_zero(manifest),
        "provenance_complete": provenance_complete(traces),
        "solver_attestation": solver_attestation(),
    }


__all__ = [
    "protected_floor_content",
    "no_total_regression",
    "contamination_zero",
    "provenance_complete",
    "solver_attestation",
    "run_invariant_suite",
]
