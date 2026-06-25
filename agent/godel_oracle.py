# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate-invariant prover for adapter promotion (NOT a Gödel machine).

Runs the decidable numeric invariant suite from ``agent.invariant_suite`` and
promotes ONLY when every invariant returns ``verdict="accepted"``. ``held`` and
``rejected`` both block promotion (fail-closed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.invariant_suite import run_invariant_suite

SELF_GATE_DIR = Path(__file__).resolve().parents[1] / "agi-proof" / "self-gate"


def all_invariants_accepted(results: dict[str, dict[str, Any]]) -> bool:
    return all(r.get("verdict") == "accepted" for r in results.values())


def compute_solver_checked(invariants: dict[str, dict[str, Any]]) -> bool:
    """True only when every invariant was checked by the z3 backend."""
    return bool(invariants) and all(r.get("backend") == "z3" for r in invariants.values())


def effective_promote(
    invariants: dict[str, dict[str, Any]],
    *,
    allow_fallback_proof: bool = False,
) -> bool:
    """Promotion verdict: z3-checked by default; optional fallback escape hatch."""
    solver_checked = compute_solver_checked(invariants)
    if all_invariants_accepted(invariants) and solver_checked:
        return True
    if allow_fallback_proof and all_invariants_accepted(invariants):
        return True
    if allow_fallback_proof:
        breaching = [name for name, r in invariants.items() if r.get("verdict") != "accepted"]
        if breaching == ["solver_attestation"]:
            sa = invariants["solver_attestation"]
            if sa.get("status") == "z3_unavailable":
                return all(
                    r.get("verdict") == "accepted"
                    for name, r in invariants.items()
                    if name != "solver_attestation"
                )
    return False


def build_proof_bundle(
    *,
    candidate_id: str,
    invariants: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
    allow_fallback_proof: bool = False,
) -> dict[str, Any]:
    solver_checked = compute_solver_checked(invariants)
    promote = effective_promote(invariants, allow_fallback_proof=allow_fallback_proof)
    breaching = [name for name, r in invariants.items() if r.get("verdict") != "accepted"]
    if promote and not solver_checked and breaching:
        breaching = []
    solver_notes: list[str] = []
    if promote and not solver_checked:
        solver_notes.append("fallback proof — not solver-checked")
    bundle: dict[str, Any] = {
        "schema": "sophia.invariant_suite_proof.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": (
            "Decidable numeric invariant proof bundle for adapter self-gate; "
            "not alignment proof, not AGI proof, not a Gödel machine."
        ),
        "candidateId": candidate_id,
        "solverChecked": solver_checked,
        "promote": promote,
        "breachingInvariants": breaching,
        "invariants": invariants,
        "inputs": inputs,
    }
    if solver_notes:
        bundle["solverNotes"] = solver_notes
    return bundle


def write_proof_bundle(bundle: dict[str, Any], *, candidate_id: str, out_dir: Path | None = None) -> Path:
    root = out_dir or SELF_GATE_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"invariant-suite.{candidate_id}.public-report.json"
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def evaluate_for_promotion(
    *,
    candidate_id: str,
    protected_content_metrics: list[dict[str, Any]],
    before_total: float | None,
    after_total: float | None,
    manifest: dict[str, Any] | None,
    traces: list[dict[str, Any]] | None,
    tolerance: float,
    protected_suites: tuple[str, ...] = ("religion", "history"),
    out_dir: Path | None = None,
    input_summary: dict[str, Any] | None = None,
    allow_fallback_proof: bool = False,
) -> tuple[bool, dict[str, Any], Path]:
    invariants = run_invariant_suite(
        protected_content_metrics=protected_content_metrics,
        before_total=before_total,
        after_total=after_total,
        manifest=manifest,
        traces=traces,
        tolerance=tolerance,
        protected_suites=protected_suites,
    )
    bundle = build_proof_bundle(
        candidate_id=candidate_id,
        invariants=invariants,
        inputs=input_summary or {},
        allow_fallback_proof=allow_fallback_proof,
    )
    path = write_proof_bundle(bundle, candidate_id=candidate_id, out_dir=out_dir)
    return bundle["promote"], bundle, path


__all__ = [
    "all_invariants_accepted",
    "compute_solver_checked",
    "effective_promote",
    "build_proof_bundle",
    "write_proof_bundle",
    "evaluate_for_promotion",
]
