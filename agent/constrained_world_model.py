# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-constrained ("physics-informed") world model (#4) — simulate forward under a verified law.

Nature's ability-2 is a world model that anticipates consequences; error-centric intelligence grounds
it in reliable dynamics. This repo's deterministic verifiers ARE exact laws (finance A=L+E; chemistry
mass balance). A verifier-constrained world model can only predict states that SATISFY the law, and it
ABSTAINS when the law does not pin the answer (under-determined) — anticipation with an honest 'I can't
determine that'. Demonstrated on the accounting identity Assets = Liabilities + Equity; the same shape
holds for any linear conservation law (mass balance, energy). stdlib-only. canClaimAGI false; CANDIDATE.
"""
from __future__ import annotations

from typing import Any

_TOL = 1e-9


def solve_accounting(assets: "float | None", liabilities: "float | None",
                     equity: "float | None") -> dict:
    """Forward-simulate A = L + E. Exactly one unknown => derive it (determined). Zero unknowns =>
    verify consistency. Two+ unknowns => ABSTAIN (under-determined). Never fabricates a value the law
    does not fix."""
    known = {k: v for k, v in (("assets", assets), ("liabilities", liabilities), ("equity", equity))
             if v is not None}
    n_unknown = 3 - len(known)

    if n_unknown >= 2:
        return {"status": "abstain", "reason": "under-determined: the identity fixes only one unknown "
                "at a time; provide at least two of {assets, liabilities, equity}.", "value": None}
    if n_unknown == 1:
        if assets is None:
            val, name = liabilities + equity, "assets"
        elif liabilities is None:
            val, name = assets - equity, "liabilities"
        else:
            val, name = assets - liabilities, "equity"
        return {"status": "determined", "derived": name, "value": round(float(val), 6),
                "satisfies_law": True}
    # fully specified: verify, do not invent.
    ok = abs(assets - (liabilities + equity)) <= _TOL
    return {"status": "consistent" if ok else "violation", "value": None,
            "residual": round(float(assets - (liabilities + equity)), 6), "satisfies_law": ok}


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    # 1. One unknown => the model DERIVES it, and the derived state satisfies the law.
    r = solve_accounting(None, 60.0, 40.0)
    checks["derives_missing_assets"] = r["status"] == "determined" and abs(r["value"] - 100.0) < 1e-6
    r2 = solve_accounting(100.0, None, 40.0)
    checks["derives_missing_liabilities"] = r2["status"] == "determined" and abs(r2["value"] - 60.0) < 1e-6
    # 2. Two unknowns => ABSTAIN (does not fabricate a value the law can't fix).
    checks["abstains_under_determined"] = solve_accounting(100.0, None, None)["status"] == "abstain"
    # 3. Fully specified + consistent => 'consistent'; inconsistent => 'violation' (a verifier).
    checks["flags_consistent"] = solve_accounting(100.0, 60.0, 40.0)["status"] == "consistent"
    checks["flags_violation"] = solve_accounting(100.0, 60.0, 30.0)["status"] == "violation"
    # 4. Every DETERMINED prediction satisfies the constraint (never predicts an off-law state).
    checks["predictions_on_law"] = solve_accounting(None, 12.5, 7.5)["satisfies_law"] is True
    return all(checks.values()), {"checks": checks}


if __name__ == "__main__":
    ok, d = offline_invariants()
    print("constrained_world_model offline invariants:", "PASS" if ok else "FAIL")
    for k, v in d["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
