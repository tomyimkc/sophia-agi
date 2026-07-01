# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CI regression for the 6 world-model prototypes + the #1 ablation + the recipe linter.

Makes the error-centric / world-model prototypes (Nature "change in tack"; arXiv 2510.15128) STANDING
checks: each module's GPU-free offline_invariants must stay green, the proactive-abstention ablation
must keep its held-out delta, and the recipe spec must keep every adopted ingredient proof-gated. No
capability claim; canClaimAGI stays false. Runnable under pytest OR as a plain script (no pytest dep).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def test_verifiability_model():
    from agent.verifiability_model import offline_invariants
    ok, d = offline_invariants()
    assert ok, d["checks"]


def test_counterfactual_traps():
    import counterfactual_traps
    ok, d = counterfactual_traps.offline_invariants()
    assert ok, d["checks"]


def test_causal_ablation():
    import causal_ablation
    ok, d = causal_ablation.offline_invariants()
    assert ok, d["checks"]


def test_constrained_world_model():
    from agent.constrained_world_model import offline_invariants
    ok, d = offline_invariants()
    assert ok, d["checks"]


def test_active_query():
    from agent.active_query import offline_invariants
    ok, d = offline_invariants()
    assert ok, d["checks"]


def test_process_world_model():
    from agent.process_world_model import offline_invariants
    ok, d = offline_invariants()
    assert ok, d["checks"]


def test_verifiability_ablation_holds_out_delta():
    import ablate_verifiability
    ok, d = ablate_verifiability.offline_invariants()
    assert ok, d["checks"]
    # the held-out proactive-gate delta must stay strong at low control cost.
    r = d["result"]
    assert r["ablation_delta"] >= 0.8 and r["control_over_abstain_rate"] <= 0.2, r


def test_recipe_spec_lints_and_adopted_are_proof_gated():
    import json

    import lint_recipe
    spec = json.loads((ROOT / "docs" / "06-Roadmap" / "recipe_spec.json").read_text())
    ok, errs, info = lint_recipe.lint(spec)
    assert ok, errs
    # every adopted ingredient must have a numeric ablation delta + a gate (no over-promotion).
    for ing in spec["ingredients"]:
        if ing["adopted"]:
            assert ing["proofStatus"] == "validated"
            assert isinstance(ing["ablationDelta"], (int, float)) and not isinstance(ing["ablationDelta"], bool)
            assert ing["gate"]


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [ok] {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  [XX] {fn.__name__}: {exc}")
    print(f"world-model regression: {'PASS' if not failed else 'FAIL'} ({len(fns) - failed}/{len(fns)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
