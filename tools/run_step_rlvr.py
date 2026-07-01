#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Process-RLVR (per-step verifier-as-reward) — offline invariants + runbook.

The step-level analogue of ``tools/run_rlvr.py``. The GRPO reward is the
process reward of ``provenance_bench.step_reward`` (every verified step rewarded,
every misstep penalised). The REAL GPU run is launched through GitHub Actions /
RunPod (never local SSH; read the wisdom-gpu-prebaked cost-guard runbook first) —
this file ships the **offline invariants** that must hold before any GPU spend,
exactly the ``run_rlvr.py --model mock`` discipline:

  1. reward is DETERMINISTIC (same derivation -> same score),
  2. BOUNDED in [-1, 1],
  3. a fully-verified clean chain scores +1,
  4. a derivation containing a real misstep scores < a clean one (monotone in the
     right direction) and a fully-wrong one scores negative,
  5. an unverifiable derivation scores 0 (fail-closed, never a silent reward),
  6. the ``agent.step_verifier`` seam is actually invoked.

``python tools/run_step_rlvr.py`` runs the invariants and exits non-zero on any
failure (CI-safe, no GPU, no model). ``canClaimAGI`` unaffected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import math_verifier as mv  # noqa: E402
from provenance_bench.step_reward import reward_for_derivation  # noqa: E402

# Physics chains (always available, sympy-independent).
_CLEAN_PHYS = [{"expr": "5 W"}, {"expr": "5 J/s"}]
_MISSTEP_PHYS = [{"expr": "5 W"}, {"expr": "5 J"}]  # dimension error
# Math chains (need sympy; skipped from the strict asserts when absent).
_CLEAN_MATH = [{"expr": "(x+1)**2"}, {"expr": "x**2 + 2*x + 1"}]
_MISSTEP_MATH = [{"expr": "(x+1)**2"}, {"expr": "x**2 - 2*x + 1"}]


def check_invariants() -> list[str]:
    failures: list[str] = []

    # (1) deterministic + (2) bounded + (3) clean -> +1 (physics, always on)
    s1, _ = reward_for_derivation(_CLEAN_PHYS, gold="5 W", domain="physics")
    s2, _ = reward_for_derivation(_CLEAN_PHYS, gold="5 W", domain="physics")
    if s1 != s2:
        failures.append(f"non-deterministic reward ({s1} != {s2})")
    if not (-1.0 <= s1 <= 1.0):
        failures.append(f"reward out of [-1,1]: {s1}")
    if s1 != 1.0:
        failures.append(f"clean physics chain should score +1, got {s1}")

    # (4) misstep scores negative and strictly below clean
    sm, dm = reward_for_derivation(_MISSTEP_PHYS, gold="5 W", domain="physics")
    if sm >= s1:
        failures.append(f"misstep ({sm}) not below clean ({s1})")
    if sm >= 0.0:
        failures.append(f"a dimension-error derivation should score < 0, got {sm}")
    if dm.get("verdict") != "rejected":
        failures.append("step_verifier seam not invoked (no rejected verdict on a dimension error)")

    # (5) unverifiable -> 0 (no checks)
    s0, d0 = reward_for_derivation([{"expr": "5 W"}], domain="physics")  # single step, no gold
    if s0 != 0.0:
        failures.append(f"unverifiable derivation should score 0, got {s0}")

    # math chains only when sympy is present (the production default for the math oracle)
    if mv.sympy_available():
        scm, _ = reward_for_derivation(_CLEAN_MATH, gold="x**2 + 2*x + 1", domain="math")
        smm, _ = reward_for_derivation(_MISSTEP_MATH, gold="x**2 + 2*x + 1", domain="math")
        if scm != 1.0:
            failures.append(f"clean math chain should score +1, got {scm}")
        if smm >= scm:
            failures.append(f"math misstep ({smm}) not below clean ({scm})")
    return failures


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="(default behaviour) run offline invariants")
    ap.parse_args(argv)

    failures = check_invariants()
    if failures:
        print("STEP-RLVR INVARIANTS: FAIL")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("STEP-RLVR INVARIANTS: OK")
    print("  ✓ deterministic, bounded [-1,1], clean=+1, misstep<clean & <0, unverifiable=0, seam invoked")
    print("  (real GPU GRPO run goes through GitHub Actions + RunPod — see wisdom-gpu-prebaked cost guard)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
