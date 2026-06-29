#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the decomposition/coverage swarm benchmark (offline, mock)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from provenance_bench import swarm_coverage_eval as cov  # noqa: E402


def test_offline_invariants() -> None:
    ok, detail = cov.offline_invariants()
    assert ok, detail["checks"]


def test_failclosed_empty_swarm() -> None:
    assert cov.swarm_answer(lambda s, u: "" if "specialist" in s else "x", "Q") == ""


def test_coverage_win_requires_ci_excluding_zero() -> None:
    tasks = [cov.CoverageTask(f"Q {i}", (f"a{i}", f"b{i}")) for i in range(8)]
    # identical arms → no win
    same = cov.run_coverage(tasks, lambda s, u: "nothing", lambda a, asp: 0, subject="m")
    assert not same.is_win and same.delta == 0.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  [ok] {fn.__name__}")
        except Exception:
            failed += 1; print(f"  [XX] {fn.__name__}"); traceback.print_exc()
    print(f"{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
