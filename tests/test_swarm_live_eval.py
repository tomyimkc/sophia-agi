#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the real-model end-to-end swarm benchmark (offline, mock model)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from provenance_bench import swarm_live_eval as sle  # noqa: E402
from agent.swarm_router import SwarmRouter  # noqa: E402


def test_offline_invariants() -> None:
    ok, detail = sle.offline_invariants()
    assert ok, detail["checks"]


def test_failclosed_abstain_on_empty_findings() -> None:
    fn = lambda system, user: "" if "specialist" in system else "Yes."
    assert sle.swarm_answer(fn, "Compare the disputed claim versus its rival, citing sources", SwarmRouter()) == sle.ABSTAIN


def test_swarm_makes_more_calls_than_solo() -> None:
    claims = ["Is false claim A accurate?"] * 5
    rep = sle.run_live(claims, lambda s, u: "No, false; no evidence.", subject="m")
    assert rep.calls > rep.n  # solo (n) + swarm (router fan-out) > n


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
