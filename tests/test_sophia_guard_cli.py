#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the sophia-guard CLI (tools/sophia_guard.py). Offline.

The CLI wraps guarded_complete so any local model can answer behind Sophia's
provenance gate. The model is injected so the repair/abstain paths run without a
live model, and the output dict must be JSON-serialisable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.model import ModelResult  # noqa: E402
from tools import sophia_guard as sg  # noqa: E402

RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
VIOLATING = "Alice wrote the Project Phoenix Charter."
CLEAN = "The Project Phoenix Charter was ratified by the committee."


def _gen(*responses):
    box = {"i": 0}

    def g(system, user):
        i = min(box["i"], len(responses) - 1)
        box["i"] += 1
        return ModelResult(text=responses[i], provider="mock", model="t", ok=True)

    return g


def _kw(**over):
    base = dict(records=RECORDS, retrieve_fn=lambda q, top_k=8: [], format_context_fn=lambda c: "(ctx)")
    base.update(over)
    return base


def test_run_clean() -> None:
    out = sg.run("q", on_fail="repair", generate=_gen(CLEAN), **_kw())
    assert out["action"] == "clean"
    assert out["ok"] is True and out["passed"] is True
    assert out["text"] == CLEAN and out["attempts"] == 1


def test_run_repairs() -> None:
    out = sg.run("q", on_fail="repair", generate=_gen(VIOLATING, CLEAN), **_kw())
    assert out["action"] == "repaired" and out["passed"] is True
    assert out["attempts"] == 2


def test_run_abstains_when_repair_fails() -> None:
    out = sg.run("q", on_fail="repair", generate=_gen(VIOLATING, VIOLATING), **_kw())
    assert out["action"] == "abstained" and out["ok"] is True


def test_output_is_json_serialisable() -> None:
    out = sg.run("q", on_fail="hedge", generate=_gen(VIOLATING), **_kw())
    json.dumps(out)  # must not raise
    assert set(out) >= {"text", "ok", "passed", "action", "attempts", "violations"}


def main() -> int:
    test_run_clean()
    test_run_repairs()
    test_run_abstains_when_repair_fails()
    test_output_is_json_serialisable()
    print("test_sophia_guard_cli: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
