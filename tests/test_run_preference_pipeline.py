#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Preference-pipeline orchestrator: both paths run end-to-end offline + decontaminate.

Proves the orchestrator composes the stage tools (generate/ingest -> label ->
decontaminate) without reimplementing them, and that the decontamination gate is
fail-closed (an eval-prompt leak => clean=False, which the CLI surfaces as exit 2).
No model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_preference_pipeline import (  # noqa: E402
    _decontam_report, pipeline_generate_label, pipeline_ingest_foreign, self_test)
from tools.gen_preference_candidates import (  # noqa: E402
    SELF_TEST_ROWS as GEN_FIXTURES, _fake_complete_factory)
from tools.ingest_foreign_tool_fuel import SELF_TEST_ROWS as FOREIGN_FIXTURES  # noqa: E402


def test_self_test_passes() -> None:
    assert self_test() == 0


def test_generate_label_path_mints_and_decontaminates() -> None:
    script = {"Socrates": [
        "No — Socrates wrote nothing himself; The Republic was written by Plato.",
        "Yes, Socrates wrote The Republic.",
    ]}
    fake = _fake_complete_factory(script)
    r = pipeline_generate_label(tasks=GEN_FIXTURES, n=2, complete_fn=fake, spec=None)
    assert r["path"] == "generate-label"
    assert r["label"]["pairs"] >= 1
    assert r["decontam"]["clean"] is True
    for p in r["pairs"]:
        assert p["metadata"]["label_source"] == "machine_verified"


def test_ingest_foreign_path_partial_coverage() -> None:
    r = pipeline_ingest_foreign(rows=FOREIGN_FIXTURES, source="ToolACE-shape")
    assert r["path"] == "ingest-foreign"
    # honest partial coverage: one trace mints, one is skipped (no mapping)
    assert r["ingest"]["pairs"] >= 1
    assert r["ingest"]["reasons"].get("all_candidates_skipped", 0) >= 1
    assert r["decontam"]["clean"] is True


def test_decontam_is_fail_closed_on_eval_prompt_leak() -> None:
    # A row whose prompt IS a held-out eval prompt must NOT pass decontamination.
    from provenance_bench.dataset_guard import eval_prompt_set
    ev = next(iter(eval_prompt_set()))
    report = _decontam_report([{"prompt": ev, "chosen": "a", "rejected": "b", "metadata": {}}])
    assert report["clean"] is False
    assert report["overlapCount"] >= 1


def test_unique_prompts_counted() -> None:
    rows = [
        {"prompt": "p1", "chosen": "a", "rejected": "b", "metadata": {}},
        {"prompt": "p1", "chosen": "a", "rejected": "b2", "metadata": {}},  # dup prompt
        {"prompt": "p2", "chosen": "a", "rejected": "b", "metadata": {}},
    ]
    report = _decontam_report(rows)
    assert report["uniquePrompts"] == 2


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} run_preference_pipeline tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
