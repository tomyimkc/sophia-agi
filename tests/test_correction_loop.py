#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Correction loop pipeline tests (no API keys required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import load_traditions, score_case, load_benchmark  # noqa: E402
from agent.correction_loop import find_failures  # noqa: E402

PENDING = ROOT / "training" / "corrections_pending"
REPORT = ROOT / "benchmark" / "model_runs" / "local-sophia-v1-psychology.report.json"


def test_find_stockholm_failure() -> None:
    assert REPORT.exists(), f"missing {REPORT}"
    failures = find_failures(REPORT)
    ids = {f["case_id"] for f in failures}
    assert "stockholm_every_kidnapping" in ids


def test_pending_correction_passes_scorer() -> None:
    path = PENDING / "correction-stockholm-every-kidnapping.json"
    assert path.exists(), f"missing {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer = next(m["content"] for m in payload["messages"] if m["role"] == "assistant")
    case = next(c for c in load_benchmark("psychology")["cases"] if c["id"] == "stockholm_every_kidnapping")
    ok, reasons = score_case(case, answer, load_traditions())
    assert ok, reasons
    assert payload["metadata"]["source"] == "correction-loop"


def main() -> int:
    test_find_stockholm_failure()
    test_pending_correction_passes_scorer()
    print("test_correction_loop: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())