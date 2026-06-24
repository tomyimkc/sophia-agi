#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for verifier-gated distillation (offline via mock teacher)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as m  # noqa: E402
from tools import distill_export as d  # noqa: E402


def test_accepts_verified_and_rejects_bad() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    client = m.ModelClient(m.resolve_config("mock"))
    # mock default answer contains "source discipline" + Decision + 中文 -> passes gate,
    # but mustInclude "Laozi" is absent -> rejected; a keyword the mock includes is accepted
    prompts = [
        {"id": "good", "prompt": "give a decision", "mustInclude": ["Decision"]},
        {"id": "bad", "prompt": "name the author", "mustInclude": ["Laozi"]},
    ]
    data = d.distill(prompts, client)
    by_id = {t["id"]: t for t in data["trajectory"]}
    assert by_id["good"]["accepted"] is True
    assert by_id["bad"]["accepted"] is False
    assert data["accepted"] == 1 and data["rejected"] == 1
    assert len(data["sft"]) == 1 and data["sft"][0]["metadata"]["source"] == "distillation"


def test_empty_teacher_output_rejected() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = ""
    try:
        client = m.ModelClient(m.resolve_config("mock"))
        data = d.distill([{"id": "e", "prompt": "x"}], client)
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert data["accepted"] == 0 and data["rejected"] == 1


def main() -> int:
    test_accepts_verified_and_rejects_bad()
    test_empty_teacher_output_rejected()
    print("test_distill_export: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
