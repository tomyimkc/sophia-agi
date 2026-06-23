#!/usr/bin/env python3
"""Offline tests for OpenRouter hidden-eval adapter."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import openrouter_client as oc  # noqa: E402
from tools import run_hidden_eval_openrouter as runner  # noqa: E402


def test_load_api_key_from_file_not_repo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "key"
        p.write_text("secret", encoding="utf-8")
        assert oc.load_api_key(api_key_file=p) == "secret"


def test_extract_text() -> None:
    assert oc.extract_text({"choices": [{"message": {"content": "ok"}}]}) == "ok"


def test_runner_uses_mocked_chat_completion() -> None:
    pack = {"packId": "unit", "visibility": "revealed-after-eval", "cases": [{"id": "c1", "domain": "philosophy", "prompt": "Did Confucius write Dao De Jing?", "materials": [], "scoring": {"maxPoints": 1, "rubric": ["r"], "mustInclude": ["No"]}}]}
    old = runner.chat_completion
    try:
        runner.chat_completion = lambda **kw: {"choices": [{"message": {"content": "No."}}]}
        out = runner.run(pack, mode="sophia_full", model="mock/model", api_key_file=None, timeout_sec=1, limit=None)
    finally:
        runner.chat_completion = old
    assert out["responses"]["c1"] == "No."
    assert out["mode"] == "sophia_full"


def main() -> int:
    test_load_api_key_from_file_not_repo()
    test_extract_text()
    test_runner_uses_mocked_chat_completion()
    print("test_openrouter_hidden_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
