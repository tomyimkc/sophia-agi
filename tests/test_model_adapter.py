#!/usr/bin/env python3
"""Tests for the unified model adapter (agent/model.py). All offline."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as m  # noqa: E402


def test_resolve_presets() -> None:
    assert m.resolve_config("mock").kind == "mock"
    glm = m.resolve_config("glm")
    assert glm.kind == "openai" and "bigmodel" in (glm.base_url or "")
    ollama = m.resolve_config("ollama:llama3.2")
    assert ollama.kind == "openai" and ollama.model == "llama3.2"
    try:
        m.resolve_config("does-not-exist")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_mock_generate_is_offline_and_structured() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    client = m.ModelClient(m.resolve_config("mock"))
    result = client.generate("system", "Should we ship the launch?")
    assert result.ok is True
    assert "Decision" in result.text and "中文摘要" in result.text
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    assert result.latency_sec >= 0


def test_mock_forced_response_and_streaming() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = "alpha beta gamma"
    tokens: list[str] = []
    try:
        client = m.ModelClient(m.resolve_config("mock"))
        result = client.generate("s", "u", on_token=tokens.append)
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert result.text == "alpha beta gamma"
    assert "".join(tokens).strip() == "alpha beta gamma"


def test_estimate_cost_known_and_unknown() -> None:
    cost, known = m.estimate_cost("glm-4.6", 1_000_000, 1_000_000)
    assert known is True and cost > 0
    cost2, known2 = m.estimate_cost("totally-unknown-model", 1000, 1000)
    assert known2 is False and cost2 == 0.0


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_openai_compatible_parses_and_costs(monkeypatch_env=None) -> None:
    os.environ["ZHIPUAI_API_KEY"] = "test-key"
    payload = {
        "model": "glm-4.6",
        "choices": [{"message": {"content": "Decision: yes. 中文摘要: 好"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }
    original = m.urllib.request.urlopen
    m.urllib.request.urlopen = lambda req, timeout=None: _FakeResp(payload)
    try:
        client = m.ModelClient(m.resolve_config("glm"))
        result = client.generate("s", "u")
    finally:
        m.urllib.request.urlopen = original
        os.environ.pop("ZHIPUAI_API_KEY", None)
    assert result.ok is True
    assert result.prompt_tokens == 1000 and result.completion_tokens == 500
    assert result.cost_usd > 0  # glm has a price entry
    assert "Decision" in result.text


def test_openai_missing_key_fails() -> None:
    os.environ.pop("ZHIPUAI_API_KEY", None)
    result = m.ModelClient(m.resolve_config("glm")).generate("s", "u")
    assert result.ok is False
    assert "API key" in (result.error or "")


def test_retry_then_fallback_to_mock() -> None:
    os.environ["ZHIPUAI_API_KEY"] = "test-key"

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url if hasattr(req, "full_url") else "url", 503, "Service Unavailable", {}, None)

    original = m.urllib.request.urlopen
    m.urllib.request.urlopen = boom
    try:
        client = m.ModelClient(m.resolve_config("glm"), [m.resolve_config("mock")], retries=2)
        result = client.generate("s", "u", sleep=lambda _: None)
    finally:
        m.urllib.request.urlopen = original
        os.environ.pop("ZHIPUAI_API_KEY", None)
    assert result.ok is True
    assert result.provider == "mock"
    assert result.fallback_used is True
    # primary tried `retries` times (transient) before falling back
    glm_attempts = [a for a in result.attempts if a.provider == "glm"]
    assert len(glm_attempts) == 2 and all(not a.ok for a in glm_attempts)


def test_complete_backward_compat_and_raises() -> None:
    os.environ["SOPHIA_MODEL_PROVIDER"] = "mock"
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    try:
        text = m.complete("s", "u")
        assert isinstance(text, str) and text
    finally:
        os.environ.pop("SOPHIA_MODEL_PROVIDER", None)
    # a provider with no key and no fallback should raise
    os.environ.pop("ZHIPUAI_API_KEY", None)
    os.environ.pop("SOPHIA_MODEL_FALLBACKS", None)
    try:
        m.complete("s", "u", spec="glm")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def main() -> int:
    test_resolve_presets()
    test_mock_generate_is_offline_and_structured()
    test_mock_forced_response_and_streaming()
    test_estimate_cost_known_and_unknown()
    test_openai_compatible_parses_and_costs()
    test_openai_missing_key_fails()
    test_retry_then_fallback_to_mock()
    test_complete_backward_compat_and_raises()
    print("test_model_adapter: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
