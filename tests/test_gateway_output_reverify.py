# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Red-team tests for served-output re-verification (sophia_mcp.gateway_wiring).

Prove the opt-in output gate (SOPHIA_MCP_OUTPUT_VERIFY=1) actually withholds a served
payload that smuggles a fabricated attribution — content the pre-dispatch checks never
inspect — while leaving clean output untouched, and is a no-op when the flag is off.
The inference handler is mocked so the "model" output is deterministic and offline.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp import boundary, gateway_wiring as gw  # noqa: E402
from sophia_mcp import tools_impl as impl  # noqa: E402

_ENV_KEYS = ("SOPHIA_MCP_GATEWAY", "SOPHIA_MCP_OUTPUT_VERIFY", "SOPHIA_MCP_KILL_SWITCH",
             "SOPHIA_MCP_ROLE", "SOPHIA_MCP_CLEARANCE")

# A known attribution trap (matches the curated trap registry) + a fabricated answer.
_TRAP_Q = "Did Confucius write the Dao De Jing?"
_FABRICATION = "Yes, Confucius wrote the Dao De Jing."


@pytest.fixture(autouse=True)
def clean_env():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    gw.reset()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    gw.reset()


def test_flag_off_by_default():
    assert boundary.output_verify_enabled() is False


def test_fabricated_attribution_is_withheld(monkeypatch):
    os.environ["SOPHIA_MCP_GATEWAY"] = "1"
    os.environ["SOPHIA_MCP_OUTPUT_VERIFY"] = "1"
    monkeypatch.setattr(impl, "openclaw_infer",
                        lambda **k: {"ok": True, "text": _FABRICATION, "model": k.get("model")})

    resp = gw.governed("sophia_openclaw_infer", {"model": "mock", "prompt": _TRAP_Q})

    assert resp["result"] is None
    assert resp["held_reason"] == "output_failed_reverification"
    assert resp["reasons"]  # the specific gate violations are surfaced
    assert resp["_governance"]["outputVerified"] is False
    # the fabricated text is NOT returned to the caller
    assert _FABRICATION not in str(resp)


def test_clean_output_passes_through_verified(monkeypatch):
    os.environ["SOPHIA_MCP_GATEWAY"] = "1"
    os.environ["SOPHIA_MCP_OUTPUT_VERIFY"] = "1"
    clean = "Paris is the capital of France."
    monkeypatch.setattr(impl, "openclaw_infer",
                        lambda **k: {"ok": True, "text": clean, "model": k.get("model")})

    resp = gw.governed("sophia_openclaw_infer",
                       {"model": "mock", "prompt": "What is the capital of France?"})

    assert resp.get("text") == clean  # untouched
    assert resp["_governance"]["verdict"] == "accepted"
    assert resp["_governance"]["outputVerified"] is True


def test_verify_off_lets_fabrication_through(monkeypatch):
    # With the flag off, behavior is unchanged: the gateway still tags untrusted, but
    # the output is not re-verified (proves the feature is genuinely opt-in).
    os.environ["SOPHIA_MCP_GATEWAY"] = "1"
    monkeypatch.setattr(impl, "openclaw_infer",
                        lambda **k: {"ok": True, "text": _FABRICATION, "model": k.get("model")})

    resp = gw.governed("sophia_openclaw_infer", {"model": "mock", "prompt": _TRAP_Q})

    assert resp.get("text") == _FABRICATION
    assert resp["_governance"]["verdict"] == "accepted"
    assert "outputVerified" not in resp["_governance"]


def test_no_text_output_is_marked_uninspectable(monkeypatch):
    os.environ["SOPHIA_MCP_GATEWAY"] = "1"
    os.environ["SOPHIA_MCP_OUTPUT_VERIFY"] = "1"
    os.environ["SOPHIA_MCP_ROLE"] = "maintainer"  # export may require approval; just metadata here
    monkeypatch.setattr(impl, "export_corpus",
                        lambda *a, **k: {"ok": True, "path": "x.jsonl", "lines": 3})

    resp = gw.governed("sophia_export_corpus", {})

    assert resp.get("ok") is True
    assert resp["_governance"]["outputVerified"] is None  # nothing inspectable, not a failure


def test_extract_output_text_pulls_results_snippets():
    text = gw._output_text({"results": [{"text": "alpha"}, {"snippet": "beta"}], "answer": "gamma"})
    assert "alpha" in text and "beta" in text and "gamma" in text


if __name__ == "__main__":
    # Minimal offline runner (mirrors the repo's other test mains) without pytest fixtures.
    import types

    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            for k in _ENV_KEYS:
                os.environ.pop(k, None)
            gw.reset()
            try:
                fn(_MP()) if fn.__code__.co_argcount else fn()
                print(f"ok {name}")
            finally:
                for k in _ENV_KEYS:
                    os.environ.pop(k, None)
                gw.reset()
    print("all passed")
