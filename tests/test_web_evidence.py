#!/usr/bin/env python3
"""Tests for Sophia web evidence adapters."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import web_evidence  # noqa: E402


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


def test_offline_gather_keeps_web_disabled() -> None:
    result = web_evidence.gather_evidence("Dao De Jing attribution", online=False, local_top_k=1)
    assert result["web"]["online"] is False
    assert result["web"]["sources"] == []
    assert "disabled" in result["warnings"][0]
    assert result["localSources"]


def test_brave_adapter_parses_results_and_redacts_key() -> None:
    original_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    original_urlopen = urllib.request.urlopen
    os.environ["BRAVE_SEARCH_API_KEY"] = "secret-brave-key"

    def fake_urlopen(request, timeout):
        assert request.full_url.startswith(web_evidence.BRAVE_ENDPOINT)
        assert request.headers["X-subscription-token"] == "secret-brave-key"
        return FakeResponse(
            b'{"web":{"results":[{"title":"Paper","url":"https://arxiv.org/abs/1234","description":"abstract"}]}}'
        )

    urllib.request.urlopen = fake_urlopen
    try:
        result = web_evidence.web_search("test query", online=True, provider="brave", top_k=1)
    finally:
        urllib.request.urlopen = original_urlopen
        if original_key is None:
            os.environ.pop("BRAVE_SEARCH_API_KEY", None)
        else:
            os.environ["BRAVE_SEARCH_API_KEY"] = original_key

    assert result["ok"] is True
    assert result["sources"][0]["quality"] == "academic"
    assert "secret-brave-key" not in str(result)


def test_tavily_adapter_uses_bearer_auth() -> None:
    original_key = os.environ.get("TAVILY_API_KEY")
    original_urlopen = urllib.request.urlopen
    os.environ["TAVILY_API_KEY"] = "secret-tavily-key"

    def fake_urlopen(request, timeout):
        assert request.full_url == web_evidence.TAVILY_ENDPOINT
        assert request.headers["Authorization"] == "Bearer secret-tavily-key"
        return FakeResponse(
            b'{"results":[{"title":"SEP","url":"https://plato.stanford.edu/entries/test","content":"entry","score":0.9}]}'
        )

    urllib.request.urlopen = fake_urlopen
    try:
        result = web_evidence.web_search("test query", online=True, provider="tavily", top_k=1)
    finally:
        urllib.request.urlopen = original_urlopen
        if original_key is None:
            os.environ.pop("TAVILY_API_KEY", None)
        else:
            os.environ["TAVILY_API_KEY"] = original_key

    assert result["ok"] is True
    assert result["sources"][0]["provider"] == "tavily"
    assert result["sources"][0]["score"] == 0.9


def test_serpapi_adapter_parses_organic_results() -> None:
    original_key = os.environ.get("SERPAPI_API_KEY")
    original_urlopen = urllib.request.urlopen
    os.environ["SERPAPI_API_KEY"] = "secret-serpapi-key"

    def fake_urlopen(request, timeout):
        assert request.full_url.startswith(web_evidence.SERPAPI_ENDPOINT)
        return FakeResponse(
            b'{"organic_results":[{"title":"Docs","link":"https://docs.python.org/3/","snippet":"official docs"}]}'
        )

    urllib.request.urlopen = fake_urlopen
    try:
        result = web_evidence.web_search("python docs", online=True, provider="serpapi", top_k=1)
    finally:
        urllib.request.urlopen = original_urlopen
        if original_key is None:
            os.environ.pop("SERPAPI_API_KEY", None)
        else:
            os.environ["SERPAPI_API_KEY"] = original_key

    assert result["ok"] is True
    assert result["sources"][0]["quality"] == "official-primary"


def main() -> int:
    test_offline_gather_keeps_web_disabled()
    test_brave_adapter_parses_results_and_redacts_key()
    test_tavily_adapter_uses_bearer_auth()
    test_serpapi_adapter_parses_organic_results()
    print("test_web_evidence: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
