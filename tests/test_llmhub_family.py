#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the llmhub aggregator wiring (preset + bare-id family detection).

The no-overclaim gate requires >=2 DISTINCT judge families. llmhub.com.cn serves
bare model ids (no 'vendor/' prefix), so without the name->family map two
genuinely-different vendors (gpt-4o + claude) would wrongly collapse to ONE
family behind the shared 'llmhub:' prefix — defeating the independence the gate
measures. These tests pin the honest behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.aggregate import (  # noqa: E402
    _distinct_families,
    _llmhub_family,
)


def test_llmhub_two_distinct_vendors_are_two_families():
    """The load-bearing case: gpt-4o (openai) + claude (anthropic) behind one
    llmhub key must count as TWO independent families."""
    assert _distinct_families(["llmhub:gpt-4o", "llmhub:claude-sonnet-4-6"]) == 2


def test_llmhub_two_same_vendor_collapse_to_one():
    """Negative control: two openai models must collapse to ONE family (no
    inflation of independence)."""
    assert _distinct_families(["llmhub:gpt-4o", "llmhub:gpt-4o-mini"]) == 1


def test_llmhub_family_map_covers_known_vendors():
    cases = {
        "claude-sonnet-4-6": "anthropic",
        "claude-opus-4-7": "anthropic",
        "gpt-4o": "openai",
        "gpt-5-mini": "openai",
        "gemini-2.5-pro": "google",
        "deepseek-v3.2": "deepseek",
        "qwen3-max": "qwen",
        "glm-5": "zhipu",
        "kimi-k2.5": "moonshot",
        "grok-4.3": "xai",
    }
    for model, fam in cases.items():
        assert _llmhub_family(model) == fam, f"{model} -> {_llmhub_family(model)} != {fam}"


def test_llmhub_unknown_model_is_own_family():
    """Conservative: an unmapped model is its OWN family — never collapses two
    unknowns into one (which would falsely inflate independence)."""
    assert _llmhub_family("totally-unknown-model-xyz") == "totally-unknown-model-xyz"
    # but two DIFFERENT unknowns are still 2 families
    assert _distinct_families(["llmhub:unknown-a", "llmhub:unknown-b"]) == 2


def test_llmhub_spec_with_base_url_suffix_still_resolves_family():
    """The per-spec '@base_url' suffix must be stripped before family detection."""
    assert _distinct_families(
        ["llmhub:gpt-4o@https://api.llmhub.com.cn/v1",
         "llmhub:claude-sonnet-4-6@https://api.llmhub.com.cn/v1"]
    ) == 2


def test_llmhub_preset_registered():
    """The llmhub preset must resolve to an OpenAI-compatible config pointing at
    the HTTPS endpoint with the LLMHUB_API_KEY env."""
    from agent.model import resolve_config

    cfg = resolve_config("llmhub:claude-sonnet-4-6")
    assert cfg.kind == "openai"
    assert cfg.base_url == "https://api.llmhub.com.cn/v1"
    assert cfg.api_key_env == "LLMHUB_API_KEY"
    assert cfg.model == "claude-sonnet-4-6"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
