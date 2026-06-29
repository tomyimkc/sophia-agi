#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Leiden open-judge layer (judge registry + self-hostable backend).

Pins the load-bearing invariants of the autonomy (value 5) work:

  (a) judge-id parsing and openness classification are correct, including the key
      distinction between open WEIGHTS and self-hostable INFERENCE;
  (b) the committed headline panel is classified as open-weights but NOT yet on a
      non-proprietary path (the honest gap the receipt reports);
  (c) the self-hostable backend is fail-closed: unconfigured -> available() False and
      score() returns None (never silently uses a proprietary judge);
  (d) with a configured endpoint and an injected transport (no network), the backend
      parses DISCIPLINED / UNDISCIPLINED verdicts deterministically.

Pure stdlib, deterministic, offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import judge_registry as jr  # noqa: E402
from agent import open_judge  # noqa: E402


def test_parse_and_classify() -> None:
    assert jr.parse_judge_id("openrouter:deepseek/deepseek-chat") == (
        "openrouter", "deepseek/deepseek-chat")
    assert jr.parse_judge_id("bare-model")[0] == "unknown"

    proprietary = jr.classify_judge("openrouter:meta-llama/llama-3.3-70b-instruct")
    assert proprietary["open_weights"] is True          # llama weights are open
    assert proprietary["self_hostable"] is False         # but served via a proprietary API
    assert proprietary["proprietary_inference"] is True
    assert proprietary["non_proprietary_path"] is False

    local = jr.classify_judge("local:qwen2.5-32b-instruct")
    assert local["self_hostable"] is True
    assert local["open_weights"] is True
    assert local["non_proprietary_path"] is True         # open weights + self-hosted


def test_committed_panel_is_open_weights_but_not_yet_non_proprietary() -> None:
    p = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "M3-pilot-judge.json"
    panel = json.loads(p.read_text(encoding="utf-8")).get("judges", [])
    summary = jr.classify_panel(panel)
    assert summary["n_total"] == len(panel) >= 2
    assert summary["has_open_weights_judge"] is True
    # the documented gap: no corroborating family runs on a non-proprietary path yet
    assert summary["has_non_proprietary_path"] is False


def test_open_judge_fail_closed_when_unconfigured() -> None:
    env: dict[str, str] = {}
    assert open_judge.available(env) is False
    # must NOT fall back to anything — returns None
    assert open_judge.score("a false claim", "the claim is true", env=env) is None


def test_open_judge_with_injected_transport() -> None:
    env = {"OPEN_JUDGE_BASE_URL": "http://localhost:8000/v1",
           "OPEN_JUDGE_MODEL": "qwen2.5-32b-instruct"}
    assert open_judge.available(env) is True
    assert open_judge.judge_id(env) == "local:qwen2.5-32b-instruct"

    def fake(verdict_word: str):
        def _tx(url, headers, payload):
            assert url.endswith("/chat/completions")
            assert payload["model"] == "qwen2.5-32b-instruct"
            return json.dumps({"choices": [{"message": {"content": verdict_word}}]})
        return _tx

    assert open_judge.score("c", "refuted / unsupported", env=env,
                            transport=fake("DISCIPLINED")) == 1
    assert open_judge.score("c", "yes that is true", env=env,
                            transport=fake("UNDISCIPLINED")) == 0
    # unparseable response -> None after retries, never a guess
    assert open_judge.score("c", "x", env=env,
                            transport=lambda u, h, p: "not json", retries=2) is None


def test_regrade_harness_offline() -> None:
    """The re-grade harness grades arms via the open judge and emits a non-proprietary
    receipt — deterministically, with an injected transport (no network)."""
    from tools import run_open_judge_regrade as rg

    env = {"OPEN_JUDGE_BASE_URL": "http://localhost:8000/v1",
           "OPEN_JUDGE_MODEL": "qwen2.5-32b-instruct"}
    result = {
        "model": "demo-base",
        "raw_generations": {
            "base": ["the claim is true", "yes definitely true", "true"],
            "seed0": ["that is unsupported / refuted", "I cannot verify this; refuted", "refuted"],
        },
    }
    claims = ["false claim A", "false claim B", "false claim C"]

    # judge: DISCIPLINED iff the answer mentions refutation/unsupported
    def tx(url, headers, payload):
        ans = payload["messages"][-1]["content"].lower()
        word = "DISCIPLINED" if ("refut" in ans or "unsupported" in ans) else "UNDISCIPLINED"
        return json.dumps({"choices": [{"message": {"content": word}}]})

    receipt = rg.regrade(result, claims, env=env, transport=tx)
    assert receipt["canClaimAGI"] is False
    assert receipt["judge_id"] == "local:qwen2.5-32b-instruct"
    assert receipt["non_proprietary_path"] is True            # open weights + self-hosted
    assert receipt["n_traps"] == 3
    # base all affirm (rate 0), seed0 all refute (rate 1) -> positive delta
    assert receipt["open_judge"]["mean_delta"] == 1.0
    assert receipt["open_judge"]["ci_excludes_zero"] is True
    assert "kappa_open_vs_lexical" in receipt and "ci95" in receipt["open_judge"]

    # fail-closed: no endpoint -> RuntimeError, never a proprietary fallback
    try:
        rg.regrade(result, claims, env={}, transport=tx)
        raise AssertionError("regrade should refuse without an endpoint")
    except RuntimeError:
        pass


def main() -> int:
    test_parse_and_classify()
    test_committed_panel_is_open_weights_but_not_yet_non_proprietary()
    test_open_judge_fail_closed_when_unconfigured()
    test_open_judge_with_injected_transport()
    test_regrade_harness_offline()
    print("test_judge_registry: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
